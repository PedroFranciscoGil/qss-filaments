"""Persistent pool of VMEX worker processes (parallel cold solves for the FD Jacobian).

PHR spreads the ``n + 1`` forward-difference VMEC solves of every Jacobian over 8
MPI groups. The VMEX environment has no mpi4py, so this module keeps
``n_workers`` long-lived worker interpreters, each of which imports
:mod:`phrport.equilibrium` once (JAX/XLA warm-up and compilation are paid once
per worker and step, not per solve).

Design
------
* Workers are fresh interpreters started with ``subprocess`` (spawn semantics:
  no fork of a JAX process, and -- unlike ``multiprocessing``'s spawn -- the
  parent's ``__main__`` is never re-imported). Each runs
  ``python -m phrport.vmex_pool --worker`` with an explicit environment:
  ``PYTHONPYCACHEPREFIX`` (= ``phrport.PYCACHE_PREFIX``), ``JAX_ENABLE_X64=1``
  and the thread caps ``PJRT_NPROC`` (XLA:CPU intra-op pool; the classic
  ``--xla_cpu_multi_thread_eigen`` flag is ignored by this jaxlib) plus
  ``OMP/OPENBLAS/MKL/VECLIB_*`` = ``threads_per_worker``. Results do not
  depend on the thread count (verified bitwise), so pool and serial
  :func:`phrport.equilibrium.evaluate` agree bitwise.
* Messages are length-prefixed pickles over the worker's stdin/stdout; the
  worker moves its own fd 1 to stderr so library prints cannot corrupt the
  protocol.
* :meth:`VmexPool.evaluate_many` preserves order and load-balances dynamically
  (one dispatcher thread per worker pulling from a shared queue).
* A solver failure is an ordinary ``EqResult(converged=False)`` from the
  worker. A worker crash (process death, broken pipe) or a timeout yields
  ``converged=False`` for the item it was running; the worker is killed and
  respawned and the batch continues.
* Opt-in (roadmap S-c): ``VmexPool(return_state=True)`` or
  ``evaluate_many(items, return_state=True)`` makes the workers attach the
  converged VMEX state to each converged result (``EqResult.vmex_state``, a
  numpy-only :class:`phrport.equilibrium.VmexStateArrays`, ~144 kB at mode 4,
  pickled as one float64 block), which
  :class:`phrport.exact_jacobian.ExactEquilibriumJacobian` accepts as ``state=``
  instead of re-solving. Off by default: the default ``eval`` message and the
  returned results are unchanged.
"""
# (phrport/__init__.py sets sys.pycache_prefix before this module is imported)
from __future__ import annotations

import os
import pickle
import queue
import select
import struct
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

from phrport import PYCACHE_PREFIX   # $PHRPORT_PYCACHE, else <repo>/.pycache (phrport/__init__.py)

SANDBOX_ROOT = Path(__file__).resolve().parents[1]   # code location: worker cwd and PYTHONPATH

_HEADER = struct.Struct("<Q")


# ---------------------------------------------------------------------------
# framing helpers (raw fds, no buffering surprises)
# ---------------------------------------------------------------------------
class WorkerDied(RuntimeError):
    pass


class WorkerTimeout(RuntimeError):
    pass


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        n = os.write(fd, view)
        view = view[n:]


def _send(fd: int, obj) -> None:
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    _write_all(fd, _HEADER.pack(len(data)) + data)


def _read_exact(fd: int, n: int, deadline: float | None) -> bytes:
    chunks, got = [], 0
    while got < n:
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise WorkerTimeout("timed out waiting for worker")
            ready, _, _ = select.select([fd], [], [], remaining)
            if not ready:
                raise WorkerTimeout("timed out waiting for worker")
        chunk = os.read(fd, n - got)
        if not chunk:
            raise WorkerDied("worker closed its pipe (process died)")
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


def _recv(fd: int, deadline: float | None = None):
    (n,) = _HEADER.unpack(_read_exact(fd, _HEADER.size, deadline))
    return pickle.loads(_read_exact(fd, n, deadline))


# ---------------------------------------------------------------------------
# worker side
# ---------------------------------------------------------------------------
def _worker_main() -> None:
    in_fd = sys.stdin.fileno()
    out_fd = os.dup(sys.stdout.fileno())     # private protocol channel
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())  # stray prints -> stderr
    sys.stdout = sys.stderr
    from phrport import equilibrium as E
    # Warm the heavy imports before announcing readiness.
    from vmex import optimize as _opt  # noqa: F401  (lazy attribute of vmex)
    E._base_input()
    E._qs_objective()
    _send(out_fd, ("ready", os.getpid()))
    while True:
        try:
            msg = _recv(in_fd)
        except (WorkerDied, EOFError, OSError):
            break
        kind = msg[0]
        if kind == "stop":
            break
        if kind == "ping":
            _send(out_fd, ("pong", os.getpid()))
        elif kind == "eval":
            _, idx, boundary, step, *rest = msg       # optional 5th item: evaluate() options
            opts = dict(rest[0]) if rest and rest[0] else {}
            t0 = time.perf_counter()
            try:
                res = E.evaluate(boundary, step, **opts) if opts else E.evaluate(boundary, step)
            except Exception:
                res = E.failure_result("worker exception: " + traceback.format_exc()[-3000:])
            res.info["worker_pid"] = os.getpid()
            res.info["t_worker"] = time.perf_counter() - t0
            _send(out_fd, ("result", idx, res))
        else:
            _send(out_fd, ("error", f"unknown message {kind!r}"))


# ---------------------------------------------------------------------------
# parent side
# ---------------------------------------------------------------------------
def _validate_item(boundary: dict, step: int) -> None:
    """Same input checks as equilibrium.vmec_input (raise in the caller, not in a worker)."""
    import math
    from .equilibrium import parse_name, max_mode, SURF_MPOL, SURF_NTOR
    if "rc(0,0)" not in boundary:
        raise ValueError("boundary must be a FULL boundary dict (missing 'rc(0,0)')")
    for name, val in boundary.items():
        kind, m, n = parse_name(name)
        if m < 0 or (m == 0 and n < 0) or (kind == "zs" and m == 0 and n == 0):
            raise ValueError(f"{name!r} is not a stellsym SurfaceRZFourier DOF")
        if not math.isfinite(float(val)):
            raise ValueError(f"non-finite boundary coefficient {name}={val!r}")
    if not (0 <= int(step) and max_mode(step) <= min(SURF_MPOL, SURF_NTOR)):
        raise ValueError(f"bad continuation step {step}")


class _Worker:
    def __init__(self, pool: "VmexPool", index: int):
        self.pool, self.index = pool, index
        self.proc: subprocess.Popen | None = None
        self.pid: int | None = None
        self.spawns = 0
        self.tasks = 0                 # items evaluated by the current process
        self.busy_since: float | None = None

    def start(self) -> None:
        p = self.pool
        self.proc = subprocess.Popen(
            [p.python, "-u", "-m", "phrport.vmex_pool", "--worker"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=p._stderr_target(self.index),
            env=p._worker_env(), cwd=str(SANDBOX_ROOT), bufsize=0, close_fds=True,
        )
        self.spawns += 1
        self.tasks = 0
        self.pid = None

    def wait_ready(self, timeout: float) -> None:
        msg = _recv(self.proc.stdout.fileno(), time.monotonic() + timeout)
        if msg[0] != "ready":
            raise WorkerDied(f"unexpected handshake {msg!r}")
        self.pid = int(msg[1])

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def kill(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.poll() is None:
                self.proc.kill()
            self.proc.wait(timeout=10)
        except Exception:
            pass
        for s in (self.proc.stdin, self.proc.stdout):
            try:
                s.close()
            except Exception:
                pass
        self.proc = None
        self.pid = None

    def respawn(self) -> None:
        self.kill()
        self.start()
        self.wait_ready(self.pool.start_timeout)

    def stop(self, timeout: float = 10.0) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.poll() is None:
                _send(self.proc.stdin.fileno(), ("stop",))
                self.proc.wait(timeout=timeout)
        except Exception:
            pass
        self.kill()

    def evaluate(self, idx: int, boundary: dict, step: int, timeout: float, opts: dict | None = None):
        msg = ("eval", idx, dict(boundary), int(step))
        _send(self.proc.stdin.fileno(), msg + (dict(opts),) if opts else msg)
        self.busy_since = time.monotonic()
        try:
            msg = _recv(self.proc.stdout.fileno(), time.monotonic() + timeout)
        finally:
            self.busy_since = None
        if msg[0] != "result" or msg[1] != idx:
            raise WorkerDied(f"protocol error: {msg[:2]!r}")
        return msg[2]


class VmexPool:
    """Persistent pool of VMEX workers; ``evaluate_many`` keeps the input order.

    Parameters
    ----------
    n_workers : number of worker processes (PHR used 8 MPI groups).
    threads_per_worker : XLA/BLAS threads per worker (``PJRT_NPROC`` etc.).
    timeout : seconds allowed for one solve before the worker is killed and the
        item reported as failed (VMEX's NITER caps make this a safety net).
    log_dir : if given, each worker's stderr goes to ``log_dir/worker_<i>.log``;
        otherwise it is inherited.
    recycle_after : if set, a worker is replaced by a fresh process after this
        many items (bounds slow allocator growth in multi-hour runs; RSS grew
        ~600 -> 660 MB over 60 solves and then flattened). Default: never.
    return_state : opt-in default for ``evaluate_many(..., return_state=None)``:
        converged results carry ``vmex_state`` (see the module docstring).
        Default False (results exactly as before).

    ``evaluate_many`` calls are serialized (thread-safe); ``close()`` (or the
    context manager) stops the workers.
    """

    def __init__(self, n_workers: int = 8, threads_per_worker: int = 1, *,
                 timeout: float = 900.0, start_timeout: float = 300.0,
                 log_dir: str | os.PathLike | None = None, python: str | None = None,
                 recycle_after: int | None = None, return_state: bool = False):
        if n_workers < 1:
            raise ValueError("n_workers must be >= 1")
        self.n_workers = int(n_workers)
        self.threads_per_worker = int(threads_per_worker)
        self.timeout = float(timeout)
        self.start_timeout = float(start_timeout)
        self.python = python or sys.executable
        self.recycle_after = int(recycle_after) if recycle_after else None
        self.return_state = bool(return_state)
        self.log_dir = Path(log_dir) if log_dir is not None else None
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self._logs: list = []
        self._lock = threading.Lock()
        self._batch_lock = threading.Lock()
        self._closed = False
        self.stats = dict(batches=0, items=0, failures=0, crashes=0, timeouts=0,
                          respawns=0, recycled=0, wall=0.0)
        self.workers = [_Worker(self, i) for i in range(self.n_workers)]
        t0 = time.perf_counter()
        for w in self.workers:
            w.start()
        try:
            for w in self.workers:
                w.wait_ready(self.start_timeout)
        except Exception:
            self.close()
            raise
        self.startup_seconds = time.perf_counter() - t0

    # -- environment -----------------------------------------------------
    def _worker_env(self) -> dict:
        env = dict(os.environ)
        t = str(self.threads_per_worker)
        env.update(
            PYTHONPYCACHEPREFIX=PYCACHE_PREFIX, JAX_ENABLE_X64="1",
            PJRT_NPROC=t, OMP_NUM_THREADS=t, OPENBLAS_NUM_THREADS=t, MKL_NUM_THREADS=t,
            VECLIB_MAXIMUM_THREADS=t, NUMEXPR_NUM_THREADS=t,
            PYTHONUNBUFFERED="1",
        )
        if env.get("PHRPORT_VMEX_DEVICE", "").lower() == "cpu":
            env["JAX_PLATFORMS"] = "cpu"   # VMEX-only workers never need the GPU
        pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(SANDBOX_ROOT) + (os.pathsep + pp if pp else "")
        return env

    def _stderr_target(self, index: int):
        if self.log_dir is None:
            return None
        f = open(self.log_dir / f"worker_{index}.log", "ab")
        self._logs.append(f)
        return f

    # -- public API --------------------------------------------------------
    @property
    def pids(self) -> list:
        return [w.pid for w in self.workers]

    def evaluate_many(self, items, *, return_state: bool | None = None):
        """Evaluate ``[(boundary, step), ...]``; returns ``[EqResult, ...]`` in input order.

        Never raises for solver failures or worker crashes (those items come back
        with ``converged=False``); raises ``ValueError`` for malformed boundaries,
        like :func:`phrport.equilibrium.evaluate`.

        ``return_state`` (None -> the pool's ``return_state``): converged results
        carry the converged VMEX state as ``EqResult.vmex_state`` (opt-in).
        """
        rs = self.return_state if return_state is None else bool(return_state)
        with self._batch_lock:
            return self._evaluate_many(items, dict(return_state=True) if rs else None)

    def _evaluate_many(self, items, opts=None):
        from .equilibrium import failure_result
        if self._closed:
            raise RuntimeError("pool is closed")
        items = [(b, int(s)) for b, s in items]
        for b, s in items:
            _validate_item(b, s)
        n = len(items)
        results: list = [None] * n
        if n == 0:
            return results
        todo: "queue.Queue[int]" = queue.Queue()
        for i in range(n):
            todo.put(i)
        t0 = time.perf_counter()

        def run(worker: _Worker) -> None:
            while True:
                try:
                    i = todo.get_nowait()
                except queue.Empty:
                    return
                boundary, step = items[i]
                if not worker.alive():
                    try:
                        worker.respawn()
                        with self._lock:
                            self.stats["respawns"] += 1
                    except Exception as exc:
                        results[i] = failure_result(f"worker {worker.index} could not be (re)started: {exc}")
                        continue
                try:
                    res = worker.evaluate(i, boundary, step, self.timeout, opts)
                    res.info["worker"] = worker.index
                    results[i] = res
                    worker.tasks += 1
                    if self.recycle_after and worker.tasks >= self.recycle_after:
                        worker.stop()
                        with self._lock:
                            self.stats["recycled"] += 1
                except WorkerTimeout:
                    with self._lock:
                        self.stats["timeouts"] += 1
                    results[i] = failure_result(
                        f"worker {worker.index} (pid {worker.pid}) timed out after {self.timeout:.0f}s; respawned",
                        dict(worker=worker.index))
                    worker.kill()
                except (WorkerDied, BrokenPipeError, OSError, EOFError, pickle.UnpicklingError) as exc:
                    with self._lock:
                        self.stats["crashes"] += 1
                    rc = None
                    if worker.proc is not None:
                        try:
                            rc = worker.proc.wait(timeout=2.0)
                        except Exception:
                            rc = worker.proc.poll()
                    results[i] = failure_result(
                        f"worker {worker.index} (pid {worker.pid}) crashed (exit {rc}): {exc}; respawned",
                        dict(worker=worker.index))
                    worker.kill()
                if not worker.alive():           # respawn eagerly so it is warm for the next item
                    try:
                        worker.respawn()
                        with self._lock:
                            self.stats["respawns"] += 1
                    except Exception:
                        pass                      # retried lazily at the next item

        threads = [threading.Thread(target=run, args=(w,), daemon=True) for w in self.workers]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        for i in range(n):          # defensive: nothing may stay None
            if results[i] is None:
                results[i] = failure_result("internal pool error: item not evaluated")
        with self._lock:
            self.stats["batches"] += 1
            self.stats["items"] += n
            self.stats["failures"] += sum(not r.converged for r in results)
            self.stats["wall"] += time.perf_counter() - t0
        return results

    def warmup(self, steps=(0, 1, 2, 3)) -> float:
        """Compile every worker for the given steps (one seed solve per worker and step)."""
        from .equilibrium import seed_boundary
        seed = dict(seed_boundary())
        t0 = time.perf_counter()
        for s in steps:
            # one item per worker: the dynamic queue gives each idle worker one item
            self._broadcast_eval(seed, int(s))
        return time.perf_counter() - t0

    def _broadcast_eval(self, boundary: dict, step: int) -> list:
        out = [None] * self.n_workers

        def one(k: int) -> None:
            w = self.workers[k]
            try:
                if not w.alive():
                    w.respawn()
                out[k] = w.evaluate(-1 - k, boundary, step, self.timeout)
            except Exception as exc:
                out[k] = exc
                w.kill()

        ths = [threading.Thread(target=one, args=(k,), daemon=True) for k in range(self.n_workers)]
        for th in ths:
            th.start()
        for th in ths:
            th.join()
        return out

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for w in self.workers:
            w.stop()
        for f in self._logs:
            try:
                f.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


if __name__ == "__main__":
    if "--worker" in sys.argv[1:]:
        _worker_main()
    else:
        print("usage: python -m phrport.vmex_pool --worker  (internal)", file=sys.stderr)
        sys.exit(2)
