"""Reference axis of the coil ordering (slot) rows: frozen at the start of every continuation step (default) or
live. Loop bookkeeping on the stub loop (recomputed at each step, stored in checkpoints, reused by a resume inside
the step); on real coils: rows invariant to surface changes within a step, and the envelope gradient consistent
with finite differences of the coil Lagrangian at the frozen axis."""
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from phrport import phr as P
from phrport import backend as B
import stubs as S

ok, why = B.xcoil_available()


class AxisStubCoilSolver(S.StubCoilSolver):
    """Stub coil solver with the frozen-axis interface of phrport.coils.CoilSolver (axis = the boundary's m = 0
    modes rc(0, n), -zs(0, n), n = 0..2)."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.cfg = SimpleNamespace(ordering_axis="frozen_step", slot_rows=True, nfp=2)
        self.calls, self._ax = [], None

    def set_ordering_axis(self, axis=None, boundary=None):
        if boundary is not None:
            axis = [[float(boundary.get(f"rc(0,{n})", 0.0)) for n in range(3)],
                    [-float(boundary.get(f"zs(0,{n})", 0.0)) for n in range(3)]]
            self.calls.append(("boundary", axis))
        else:
            self.calls.append(("checkpoint", axis))
        self._ax = [list(axis[0]), list(axis[1])]
        return self._ax

    @property
    def ordering_axis(self):
        return self._ax


def _runner(out, cs, **cfgkw):
    eq = S.StubEquilibrium()
    cfg = P.PHRConfig(**dict(dict(flux_metric_scale=100.0, n_continuation_steps=2, al_outer_iter=2, max_nfev=4),
                             **cfgkw))
    return P.PHRRunner(cfg, evaluate=eq.evaluate, evaluate_many=S.StubPool(eq).evaluate_many, coil_solver=cs,
                       free_dof_names=S.free_dof_names, seed_boundary=S.seed_boundary(), out_dir=str(out),
                       log_to_stdout=False)


def test_axis_frozen_per_step_checkpointed_and_resumed(tmp_path):
    cs = AxisStubCoilSolver()
    out = tmp_path / "run"
    _runner(out, cs).run()
    assert [k for k, _ in cs.calls] == ["boundary", "boundary"]                  # once per continuation step
    ck0 = json.load(open(out / "al1_checkpoint_step00_iter01.json"))
    ck1 = json.load(open(out / "al1_checkpoint_step01_iter01.json"))
    assert ck0["ordering_axis"] == cs.calls[0][1] and ck1["ordering_axis"] == cs.calls[1][1]
    # a resume inside step 0 reuses the checkpoint's axis; the next step recomputes it
    cs2 = AxisStubCoilSolver()
    r = _runner(tmp_path / "res", cs2)
    r.resume_from = str(out / "al1_checkpoint_step00_iter00.json")
    r.run()
    assert cs2.calls[0] == ("checkpoint", json.load(open(out / "al1_checkpoint_step00_iter00.json"))["ordering_axis"])
    assert cs2.calls[1][0] == "boundary"
    # --start-step: a new step, so the axis comes from the boundary
    cs3 = AxisStubCoilSolver()
    r = _runner(tmp_path / "ss", cs3, start_step=1)
    r.resume_from = str(out / "al1_checkpoint_step00_iter01.json")
    r.run()
    assert cs3.calls[0][0] == "boundary"


@pytest.fixture(scope="module")
def active():
    """PHR seed, ordering rows with a 15 degree gap (above the natural 12.7: rows active, multipliers > 0)."""
    from phrport import equilibrium as E
    from phrport.coils import CoilConfig, CoilSolver
    b = E.seed_boundary()
    out = {}
    for mode in ("frozen_step", "live"):
        cs = CoilSolver(CoilConfig(slot_rows=True, slot_gap_deg=15.0, ordering_axis=mode))
        cs.set_ordering_axis(boundary=b)
        r = cs.solve(b, warm=None)
        out[mode] = (cs, r)
    return b, out


def _moved(b, h, names=("rc(0,1)", "zs(0,1)", "rc(1,1)")):
    b2 = dict(b)
    for k in names:
        b2[k] = b2.get(k, 0.0) + h
    return b2


@pytest.mark.xcoil
@pytest.mark.skipif(not ok, reason=f"XCoil unavailable: {why}")
def test_rows_invariant_within_a_step(active):
    b, out = active
    for mode, (cs, r) in out.items():
        sl = cs.P.blocks["slot"]
        g0 = cs.lagrangian_parts(b, r.x)[1][sl]
        g1 = cs.lagrangian_parts(_moved(b, 2e-3), r.x)[1][sl]
        if mode == "frozen_step":
            assert np.array_equal(g0, g1)                       # coils only
        else:
            assert np.max(np.abs(g1 - g0)) > 1e-6               # the live axis moves with the surface


@pytest.mark.xcoil
@pytest.mark.skipif(not ok, reason=f"XCoil unavailable: {why}")
def test_envelope_gradient_matches_lagrangian_fd(active):
    """d/ds [c0 + c0_ref lam^T g] at fixed coils and multipliers: the envelope gradient against central differences,
    in both modes (each about its own axis definition)."""
    b, out = active
    names = ["rc(0,1)", "zs(0,1)", "rc(1,1)", "zs(1,1)"]
    for mode, (cs, r) in out.items():
        assert np.max(r.lam[cs.P.blocks["slot"]]) > 0           # ordering rows active
        G = np.asarray(cs.envelope_grad(b, r, names))
        lam = np.asarray(r.lam)
        L = lambda bb: (lambda c0, g: c0 + cs.cfg.c0_ref * float(lam @ g))(*cs.lagrangian_parts(bb, r.x, names))
        h = 1e-6
        fd = np.array([(L(_moved(b, h, (n,))) - L(_moved(b, -h, (n,)))) / (2 * h) for n in names])
        assert np.max(np.abs(G - fd)) <= 1e-5 * np.max(np.abs(fd)), (mode, G, fd)
