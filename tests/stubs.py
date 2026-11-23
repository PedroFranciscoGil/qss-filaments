"""Cheap analytic STUB components with the phrport interfaces (equilibrium / pool / coils).

Only for exercising phrport.phr: the 'physics' is a smooth toy model on the real
simsopt SurfaceRZFourier DOF layout of input.nfp2_QA (mpol = ntor = 5, nfp = 2,
stellsym: 121 DOFs; free DOFs per continuation step 8 / 24 / 48 / 80).
"""
import collections
import zlib
from dataclasses import dataclass, field

import numpy as np

MPOL = 5
NTOR = 5
NFP = 2
SEED_ELLIPSE = 0.07


def _names_helper(prefix, include0, mpol=MPOL, ntor=NTOR):
    names = [f"{prefix}(0,0)"] if include0 else []
    names += [f"{prefix}(0,{n})" for n in range(1, ntor + 1)]
    for m in range(1, mpol + 1):
        names += [f"{prefix}({m},{n})" for n in range(-ntor, ntor + 1)]
    return names


def all_dof_names(mpol=MPOL, ntor=NTOR):
    """simsopt SurfaceRZFourier (stellsym) local DOF order: rc block then zs block."""
    return _names_helper("rc", True, mpol, ntor) + _names_helper("zs", False, mpol, ntor)


def parse(name):
    pre, rest = name.split("(")
    m, n = rest.rstrip(")").split(",")
    return pre, int(m), int(n)


def free_dof_names(step, mpol=MPOL, ntor=NTOR):
    """PHR mask: fix_all; fixed_range(0..mm, -mm..mm, fixed=False); fix rc(0,0); mm = step+1."""
    mm = step + 1
    out = []
    for nm in all_dof_names(mpol, ntor):
        pre, m, n = parse(nm)
        if m <= mm and abs(n) <= mm and nm != "rc(0,0)":
            out.append(nm)
    return out


def seed_boundary():
    b = collections.OrderedDict((k, 0.0) for k in all_dof_names())
    b["rc(0,0)"] = 1.0
    b["rc(1,0)"] = 0.2
    b["zs(1,0)"] = 0.2
    b["rc(2,1)"] += SEED_ELLIPSE
    b["zs(2,1)"] += SEED_ELLIPSE
    return b


@dataclass
class StubEqResult:
    converged: bool
    qs_residuals: np.ndarray
    qs_total: float
    aspect: float
    mean_iota: float
    iota_axis: float
    iota_edge: float
    elongation: float
    well: float
    message: str = ""
    info: dict = field(default_factory=dict)


def _fail(msg):
    nan = float("nan")
    return StubEqResult(False, np.zeros(0), nan, nan, nan, nan, nan, nan, nan, msg)


NAMES = all_dof_names()
IDX = {k: i for i, k in enumerate(NAMES)}


def _vec(boundary):
    return np.array([float(boundary.get(k, 0.0)) for k in NAMES])


class StubEquilibrium:
    """evaluate(boundary, step) -> StubEqResult. Toy QS objective that prefers the
    axisymmetric ellipse; mean iota needs the (1,1)/(2,2)/(3,3) helical modes;
    elongation grows with the (1,1) mode; 'VMEC' fails for large deformations."""

    def __init__(self, n_qs=40, seed=7, fail_step=None, fail_fn=None, band33=0.09,
                 sporadic_fail_mod=None):
        rng = np.random.default_rng(seed)
        w = np.array([1.0 / (1.0 + parse(k)[1] + abs(parse(k)[2])) for k in NAMES])
        self.A = rng.standard_normal((n_qs, len(NAMES))) * w[None, :] * 0.15
        self.B = rng.standard_normal((n_qs, len(NAMES))) * w[None, :] * 0.05
        self.vstar = _vec(seed_boundary())
        self.vstar[IDX["rc(2,1)"]] = 0.0
        self.vstar[IDX["zs(2,1)"]] = 0.0
        self.vstar[IDX["rc(1,0)"]] = 0.22
        self.vstar[IDX["zs(1,0)"]] = 0.22
        self.band33 = band33
        self.sporadic_fail_mod = sporadic_fail_mod
        self.fail_step = fail_step
        self.fail_fn = fail_fn
        self.calls = 0

    def iota(self, v):
        g = lambda k: v[IDX[k]]  # noqa: E731
        return (0.0403 + 1.2 * (g("rc(1,1)") + g("zs(1,1)")) + 2.0 * (g("rc(2,2)") + g("zs(2,2)"))
                + 2.5 * (g("rc(3,3)") + g("zs(3,3)")) + 3.0 * (g("rc(4,4)") + g("zs(4,4)")))

    def __call__(self, boundary, step):
        return self.evaluate(boundary, step)

    def evaluate(self, boundary, step):
        self.calls += 1
        v = _vec(boundary)
        g = lambda k: v[IDX[k]]  # noqa: E731
        if self.fail_step is not None and step >= self.fail_step:
            return _fail(f"stub: forced failure at step {step}")
        if self.fail_fn is not None and self.fail_fn(boundary, step):
            return _fail("stub: fail_fn")
        if self.sporadic_fail_mod and zlib.crc32(v.tobytes()) % self.sporadic_fail_mod == 0:
            return _fail("stub: sporadic (hash-deterministic) failure")
        others = np.delete(np.abs(v), IDX["rc(0,0)"])
        h11 = g("rc(1,1)") ** 2 + g("zs(1,1)") ** 2
        if (np.max(others) > 0.6 or g("rc(1,0)") < 0.05 or g("zs(1,0)") < 0.05 or h11 > 0.12
                or g("rc(3,3)") > self.band33):
            return _fail("stub: 'VMEC' did not converge (large deformation)")
        d = v - self.vstar
        qs = self.A @ d + (self.B @ d) ** 2
        qs = qs * (1.0 + 0.01 * step)            # mild resolution dependence
        a = 0.5 * (g("rc(1,0)") + g("zs(1,0)"))
        aspect = 0.896 * g("rc(0,0)") / a * (1.0 + 0.3 * float(np.sum(others ** 2)) - 0.3 * 0.0698)
        iota = self.iota(v)
        elong = 1.5 + 50.0 * h11 + 30.0 * (g("rc(1,0)") - g("zs(1,0)")) ** 2 \
            + 10.0 * (g("rc(2,2)") ** 2 + g("zs(2,2)") ** 2)
        well = -0.0167 + 0.2 * g("rc(2,0)") - 0.5 * h11
        return StubEqResult(True, qs, float(np.sum(qs ** 2)), float(aspect), float(iota),
                            float(iota - 0.008), float(iota + 0.022), float(elong), float(well))


class StubPool:
    """VmexPool-like: evaluate_many(items) -> list (order preserved)."""

    def __init__(self, evaluator):
        self.ev = evaluator
        self.batches = []

    def evaluate_many(self, items):
        self.batches.append(len(items))
        return [self.ev.evaluate(b, s) for b, s in items]

    def close(self):
        pass


@dataclass
class StubCoilResult:
    ok: bool
    c0_local: float
    flux_F: float
    x: np.ndarray
    currents: np.ndarray
    lam: np.ndarray
    certified: bool
    linked: bool
    info: dict = field(default_factory=dict)


class StubCoilSolver:
    """c0_local(s): smooth positive toy flux; coil-friendly shapes have small helical
    content. Warm starts return the same optimum (unique minimiser) but are counted."""

    WEIGHTS = {"rc(1,1)": 40.0, "zs(1,1)": 40.0, "rc(2,2)": 4.0, "zs(2,2)": 4.0,
               "rc(3,3)": 1.0, "zs(3,3)": 1.0, "rc(4,4)": 0.25, "zs(4,4)": 0.25}

    def __init__(self, c0_base=0.004, raise_fn=None, notok_fn=None, nan_fn=None, min_cs_fn=None):
        self.c0_base = c0_base
        self.raise_fn, self.notok_fn, self.nan_fn = raise_fn, notok_fn, nan_fn
        self.min_cs_fn = min_cs_fn
        self.n_cold = 0
        self.n_warm = 0
        self.n_grad = 0

    def _w(self, name):
        if name in self.WEIGHTS:
            return self.WEIGHTS[name]
        pre, m, n = parse(name)
        if name in ("rc(0,0)", "rc(1,0)", "zs(1,0)"):
            return 0.0
        return 8.0

    def c0(self, boundary):
        a = 0.5 * (boundary["rc(1,0)"] + boundary["zs(1,0)"])
        s = sum(self._w(k) * float(v) ** 2 for k, v in boundary.items())
        return self.c0_base * (0.005 + s) * (a / 0.2) ** 2

    def solve(self, boundary, warm=None):
        if warm is None:
            self.n_cold += 1
        else:
            self.n_warm += 1
        if self.raise_fn is not None and self.raise_fn(boundary):
            raise RuntimeError("stub coil solve: start build failed")
        c0 = float("nan") if (self.nan_fn is not None and self.nan_fn(boundary)) else self.c0(boundary)
        ok = not (self.notok_fn is not None and self.notok_fn(boundary))
        min_cs = 0.2 if self.min_cs_fn is None else float(self.min_cs_fn(boundary))
        info = {"min_cs_dist": min_cs, "mean_bn_over_b_sq": 1.7 * c0 if np.isfinite(c0) else c0,
                "warm": warm is not None, "total_length": 16.0, "max_curvature": 3.0,
                "min_cc_distance": 0.12}
        return StubCoilResult(ok, c0, 0.9 * c0, np.zeros(66), np.full(4, 312500.0), np.zeros(3),
                              True, False, info)

    def envelope_grad(self, boundary, result, names):
        self.n_grad += 1
        a = 0.5 * (boundary["rc(1,0)"] + boundary["zs(1,0)"])
        s = sum(self._w(k) * float(v) ** 2 for k, v in boundary.items())
        out = np.zeros(len(names))
        for i, k in enumerate(names):
            d = self.c0_base * (2.0 * self._w(k) * float(boundary[k])) * (a / 0.2) ** 2
            if k in ("rc(1,0)", "zs(1,0)"):
                d += self.c0_base * (0.005 + s) * 2.0 * (a / 0.2) * (0.5 / 0.2)
            out[i] = d
        return out
