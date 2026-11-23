"""PHR-faithful fixed-boundary equilibrium evaluation with VMEX.

Reproduces what ``qss_script_jax_qa_phr_envelope.py`` computes with VMEC2000
through simsopt at every surface evaluation:

* input ``input.nfp2_QA`` (read-only), VMEC resolution ``mpol = ntor = 3 + step``
  (the script only changes ``vmec.indata.mpol/ntor``), angular grid inherited
  from the input deck: ``ntheta = 16``, ``nzeta = 14`` (VMEC2000's read of the
  deck fills ``2*mpol+6`` / ``2*ntor+4`` at the deck's mpol = ntor = 5, and the
  script never changes them), radial ladder ``NS 16/50, FTOL 1e-16/1e-11,
  NITER 600/2000``; axis guess zeroed (simsopt ``Vmec.set_indata``) so VMEC
  guesses the axis on every solve (cold start, history independent);
* QS: ``QuasisymmetryRatioResidual(np.arange(0, 1.01, 0.1), 1, 0)`` on the
  wout tables (VMEX's wout-path implementation is the simsopt algorithm, 63x64
  grid -> 44352 residuals) -- NOT VMEX's ``residuals_state``;
* aspect = ``wout.aspect``; mean_iota = ``mean(wout.iotas[1:])``; iota_axis =
  ``wout.iotaf[0]``; iota_edge = ``wout.iotaf[-1]``; vacuum well = simsopt's
  ``Vmec.vacuum_well`` formula on ``gmnc(m=0,n=0)``;
* elongation: PHR's ``MaxElongationPen(vmec, t=0)`` (vendored in
  :mod:`phrport.elongation`);
* the wout fed to the metrics carries ``bsubvmnc`` exactly as VMEC2000 writes
  it (``bcovar.f`` IEQUI=1 correction before the jxbforce filter), which VMEX
  omits for stellarator-symmetric runs -- see :func:`vmec2000_bsubvmnc`. Without
  it the QS residual vector differs from VMEC2000 by ~1.5e-4 (l2, relative).

VMEC2000 (simsopt) raises ``ObjectiveFailure`` for ``ierr != 11`` (any final-stage
non-convergence, Jacobian failure, ...). VMEX instead can return an unconverged
state, so :func:`evaluate` checks convergence explicitly and maps every failure
to ``EqResult(converged=False)``; it never raises on a failed solve. VMEX's
non-VMEC2000 recovery heuristics (``jacobian_retries``, ``coarse_grid_retry``)
are switched off so failures happen where VMEC2000 would fail (replaying all 284
evaluations of the PHR scale100 run: the 3 VMEC2000 failures fail in VMEX with
"INITIAL JACOBIAN CHANGED SIGN", the other 281 converge).

Known limit of any parity: at rare ill-conditioned boundaries (the ns=50 stage
passes through a growing-residual phase) the converged state depends on
roundoff; VMEC2000 itself lands on a different state under a 1-ulp boundary
change (1 of the 284 PHR evaluations; QS differs by ~9% there, everything else
agrees to <= 1.4e-7).

Boundary dicts use simsopt ``SurfaceRZFourier`` names ``'rc(m,n)'`` / ``'zs(m,n)'``
(n in field-period units, i.e. the VMEC ``RBC(n,m)``/``ZBS(n,m)`` index). A dict
is the FULL boundary: absent modes are zero; modes outside the VMEC resolution
of the step (m >= mpol or |n| > ntor) are ignored, exactly like VMEC2000 ignores
the surface modes simsopt copies into ``rbc`` beyond ``mpol``/``ntor``.

Opt-in extras (defaults unchanged, results byte-identical):

* ``evaluate(boundary, step, return_state=True)`` attaches the converged VMEX
  state as ``EqResult.vmex_state`` (:class:`VmexStateArrays`: numpy only,
  picklable, keyed by :func:`input_fingerprint`) so that
  :class:`phrport.exact_jacobian.ExactEquilibriumJacobian` can skip its cold
  re-solve (roadmap S-c); :class:`phrport.vmex_pool.VmexPool` ships it back
  from its workers with ``return_state=True``.
* :func:`evaluate_resolution` -- the same solve + PHR metrics at an arbitrary
  VMEC resolution / angular grid / radial ladder (roadmap G3: PHR grid vs
  mpol 8, 22x20, ns 101).
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import math
import re
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path

import numpy as np

from phrport import problem as _PROB

__all__ = [
    "SEED_ELLIPSE", "INPUT_FILE", "NTHETA", "NZETA", "NS_ARRAY", "FTOL_ARRAY",
    "NITER_ARRAY", "QS_SURFACES", "N_QS_RESIDUALS", "SURF_MPOL", "SURF_NTOR",
    "EqResult", "seed_boundary", "free_dof_names", "all_dof_names", "max_mode",
    "vmec_resolution", "vmec_input", "evaluate", "evaluate_wout", "failure_result",
    "boundary_from_free_dofs", "free_dofs_from_boundary", "parse_name",
    "vmec2000_bsubvmnc", "vmec2000_wout", "vacuum_well_from_wout", "VMEC2000_BSUBV", "write_wout",
    "VmexStateArrays", "input_fingerprint", "vmec_input_at", "NS_LADDERS", "resolve_ns_ladder",
    "evaluate_resolution", "PHYSICS", "QI_SURFACES", "QI_MIRROR_S", "QI_OPTIONS", "qi_metrics",
]

# ---------------------------------------------------------------------------
# PHR constants
# ---------------------------------------------------------------------------
SEED_ELLIPSE = float(_PROB.get("equilibrium", "seed_ellipse", 0.07))
# the profile's equilibrium.input_file, else $PHRPORT_INPUT, else the repository's inputs/input.nfp2_QA (PHR's deck)
INPUT_FILE = (Path(_PROB.get("equilibrium", "input_file", "")) if _PROB.get("equilibrium", "input_file", "")
              else Path(os.environ["PHRPORT_INPUT"]) if os.environ.get("PHRPORT_INPUT")
              else _PROB.REPO_ROOT / "inputs" / "input.nfp2_QA")

# Angular grid: inherited from the input deck (see module docstring).
NTHETA = 16
NZETA = 14
# Radial ladder of input.nfp2_QA (identical to the deck; set explicitly).
NS_ARRAY = tuple(int(v) for v in _PROB.get("equilibrium", "ns_array", (16, 50)))
FTOL_ARRAY = tuple(float(v) for v in _PROB.get("equilibrium", "ftol_array", (1.0e-16, 1.0e-11)))
NITER_ARRAY = tuple(int(v) for v in _PROB.get("equilibrium", "niter_array", (600, 2000)))
# A profile may fix the VMEC resolution (mpol = ntor = VMEC_MPOL at every step, VMEC's
# 2M+6 x 2N+4 angular grid); None = PHR's continuation resolution 3 + step.
VMEC_MPOL = _PROB.get("equilibrium", "vmec_mpol", None)
# A profile may also fix the VMEC toroidal resolution separately (default: ntor = mpol), e.g. to resolve
# boundary modes |n| > mpol such as the coil-ripple band below.
VMEC_NTOR = _PROB.get("equilibrium", "vmec_ntor", None)
# deck modes |n| > deck_ntor_max are dropped from the seed
DECK_NTOR_MAX = _PROB.get("equilibrium", "deck_ntor_max", None)
# Profile opt-in (28 Oct, tokamak cold start): continuation step k frees |n| <= max_mode + NTOR_EXTRA (capped at the
# surface ntor) instead of the square box, so axis torsion and bean rotation are available at low poloidal order.
NTOR_EXTRA = int(_PROB.get("equilibrium", "ntor_extra", 0))
# Ripple band (opt-in; ported from the LP QA thread): extra free boundary modes m <= m_max,
# n_min <= |n| <= n_max from continuation step from_step on, so that the boundary can follow the
# discrete-coil ripple (outside the max_mode box).
_RB = _PROB.get("equilibrium", "ripple_band", None)
RIPPLE_BAND = None if not _RB else (int(_RB.get("m_max", 3)), int(_RB.get("n_min", 6)),
                                    int(_RB.get("n_max", 10)), int(_RB.get("from_step", 4)))

# Radial ladders as (ns, ftol, niter) stages, for evaluate_resolution (roadmap G3).
# "phr" is the deck's ladder above (evaluate_resolution(b, 6, 16, 14, "phr") is
# bitwise evaluate(b, 3)); "ns101" refines to ns = 101 through PHR's ns = 50 stage;
# "ns101_b" reaches ns = 101 by a different path (no ns = 50 stage, coarse stage at
# 25 surfaces) so that ladder (gauge/path) sensitivity can be separated from
# resolution sensitivity.
NS_LADDERS = {
    "phr": tuple(zip(NS_ARRAY, FTOL_ARRAY, NITER_ARRAY)),
    "ns101": ((16, 1.0e-16, 600), (50, 1.0e-13, 3000), (101, 1.0e-14, 8000)),
    "ns101_b": ((25, 1.0e-12, 2000), (101, 1.0e-14, 8000)),
}

# QuasisymmetryRatioResidual(vmec, np.arange(0, 1.01, 0.1), helicity_m=1, helicity_n=0)
QS_SURFACES = np.arange(0, 1.01, 0.1)
if _PROB.get("equilibrium", "qs_surfaces", None) is not None:      # opt-in (profile): QS surface list, e.g. a core-weighted grid
    QS_SURFACES = np.asarray(_PROB.get("equilibrium", "qs_surfaces", None), dtype=float)
QS_HELICITY_M = int(_PROB.get("equilibrium", "helicity_m", 1))
QS_HELICITY_N = int(_PROB.get("equilibrium", "helicity_n", 0))
QS_NTHETA = 63           # simsopt defaults
QS_NPHI = 64
N_QS_RESIDUALS = len(QS_SURFACES) * QS_NTHETA * QS_NPHI      # 44352

# Physics block of the outer problem: "qs" (PHR, default) or "qi" (opt-in profile key
# equilibrium.physics).  For "qi" the qs_* fields of EqResult carry the continuous
# constructed-QI residual (phrport.qi_residual, Goodman et al. JPP 2023) instead of the
# QS residual, and EqResult.mirror the mirror ratio at s = QI_MIRROR_S.
PHYSICS = str(_PROB.get("equilibrium", "physics", "qs"))
if PHYSICS not in ("qs", "qi"):
    raise ValueError(f"equilibrium.physics must be 'qs' or 'qi', got {PHYSICS!r}")
QI_SURFACES = tuple(float(v) for v in _PROB.get("equilibrium", "qi_surfaces", (0.1, 0.3, 0.5, 0.7, 0.9)))
QI_MIRROR_S = float(_PROB.get("equilibrium", "qi_mirror_s", 1.0 / 51.0))
QI_OPTIONS = dict(dict(nphi=101, nalpha=21, n_bounce=31, mboz=16, nboz=16, oversample=2, alpha_weights=False),
                  **_PROB.get("equilibrium", "qi_options", {}))
# Mercier metric (phrport.mercier: smooth min of s DMerc); opt-in with equilibrium.mercier_options (s_min, s_max, tau).
MERC_OPTIONS = _PROB.get("equilibrium", "mercier_options", None)
MERC_JAC_BATCH = int(_PROB.get("equilibrium", "mercier_jac_batch", 8))     # DOF columns per DMerc pushforward
if PHYSICS == "qi":
    N_QS_RESIDUALS = len(QI_SURFACES) * int(QI_OPTIONS["nalpha"]) * int(QI_OPTIONS["nphi"])
    if float(QI_OPTIONS.get("bmin_weight", 0.0)) > 0.0:     # opt-in well-bottom flatness rows (qi_residual)
        N_QS_RESIDUALS += (len(QI_SURFACES) + len(QI_OPTIONS.get("bmin_surfaces", ()))) * int(QI_OPTIONS["nalpha"])
    if QI_OPTIONS.get("djds_surfaces"):                       # opt-in shallow-well dJ/ds rows (qi_residual)
        N_QS_RESIDUALS += (len(QI_OPTIONS["djds_surfaces"]) - 1) * len(QI_OPTIONS.get("djds_depths", (0.05, 0.1))) * int(QI_OPTIONS.get("djds_nalpha", 24))
        if QI_OPTIONS.get("jspread_weight") is not None:
            N_QS_RESIDUALS += len(QI_OPTIONS["djds_surfaces"]) * len(QI_OPTIONS.get("jspread_depths", (0.05, 0.1))) * int(QI_OPTIONS.get("djds_nalpha", 24))
    if QI_OPTIONS.get("bminu_surfaces"):                      # opt-in field-line B_min uniformity rows (qi_residual)
        N_QS_RESIDUALS += len(QI_OPTIONS["bminu_surfaces"]) * int(QI_OPTIONS.get("bminu_nalpha", 24))


def qi_metrics(state, runtime) -> dict:
    """QI physics block (numpy): qs_residuals, qs_total, mirror from a converged state."""
    from .qi_residual import qi_physics
    o = dict(QI_OPTIONS)
    out = qi_physics(state, runtime, surfaces=QI_SURFACES, mirror_s=QI_MIRROR_S,
                     mboz=o.pop("mboz"), nboz=o.pop("nboz"), oversample=o.pop("oversample"), **o)
    return dict(qs_residuals=np.asarray(out["qs"], dtype=np.float64),
                qs_total=float(out["qs_total"]), mirror=float(out["mirror"]),
                **({"maxj": float(out["maxj"]), "maxj_fraction": float(out["maxj_fraction"])}
                   if "maxj" in out else {}),
                **({"bmaxsp": float(out["bmaxsp"]), "bmaxsp_profile": np.asarray(out["bmaxsp_profile"]).tolist()}
                   if "bmaxsp" in out else {}))


def _runtime_for(inp, state):
    from vmex.core.solver import prepare_runtime, resolution_from_input
    return prepare_runtime(inp, resolution_from_input(inp, ns=int(np.shape(state.R_cos)[0])))

# MaxElongationPen(v, t=0.0) defaults
ELONG_NTHETA = 50
ELONG_NPHI = 50

# Surface Fourier resolution of the PHR boundary object: simsopt builds
# vmec.boundary with the INPUT FILE's mpol/ntor (5/5); continuation only changes
# vmec.indata.mpol/ntor. The DOF order (and hence free_dof_names) is defined on
# this surface. Checked against the deck by the tests.
SURF_MPOL = int(_PROB.get("equilibrium", "surf_mpol", 5))
SURF_NTOR = int(_PROB.get("equilibrium", "surf_ntor", 5))
NFP = int(_PROB.get("equilibrium", "nfp", 2))
# profiles may free the major radius rc(0,0) (PHR fixes it), e.g. with a volume instead of an aspect constraint
FREE_R00 = bool(_PROB.get("equilibrium", "free_r00", False))

# Write bsubvmnc the way VMEC2000's wrout.f does (bcovar IEQUI=1 correction);
# VMEX skips it for lasym=False. See vmec2000_bsubvmnc(). False = raw VMEX wout.
VMEC2000_BSUBV = True

# VMEX solve options: VMEC2000 failure semantics.
VMEX_SOLVE_KWARGS = dict(
    **({'device': 'cpu'} if os.environ.get('PHRPORT_VMEX_DEVICE', '').lower() == 'cpu' else {}),
    raise_on_max_iterations=False,   # get the state back, flag converged=False
    jacobian_retries=0,              # VMEC2000: fatal after 75 Jacobian resets
    coarse_grid_retry=False,         # VMEC++-only recovery; VMEC2000 fails
    verbose=False,
)

_NAME_RE = re.compile(r"^(rc|zs)\((-?\d+),(-?\d+)\)$")


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------
@dataclass
class EqResult:
    """Outcome of one fixed-boundary solve.

    On failure (``converged=False``) every float is NaN and ``qs_residuals`` is a
    NaN vector of the regular length (44352), so a caller can substitute PHR's
    1e12 sentinel without knowing the length. ``info`` carries diagnostics
    (wall time, iterations, fsq, ier_flag); it is not part of the PHR contract.
    ``vmex_state`` is ``None`` unless the converged state was requested
    (``evaluate(..., return_state=True)``; never set on failures).
    """

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
    vmex_state: "VmexStateArrays | None" = field(default=None, repr=False, compare=False)
    # plasma size (wout volume_p / Aminor_p; NaN on failures or when not measured)
    volume: float = float("nan")
    aminor: float = float("nan")
    # QI profiles: mirror ratio at s = QI_MIRROR_S (NaN otherwise)
    mirror: float = float("nan")
    # max-J constraint value (qi_residual.constructed_maximum_j; NaN when not configured)
    maxj: float = float("nan")
    # Mercier metric: smooth min of s DMerc (phrport.mercier; NaN when not configured)
    merc: float = float("nan")
    # field-line B_max spread constraint value (qi_residual.bmax_spread; NaN when not configured)
    bmaxsp: float = float("nan")


def failure_result(message: str, info: dict | None = None) -> EqResult:
    nan = float("nan")
    return EqResult(False, np.full(N_QS_RESIDUALS, nan), nan, nan, nan, nan, nan, nan, nan,
                    message=message, info=dict(info or {}))


# ---------------------------------------------------------------------------
# Converged-state transfer (roadmap S-c)
# ---------------------------------------------------------------------------
def _hash_value(h, v) -> None:
    if isinstance(v, (np.ndarray, list, tuple)) or (hasattr(v, "__array__") and not np.isscalar(v)):
        a = np.asarray(v)
        if a.dtype.kind in "biufc":
            a = np.ascontiguousarray(a)
            h.update(f"{a.dtype.str}{a.shape}".encode())
            h.update(a.tobytes())
            return
        h.update(repr(a.tolist()).encode())
        return
    h.update(f"{type(v).__name__}:{v!r}".encode())


def input_fingerprint(inp) -> str:
    """SHA-256 over every field of a VMEX ``VmecInput`` (names, dtypes, shapes, exact bytes).

    Two inputs with the same fingerprint run the identical solve (same
    boundary bits, resolution, grid, ladder, profiles); used to tie a
    :class:`VmexStateArrays` to the input it solves.
    """
    import dataclasses
    import hashlib
    h = hashlib.sha256()
    for f in dataclasses.fields(inp):
        h.update(f.name.encode() + b"=")
        _hash_value(h, getattr(inp, f.name))
        h.update(b";")
    return h.hexdigest()


@dataclass(frozen=True, eq=False)
class VmexStateArrays:
    """Host (numpy) copy of a converged VMEX ``SpectralState`` (roadmap S-c).

    Everything :class:`phrport.exact_jacobian.ExactEquilibriumJacobian` needs to
    skip its cold re-solve: the converged state (the solver's internal
    normalization, (ns, mnmax) per field) plus the convergence diagnostics.
    Numpy only, so it pickles as one contiguous float64 block (``data``) and a
    :class:`phrport.vmex_pool.VmexPool` worker returns it cheaply (mode 4: 5 x
    (50, 72) float64 = 144 kB; fields whose bits are ALL zero -- ``L_cos`` for
    stellarator-symmetric runs -- are recorded by name and not shipped).
    The round trip is bitwise (no dtype conversion, -0.0 kept).

    ``key`` is :func:`input_fingerprint` of the ``VmecInput`` that was solved;
    consumers recompute it from (boundary, step) and refuse a state that
    belongs to another input.
    """

    key: str
    step: "int | None"
    names: tuple                 # SpectralState fields stored in ``data`` (in field order)
    data: np.ndarray             # (len(names), ns, mnmax) float64
    zero_names: tuple            # SpectralState fields that are exactly zero (all bits)
    shape: tuple                 # (ns, mnmax)
    converged: bool
    iterations: int
    fsq: tuple                   # (fsqr, fsqz, fsql)

    @classmethod
    def from_equilibrium(cls, eq, inp=None, step=None) -> "VmexStateArrays":
        """From a VMEX ``Equilibrium`` (``opt.solve_equilibrium`` output)."""
        import dataclasses
        import jax
        inp = eq.inp if inp is None else inp
        host = jax.device_get(eq.state)
        names, arrs, zeros, shape = [], [], [], None
        for f in dataclasses.fields(host):
            a = np.asarray(getattr(host, f.name))
            if a.dtype != np.float64:
                raise TypeError(f"state field {f.name} has dtype {a.dtype}, expected float64")
            if shape is None:
                shape = tuple(a.shape)
            elif tuple(a.shape) != shape:
                raise ValueError(f"state field {f.name} has shape {a.shape}, expected {shape}")
            if not np.ascontiguousarray(a).view(np.uint64).any():
                zeros.append(f.name)
            else:
                names.append(f.name)
                arrs.append(a)
        data = (np.stack(arrs, axis=0) if arrs else np.zeros((0,) + shape))
        r = eq.result
        return cls(key=input_fingerprint(inp), step=None if step is None else int(step),
                   names=tuple(names), data=np.ascontiguousarray(data, dtype=np.float64),
                   zero_names=tuple(zeros), shape=tuple(int(s) for s in shape),
                   converged=bool(r.converged), iterations=int(r.iterations),
                   fsq=(float(r.fsqr), float(r.fsqz), float(r.fsql)))

    @property
    def nbytes(self) -> int:
        return int(self.data.nbytes)

    def array(self, name: str) -> np.ndarray:
        if name in self.names:
            return self.data[self.names.index(name)]
        if name in self.zero_names:
            return np.zeros(self.shape, dtype=np.float64)
        raise KeyError(name)

    def matches(self, inp) -> bool:
        return self.key == input_fingerprint(inp)

    def spectral_state(self, device=None):
        """VMEX ``SpectralState`` of JAX float64 arrays COMMITTED to ``device``.

        Default: the current default device (``jax.default_device`` context, e.g.
        :func:`vmex_device_ctx`, else the backend's first device) -- where a
        solve in the same context leaves its state. Committed like the solver's
        output, so jitted consumers (the exact Jacobian) hit their compilation
        cache instead of retracing for uncommitted inputs.
        """
        import dataclasses
        import jax
        from vmex.core.solver import SpectralState
        if device is None:
            device = jax.config.jax_default_device
        if device is None:
            device = jax.devices()[0]
        elif isinstance(device, str):
            device = jax.devices(device)[0]
        kw = {f.name: jax.device_put(np.ascontiguousarray(self.array(f.name)), device)
              for f in dataclasses.fields(SpectralState)}
        return SpectralState(**kw)


# ---------------------------------------------------------------------------
# DOF bookkeeping (simsopt SurfaceRZFourier order)
# ---------------------------------------------------------------------------
def _names_helper(prefix: str, include0: bool, mpol: int, ntor: int) -> list[str]:
    # == simsopt SurfaceRZFourier._make_names_helper (and JaxSurfaceRZFourier._make_names)
    names = [f"{prefix}(0,0)"] if include0 else []
    names += [f"{prefix}(0,{n})" for n in range(1, ntor + 1)]
    for m in range(1, mpol + 1):
        names += [f"{prefix}({m},{n})" for n in range(-ntor, ntor + 1)]
    return names


def all_dof_names(mpol: int = SURF_MPOL, ntor: int = SURF_NTOR) -> list[str]:
    """Full local DOF names of the stellsym PHR boundary surface (121 for 5/5)."""
    return _names_helper("rc", True, mpol, ntor) + _names_helper("zs", False, mpol, ntor)


def parse_name(name: str) -> tuple[str, int, int]:
    mt = _NAME_RE.match(name)
    if mt is None:
        raise ValueError(f"bad boundary mode name {name!r}; expected 'rc(m,n)' or 'zs(m,n)'")
    return mt.group(1), int(mt.group(2)), int(mt.group(3))


def max_mode(step: int) -> int:
    return int(step) + 1


def vmec_resolution(step: int) -> int:
    """mpol = ntor of the VMEC solve in continuation step ``step`` (0-based)."""
    if VMEC_MPOL is not None:
        return int(VMEC_MPOL)
    return 3 + int(step)


def free_dof_names(step: int) -> list[str]:
    """Free surface DOFs of continuation step ``step`` in simsopt local-DOF order.

    PHR mask: ``surf.fix_all(); surf.fixed_range(mmin=0, mmax=mm, nmin=-mm,
    nmax=mm, fixed=False); surf.fix('rc(0,0)')`` with ``mm = step + 1`` on the
    5/5 boundary surface. ``surf.x`` lists the free DOFs in the order of the full
    DOF vector: the rc block then the zs block, m-major, n from -ntor..ntor
    (m=0: n>=0; zs(0,0) absent). Sizes 8/24/48/80 for steps 0..3.
    """
    mm = max_mode(step)
    if mm > min(SURF_MPOL, SURF_NTOR):
        raise ValueError(f"step {step}: max_mode {mm} exceeds the surface resolution")
    band = RIPPLE_BAND if (RIPPLE_BAND is not None and int(step) >= RIPPLE_BAND[3]) else None
    if band is not None and band[2] > SURF_NTOR:
        raise ValueError(f"ripple band |n| <= {band[2]} exceeds the surface resolution ntor {SURF_NTOR}")
    out = []
    for name in all_dof_names():
        _, m, n = parse_name(name)
        in_box = m <= mm and abs(n) <= min(mm + NTOR_EXTRA, SURF_NTOR)
        in_band = band is not None and m <= band[0] and band[1] <= abs(n) <= band[2]
        if (in_box or in_band) and (name != "rc(0,0)" or FREE_R00):
            out.append(name)
    return out


@lru_cache(maxsize=1)
def _base_input():
    from vmex.core.input import VmecInput
    return VmecInput.from_file(str(INPUT_FILE))


def seed_boundary() -> "OrderedDict[str, float]":
    """PHR initial boundary: input.nfp2_QA + rotating-ellipse seed.

    All 121 DOFs of the 5/5 boundary surface in simsopt order (zeros included),
    with ``rc(2,1) = zs(2,1) = SEED_ELLIPSE`` added to the (axisymmetric) deck.
    """
    inp = _base_input()
    assert int(inp.nfp) == NFP
    if not _PROB.ACTIVE:
        assert int(inp.mpol) == SURF_MPOL and int(inp.ntor) == SURF_NTOR
    rbc = np.asarray(inp.rbc)
    zbs = np.asarray(inp.zbs)
    dn = int(inp.ntor)
    out = OrderedDict()
    for name in all_dof_names():
        kind, m, n = parse_name(name)
        arr = rbc if kind == "rc" else zbs
        # deck arrays are dense [n + ntor_deck, m], m = 0..mpol-1 (m = mpol is 0 in simsopt too);
        # a profile surface smaller than the deck truncates it (VMEC ignores those modes anyway)
        nmax_deck = dn if DECK_NTOR_MAX is None else min(dn, int(DECK_NTOR_MAX))
        out[name] = float(arr[n + dn, m]) if (m < arr.shape[1] and abs(n) <= nmax_deck) else 0.0
    # The deck is axisymmetric -> PHR adds the (m=2, n=1) rotating ellipse.
    axisym = all(abs(v) <= 1e-8 for k, v in out.items() if parse_name(k)[2] != 0)
    if SEED_ELLIPSE != 0.0 and axisym:
        out["rc(2,1)"] += SEED_ELLIPSE
        out["zs(2,1)"] += SEED_ELLIPSE
    return out


def boundary_from_free_dofs(x, step: int, base: dict | None = None) -> "OrderedDict[str, float]":
    """Full boundary = ``base`` (default: seed) with the free DOFs of ``step`` set to ``x``."""
    names = free_dof_names(step)
    x = np.asarray(x, dtype=float).ravel()
    if x.size != len(names):
        raise ValueError(f"step {step}: expected {len(names)} free DOFs, got {x.size}")
    out = OrderedDict(seed_boundary() if base is None else base)
    for nm, v in zip(names, x):
        out[nm] = float(v)
    return out


def free_dofs_from_boundary(boundary: dict, step: int) -> np.ndarray:
    return np.array([float(boundary.get(nm, 0.0)) for nm in free_dof_names(step)])


# ---------------------------------------------------------------------------
# VMEX input
# ---------------------------------------------------------------------------
def vmec_input(boundary: dict, step: int):
    """VMEX ``VmecInput`` reproducing PHR's VMEC2000 run for ``boundary`` at ``step``."""
    M = vmec_resolution(step)
    N = M if VMEC_NTOR is None else int(VMEC_NTOR)
    if VMEC_MPOL is not None or int(step) >= 4:
        # roadmap A6 (mode 5 and up; PHR never ran these steps, so steps 0-3 keep PHR's 16 x 14 grid):
        # VMEC's angular-grid rule ntheta >= 2 mpol + 6, nzeta >= 2 ntor + 4
        return vmec_input_at(boundary, M, max(NTHETA, 2 * M + 6), max(NZETA, 2 * N + 4),
                             NS_ARRAY, FTOL_ARRAY, NITER_ARRAY, ntor=None if VMEC_NTOR is None else N)
    return vmec_input_at(boundary, M, NTHETA, NZETA, NS_ARRAY, FTOL_ARRAY, NITER_ARRAY,
                         ntor=None if VMEC_NTOR is None else N)


def vmec_input_at(boundary: dict, mpol: int, ntheta: int, nzeta: int, ns_array, ftol_array,
                  niter_array, *, ntor: int | None = None):
    """PHR's VMEX input (deck + boundary, zeroed axis guess) at any resolution/grid/ladder.

    ``vmec_input(boundary, step)`` is ``vmec_input_at(boundary, 3 + step, 16, 14,
    NS_ARRAY, FTOL_ARRAY, NITER_ARRAY)``. ``ntor`` defaults to ``mpol``. Boundary
    modes with ``m >= mpol`` or ``|n| > ntor`` are ignored (VMEC2000 semantics).
    """
    if "rc(0,0)" not in boundary:
        raise ValueError("boundary must be a FULL boundary dict (missing 'rc(0,0)'); "
                         "use boundary_from_free_dofs() for free-DOF vectors")
    M = int(mpol)
    N = M if ntor is None else int(ntor)
    if M < 1 or N < 0:
        raise ValueError(f"bad VMEC resolution mpol={mpol}, ntor={ntor}")
    if not (len(ns_array) == len(ftol_array) == len(niter_array) and len(ns_array) > 0):
        raise ValueError("ns_array, ftol_array and niter_array must be non-empty and of equal length")
    base = _base_input()
    inp = base.change_resolution(mpol=M, ntor=N, ntheta=int(ntheta), nzeta=int(nzeta))
    rbc = np.zeros((2 * N + 1, M))
    zbs = np.zeros((2 * N + 1, M))
    for name, val in boundary.items():
        kind, m, n = parse_name(name)
        if m < 0 or (m == 0 and n < 0) or (kind == "zs" and m == 0 and n == 0):
            raise ValueError(f"{name!r} is not a stellsym SurfaceRZFourier DOF")
        v = float(val)
        if not math.isfinite(v):
            raise ValueError(f"non-finite boundary coefficient {name}={val!r}")
        if m < M and abs(n) <= N:      # VMEC reads m = 0..mpol-1, |n| <= ntor
            (rbc if kind == "rc" else zbs)[n + N, m] = v
    return replace(
        inp, rbc=rbc, zbs=zbs,
        raxis_c=np.zeros(N + 1), zaxis_s=np.zeros(N + 1),    # simsopt zeroes the axis
        ns_array=np.array(ns_array), ftol_array=np.array(ftol_array),
        niter_array=np.array(niter_array),
    )


# ---------------------------------------------------------------------------
# Metrics from a wout (VMEX WoutData, file layout (ns, mnmax))
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _qs_objective():
    from vmex import optimize as opt
    return opt.QuasisymmetryRatioResidual(QS_SURFACES, QS_HELICITY_M, QS_HELICITY_N,
                                          ntheta=QS_NTHETA, nphi=QS_NPHI)


def vacuum_well_from_wout(wout) -> float:
    """simsopt ``Vmec.vacuum_well`` on VMEX's (ns, mnmax_nyq) ``gmnc``."""
    gmnc = np.asarray(wout.gmnc, dtype=float)
    ns = int(wout.ns)
    if gmnc.shape[0] != ns:
        raise ValueError(f"expected (ns, mnmax_nyq) gmnc, got {gmnc.shape}")
    xm = np.asarray(wout.xm_nyq); xn = np.asarray(wout.xn_nyq)
    if not (xm[0] == 0 and xn[0] == 0):
        raise ValueError("gmnc mode 0 is not (0,0)")
    dVds = 4 * np.pi * np.pi * np.abs(gmnc[1:, 0])
    dVds_s0 = 1.5 * dVds[0] - 0.5 * dVds[1]
    dVds_s1 = 1.5 * dVds[-1] - 0.5 * dVds[-2]
    return float((dVds_s0 - dVds_s1) / dVds_s0)


def vmec2000_bsubvmnc(eq) -> np.ndarray:
    """``bsubvmnc`` exactly as VMEC2000's wrout.f writes it, for a VMEX ``Equilibrium``.

    VMEC2000 (simsopt's build, i.e. what PHR ran) writes ``bsubv`` after the
    ``bcovar.f`` IEQUI=1 block -- the backward recurrence
    ``bsubv(js) = 2 bsubv_e(js) - bsubv(js+1)`` from the lambda-force
    ``bsubv_e`` plus the per-surface shift that keeps ``<bsubv>`` -- and then the
    ``jxbforce.f`` low-pass filter. VMEX applies that correction only on its
    lasym lane, so for stellarator-symmetric runs its ``bsubvmnc`` differs from
    VMEC2000 by up to ~5e-5 (relative to max, largest near the axis), which moves
    the QS residuals by ~1e-4 relative (they use B_v). The pieces are VMEX's own
    ports (``nyquist.bsubv_lambda_full_mesh`` / ``apply_bsubv_equif_correction`` /
    ``filter_bsubuv_symmetric`` / ``wrout_cos_coeffs``); with them the table
    matches VMEC2000 to ~4e-12 (checked on the PHR final surface).
    """
    import jax
    from vmex.core import nyquist as NQ
    from vmex.core import wout as W
    from vmex.core.fourier import Resolution, mode_table, trig_tables
    from vmex.core.postprocess import expand_mode_columns
    from vmex.core.setup import boundary_from_input, flux_profiles, radial_grids
    from vmex.core.solver import resolution_from_input

    inp, state = eq.inp, eq.state
    if bool(inp.lasym):
        raise NotImplementedError("lasym runs already carry the correction in VMEX")
    ns = int(np.shape(state.R_cos)[0])
    res = resolution_from_input(inp, ns=ns)
    mpol, ntor, nfp = int(res.mpol), int(res.ntor), int(res.nfp)
    modes = mode_table(mpol, ntor)
    mnyq_grid = max(res.ntheta1 // 2, mpol - 1)
    nnyq_grid = max(int(res.nzeta) // 2, ntor)
    trig = trig_tables(Resolution(mpol=mnyq_grid + 1, ntor=nnyq_grid, ntheta=int(res.ntheta),
                                  nzeta=int(res.nzeta), nfp=nfp, lasym=False, ns=ns))
    grids = radial_grids(ns)
    s = np.asarray(grids.s_full, dtype=float)
    bnd = boundary_from_input(inp, modes=modes, trig=trig, lconm1=True)
    signgs = int(bnd.signgs)
    prof = flux_profiles(inp, grids, r00=bnd.r00, signgs=signgs, lflip=bnd.lflip)
    # same compiled lane wout_from_state uses (no recompilation)
    geometry, jacobian, metrics, fields = jax.device_get(W._wout_field_state_lane(
        state, grids.s_full, prof["phips"], prof["phipf"], prof["chips"], prof["mass"],
        prof["icurv"], res=res, signgs=signgs, gamma=float(inp.gamma),
        ncurr=int(inp.ncurr)))[:4]
    bsubv_e = NQ.bsubv_lambda_full_mesh(geometry=geometry, jacobian=jacobian, metrics=metrics,
                                        fields=fields, s=s, phipf=np.asarray(prof["phipf"], dtype=float))
    bsubv_c = NQ.apply_bsubv_equif_correction(bsubv=np.asarray(fields.bsubv, dtype=float),
                                              bsubv_e=bsubv_e, trig=trig)
    _, bsubv_f = NQ.filter_bsubuv_symmetric(bsubu=np.asarray(fields.bsubu, dtype=float), bsubv=bsubv_c,
                                            trig=trig, mmax_force=mpol - 1, nmax_force=ntor, s=s)
    mnyq, nnyq = NQ.nyquist_limits(trig)
    nyq_modes = mode_table(max(mnyq, mpol - 1) + 1, max(nnyq, ntor))
    tab = NQ.wrout_cos_coeffs(f=bsubv_f, modes=nyq_modes, trig=trig)
    tab[0, :] = 0.0                                   # wrout: first (axis) surface zeroed
    xm_n = np.asarray(nyq_modes.m, dtype=float)
    xn_n = np.asarray(nyq_modes.n, dtype=float) * nfp
    w = eq.wout
    if tab.shape != np.shape(w.bsubvmnc) or not (np.array_equal(xm_n, np.asarray(w.xm_nyq))
                                                 and np.array_equal(xn_n, np.asarray(w.xn_nyq))):
        tab = expand_mode_columns(tab, xm_n, xn_n, np.asarray(w.xm_nyq), np.asarray(w.xn_nyq))
    return tab


def vmec2000_wout(eq):
    """``eq.wout`` with ``bsubvmnc`` replaced by the VMEC2000 table (see ``vmec2000_bsubvmnc``).

    Only ``bsubvmnc`` is replaced; its (m,n) = (0,0) column -- and hence ``bvco``
    -- is unchanged by the correction (it preserves the surface average). The
    other derived current diagnostics (``currvmnc``, ``jcurv``, ...) are left as
    VMEX computes them; none of them enters a PHR metric.
    """
    import dataclasses
    w = eq.wout
    if bool(w.lasym) or not VMEC2000_BSUBV:
        return w
    return dataclasses.replace(w, bsubvmnc=vmec2000_bsubvmnc(eq))


def evaluate_wout(wout, *, elongation: bool = True) -> dict:
    """PHR metrics of a VMEX ``WoutData`` (also usable on ``vmex.read_wout`` output)."""
    from .elongation import max_elongation_vmex
    if PHYSICS == "qi":      # the QI block needs the state (qi_metrics); callers fill it in
        qs = dict(residuals1d=np.zeros(0), total=0.0)
    else:
        qs = _qs_objective().compute(wout)
    res = np.asarray(qs["residuals1d"], dtype=np.float64)
    iotas = np.asarray(wout.iotas, dtype=np.float64)
    iotaf = np.asarray(wout.iotaf, dtype=np.float64)
    out = dict(
        qs_residuals=res,
        qs_total=float(np.asarray(qs["total"])),
        aspect=float(wout.aspect),
        mean_iota=float(np.mean(iotas[1:])),
        iota_axis=float(iotaf[0]),
        iota_edge=float(iotaf[-1]),
        well=vacuum_well_from_wout(wout),
        volume=float(wout.volume_p),
        aminor=float(wout.Aminor_p),
    )
    out["elongation"] = (max_elongation_vmex(wout, ELONG_NTHETA, ELONG_NPHI)
                         if elongation else float("nan"))
    return out


# ---------------------------------------------------------------------------
# One solve
# ---------------------------------------------------------------------------
def vmex_device_ctx():
    """Device for VMEX work. PHRPORT_VMEX_DEVICE=cpu pins VMEX (solves, metrics, tangents) to the CPU
    while XCoil keeps the default (GPU) backend: VMEX forward solves are slower on small GPUs."""
    import contextlib
    if os.environ.get("PHRPORT_VMEX_DEVICE", "").lower() == "cpu":
        import jax
        return jax.default_device(jax.devices("cpu")[0])
    return contextlib.nullcontext()


def _solve_and_measure(boundary: dict, step: int, return_state: bool = False):
    with vmex_device_ctx():
        return _solve_and_measure_impl(boundary, step, return_state)


def _solve_and_measure_impl(boundary: dict, step: int, return_state: bool = False):
    """-> (EqResult, wout | None); the wout (VMEC2000 bsubvmnc) only for converged solves."""
    inp = vmec_input(boundary, step)
    info = dict(step=int(step), mpol=vmec_resolution(step), ntheta=int(inp.ntheta), nzeta=int(inp.nzeta))
    return _solve_input_and_measure(inp, info, step=int(step), return_state=return_state)


def _solve_input_and_measure(inp, info: dict, *, step=None, return_state: bool = False):
    t0 = time.perf_counter()
    try:
        from vmex import optimize as opt
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            eq = opt.solve_equilibrium(inp, **VMEX_SOLVE_KWARGS)
    except Exception as exc:  # VmecJacobianError, VmecNumericalError, ...
        info["t_solve"] = time.perf_counter() - t0
        return failure_result(f"VMEX raised {type(exc).__name__}: {exc}"[:2000], info), None
    r = eq.result
    info.update(t_solve=time.perf_counter() - t0, iterations=int(r.iterations),
                ier_flag=int(r.ier_flag), fsqr=float(r.fsqr), fsqz=float(r.fsqz),
                fsql=float(r.fsql), jacobian_resets=int(getattr(r, "jacobian_resets", -1)))
    if not bool(r.converged):
        return failure_result(
            f"VMEX not converged (ier_flag={r.ier_flag}, iterations={r.iterations}, "
            f"fsq=({r.fsqr:.3e},{r.fsqz:.3e},{r.fsql:.3e}))", info), None
    t1 = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            wout = vmec2000_wout(eq)
            m = evaluate_wout(wout)
    except Exception as exc:
        info["t_metrics"] = time.perf_counter() - t1
        return failure_result(f"metric evaluation failed: {type(exc).__name__}: {exc}"[:2000], info), None
    info["t_metrics"] = time.perf_counter() - t1
    info["t_total"] = time.perf_counter() - t0
    scal = [m[k] for k in ("qs_total", "aspect", "mean_iota", "iota_axis", "iota_edge",
                           "elongation", "well")]
    if not (all(math.isfinite(v) for v in scal) and np.all(np.isfinite(m["qs_residuals"]))):
        return failure_result("non-finite metrics: " + ", ".join(f"{v:.3e}" for v in scal), info), None
    if PHYSICS == "qi":
        t2 = time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                m.update(qi_metrics(eq.state, eq.runtime))
        except Exception as exc:
            return failure_result(f"QI metric evaluation failed: {type(exc).__name__}: {exc}"[:2000], info), None
        info["t_qi"] = time.perf_counter() - t2
        if not (math.isfinite(m["qs_total"]) and math.isfinite(m["mirror"])
                and math.isfinite(m.get("maxj", 0.0)) and math.isfinite(m.get("bmaxsp", 0.0))
                and np.all(np.isfinite(m["qs_residuals"]))):
            return failure_result("non-finite QI metrics", info), None
    if MERC_OPTIONS is not None:     # any physics (QS or QI)
        try:
            from .mercier import mercier_metric
            m["merc"] = float(mercier_metric(eq.state, eq.runtime, **MERC_OPTIONS)["soft_min"])
        except Exception as exc:  # noqa: BLE001
            return failure_result(f"Mercier metric failed: {type(exc).__name__}: {exc}"[:2000], info), None
        if not math.isfinite(m["merc"]):
            return failure_result("non-finite Mercier metric", info), None
    res = EqResult(True, m["qs_residuals"], m["qs_total"], m["aspect"], m["mean_iota"],
                   m["iota_axis"], m["iota_edge"], m["elongation"], m["well"], "", info)
    res.volume, res.aminor = m["volume"], m["aminor"]
    res.mirror = float(m.get("mirror", float("nan")))
    res.maxj = float(m.get("maxj", float("nan")))
    res.merc = float(m.get("merc", float("nan")))
    res.bmaxsp = float(m.get("bmaxsp", float("nan")))
    if return_state:
        t2 = time.perf_counter()
        try:
            res.vmex_state = VmexStateArrays.from_equilibrium(eq, inp, step=step)
        except Exception as exc:  # noqa: BLE001  (the metrics stay valid; consumers re-solve)
            info["state_error"] = f"{type(exc).__name__}: {exc}"[:500]
        info["t_state"] = time.perf_counter() - t2
    return res, wout


def _with_size(res: EqResult, m: dict) -> EqResult:
    """Attach volume / minor radius (and the QI mirror ratio) from a metric dict to an EqResult."""
    res.volume, res.aminor = float(m.get("volume", float("nan"))), float(m.get("aminor", float("nan")))
    res.mirror = float(m.get("mirror", float("nan")))
    res.maxj = float(m.get("maxj", float("nan")))
    res.merc = float(m.get("merc", float("nan")))
    res.bmaxsp = float(m.get("bmaxsp", float("nan")))
    return res


def evaluate(boundary: dict, step: int, *, return_state: bool = False) -> EqResult:
    """One cold VMEX fixed-boundary solve at PHR resolution/grid; never raises on failure.

    Raises ``ValueError`` only for malformed input (bad names, missing rc(0,0),
    non-finite values) -- programming errors, not solver failures.

    ``return_state=True`` (opt-in) also attaches the converged state as
    ``res.vmex_state`` (:class:`VmexStateArrays`, converged results only); the
    metrics are bitwise those of the default call.
    """
    return _solve_and_measure(boundary, step, return_state)[0]


def resolve_ns_ladder(ns_ladder, ftol_array=None, niter_array=None):
    """-> (ns, ftol, niter) tuples from a ladder spec.

    ``ns_ladder``: a key of :data:`NS_LADDERS`; a sequence of ``(ns, ftol, niter)``
    stages; or a sequence of ``ns`` ints together with ``ftol_array`` and
    ``niter_array`` (PHR's NS_ARRAY alone means PHR's ftol/niter).
    """
    if isinstance(ns_ladder, str):
        if ns_ladder not in NS_LADDERS:
            raise ValueError(f"unknown ns ladder {ns_ladder!r}; known: {sorted(NS_LADDERS)}")
        ns_ladder = NS_LADDERS[ns_ladder]
    stages = list(ns_ladder)
    if not stages:
        raise ValueError("empty ns ladder")
    if all(np.ndim(s) == 1 and len(s) == 3 for s in stages):
        if ftol_array is not None or niter_array is not None:
            raise ValueError("give ftol/niter either inside the (ns, ftol, niter) stages or separately")
        ns = tuple(int(s[0]) for s in stages)
        ftol = tuple(float(s[1]) for s in stages)
        niter = tuple(int(s[2]) for s in stages)
    elif all(np.ndim(s) == 0 for s in stages):
        ns = tuple(int(s) for s in stages)
        if ftol_array is None and niter_array is None and ns == tuple(NS_ARRAY):
            ftol, niter = tuple(FTOL_ARRAY), tuple(NITER_ARRAY)
        elif ftol_array is None or niter_array is None:
            raise ValueError("an ns-only ladder needs ftol_array and niter_array "
                             "(or use (ns, ftol, niter) stages / a NS_LADDERS key)")
        else:
            ftol = tuple(float(v) for v in ftol_array)
            niter = tuple(int(v) for v in niter_array)
    else:
        raise ValueError(f"bad ns ladder {ns_ladder!r}")
    if not (len(ns) == len(ftol) == len(niter)):
        raise ValueError("ns, ftol and niter stages differ in length")
    if any(n < 3 for n in ns) or any(b <= a for a, b in zip(ns, ns[1:])):
        raise ValueError(f"ns ladder must be increasing with ns >= 3, got {ns}")
    if any(not (f > 0.0 and math.isfinite(f)) for f in ftol) or any(i < 1 for i in niter):
        raise ValueError(f"bad ftol/niter stages {ftol} / {niter}")
    return ns, ftol, niter


def evaluate_resolution(boundary: dict, mpol: int, ntheta: int, nzeta: int, ns_ladder="phr", *,
                        ntor: int | None = None, ftol_array=None, niter_array=None,
                        return_state: bool = False) -> EqResult:
    """PHR metrics of ``boundary`` from a cold VMEX solve at a chosen resolution (roadmap G3).

    Same deck, boundary handling, zeroed axis guess, solver options (VMEC2000
    failure semantics) and metric code (VMEC2000 ``bsubvmnc``, simsopt QS on
    the 63x64 grid at s = 0, 0.1, .., 1, PHR elongation) as :func:`evaluate`;
    only ``mpol``/``ntor`` (default ``ntor = mpol``), the angular grid
    ``ntheta x nzeta`` and the radial ladder change. ``ns_ladder``: see
    :func:`resolve_ns_ladder` (default ``"phr"``: NS 16/50, FTOL 1e-16/1e-11,
    NITER 600/2000). ``evaluate_resolution(b, 3 + step, 16, 14, "phr")`` is
    bitwise ``evaluate(b, step)``. Boundary modes with ``m >= mpol`` or
    ``|n| > ntor`` are dropped (the PHR surface is 5/5, so mpol >= 6 keeps all).
    Never raises on a failed solve (``converged=False``); ``ValueError`` for
    malformed input. ``info`` records the resolution and ladder.

    G3 comparison: ``evaluate_resolution(b, 8, 22, 20, "ns101")`` (VMEC's
    2M+6 / 2N+4 grid rule at mpol = ntor = 8) and ``"ns101_b"`` (second ladder).
    """
    ns, ftol, niter = resolve_ns_ladder(ns_ladder, ftol_array, niter_array)
    inp = vmec_input_at(boundary, mpol, ntheta, nzeta, ns, ftol, niter, ntor=ntor)
    info = dict(step=None, mpol=int(mpol), ntor=int(mpol if ntor is None else ntor),
                ntheta=int(ntheta), nzeta=int(nzeta), ns_array=list(ns), ftol_array=list(ftol),
                niter_array=list(niter))
    with vmex_device_ctx():
        return _solve_input_and_measure(inp, info, step=None, return_state=return_state)[0]


def write_wout(boundary: dict, step: int, path) -> EqResult:
    """Solve like :func:`evaluate` and, if converged, write a VMEC2000-layout ``wout_*.nc``.

    The file carries the VMEC2000 ``bsubvmnc`` (see :func:`vmec2000_bsubvmnc`), so
    simsopt's ``Vmec(path)`` / ``QuasisymmetryRatioResidual`` reproduce the
    returned metrics. Nothing is written for a failed solve (``res.converged``
    is False, ``res.info['wout']`` absent).
    """
    res, wout = _solve_and_measure(boundary, step)
    if wout is not None:
        from vmex.core.wout import write_wout as _write
        res.info["wout"] = str(_write(path, wout))
    return res
