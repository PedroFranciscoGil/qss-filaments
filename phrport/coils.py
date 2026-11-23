"""XCoil coil layer for the PHR envelope port (component "coils").

What this module does
---------------------
``CoilSolver.solve(boundary)`` solves, on a fixed plasma boundary, the PHR coil subproblem

    min_x  c0_local(x; s)   s.t.  per-coil length <= 4, pointwise curvature <= 3, MSC <= 4,
                                  coil-coil >= 0.1, coil-surface >= 0.15          (PHR thresholds)

with XCoil's device-resident PHR augmented Lagrangian + Levenberg-Marquardt solver (a sandbox
COPY of xcoil/fixed/device_lm.py ``_build.outer`` and of ParetoSolver.full / correct), where

  * the objective residual is PHR's "local" squared flux (simsopt fluxobjective
    squared_flux_pure(definition="local")):
        c0_local = 0.5 * sum_ij (B.n)^2/|B|^2 |N_ij| / (nphi ntheta)
    on PHR's flux quadrature: the simsopt Vmec boundary grid of the PHR script
    (range='full torus', nphi = ntheta = 50, phi_j = j/50, theta_i = i/50, no offset).
    The 2500-point grid is folded exactly onto 13 x 50 points with multiplicities 2 / 4 using the
    nfp and stellarator symmetry of grid, boundary and coil set (the sum is identical).
    Residual r_q = sqrt(0.5 w_q / c0_ref) (B.n)_q/|B_q|, w_q = mult_q |N_q| / (nphi ntheta), so
    the solver objective is F = c0_local / c0_ref + w_arc mean(arclength residual^2).
    The Jacobian is analytic: d(n.B/|B|) = (1/|B|) (n - rho u) . dB  (rho = n.B/|B|, u = B/|B|),
    evaluated with a copy of XCoil's coil-scan Biot-Savart Jacobian kernel (fp64 or mixed
    precision), with the "normal" replaced by m = n - rho u;
  * three free currents with fixed total (as PHR: base_currents 0..2 free, the 4th = total - sum):
    I_c = I0 (1 + y_c), c < 3, I_3 = I0 (1 - y_0 - y_1 - y_2), I0 = total_current / 4.
    The local flux is invariant under a common current scale, so this is the same feasible set
    as PHR's; current Jacobian columns come from the per-coil partial fields of the same kernel;
  * the surface enters only through traced arrays (flux points / normals / weights and the
    coil-surface distance grid), so ONE compiled program serves every surface (no per-surface
    recompiles; compile once per process and CoilConfig);
  * certification (geometric, FP64, against the TRUE limits; XCoil verify/certify functions:
    4096-sample max curvature with golden refinement, MSC, exact segment / triangle distances,
    Gauss linking numbers) with the triangulated surface as a traced argument too.

Deviation from XCoil's device AL-LM (option ``lagr_curv``, default on): the LM model matrix
also contains the PSD-clipped per-coil Hessian of lam+^T g over the length / curvature / MSC rows
(XCoil's IPM "lagrangian_hessian=blockdiag" term, problem_h.lagrangian_blockdiag); it gives
slightly lower c0 and ~10x smaller KKT residuals on the PHR checkpoint surfaces.

``CoilSolver.envelope_grad(boundary, result, names)`` returns d c0_local*/d boundary[names] by the
envelope theorem: the Lagrangian  c0(x*, s) + c0_ref lam*^T g(x*, s)  differentiated by JAX with
respect to the boundary Fourier coefficients at fixed coils x* and fixed KKT multipliers lam*.
lam* (= result.lam) is re-estimated at x* by NNLS on the near-active rows,
min ||grad_x F + A_act^T z||, z >= 0 (XCoil CoilProblem.kkt; the PHR script recovers its nu from
the same stationarity by least squares); the AL's own max(0, lam + rho g) is kept in
info["lam_plus"] (it is noisier: 15-20% error on a cancellation-dominated component in the tests).
Only the coil-surface rows of g depend on the surface. The arclength gauge term of F does not
depend on the surface; its effect on c0* through x*(s) is not included (measured < 1e-4 relative).

Phase-0 additions (roadmap R1/G4, S-a, S-b; all opt-in, the default solve is bit-identical):
  * every result: info['c0_lagr'] = c0 + c0_ref lam.g (Lagrangian value), info['elasticity'][block]
    = c0_ref sum lam[block] / c0, info['bn_ratio_int'] = sum w|B.n| / sum w|B| on the 50x50 grid
    (ratio of integrals; 'mean_bn_area' is the area-weighted mean of |B.n|/|B| and
    'mean_bn_unweighted' the grid-point mean PHR reports as avg_Bn_over_B), info['v_soft'],
    info['soft_ok'], info['d_cs_soft'];
  * flux_metrics_full_grid / flux_metrics_unfolded: the same metrics on unfolded full-torus grids
    with a grid-convergence check (default 128x128 vs 200x200, refined to 400x400 if needed);
  * final_coil_stage(boundary): every start; every ok unlinked result polished to certified
    v <= 1e-5 (tightening the solver-row limits by the row/certificate mismatch where needed);
    the lowest POLISHED Lagrangian among those that reach it wins (ranking after the polish:
    certification at the cold stage fails mostly through the curvature-sampling gap the polish
    removes); all metrics, every polished candidate and the chosen start in info['final_stage'];
  * polish(..., ftol, viol_tol);
  * solve(..., should_abort=callable) (S-a): checked before every fallback start, aborts with
    ok=False and info['aborted'] = True;
  * solve(..., certify=False) (S-b): no certification (runtime argument, no recompilation),
    ok = info['soft_ok'] from the FP64 solver rows plus the Gauss linking check;
    CoilSolver(cold_cache=True) / solve(..., reuse=True): the most recent cold solve (per-start
    solver states and finished results, keyed by a surface hash) is reused by an identical later
    cold solve (bit-identical; certifying a surface first solved without certification replays
    the stored states instead of re-solving).

Phase-2 additions (roadmap B2, anchored coil oracle of phrport.phr --coil-oracle anchored; opt-in, the
default solve is bit-identical): CoilConfig.rho_warm_reset and solve(..., rho_reset=, warm_k_max=,
warm_fallback=) -- a warm solve restarted at rho0 with lam_al kept, a runtime outer budget, and a warm
result returned without the cold fallback (info['warm_opts']).

XCoil (formerly XCoil) is the coil backend: the installed ``xcoil`` package (pyproject.toml pins the commit);
PHRPORT_XCOIL selects a local checkout (phrport.backend).
"""
from __future__ import annotations

import phrport  # noqa: F401  (sets sys.pycache_prefix first)

import copy
import hashlib
import math
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict, replace

from phrport.backend import select_xcoil as _select_xcoil              # noqa: E402

XCOIL_BACKEND = _select_xcoil()   # puts a selected XCoil tree first on sys.path (see phrport.backend)

import numpy as np                                                          # noqa: E402
from phrport import problem as _PROB
import jax                                                                  # noqa: E402
import jax.numpy as jnp                                                     # noqa: E402
from jax import lax                                                         # noqa: E402

try:
    import xcoil                                                            # noqa: E402,F401  (x64)
    from xcoil.curves import fourier as fr                                  # noqa: E402
    from xcoil.curves import symmetry as sym_mod                            # noqa: E402
    from xcoil.constraints import pointwise as pw                           # noqa: E402
    from xcoil.field import residuals as rs                                 # noqa: E402
    from xcoil.field.jacobian import MU0_4PI, copy_tables                   # noqa: E402
    from xcoil.homotopy import boundary as bd                               # noqa: E402
    from xcoil.homotopy import problem_h as ph                              # noqa: E402
    from xcoil.homotopy.config import for_device                            # noqa: E402
    from xcoil.fixed.device_lm import _lancelot                             # noqa: E402
    from xcoil.fixed import starts as starts_mod                            # noqa: E402
    from xcoil.verify import certify as cf                                  # noqa: E402
except ImportError as _e:                                                   # pragma: no cover
    from phrport.backend import XcoilUnavailable, INSTALL_HINT
    raise XcoilUnavailable(f"the XCoil coil backend ({XCOIL_BACKEND.describe()}) could not be imported: {_e}. "
                           f"{INSTALL_HINT}") from _e
try:                                                                        # Lp-curvature rows: XCoil >= 1.4.0
    from xcoil.constraints.lp_curvature import lp_curvature                 # noqa: E402
except ImportError:                                                         # an older XCoil checkout
    lp_curvature = None
try:                                                                        # ordering rows: XCoil >= 1.3.0
    from xcoil.constraints import ordering as od                            # noqa: E402
except ImportError:                                                         # an older XCoil checkout
    od = None
try:                                                                        # max-force rows: XCoil >= 1.2.0
    from xcoil.constraints import forces as fc_forces                       # noqa: E402
except ImportError:                                                         # an older XCoil checkout
    fc_forces = None

jax.config.update("jax_enable_x64", True)
F64 = jnp.float64
_NAME_RE = re.compile(r"^\s*(rc|zs)\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*$")


# ================================================================================================
# public dataclasses
# ================================================================================================

@dataclass
class CoilConfig:
    # --- interface fields (PHR-matched defaults) ---------------------------------------------
    order: int = int(_PROB.get("coils", "order", 5))
    n_coils: int = int(_PROB.get("coils", "n_coils", 4))
    per_coil_length: float = float(_PROB.get("coils", "per_coil_length", 4.0))
    max_curvature: float = float(_PROB.get("coils", "max_curvature", 3.0))
    max_msc: float = float(_PROB.get("coils", "max_msc", 4.0))
    min_cc: float = float(_PROB.get("coils", "min_cc", 0.1))
    min_cs: float = float(_PROB.get("coils", "min_cs", 0.15))
    # problem profiles: 'total' caps the summed base-coil length per half period at total_length
    # (XCoil per_coil_length=False) instead of each coil at per_coil_length (PHR)
    length_mode: str = str(_PROB.get("coils", "length_mode", "per_coil"))
    total_length: float = float(_PROB.get("coils", "total_length", 16.0))
    curvature_margin: float = float(_PROB.get("coils", "curvature_margin", 0.0))  # XCoil beta on the curvature / MSC rows (PHR: none); removed at certification
    arc_penalty: bool = False           # arclength-variation penalty of simsopt auglag_qa.py (PHR: none)
    free_currents: bool = True          # 3 free currents, fixed total (PHR)
    precision: str = str(_PROB.get("coils", "precision", "mixed"))  # "mixed" (FP32 kernels, FP64 accumulation) or "fp64"
    start: str = str(_PROB.get("coils", "start", "offset_3d_L85"))   # lowA_ncsx: profile opt-in (default unchanged)
    link_guard: str = str(_PROB.get("coils", "link_guard", "none"))      # none | link | link_floor (opt-in)
    link_floor_frac: float = float(_PROB.get("coils", "link_floor_frac", 0.15))
    # --- extras (all have defaults; not part of the frozen interface) ------------------------
    nfp: int = int(_PROB.get("equilibrium", "nfp", 2))
    total_current: float = float(_PROB.get("coils", "total_current", 1.25e6))       # sum of the base-coil currents [A] (PHR B0 = 1 T)
    nq: int = int(_PROB.get("coils", "nq", 128))   # coil quadrature (PHR COIL_QUADPOINTS); lowA_ncsx: profile opt-in (default unchanged)
    phr_grid: tuple = tuple(_PROB.get("coils", "phr_grid", (50, 50)))          # PHR flux grid (simsopt Vmec default, full torus)
    dist_grid: tuple = tuple(_PROB.get("coils", "dist_grid", (128, 64)))        # coil-surface soft-min grid (full torus, phi x theta); XCoil's
                                        # qa_auglag default (64, 64) under-resolves phi: up to 1.1 % (8 % on
                                        # folded trial surfaces) certified d_cs violation
    verify_grid: tuple = (64, 128)      # certification triangulation, per field period x theta
    nq_verify: int = 768                # certification length quadrature
    mpol_min: int = 5                   # boundary mode table at least this big (input.nfp2_QA)
    ntor_min: int = 5
    c0_ref: float = 1e-3                # F = c0_local / c0_ref + w_arc * arc term
    w_arc: float = 1e-3                 # XCoil arclength gauge regularizer (f_ref units)
    softmin_alpha: float = 100.0
    inner: int = 20                     # XCoil budget used in the PHR gap studies
    inner_first: int = 400
    first_prog: float = 0.01
    rho0: float = 10.0
    max_outer: int = 60
    warm_max_outer: int = int(_PROB.get("coils", "warm_max_outer", 15))   # profile: more for 6+ coils
    viol_tol: float = float(_PROB.get("coils", "viol_tol", 1e-3))
    ftol: float = float(_PROB.get("coils", "ftol", 1e-3))
    rho_corr: float = 1e6
    cc_rows: int = 128
    ok_viol: float = float(_PROB.get("coils", "ok_viol", 2e-2))              # ok=False if the certified true-limit violation exceeds this
    fallback_starts: tuple = ("clear", "L85_tilt+8", "L70", "L85_tilt-8", "L100")
    max_starts: int = 3                 # cold solves (primary + fallbacks) before giving up; starts that
                                        # cannot be built (e.g. L85 shorter than the clearing offset) are
                                        # skipped without counting
    certify: bool = True
    distance_margin: float = float(_PROB.get("coils", "distance_margin", 0.0))    # XCoil beta_d on the coil-coil / coil-surface rows (PHR: none)
    kkt_act_tol: float = 1e-3           # rows with g > -tol enter the NNLS multiplier estimate
    lagr_curv: bool = True              # add the PSD per-coil Hessian of lam+^T g (length, curvature,
                                        # MSC rows) to the LM model (XCoil lagrangian_blockdiag)
    # --- max force (opt-in, xcoil.constraints.forces): pointwise rows |dF/dl|_q / F_lim - 1 <= 0 on every base coil and
    # quadrature point, F_lim from a per-turn limit at a reference scale (AL coil paper Table II convention)
    max_force: float = float(_PROB.get("coils", "max_force", 0.0))          # per-turn limit [MN/m] at the reference scale (0 = off)
    force_scale: float = float(_PROB.get("coils", "force_scale", 10.1266))  # length factor to the reference scale (LP QA ARIES R0)
    force_current: float = float(_PROB.get("coils", "force_current", 72.14e6))  # total current per half period at that scale [A]
    force_turns: float = float(_PROB.get("coils", "force_turns", 0.0))      # turns per coil (0 = 800 / n_coils)
    force_radius: float = float(_PROB.get("coils", "force_radius", 0.30))   # conductor radius at that scale [m]
    force_margin: float = float(_PROB.get("coils", "force_margin", 0.0))    # optimization margin on the force rows (rows at (1 - m) F_lim; removed at certification)
    # --- Lp curvature (opt-in): per-coil rows J_c / lp_curvature_max - 1 <= 0 with the simsopt LpCurveCurvature
    # functional J_c = (1/p) int max(kappa - kappa_0, 0)^p dl (kappa_0 = lp_curvature_threshold); 0 = off
    lp_curvature_max: float = float(_PROB.get("coils", "lp_curvature_max", 0.0))
    lp_curvature_p: float = float(_PROB.get("coils", "lp_curvature_p", 2.0))
    lp_curvature_threshold: float = float(_PROB.get("coils", "lp_curvature_threshold", 0.0))
    # --- phase 2 (roadmap B2; solve-time only, not part of the compiled programs) ---------------
    rho_warm_reset: bool = False        # warm starts restart the AL penalty at rho0 (keeping the warm
                                        # result's lam_al) instead of inheriting its final rho;
                                        # solve(..., rho_reset=) overrides it per call
    # --- toroidal ordering ("slot") rows (opt-in; off = the previous programs bit for bit) --------
    # neighbouring base coils, and each end coil with its stellarator image across phi = 0 / pi/nfp,
    # must not cross in the (theta, phi) projection about the boundary's geometric axis (theta-mean of
    # the distance grid): for every point k of the left coil, softmin_k' [phi_R,k' - phi_L,k
    # + slot_kappa (1 - cos(theta_L,k - theta_R,k'))] >= slot_gap_deg; soft-max over slot_seg segments
    slot_rows: bool = bool(_PROB.get("coils", "slot_rows", False))
    slot_gap_deg: float = float(_PROB.get("coils", "slot_gap_deg", 1.0))
    slot_kappa: float = float(_PROB.get("coils", "slot_kappa", 20.0))      # theta-window penalty [rad]
    slot_beta: float = float(_PROB.get("coils", "slot_beta", 500.0))       # soft-min sharpness [1/rad]
    slot_seg: int = int(_PROB.get("coils", "slot_seg", 16))                # rows per coil pair
    # reference axis of the slot rows' poloidal angle: "frozen_step" = the boundary's geometric axis (m = 0 modes) at
    # the start of each continuation step (set by the outer loop; standalone solves use the given boundary's axis),
    # fixed within the step so the rows depend on the coils only; "live" = recomputed from the current boundary at
    # every evaluation (the rows then also depend on the surface, and the envelope gradient carries that term)
    ordering_axis: str = str(_PROB.get("coils", "ordering_axis", "frozen_step"))
    # seed_sensitivity (Nov 2026, opt-in): JSON of base-coil points (C, n, 3) + currents; the first cold solve of the
    # process starts from these coils (label 'seed') instead of cfg.start; "" = off
    seed_coils: str = str(_PROB.get("coils", "seed_coils", ""))


# CoilConfig fields read by CoilSolver.solve only: they do not enter the compiled programs, so they are
# left out of the program-cache key (a solver differing only in them shares the compiled programs)
_SOLVE_ONLY_FIELDS = ("rho_warm_reset", "seed_coils")
_SEED_USED = [False]                    # seed_coils: consumed by the first cold solve of the process


@dataclass
class CoilResult:
    ok: bool
    c0_local: float
    flux_F: float
    x: np.ndarray
    currents: np.ndarray
    lam: np.ndarray
    certified: bool
    linked: bool
    info: dict


# ================================================================================================
# boundary dictionaries
# ================================================================================================

def parse_name(name):
    """'rc(m,n)' / 'zs(m,n)' -> ('rc', m, n)."""
    mt = _NAME_RE.match(str(name))
    if not mt:
        raise ValueError(f"bad boundary coefficient name {name!r}")
    return mt.group(1), int(mt.group(2)), int(mt.group(3))


def boundary_modes(boundary):
    """dict -> {('rc'|'zs', m, n): value}; ignores entries that are not rc/zs names."""
    out = {}
    for k, v in boundary.items():
        try:
            key = parse_name(k)
        except ValueError:
            continue
        out[key] = float(v)
    return out


def _ceil_sig(v, digits=2):
    """v > 0 rounded UP to `digits` significant digits (0 for v <= 0)."""
    if not (v > 0.0) or not math.isfinite(v):
        return 0.0
    e = math.floor(math.log10(v)) - digits + 1
    return float(f"{math.ceil(v / 10.0 ** e * (1.0 - 1e-12))}e{e}")


def _mode_list(mpol, ntor):
    return [(m, n) for m in range(mpol + 1) for n in range(-ntor, ntor + 1) if not (m == 0 and n < 0)]


# ================================================================================================
# jnp kernels (sandbox copies / variants of XCoil field kernels with per-coil currents)
# ================================================================================================

def _field_I(xc, Iphys, cb, sym, X, kdt=None):
    """B (N, 3) at points X from base-coil DOFs xc with per-PHYSICAL-coil currents Iphys (J,).
    kdt=None: FP64; kdt=float32: FP64 curves and separations, kdt kernel, FP64 sums (as
    xcoil.field.mixed.field_on)."""
    g, gd, _ = fr.coils_curves(xc, cb, sym.C)
    G, Gd = sym_mod.expand(g, sym), sym_mod.expand(gd, sym)
    Q = G.shape[1]
    f64 = xc.dtype

    def step(B, c):
        gc, gdc, Ii = c
        D = X[:, None, :] - gc[None]
        gk = gdc
        if kdt is not None:
            D = D.astype(kdt)
            gk = gdc.astype(kdt)
        inv3 = jnp.sum(D * D, -1) ** -1.5
        cr = jnp.cross(gk[None], D)
        return B + (Ii / Q) * jnp.sum((cr * inv3[..., None]).astype(f64), 1), None

    B, _ = lax.scan(step, jnp.zeros((X.shape[0], 3), f64), (G, Gd, Iphys))
    return MU0_4PI * B


def _local_rJ(xc, Ic, cb, sym, X, n, sw, kdt=None, Bp=None):
    """Local-flux residual r_q = sw_q (n_q.B_q)/|B_q| and its Jacobian.

    Returns r (Ns,), J_x (Ns, C*3K) and R_unit (C, Ns) with R_unit[c] = d r / d I_c (per unit
    current of base coil c and all its symmetry copies).  Uses
        d r_q = (sw_q/|B_q|) m_q . dB_q,   m_q = n_q - rho_q u_q,
    and a copy of xcoil.field.mixed.residual_and_jacobian (per-coil currents added)."""
    f64 = xc.dtype
    Iphys = Ic[sym.coil] * sym.sign
    B = _field_I(xc, Iphys, cb, sym, X, kdt)
    if Bp is not None:                       # finite beta: plasma field (coil-independent) on the grid
        B = B + Bp
    Bm = jnp.linalg.norm(B, axis=1)
    u = B / Bm[:, None]
    rho = jnp.sum(B * n, 1) / Bm
    r = sw * rho
    m = n - rho[:, None] * u
    cq = sw / Bm
    C, K = sym.C, cb.K
    Xc = xc.reshape(C, 3, K)
    g = jnp.einsum("qk,cdk->cqd", cb.B0, Xc)
    gd = jnp.einsum("qk,cdk->cqd", cb.B1, Xc)
    Rc, sc = copy_tables(sym)
    nq = g.shape[1]
    Ns = X.shape[0]
    kd = f64 if kdt is None else kdt
    mb = m.astype(kd)[:, None, None, :]
    B0k, B1k = cb.B0.astype(kd), cb.B1.astype(kd)

    def per_coil(_, inp):
        gcc, gdc, Rj, sj, Icc = inp
        Gj = jnp.einsum("jab,kb->jka", Rj, gcc)                         # FP64 (ncopy, nq, 3)
        Gdj = jnp.einsum("jab,kb->jka", Rj, gdc).astype(kd)
        D = (X[:, None, None, :] - Gj[None]).astype(kd)                  # FP64 separation -> kd
        rho2 = jnp.sum(D * D, axis=-1)
        inv3 = rho2 ** -1.5
        inv5 = inv3 / rho2
        uu = jnp.sum(mb * jnp.cross(Gdj[None], D), axis=-1)             # m . (gamma' x Delta)
        a1 = ((cq * (MU0_4PI / nq))[:, None] * sj[None, :]).astype(kd)   # unit current
        r_unit = jnp.einsum("sj,sjk->s", a1, uu * inv3).astype(f64)
        a = a1 * Icc.astype(kd)
        dgp = a[:, :, None, None] * jnp.cross(D, mb) * inv3[..., None]
        dg = a[:, :, None, None] * (-jnp.cross(mb, Gdj[None]) * inv3[..., None]
                                    + 3.0 * (uu * inv5)[..., None] * D)
        Rk = Rj.astype(kd)
        Gh = jnp.einsum("jba,sjkb->ska", Rk, dg)
        Ghp = jnp.einsum("jba,sjkb->ska", Rk, dgp)
        Jc = jnp.einsum("skd,km->sdm", Gh, B0k) + jnp.einsum("skd,km->sdm", Ghp, B1k)
        return None, (Jc.reshape(Ns, 3 * K).astype(f64), r_unit)

    _, (Js, Ru) = lax.scan(per_coil, None, (g, gd, Rc, sc, Ic))
    return r, jnp.transpose(Js, (1, 0, 2)).reshape(Ns, C * 3 * K), Ru


# ================================================================================================
# Lp curvature (simsopt LpCurveCurvature)
# ================================================================================================

# ================================================================================================
# toroidal ordering ("slot") rows
# ================================================================================================

SLOT_BETA_ROW = 50.0        # soft-max sharpness over a segment's points (normalized row units)


def _slot_nmax(dist_grid, nfp):
    """Axis harmonics of the ordering rows (as many as the distance grid resolves, at most 8)."""
    return max(1, min(8, int(dist_grid[0]) // (2 * nfp) - 1))


def slot_pair_names(C):
    return [f"{c}|{c + 1}" for c in range(C - 1)] + ["plane0|0", f"{C - 1}|plane"]


def boundary_axis(boundary, nfp, nmax):
    """Geometric axis (a, b) of a boundary dict {rc(m,n), zs(m,n)}: its m = 0 modes (xcoil.constraints.ordering)."""
    xm, xn, rc, zs = [], [], [], []
    for name, v in boundary.items():
        kind, m, n = parse_name(name)
        if m != 0:
            continue
        xm.append(m); xn.append(n)
        rc.append(float(v) if kind == "rc" else 0.0)
        zs.append(float(v) if kind == "zs" else 0.0)
    if od is None:                       # XCoil without the ordering module: no ordering rows, the axis is unused
        return np.zeros(nmax + 1), np.zeros(nmax + 1)
    return od.axis_from_modes(xm, xn, rc, zs, nfp, nmax)


# ================================================================================================
# compiled programs (one set per CoilConfig; surface arrays are traced arguments)
# ================================================================================================

def _phr_flux_grid(nfp, nphi, ntheta):
    """Representative points of PHR's full-torus grid phi_j = 2 pi j/nphi, theta_i = 2 pi i/ntheta
    and their multiplicities under the nfp rotation and stellarator symmetry (exact folding)."""
    if nphi % nfp:
        raise ValueError("PHR grid folding needs nphi % nfp == 0")
    P = nphi // nfp
    rows = list(range(P // 2 + 1))
    mult_row = [nfp * (1 if (j == 0 or 2 * j == P) else 2) for j in rows]
    assert sum(mult_row) == nphi
    PH, TH = np.meshgrid(2 * np.pi * np.asarray(rows) / nphi, 2 * np.pi * np.arange(ntheta) / ntheta,
                         indexing="ij")
    mult = np.repeat(np.asarray(mult_row, float), ntheta)
    return TH.ravel(), PH.ravel(), mult


def _full_grid_angles(nphi, ntheta):
    """XCoil 'full' grid: phi_j = (j + 1/2) 2 pi / nphi, theta_i = 2 pi i / ntheta."""
    PH, TH = np.meshgrid((np.arange(nphi) + 0.5) * 2 * np.pi / nphi, 2 * np.pi * np.arange(ntheta) / ntheta,
                         indexing="ij")
    return TH.ravel(), PH.ravel()



def _lk_dmin(g, gd, G, Gd, self_row):
    """Rounded Gauss linking numbers (Nphys,) and min point distance of one base coil (g, gd: (n, 3)) with every
    physical coil (G, Gd: (Nphys, n, 3)); self_row masks the base coil's own identity copy."""
    n = g.shape[0]
    dx = g[:, 0][None, :, None] - G[:, None, :, 0]
    dy = g[:, 1][None, :, None] - G[:, None, :, 1]
    dz = g[:, 2][None, :, None] - G[:, None, :, 2]
    a, b = gd[None, :, None, :], Gd[:, None, :, :]
    cx = a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]
    cy = a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]
    cz = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    r2 = dx * dx + dy * dy + dz * dz
    msk = self_row[:, None, None]
    lk = jnp.sum((cx * dx + cy * dy + cz * dz) * lax.rsqrt(jnp.where(msk, 1.0, r2)) ** 3, axis=(1, 2)) / (n * n * 4 * jnp.pi)
    dmin = jnp.sqrt(jnp.min(jnp.where(msk, jnp.inf, r2)))
    return jnp.where(self_row, 0.0, jnp.rint(lk)), dmin


class _Programs:
    """Jitted solver / evaluation / certification programs for one CoilConfig."""

    def __init__(self, cfg: CoilConfig):
        self.cfg = cfg
        C = int(cfg.n_coils)
        if cfg.precision not in ("mixed", "fp64"):
            raise ValueError(f"precision {cfg.precision!r}")
        kdt = jnp.float32 if cfg.precision == "mixed" else None
        self.kdt = kdt
        ov = dict(order=int(cfg.order), n_coils=C, kappa_lim=float(cfg.max_curvature),
                  msc_lim=float(cfg.max_msc), dcc=float(cfg.min_cc), dcs=float(cfg.min_cs),
                  beta_curv=float(cfg.curvature_margin), beta_d=float(cfg.distance_margin), per_coil_length=(cfg.length_mode == "per_coil"),
                  nq_opt=int(cfg.nq), grid_dist=tuple(cfg.dist_grid), w_arc=float(cfg.w_arc),
                  w_msc0=0.0, w_msc_final=0.0, softmin_alpha=float(cfg.softmin_alpha),
                  verify_grid=tuple(cfg.verify_grid), nq_verify=int(cfg.nq_verify), arc_var_lim=0.0)
        if cfg.arc_penalty:      # simsopt auglag_qa.py arclength penalty (xcoil user_problem.ARC)
            ov.update(arc_pen_thresh=1e-2, arc_pen_weight=1.0, arc_pen_intervals=-1)
        else:
            ov.update(arc_pen_thresh=0.0)
        hcfg = for_device("qa_auglag", **ov)
        self.hcfg = hcfg
        lim = ph.limits_from(hcfg)
        self.lim = lim
        cb = fr.make_basis("fourier", int(cfg.nq), int(cfg.order))
        sym = sym_mod.build(C, int(cfg.nfp), True)
        self.cb, self.sym = cb, sym
        lguard = str(cfg.link_guard)
        if lguard not in ("none", "link", "link_floor"):
            raise ValueError(f"link_guard {lguard!r}")
        if lguard != "none":
            cbl = fr.make_basis("fourier", 256, int(cfg.order))
            smask = np.stack([(np.asarray(sym.coil) == c) & np.asarray(sym.ident) for c in range(C)])
            floor_abs = float(cfg.link_floor_frac) * float(cfg.min_cc)

            def topo(xc):
                gg, gd, _ = fr.coils_curves(xc, cbl, C)
                G, Gd = sym_mod.expand(gg, sym), sym_mod.expand(gd, sym)
                out = [_lk_dmin(gg[c], gd[c], G, Gd, jnp.asarray(smask[c])) for c in range(C)]
                return jnp.stack([o[0] for o in out]), jnp.min(jnp.stack([o[1] for o in out]))

            def guard_ok(xc1, lk0, d0):
                lk1, d1 = topo(xc1)
                ok = jnp.all(lk1 == lk0)
                if lguard == "link_floor":
                    ok = ok & (d1 >= jnp.minimum(floor_abs, d0))
                return ok
            self._topo = topo
        K = cb.K
        nc = C * 3 * K
        ny = (C - 1) if cfg.free_currents else 0
        n = nc + ny
        nqq = cb.B0.shape[0]
        self.nc, self.ny, self.n = nc, ny, n
        if cfg.length_mode not in ("per_coil", "total"):
            raise ValueError(f"length_mode {cfg.length_mode!r}")
        cap = jnp.asarray(float(cfg.per_coil_length if cfg.length_mode == "per_coil" else cfg.total_length), F64)
        self.cap = cap
        self.blocks = pw.blocks_for(C, nqq, lim)
        m_pw = pw.n_constraints(C, nqq, lim.n_seg, lim.per_coil_length, lim.arc_var_lim > 0)
        slot_on = bool(cfg.slot_rows)
        n_slot = (C + 1) * int(cfg.slot_seg) if slot_on else 0
        if slot_on and od is None:
            from phrport.backend import XcoilUnavailable
            raise XcoilUnavailable("coils.slot_rows needs XCoil >= 1.3.0 (xcoil.constraints.ordering); the selected "
                                   f"tree ({XCOIL_BACKEND.describe()}) does not have it")
        if cfg.ordering_axis not in ("frozen_step", "live"):
            raise ValueError(f"coils.ordering_axis must be 'frozen_step' or 'live', not {cfg.ordering_axis!r}")
        if slot_on:
            if nqq % int(cfg.slot_seg):
                raise ValueError("slot_seg must divide the coil quadrature")
            self.blocks["slot"] = slice(m_pw, m_pw + n_slot)
        self.m = m_pw + n_slot
        # Lp curvature rows (opt-in): one row per base coil, after the slot rows
        self.lp_on = float(cfg.lp_curvature_max) > 0.0
        if self.lp_on and lp_curvature is None:
            from phrport.backend import XcoilUnavailable
            raise XcoilUnavailable("coils.lp_curvature_max needs XCoil >= 1.4.0 (xcoil.constraints.lp_curvature); the "
                                   f"selected tree ({XCOIL_BACKEND.describe()}) does not have it")
        if self.lp_on:
            self.blocks["lpcurv"] = slice(self.m, self.m + C)
            self.m = self.m + C
        lp_args = (float(cfg.lp_curvature_p), float(cfg.lp_curvature_threshold), float(cfg.lp_curvature_max))
        # max-force rows (opt-in): C * nqq pointwise rows appended after XCoil's (and the slot) rows
        self.force_on = float(cfg.max_force) > 0.0
        if self.force_on and fc_forces is None:
            from phrport.backend import XcoilUnavailable
            raise XcoilUnavailable("coils.max_force needs XCoil >= 1.2.0 (xcoil.constraints.forces); the selected "
                                   f"tree ({XCOIL_BACKEND.describe()}) does not have it")
        m_f0 = self.m
        if self.force_on:
            self.blocks = dict(self.blocks, force=slice(m_f0, m_f0 + C * nqq))
            self.m = m_f0 + C * nqq
            turns = float(cfg.force_turns) or 800.0 / C
            self.F_lim = fc_forces.normalized_limit(float(cfg.max_force) * 1e6, turns, cfg.force_scale,
                                                    cfg.force_current, cfg.total_current)
            self.F_delta = fc_forces.regularization_circ(float(cfg.force_radius) / float(cfg.force_scale))
            # per-turn force at the reference scale = |dF/dl| (normalized) * F_to_ref
            self.F_to_ref = (float(cfg.force_current) / float(cfg.total_current)) ** 2 / (float(cfg.force_scale) * turns)
        m = self.m
        nfp_ = int(cfg.nfp)
        dgrid = tuple(int(v) for v in cfg.dist_grid)
        slot_args = (nfp_, math.radians(float(cfg.slot_gap_deg)), float(cfg.slot_kappa),
                     float(cfg.slot_beta), int(cfg.slot_seg))
        ax_nmax = _slot_nmax(dgrid, nfp_)
        self.ax_nmax = ax_nmax
        live_axis = cfg.ordering_axis == "live"

        def slot_axis(Sd, AX):
            """(a, b) of the ordering rows: the frozen axis AX (2, nmax + 1), or the current boundary's (live)."""
            if live_axis:
                return od.axis_from_points(Sd, dgrid[0], dgrid[1], nfp_, ax_nmax)
            return AX[0], AX[1]
        self.slot_on, self.n_slot = slot_on, n_slot
        I0 = float(cfg.total_current) / C
        self.I0 = I0
        c_f = 1.0 / hcfg.f_ref
        sqcf = math.sqrt(c_f)
        c0_ref = float(cfg.c0_ref)
        off = np.zeros(m)
        off[self.blocks["curv"]] = lim.beta
        off[self.blocks["msc"]] = lim.beta
        off[self.blocks["cc"]] = lim.beta_d
        off[self.blocks["cs"]] = lim.beta_d
        if self.force_on:
            off[self.blocks["force"]] = float(cfg.force_margin)
        off = jnp.asarray(off)
        cc_sl = self.blocks["cc"]
        cc_K = min(int(cfg.cc_rows), cc_sl.stop - cc_sl.start)   # small coil sets have fewer rows than cc_rows
        pen = bool(cfg.arc_penalty)
        Cnq = C * nqq

        # ---- the flux grid (PHR) and its symmetry multiplicities --------------------------------
        nphi, nth = (int(v) for v in cfg.phr_grid)
        self.flux_TH, self.flux_PH, self.flux_mult = _phr_flux_grid(int(cfg.nfp), nphi, nth)
        self.n_full = nphi * nth
        umean_w = jnp.asarray(self.flux_mult / self.n_full)        # unweighted 2500-point mean
        self.dist_TH, self.dist_PH = _full_grid_angles(*cfg.dist_grid)

        def currents(y):
            if ny:
                return I0 * jnp.concatenate([1.0 + y, jnp.reshape(1.0 - jnp.sum(y), (1,))])
            return jnp.full((C,), I0, F64)
        self.currents = currents

        def split(x):
            return x[:nc], x[nc:]

        def reg_r(xc):
            r = rs.regularizer_residual(xc, jnp.asarray(1.0, F64), cb, C, hcfg, jnp.asarray(1.0, F64))
            return sqcf * jnp.concatenate([r[:Cnq], r[2 * Cnq + 1:]])

        def reg_J(xc):
            J = rs.regularizer_jacobian(xc, jnp.asarray(1.0, F64), cb, C, hcfg, jnp.asarray(1.0, F64))
            return sqcf * jnp.concatenate([J[:Cnq], J[2 * Cnq + 1:]])

        def sw_of(wf):
            return jnp.sqrt(0.5 * wf / c0_ref)

        def r_only(x, surf, kd=kdt):
            xc, y = split(x)
            Xf, nf, wf = surf[:3]
            I = currents(y)
            B = _field_I(xc, I[sym.coil] * sym.sign, cb, sym, Xf, kd)
            if len(surf) > 5:
                B = B + surf[5]
            rho = jnp.sum(B * nf, 1) / jnp.linalg.norm(B, axis=1)
            return jnp.concatenate([sw_of(wf) * rho, reg_r(xc)])

        def rJ(x, surf, kd=kdt):
            xc, y = split(x)
            Xf, nf, wf = surf[:3]
            I = currents(y)
            rB, JB, Ru = _local_rJ(xc, I, cb, sym, Xf, nf, sw_of(wf), kd, surf[5] if len(surf) > 5 else None)
            rR, JR = reg_r(xc), reg_J(xc)
            if ny:
                Jy = I0 * (Ru[:-1] - Ru[-1:]).T                       # (Ns, C-1)
                J = jnp.concatenate([jnp.concatenate([JB, Jy], 1),
                                     jnp.concatenate([JR, jnp.zeros((JR.shape[0], ny), F64)], 1)], 0)
            else:
                J = jnp.concatenate([JB, JR], 0)
            return jnp.concatenate([rB, rR]), J

        def g_slot(xc, Sd, AX):
            a, b = slot_axis(Sd, AX)
            return od.rows(fr.coils_curves(xc, cb, C)[0], a, b, *slot_args)

        lp_on = self.lp_on

        def g_lp(xc):
            _, gd, gdd = fr.coils_curves(xc, cb, C)
            return lp_curvature(gd, gdd, lp_args[0], lp_args[1]) / lp_args[2] - 1.0

        def g_geo(xc, Sd, AX, kd=None):
            gv = pw.values_x(xc, cb, sym, Sd, lim, cap, kd)
            parts = [gv] + ([g_slot(xc, Sd, AX)] if slot_on else []) + ([g_lp(xc)] if lp_on else [])
            return jnp.concatenate(parts) if len(parts) > 1 else gv
        self._g_geo = g_geo

        force_on = self.force_on
        if force_on:
            F_lim, F_delta, F_margin = self.F_lim, self.F_delta, float(cfg.force_margin)

            def g_force(x):
                """Max-force rows |dF/dl|_q / F_lim - 1 (C * nqq, FP64), coil DOFs and currents only."""
                xc, y = split(x)
                return (fc_forces.coil_forces(xc, currents(y), cb, sym, F_delta) / F_lim - (1.0 - F_margin)).ravel()
            A_force = jax.jacfwd(g_force)
            self._force_rows = jax.jit(g_force)
            self._forces = jax.jit(lambda xc, I: fc_forces.coil_forces(xc, I, cb, sym, F_delta))

        def g_only(x, surf, kd=kdt):
            g = g_geo(x[:nc], surf[3], surf[4], kd)
            return jnp.concatenate([g, g_force(x)]) if force_on else g

        def A_of(x, surf, h, kd=kdt):
            xc = x[:nc]
            Sd = surf[3]
            full = lambda _: pw.constraint_jacobian(xc, cb, sym, Sd, lim, cap, kd)
            if not cc_K or lim.n_seg:
                A = full(None)
            else:
                hcc = h[cc_sl]
                _, idx = lax.top_k(hcc, cc_K)
                part = lambda _: pw.constraint_jacobian(xc, cb, sym, Sd, lim, cap, kd,
                                                        cc_rows=(idx // nqq, idx % nqq))
                A = lax.cond(jnp.sum(hcc > 0) <= cc_K, part, full, None)
            if slot_on:
                A = jnp.concatenate([A, jax.jacfwd(g_slot)(xc, Sd, surf[4])], 0)
            if lp_on:
                A = jnp.concatenate([A, jax.jacfwd(g_lp)(xc)], 0)
            if ny:
                A = jnp.concatenate([A, jnp.zeros((A.shape[0], ny), F64)], 1)
            if force_on:
                A = jnp.concatenate([A, A_force(x)], 0)
            return A

        self._r_only, self._rJ, self._g_only, self._A_of = r_only, rJ, g_only, A_of
        lagr_curv = bool(cfg.lagr_curv)
        bl = self.blocks
        kl_ = lim.kappa_lim
        msc_l = lim.msc_lim

        def geom_hess(xc, z):
            """Per-coil PSD-clipped Hessian of z^T g over the length, curvature and MSC rows (copy of
            xcoil.homotopy.problem_h.lagrangian_blockdiag without Tables), block diagonal (nc, nc)."""
            xcc = xc.reshape(C, -1)
            zk = z[bl["curv"]].reshape(C, nqq)
            zm = z[bl["msc"]]
            zL = z[bl["L"]] if lim.per_coil_length else jnp.full(C, z[bl["L"]][0])   # total-length row: one z

            def coil_term(v, zkc, zmc, zLc):
                Lv, kap, M = pw._coil_geom(v, cb)
                return jnp.dot(zkc, kap) / kl_ + zmc * M / msc_l + zLc * Lv / cap
            Hc = jax.vmap(jax.hessian(coil_term))(xcc, zk, zm, zL)
            w, V = jnp.linalg.eigh(0.5 * (Hc + jnp.swapaxes(Hc, 1, 2)))
            Hc = jnp.einsum("cij,cj,ckj->cik", V, jnp.maximum(w, 0.0), V)
            return jax.scipy.linalg.block_diag(*[Hc[c] for c in range(C)])
        self._geom_hess = geom_hess

        def outer(x, lam, rho, mu, nu, omega, surf, inner, prog=0.0):
            """One outer AL iteration (copy of xcoil.fixed.device_lm._build.outer: inner LM loop
            to |grad Phi|_inf / (1 + |grad F|_inf) <= omega, or - prog > 0 - until 10 accepted
            steps gain less than prog * Phi; at most `inner` accepted steps, 4 inner trials)."""
            In = jnp.eye(n, dtype=F64)

            def phi_of(r, g):
                h = jnp.maximum(0.0, g + lam / rho)
                return r @ r + 0.5 * rho * (h @ h)

            def derivs(x, g=None):
                r, J = rJ(x, surf)
                if g is None:
                    g = g_only(x, surf)
                h = jnp.maximum(0.0, g + lam / rho)
                A = A_of(x, surf, h)
                gF = 2.0 * (J.T @ r)
                grad = gF + rho * (A.T @ h)
                Aa = A * (h > 0)[:, None]
                H = 2.0 * (J.T @ J) + rho * (Aa.T @ Aa)
                if pen:
                    Hp = rs.arc_penalty_curvature(x[:nc], cb, C, hcfg, c_f)
                    H = H.at[:nc, :nc].add(Hp)
                if lagr_curv:
                    H = H.at[:nc, :nc].add(geom_hess(x[:nc], rho * h))
                stat = jnp.max(jnp.abs(grad)) / (1.0 + jnp.max(jnp.abs(gF)))
                return r, g, grad, H, stat

            r, g, grad, H, stat = derivs(x)
            if lguard != "none":
                lk0, d0 = topo(x[:nc])                    # topology of the outer iteration's start
            mu = jnp.where(mu > 0, mu, 1e-3 * jnp.max(jnp.diag(H)))
            i32 = jnp.int32
            phi0 = phi_of(r, g)
            c0 = (x, r, g, grad, H, stat, phi0, mu.astype(F64), jnp.asarray(nu, F64),
                  jnp.asarray(0, i32), jnp.asarray(0, i32), jnp.asarray(1, i32), jnp.asarray(0, i32),
                  phi0, jnp.asarray(False))

            def cond(c):
                go = (~c[14]) if prog > 0.0 else (c[5] > omega)
                return go & (c[9] < inner) & (c[10] < 4 * inner) & (c[7] < 1e12)

            def body(c):
                x, r, g, grad, H, stat, phi, mu, nu, acc, tr, nj, nr, phi_ref, stalled = c
                L = jnp.linalg.cholesky(H + mu * In)
                ok = jnp.all(jnp.isfinite(L))
                d = -jax.scipy.linalg.cho_solve((L, True), grad)
                pred = -(grad @ d + 0.5 * d @ (H @ d))
                x1 = x + d
                if lguard != "none":                   # coils may not pass through each other
                    ok = ok & guard_ok(x1[:nc], lk0, d0)
                r1 = r_only(x1, surf)
                g1 = g_only(x1, surf)
                phi1 = phi_of(r1, g1)
                ratio = jnp.where(pred > 0, (phi - phi1) / pred, -1.0)
                accept = ok & (ratio > 1e-4) & jnp.isfinite(phi1)

                def on_accept(_):
                    rr, gg, gr, HH, st = derivs(x1, g1)
                    mu1 = mu * jnp.maximum(1.0 / 3.0, 1.0 - (2.0 * ratio - 1.0) ** 3)
                    ph1 = phi_of(rr, gg)
                    a1 = acc + 1
                    chk = (a1 % 10) == 0
                    stl = chk & ((phi_ref - ph1) < prog * ph1)
                    pref = jnp.where(chk, ph1, phi_ref)
                    return (x1, rr, gg, gr, HH, st, ph1, mu1, jnp.asarray(2.0, F64), a1, tr + 1, nj + 1,
                            nr + 1, pref, stl)

                def on_reject(_):
                    return (x, r, g, grad, H, stat, phi, mu * jnp.where(ok, nu, 10.0), jnp.where(ok, 2.0 * nu, nu),
                            acc, tr + 1, nj, nr + 1, phi_ref, stalled)
                return lax.cond(accept, on_accept, on_reject, None)

            x, r, g, grad, H, stat, phi, mu, nu, acc, tr, nj, nr, _, _ = lax.while_loop(cond, body, c0)
            mu = jnp.where(mu < 1e12, mu, -1.0)
            return x, g, stat, mu, nu, acc, tr, nj, nr, r @ r

        def stop_test(g, F, Fprev, lam, ftol, viol_tol):
            """XCoil ParetoSolver stop rule: true-limit violation, objective stall, complementarity."""
            compl = jnp.sum(lam * jnp.maximum(0.0, -g))
            return (jnp.max(g - off) <= viol_tol) & (jnp.abs(F - Fprev) <= ftol * F) & (compl <= ftol * F)

        inner = int(cfg.inner)
        inner_first = int(cfg.inner_first)
        first_prog = float(cfg.first_prog)
        rho0 = float(cfg.rho0)
        rho_corr = float(cfg.rho_corr)

        def advance(c, surf, ftol, viol_tol, k_end):
            def cond(c):
                return (~c[9]) & (c[8] < k_end)

            def body(c):
                x, lam, rho, mu, nu, omega, eta, Fp, k, done, trs, njs, lamk, st = c
                x, g, stat, mu, nu, acc, tr, nj, nr, F = outer(x, lam, rho, mu, nu, omega, surf, inner)
                lamk = jnp.maximum(0.0, lam + rho * g)           # KKT multipliers of this subproblem
                lam, rho, omega, eta, _ = _lancelot(g, lam, rho, omega, eta)
                return (x, lam, rho, mu, nu, omega, eta, F, k + 1, stop_test(g, F, Fp, lam, ftol, viol_tol),
                        trs + tr, njs + nj, lamk, stat)
            return lax.while_loop(cond, body, c)

        def solve_cold(x0, surf, ftol, viol_tol, max_outer):
            """ParetoSolver.full (sequential): first outer from the start with inner_first LM steps
            and the progress stop, then outers of `inner` steps with LANCELOT updates."""
            rho = jnp.asarray(rho0, F64)
            omega, eta = 1.0 / rho, 0.1 / rho ** 0.1
            x, g, stat, mu, nu, acc, tr, nj, nr, F = outer(x0, jnp.zeros(m, F64), rho, jnp.asarray(-1.0, F64),
                                                          jnp.asarray(2.0, F64), omega, surf, inner_first,
                                                          prog=first_prog)
            lamk = jnp.maximum(0.0, rho * g)
            lam, rho, omega, eta, _ = _lancelot(g, jnp.zeros(m, F64), rho, omega, eta)
            done = stop_test(g, F, jnp.inf, lam, ftol, viol_tol)
            c = (x, lam, rho, mu, nu, omega, eta, F, jnp.asarray(1), done, tr, nj, lamk, stat)
            return advance(c, surf, ftol, viol_tol, max_outer)

        def solve_warm(x0, lam0, rho_in, surf, ftol, viol_tol, k_max):
            """ParetoSolver.correct (sequential): multipliers inherited, penalty capped at rho_corr,
            previous objective = that of the start (an accurate start stops after one outer)."""
            rho = jnp.minimum(rho_in, rho_corr)
            omega, eta = 1.0 / rho, 0.1 / rho ** 0.1
            r0 = r_only(x0, surf)
            c = (x0, lam0, rho, jnp.asarray(-1.0, F64), jnp.asarray(2.0, F64), omega, eta, r0 @ r0,
                 jnp.asarray(0), jnp.asarray(False), jnp.asarray(0), jnp.asarray(0), lam0, jnp.asarray(jnp.inf, F64))
            return advance(c, surf, ftol, viol_tol, k_max)

        self._outer = outer
        hp = jax.default_matmul_precision("highest")
        self._hp = hp
        self._solve_cold = jax.jit(solve_cold)
        self._solve_warm = jax.jit(solve_warm)

        # ---- FP64 evaluation ----------------------------------------------------------------------
        def c0_of(xc, I, Xf, nf, wf, Bp=None):
            B = _field_I(xc, I[sym.coil] * sym.sign, cb, sym, Xf, None)
            if Bp is not None:
                B = B + Bp
            rho = jnp.sum(B * nf, 1) / jnp.linalg.norm(B, axis=1)
            return 0.5 * jnp.sum(wf * rho ** 2), rho
        self._c0_of = c0_of

        def eval64(x, surf):
            """FP64 metrics at x plus the KKT pieces (grad F, g, A) for the multiplier estimate."""
            xc, y = split(x)
            Xf, nf, wf, Sd = surf[:4]
            Bp = surf[5] if len(surf) > 5 else None
            I = currents(y)
            c0, rho = c0_of(xc, I, Xf, nf, wf, Bp)
            rr = reg_r(xc)
            F = c0 / c0_ref + rr @ rr
            g = g_only(x, surf, None)
            r, J = rJ(x, surf, None)
            A = A_of(x, surf, jnp.ones(m, F64), None)
            gF = 2.0 * (J.T @ r)
            ar = jnp.abs(rho)
            # ratio of integrals sum w|B.n| / sum w|B| (the field-error metric
            # bn_over_b_avg) on the same folded 50x50 PHR grid: |B.n| = |rho| |B|
            Bt = _field_I(xc, I[sym.coil] * sym.sign, cb, sym, Xf, None)
            Bm = jnp.linalg.norm(Bt if Bp is None else Bt + Bp, axis=1)
            return dict(c0=c0, F=F, reg=rr @ rr, g=g, gF=gF, A=A, mean_unw=jnp.sum(umean_w * ar),
                        mean_area=jnp.sum(wf * ar) / jnp.sum(wf), max_bn=jnp.max(ar), area=jnp.sum(wf), I=I,
                        bn_ratio_int=jnp.sum(wf * ar * Bm) / jnp.sum(wf * Bm))
        self._eval64 = jax.jit(eval64)
        # FP64 field of given coils at given points (unfolded-grid flux metrics, fixed chunk size)
        self._B64 = jax.jit(lambda xc, I, X: _field_I(xc, I[sym.coil] * sym.sign, cb, sym, X, None))

        def kkt_parts(x, surf):
            r, J = rJ(x, surf, None)
            g = g_only(x, surf, None)
            return 2.0 * (J.T @ r), g, A_of(x, surf, jnp.ones(m, F64), None)
        self._kkt_parts = jax.jit(kkt_parts)
        self._c0_explicit = jax.jit(lambda xc, I, Xf, nf, wf: c0_of(xc, I, Xf, nf, wf)[0])
        if force_on:
            self._g64 = jax.jit(lambda x, Sd, AX: jnp.concatenate([g_geo(x[:nc], Sd, AX), g_force(x)]))
        else:
            self._g64 = jax.jit(lambda x, Sd, AX: g_geo(x[:nc], Sd, AX))

        # ---- certification (geometry vs TRUE limits, FP64) ---------------------------------------
        order = int(cfg.order)
        mk = lambda nqv: fr.make_basis("fourier", nqv, order)
        cbq, cbk, cbs, cbl = mk(int(cfg.nq_verify)), mk(cf.N_KAPPA), mk(cf.N_SEG), mk(cf.N_LINK)
        z3 = jnp.zeros((1, 3), F64)
        z1 = jnp.ones((1,), F64)
        vt = float(cfg.viol_tol)

        def cert(xc, tri, tri_c, I=None):
            """I: per-base-coil currents (only the max-force check uses them)."""
            V = cf.VerifyTables(z3, z3, z1, z3, z3, z1, tri, tri_c, cbq, cbk, cbs, cbl, sym,
                                jnp.asarray(1.0, F64), jnp.asarray(I0, F64), "fourier", order)
            _, gdq, _ = fr.coils_curves(xc, cbq, C)
            Lc = pw.lengths(gdq)
            kl = lim.kappa_lim
            kmax = cf.max_curvature(xc, V, C)
            _, gdk, gddk = fr.coils_curves(xc, cbk, C)
            msc = pw.msc(gdk, gddk)
            dcc = cf.coil_coil_distance(xc, V)
            dcs = cf.coil_surface_distance(xc, V)
            Lk, Lraw = cf.linking_numbers(xc, V)
            Lcap = jnp.max(Lc) if cfg.length_mode == "per_coil" else jnp.sum(Lc)
            rows = [(Lcap - cap) / cap, (kmax - kl) / kl, jnp.max(msc - lim.msc_lim) / lim.msc_lim,
                    (lim.dcc - dcc) / lim.dcc, (lim.dcs - dcs) / lim.dcs]
            extra = {}
            if force_on:            # true max on the certification quadrature (nq_verify points per coil)
                Fq = fc_forces.coil_forces(xc, I, cbq, sym, F_delta)
                Fmax = jnp.max(Fq, axis=1)
                rows.append((jnp.max(Fmax) - F_lim) / F_lim)
                extra = dict(max_force=Fmax, max_force_ref_MN=Fmax * self.F_to_ref / 1e6)
            if lp_on:               # Lp curvature on the curvature-certification quadrature
                lpv = lp_curvature(gdk, gddk, lp_args[0], lp_args[1])
                rows.append((jnp.max(lpv) - lp_args[2]) / lp_args[2])
                extra = dict(extra, lp_curvature=lpv)
            gg = jnp.stack(rows)
            v = jnp.max(jnp.maximum(gg, 0.0))
            finite = jnp.all(jnp.isfinite(gg)) & jnp.all(jnp.isfinite(xc))
            unlinked = jnp.all(Lk == 0)
            return dict(L=Lc, max_kappa=kmax, msc=msc, d_cc=dcc, d_cs=dcs, linking_raw_max=jnp.max(jnp.abs(Lraw)),
                        g=gg, v=v, linked=~unlinked, finite=finite,
                        certified=(v <= vt) & unlinked & finite, **extra)
        self._cert = jax.jit(cert)
        if od is not None:
            self._slot_report = jax.jit(lambda xc, Sd, AX: od.report(fr.coils_curves(xc, cbq, C)[0],
                                                                     *slot_axis(Sd, AX), nfp_, float(cfg.slot_kappa)))

        # soft linking check (S-b): the certification's Gauss linking numbers alone (same N_LINK
        # samples, hence the same rounded integers), without the curvature / distance work
        V_link = cf.VerifyTables(z3, z3, z1, z3, z3, z1, jnp.zeros((1, 3, 3), F64), z3, cbq, cbk, cbs, cbl, sym,
                                 jnp.asarray(1.0, F64), jnp.asarray(I0, F64), "fourier", order)

        def link(xc):
            Lk, Lraw = cf.linking_numbers(xc, V_link)
            return jnp.any(Lk != 0), jnp.max(jnp.abs(Lraw))
        self._link = jax.jit(link)


_PROGRAMS: dict = {}


def _programs(cfg: CoilConfig) -> _Programs:
    key = tuple(sorted((k, (tuple(v) if isinstance(v, (list, tuple)) else v)) for k, v in asdict(cfg).items()
                       if k not in _SOLVE_ONLY_FIELDS))
    if key not in _PROGRAMS:
        _PROGRAMS[key] = _Programs(cfg)
    return _PROGRAMS[key]


# ================================================================================================
# differentiable surface tables (one compile per boundary mode table)
# ================================================================================================

class _SurfaceMap:
    """(rc, zs) on the mode table (m = 0..mpol, n = -ntor..ntor) -> traced solver arrays:
    PHR flux grid points / unit normals / c0 weights and the coil-surface distance grid."""

    def __init__(self, P: _Programs, mpol, ntor):
        self.P = P
        self.mpol, self.ntor = int(mpol), int(ntor)
        modes = _mode_list(self.mpol, self.ntor)
        self.modes = modes
        self.xm = np.array([mm for mm, _ in modes], float)
        self.xn = np.array([nn for _, nn in modes], float)
        nm = len(modes)
        self.nm = nm
        self.index = {}
        for k, (mm, nn) in enumerate(modes):
            self.index[("rc", mm, nn)] = k
            self.index[("zs", mm, nn)] = nm + k
        nfp = int(P.cfg.nfp)

        def mats(TH, PH):
            a = self.xm[None, :] * TH[:, None] - nfp * self.xn[None, :] * PH[:, None]
            c, s = np.cos(a), np.sin(a)
            return dict(c=jnp.asarray(c), s=jnp.asarray(s),
                        cm=jnp.asarray(c * self.xm), sm=jnp.asarray(s * self.xm),
                        cn=jnp.asarray(c * nfp * self.xn), sn=jnp.asarray(s * nfp * self.xn),
                        cp=jnp.asarray(np.cos(PH)), sp=jnp.asarray(np.sin(PH)))
        Mf = mats(P.flux_TH, P.flux_PH)
        Md = mats(P.dist_TH, P.dist_PH)
        wfac = jnp.asarray(P.flux_mult * (2 * np.pi / P.cfg.phr_grid[0]) * (2 * np.pi / P.cfg.phr_grid[1]))

        def xyz(M, rc, zs, derivs=True):
            R = M["c"] @ rc
            Z = M["s"] @ zs
            X = jnp.stack([R * M["cp"], R * M["sp"], Z], -1)
            if not derivs:
                return X
            R_t = -(M["sm"] @ rc)
            Z_t = M["cm"] @ zs
            R_p = M["sn"] @ rc
            Z_p = -(M["cn"] @ zs)
            Xt = jnp.stack([R_t * M["cp"], R_t * M["sp"], Z_t], -1)
            Xp = jnp.stack([R_p * M["cp"] - R * M["sp"], R_p * M["sp"] + R * M["cp"], Z_p], -1)
            return X, Xt, Xp

        def tables(coef):
            rc, zs = coef[:nm], coef[nm:]
            X, Xt, Xp = xyz(Mf, rc, zs)
            N = jnp.cross(Xp, Xt)
            nN = jnp.linalg.norm(N, axis=-1)
            nf = N / nN[:, None]
            wf = wfac * nN                                  # c0 = 0.5 sum wf rho^2  (PHR local flux)
            Sd = xyz(Md, rc, zs, derivs=False)
            return X, nf, wf, Sd
        self._tables_py = tables
        self._tables = jax.jit(tables)

        c0_ref = float(P.cfg.c0_ref)
        nc = P.nc

        def lagr(coef, x, lamk, AX, Bp=None):
            Xf, nf, wf, Sd = tables(coef)
            xc, y = x[:nc], x[nc:]
            c0, _ = P._c0_of(xc, P.currents(y), Xf, nf, wf, Bp)
            g = P._g_geo(xc, Sd, AX)           # AX: the frozen ordering axis, a constant (no surface derivative)
            if P.force_on:                   # force rows: coils only, zero derivative in the boundary
                g = jnp.concatenate([g, P._force_rows(x)])
            return c0 + c0_ref * (lamk @ g)

        def parts(coef, x, AX):
            Xf, nf, wf, Sd = tables(coef)
            xc, y = x[:nc], x[nc:]
            c0, _ = P._c0_of(xc, P.currents(y), Xf, nf, wf)
            g = P._g_geo(xc, Sd, AX)
            if P.force_on:
                g = jnp.concatenate([g, P._force_rows(x)])
            return c0, g
        self._lag_grad = jax.jit(jax.grad(lagr))
        self._lag_grad_bp = jax.jit(jax.grad(lagr))                       # same, traced with a Bp table

        def c0_bp(coef, x, Bp):
            Xf, nf, wf, _ = tables(coef)
            xc, y = x[:nc], x[nc:]
            return P._c0_of(xc, P.currents(y), Xf, nf, wf, Bp)[0]
        self._bp_cot = jax.jit(jax.grad(c0_bp, argnums=2))                # d c0 / d Bp (Nf, 3)
        self._parts = jax.jit(parts)

    def coef(self, modes: dict):
        v = np.zeros(2 * self.nm)
        for (kind, mm, nn), val in modes.items():
            k = self.index.get((kind, mm, nn))
            if k is None:
                if val != 0.0 and not (kind == "zs" and mm == 0 and nn == 0):
                    raise ValueError(f"mode {kind}({mm},{nn}) outside the table")
                continue
            v[k] = val
        return v


# ================================================================================================
# the solver
# ================================================================================================

class CoilSolver:
    def __init__(self, cfg: CoilConfig | None = None, cold_cache: bool = False):
        """cold_cache (opt-in, S-b): keep the most recent cold solve (per-start solver states and
        finished results, keyed by a hash of the surface) so an identical later cold solve is
        reused instead of re-solved (bit-identical result; info['cache_hit'] = True)."""
        self.cfg = cfg or CoilConfig()
        self.P = _programs(self.cfg)
        self._maps = {}
        self.plasma = None       # finite beta: callable (boundary, TH, PH) -> B_plasma (N, 3) [T]
        self.plasma_sign = 1.0
        self.timing = dict(compile_cold=None, compile_warm=None)
        self.cold_cache = bool(cold_cache)
        self._cold = None               # the cold cache entry (one surface)
        self._aux = {}                  # tightened-limit solvers of final_coil_stage
        self._ordering_axis = None      # frozen reference axis of the slot rows (a, b), set_ordering_axis

    # ---- reference axis of the toroidal ordering rows --------------------------------------------
    def set_ordering_axis(self, axis=None, boundary=None):
        """Freeze the reference axis of the ordering (slot) rows: axis = (a, b) as stored in a checkpoint, or the
        geometric axis (m = 0 modes) of `boundary`; neither = unfreeze (each surface then uses its own axis).
        Returns the axis as plain lists (for checkpoints)."""
        if boundary is not None:
            axis = boundary_axis(boundary, int(self.cfg.nfp), self.P.ax_nmax)
        if axis is None:
            self._ordering_axis = None
            return None
        a, b = (np.asarray(v, float) for v in axis)
        n = self.P.ax_nmax + 1
        if a.shape != (n,) or b.shape != (n,):
            raise ValueError(f"ordering axis has {a.shape[0]} / {b.shape[0]} harmonics, the coil problem uses {n} "
                             f"(coils.dist_grid and nfp fix it)")
        self._ordering_axis = (a, b)
        return [a.tolist(), b.tolist()]

    @property
    def ordering_axis(self):
        """The frozen axis as [a, b] lists, or None."""
        return None if self._ordering_axis is None else [v.tolist() for v in self._ordering_axis]

    def axis_for(self, boundary):
        """(2, nmax + 1) reference axis of the ordering rows on `boundary`: the frozen one, else the boundary's own
        geometric axis (standalone solves; with ordering_axis = "live" the programs recompute it anyway)."""
        if self._ordering_axis is not None and self.cfg.ordering_axis == "frozen_step":
            return np.stack(self._ordering_axis)
        return np.stack(boundary_axis(boundary, int(self.cfg.nfp), self.P.ax_nmax))

    # ---- surfaces -------------------------------------------------------------------------------
    def _map_for(self, modes, names=()):
        keys = [k for k, v in modes.items() if v != 0.0] + [parse_name(nm_) for nm_ in names]
        mp = max([self.cfg.mpol_min] + [mm for (_, mm, nn) in keys])
        nt = max([self.cfg.ntor_min] + [abs(nn) for (_, mm, nn) in keys])
        key = (mp, nt)
        if key not in self._maps:
            self._maps[key] = _SurfaceMap(self.P, mp, nt)
        return self._maps[key]

    def surface(self, boundary: dict, names=()):
        """Everything surface-dependent: mode table, coefficient vector, traced arrays, the XCoil
        FourierBoundary (starts) and the certification triangulation."""
        modes = boundary_modes(boundary)
        smap = self._map_for(modes, names)
        coef = smap.coef(modes)
        tabs = tuple(smap._tables(jnp.asarray(coef))) + (jnp.asarray(self.axis_for(boundary), F64),)
        if self.plasma is not None:          # finite beta: plasma field on the flux grid (5th table)
            tabs = tabs + (jnp.asarray(self.plasma(boundary, self.P.flux_TH, self.P.flux_PH), F64),)   # (6th table)
        if not all(bool(jnp.all(jnp.isfinite(t))) for t in tabs):
            raise FloatingPointError("non-finite surface tables")
        nm = smap.nm
        b = bd.from_modes(smap.xm, smap.xn, coef[:nm], coef[nm:], int(self.cfg.nfp), True, name="phrport")
        return dict(modes=modes, map=smap, coef=coef, tabs=tabs, b=b)

    def _triangulation(self, b):
        npp, nth = (int(v) for v in self.cfg.verify_grid)
        tri, tri_c = cf._triangulation(b, npp, nth)
        return tri, tri_c

    # ---- starts ---------------------------------------------------------------------------------
    def start(self, b, label, cache=None, family=None, _alt=False):
        """One XCoil start (fixed/starts.py helpers, computed lazily: only the sizes a label
        needs): family cfg.start (offset_3d; or `family`: offset_3d / offset_planar / circle),
        label 'L70' / 'L85' / 'L100' (fraction of the total-length cap C x per-coil cap), 'clear'
        (smallest offset clearing 1.1 d_cs) or 'L85_tilt+-8'. Returns x0 = (curve DOFs, zero
        current DOFs). `cache` holds the sizes of ONE family."""
        cfg = self.cfg
        cache = {} if cache is None else cache
        _alt = _alt or (family is not None)   # explicit family (final stage): reject, no cross-family fallback
        if family is None:
            family = cfg.start.rsplit("_", 1)[0] if "_" in cfg.start else cfg.start
        C = int(cfg.n_coils)
        phis = sym_mod.base_toroidal_angles(C, b.nfp, True)
        fams = dict(offset_3d=lambda v: starts_mod.offset_3d_points(b, phis, v),
                    offset_planar=lambda v: starts_mod.offset_planar_points(b, phis, v),
                    circle=lambda v: starts_mod.circle_points(b, phis, v))
        fun = fams[family]
        sel = b.rc[(b.xm == 1) & (b.xn == 0)]
        rc10 = float(sel[0]) if sel.size else 0.0
        if not np.isfinite(rc10) or rc10 <= 1e-6:  # degenerate boundary: the start bisection would never terminate
            raise RuntimeError(f'degenerate boundary for coil starts (rc(1,0)={rc10})')
        lo = 0.5 * rc10 if family == "circle" else 0.2 * float(cfg.min_cs)
        cap = float(cfg.per_coil_length) * C if cfg.length_mode == "per_coil" else float(cfg.total_length)

        def size(lb):
            if lb not in cache:
                if lb == "clear":
                    cache[lb] = starts_mod._size_for_clearance(fun, b, float(cfg.min_cs), lo, 4 * rc10)
                else:
                    v = starts_mod._size_for_length(fun, int(lb[1:]) / 100.0 * cap, lo, 4 * rc10)
                    if v is None:
                        raise ValueError(f"start {lb}: length target below the smallest offset")
                    cache[lb] = v
            return cache[lb]
        base, _, tilt = label.partition("_tilt")
        P = fun(size(base))
        if tilt:
            P = starts_mod._tilt(P, phis, float(tilt))
        x = starts_mod._fit(P, int(cfg.order), "fourier")[: self.P.nc]
        x0 = np.concatenate([np.asarray(x, float), np.zeros(self.P.ny)])
        # lowA_ncsx admissible starts (opt-in: only with coils.link_guard != none, as XCoil main):
        # a start with any nonzero Gauss linking number is rejected (the guard keeps start topology).
        # Called without `family` (solve path): fall back to the same label in the other families.
        topo = getattr(self.P, "_topo", None)
        if str(cfg.link_guard) != "none" and topo is not None:
            lk, dmin = topo(jnp.asarray(x0[: self.P.nc]))
            if np.any(np.asarray(lk) != 0):
                if not _alt:
                    for alt in ("offset_planar", "circle", "offset_3d"):
                        if alt == family:
                            continue
                        try:
                            return self.start(b, label, cache.setdefault(("_alt", alt), {}), family=alt, _alt=True)
                        except ValueError:
                            continue
                raise ValueError(f"start {family}:{label} linked (|Lk| max {int(np.max(np.abs(np.asarray(lk))))})")
        return x0

    def seed_start(self):
        """seed_coils start: the given coils fitted at cfg.order, currents rescaled to cfg.total_current.

        Accepted files (see docs/seeding.md):
          * phrport-coils-v1 JSON or NPZ (final_coils.json / coils_certified.json of a run, qss-coils output):
            base_curve_dofs + base_currents_A (any order; nfp and the number of base coils must match);
          * a points JSON {"points": [[[x, y, z], ...] per base coil], "currents_A": [...]} (e.g. sampled from
            simsopt curves: [c.gamma().tolist() for c in base_curves]).
        """
        import json
        path = self.cfg.seed_coils
        C = int(self.cfg.n_coils)
        if path.endswith(".npz"):
            z = np.load(path, allow_pickle=False)
            d = {k: z[k] for k in z.files}
        else:
            d = json.load(open(path))
        if "base_curve_dofs" in d:                       # phrport-coils-v1 (JSON or NPZ)
            order_f = int(np.asarray(d["order"]))
            nfp_f = int(np.asarray(d.get("nfp", self.cfg.nfp)))
            if nfp_f != int(self.cfg.nfp):
                raise ValueError(f"seed_coils: file has nfp {nfp_f}, the problem has {self.cfg.nfp}")
            dofs = np.asarray(d["base_curve_dofs"], float)
            if dofs.shape[0] != C:
                raise ValueError(f"seed_coils: file has {dofs.shape[0]} base coils, coils.n_coils is {C}")
            cbf = fr.make_basis("fourier", max(int(self.cfg.nq), 4 * order_f + 8), order_f)
            P = np.asarray(fr.coils_curves(jnp.asarray(dofs.ravel()), cbf, C)[0])
            I = np.asarray(d["base_currents_A"], float)
        else:
            P = np.asarray(d["points"], float)
            I = np.asarray(d["currents_A"], float)
        if P.ndim != 3 or P.shape[0] != C or P.shape[2] != 3:
            raise ValueError(f"seed_coils: points shape {P.shape}, expected ({C}, n, 3)")
        x = starts_mod._fit(P, int(self.cfg.order), "fourier")[: self.P.nc]
        I = I * float(self.cfg.total_current) / float(np.sum(I))
        y = (I[: self.P.ny] / self.P.I0 - 1.0) if self.P.ny else np.zeros(0)
        print(f"[seed_coils] first cold solve starts from {path} (currents {np.round(I).tolist()})", flush=True)
        return np.concatenate([np.asarray(x, float), np.asarray(y, float)])

    # ---- core -----------------------------------------------------------------------------------
    def _run_cold(self, x0, tabs):
        P, cfg = self.P, self.cfg
        t0 = time.perf_counter()
        with P._hp:
            c = P._solve_cold(jnp.asarray(x0, F64), tabs, jnp.asarray(cfg.ftol, F64),
                              jnp.asarray(cfg.viol_tol, F64), jnp.asarray(int(cfg.max_outer)))
        jax.block_until_ready(c[0])
        dt = time.perf_counter() - t0
        if self.timing["compile_cold"] is None:
            self.timing["compile_cold"] = dt
        return c, dt

    def _run_warm(self, x0, lam0, rho0, tabs, k_max=None, ftol=None, viol_tol=None):
        P, cfg = self.P, self.cfg
        t0 = time.perf_counter()
        with P._hp:
            c = P._solve_warm(jnp.asarray(x0, F64), jnp.asarray(lam0, F64), jnp.asarray(rho0, F64), tabs,
                              jnp.asarray(cfg.ftol if ftol is None else ftol, F64),
                              jnp.asarray(cfg.viol_tol if viol_tol is None else viol_tol, F64),
                              jnp.asarray(int(cfg.warm_max_outer if k_max is None else k_max)))
        jax.block_until_ready(c[0])
        dt = time.perf_counter() - t0
        if self.timing["compile_warm"] is None:
            self.timing["compile_warm"] = dt
        return c, dt

    @staticmethod
    def _nnls(gF, g, A, act_tol):
        """KKT multipliers at x: min ||grad F + A_act^T z||, z >= 0, over rows with g > -act_tol
        (XCoil CoilProblem.kkt; PHR recovers its nu from the same stationarity by least squares)."""
        from scipy.optimize import nnls
        act = np.where(g > -act_tol)[0]
        z = np.zeros(g.size)
        if act.size:
            try:
                zz, _ = nnls(A[act].T, -gF, maxiter=50 * act.size)
                z[act] = zz
            except (RuntimeError, ValueError):  # ValueError: non-finite data (e.g. a diverged warm start)
                pass
        return z, float(np.max(np.abs(gF + A.T @ z)) / (1.0 + np.max(np.abs(gF))))

    def _finish(self, c, S, label, t_solve, tri=None, certify=None):
        """FP64 metrics, KKT multipliers (NNLS at x*), Lagrangian value / elasticities and - certify
        (None: cfg.certify) - certification of a solver state c.

        Always (R1 / G4): info['c0_lagr'] = c0 + c0_ref lam.g (the Lagrangian of the coil subproblem
        at the NNLS multipliers lam), info['lam_block_sum'][block] = sum lam[block] and
        info['elasticity'][block] = c0_ref sum lam[block] / c0 (= |d ln c0* / d ln limit| of the
        normalized rows: c0* falls when a cap L / kappa / MSC is raised and rises when a minimum
        distance cc / cs is raised), info['bn_ratio_int'] = sum w|B.n| / sum w|B| (the ratio of
        integrals, 50x50 PHR grid), info['v_soft'] = max(0, g) of the FP64 solver rows,
        info['d_cs_soft'] (soft-min coil-surface distance) and info['soft_ok'].
        certify=False (S-b): no certification (no triangulation, no curvature / distance
        verification); linked = the certification's Gauss linking numbers alone (same samples), and
        ok = info['soft_ok'] = finite & not linked & v_soft <= ok_viol. info['min_cs_dist'] then
        carries d_cs_soft (input of the PHR phantom guard, which otherwise reads cert.d_cs).
        certify=True: soft_ok uses the certified linking result."""
        P, cfg = self.P, self.cfg
        certify = bool(cfg.certify) if certify is None else bool(certify)
        x = np.asarray(c[0])
        lam_plus = np.asarray(c[12])                     # max(0, lam + rho g) of the last AL subproblem
        ev = jax.tree_util.tree_map(np.asarray, P._eval64(jnp.asarray(x), S["tabs"]))
        g, gF, A = ev["g"], ev["gF"], ev["A"]
        lam, kkt = self._nnls(gF, g, A, cfg.kkt_act_tol)
        kkt_al = float(np.max(np.abs(gF + A.T @ lam_plus)) / (1.0 + np.max(np.abs(gF))))
        blk = {k: float(np.max(g[s])) for k, s in P.blocks.items() if s.stop > s.start}
        info = dict(start=label, outer=int(c[8]), stopped=bool(c[9]), rho=float(c[2]), mu=float(c[3]),
                    lm_trials=int(c[10]), jac_evals=int(c[11]), stat_last=float(c[13]),
                    lam_al=np.asarray(c[1]), lam_plus=lam_plus, t_solve=t_solve, F_solver=float(c[7]),
                    kkt=kkt, kkt_al=kkt_al, reg=float(ev["reg"]), g_block_max=blk, g_max=float(np.max(g)),
                    mean_bn_unweighted=float(ev["mean_unw"]), mean_bn_area=float(ev["mean_area"]),
                    max_bn=float(ev["max_bn"]), area=float(ev["area"]),
                    n_active={k: int(np.sum(lam[s] > 0)) for k, s in P.blocks.items() if s.stop > s.start})
        c0 = float(ev["c0"])
        c0_ref = float(cfg.c0_ref)
        # ---- R1 / G4 reporting (numpy only; nothing above or below depends on it)
        info["bn_ratio_int"] = float(ev["bn_ratio_int"])
        info["c0_lagr"] = c0 + c0_ref * float(lam @ g)
        lsum = {k: float(np.sum(lam[s])) for k, s in P.blocks.items() if s.stop > s.start}
        info["lam_block_sum"] = lsum
        info["elasticity"] = {k: (c0_ref * v_ / c0 if (np.isfinite(c0) and c0 > 0.0) else float("nan"))
                              for k, v_ in lsum.items()}
        v_soft = float(np.max(np.maximum(g, 0.0)))
        info["v_soft"] = v_soft
        cs = P.blocks["cs"]
        info["d_cs_soft"] = float(cfg.min_cs * (1.0 + cfg.distance_margin - np.max(g[cs])))
        info["certify"] = certify
        if P.slot_on:
            info["slot"] = self._slot_info(x, S)
        certified, linked, v = False, False, v_soft
        if certify:
            t1 = time.perf_counter()
            if tri is None:
                tri = self._triangulation(S["b"])
            ce = jax.tree_util.tree_map(np.asarray, P._cert(jnp.asarray(x[:P.nc]), *tri, jnp.asarray(ev["I"], F64)))
            info["t_cert"] = time.perf_counter() - t1
            info["cert"] = dict(L=ce["L"].tolist(), max_kappa=float(ce["max_kappa"]), msc=ce["msc"].tolist(),
                                d_cc=float(ce["d_cc"]), d_cs=float(ce["d_cs"]), v=float(ce["v"]),
                                g=ce["g"].tolist(), linking_raw_max=float(ce["linking_raw_max"]))
            if P.force_on:
                info["cert"].update(max_force=ce["max_force"].tolist(), max_force_ref_MN=ce["max_force_ref_MN"].tolist())
            if P.lp_on:
                info["cert"].update(lp_curvature=ce["lp_curvature"].tolist())
            certified, linked, v = bool(ce["certified"]), bool(ce["linked"]), float(ce["v"])
            finite = bool(ce["finite"])
        else:
            t1 = time.perf_counter()
            lk, lraw = P._link(jnp.asarray(x[:P.nc]))
            linked = bool(lk)
            info["t_link"] = time.perf_counter() - t1
            info["linking_raw_max_soft"] = float(lraw)
            info["min_cs_dist"] = info["d_cs_soft"]
            finite = True
        finite = finite and np.isfinite(c0) and np.all(np.isfinite(x))
        ok = bool(finite and not linked and v <= cfg.ok_viol)
        info["soft_ok"] = bool(np.isfinite(c0) and np.all(np.isfinite(x)) and not linked and v_soft <= cfg.ok_viol)
        info["v"] = v
        res = CoilResult(ok=ok, c0_local=c0 if finite else float("nan"), flux_F=float(ev["F"]), x=x,
                         currents=np.asarray(ev["I"]), lam=lam, certified=certified, linked=linked, info=info)
        return res, tri

    def _slot_info(self, x, S):
        P = self.P
        win, pen, ext = (np.asarray(v) for v in P._slot_report(jnp.asarray(np.asarray(x)[:P.nc]), S["tabs"][3],
                                                               S["tabs"][4]))
        win = np.where(np.isfinite(win), win, np.nan)
        return dict(pairs=slot_pair_names(int(self.cfg.n_coils)), gap_deg=win.tolist(), pen_gap_deg=pen.tolist(),
                    min_gap_deg=float(np.nanmin(win)), crossing=bool(np.nanmin(win) < 0.0),
                    phi_extent_deg=ext.tolist())

    def slot_metrics(self, boundary: dict, x):
        """Toroidal ordering metrics of coils x (solver layout) on `boundary`, whether or not the slot rows
        are on: per pair the windowed hard gap [deg] (negative = crossing in the (theta, phi) projection),
        the theta-penalized gap and each base coil's toroidal extent [deg]."""
        return self._slot_info(x, self.surface(boundary))

    def _fail(self, info):
        P = self.P
        return CoilResult(ok=False, c0_local=float("nan"), flux_F=float("nan"), x=np.zeros(P.n),
                          currents=np.full(P.cfg.n_coils, P.I0), lam=np.zeros(P.m), certified=False,
                          linked=False, info=info)

    # ---- cold cache (S-b) -----------------------------------------------------------------------
    @staticmethod
    def surface_key(S) -> str:
        """Hash of a surface (mode table + coefficient vector + ordering axis; independent of the dict order)."""
        smap = S["map"]
        h = hashlib.sha1(np.asarray([smap.mpol, smap.ntor], np.int64).tobytes())
        h.update(np.ascontiguousarray(S["coef"], dtype=np.float64).tobytes())
        h.update(np.ascontiguousarray(np.asarray(S["tabs"][4]), dtype=np.float64).tobytes())   # ordering axis
        return h.hexdigest()

    def clear_cold_cache(self):
        self._cold = None

    @staticmethod
    def _call_abort(should_abort):
        try:
            return bool(should_abort())
        except Exception:  # noqa: BLE001  (contract: never raise; a failing predicate never aborts)
            return False

    def solve(self, boundary: dict, warm: CoilResult | None = None, should_abort=None, certify=None,
              reuse=None, rho_reset=None, warm_k_max=None, warm_fallback=True) -> CoilResult:
        """Coil solve on `boundary` (simsopt-style 'rc(m,n)', 'zs(m,n)' names). Cold: deterministic
        XCoil start (cfg.start, then cfg.fallback_starts if the result is linked / not ok).
        warm: start from warm.x with its multipliers and penalty (ParetoSolver.correct); falls back
        to cold if the warm result is not ok. Never raises: failures return ok=False.

        Opt-in extras (defaults reproduce the previous behaviour bit for bit):
          should_abort (S-a): callable checked before every FALLBACK start (the cold start after a
            warm result that is not ok, and every further cold start); when it returns True the
            solve returns at once with ok=False and info['aborted'] = True (the best result so
            far, which is not ok, or a NaN failure if there is none). Exceptions count as False.
          certify (S-b): None = cfg.certify; False skips the certification (runtime argument, no
            recompilation): ok = info['soft_ok'] from the FP64 solver rows plus a linking check.
          reuse (S-b): None = self.cold_cache; True reads/writes the cache of the most recent cold
            solve (keyed by surface_key): a pure cold solve (warm=None) on an identical surface
            returns a copy of the cached result (info['cache_hit'] = True; a certified result is
            preferred and also serves a certify=False request), and the per-start solver states are
            replayed instead of re-solved (e.g. certifying a surface first solved with
            certify=False: bit-identical to a fresh certified solve).
          Phase 2 (roadmap B2, anchored coil oracle; warm solves only):
          rho_reset: None = cfg.rho_warm_reset; True restarts the warm AL at rho = cfg.rho0 with the
            warm result's lam_al kept (instead of inheriting its final rho).
          warm_k_max: outer budget of the warm solve (None = cfg.warm_max_outer; runtime argument).
          warm_fallback: False returns the warm result even when it is not ok (no cold fallback; the
            caller decides), True (default) falls back to the cold starts.
          When any of the three is not at its default, the warm result carries info['warm_opts']."""
        t_all = time.perf_counter()
        info_fail = dict(attempts=[])
        certify = bool(self.cfg.certify) if certify is None else bool(certify)
        reuse = self.cold_cache if reuse is None else bool(reuse)
        try:
            t0 = time.perf_counter()
            S = self.surface(boundary)
            t_tab = time.perf_counter() - t0
            tri = None
            attempts = []
            n_tried = 0                     # attempts already made: every later start is a fallback
            last = None                     # most recent (not ok) result, returned on abort
            key = ce = None
            if reuse:
                key = self.surface_key(S)
                ce = self._cold if (self._cold is not None and self._cold["key"] == key) else None
                if ce is not None and warm is None:
                    hit = ce["results"].get(True) or ce["results"].get(certify)    # certified serves any
                    if hit is not None:
                        res = copy.deepcopy(hit)
                        res.info.update(cache_hit=True, t_tables=t_tab, t_total=time.perf_counter() - t_all)
                        return res
            if warm is not None and warm.x is not None and np.size(warm.x) == self.P.n and np.all(np.isfinite(warm.x)) \
                    and np.size(warm.lam) == self.P.m:
                wi = warm.info if isinstance(warm.info, dict) else {}
                reset = bool(self.cfg.rho_warm_reset) if rho_reset is None else bool(rho_reset)
                rho_w = float(self.cfg.rho0) if reset else float(wi.get("rho", self.cfg.rho0))
                lam_w = np.asarray(wi.get("lam_al", warm.lam))
                c, dt = self._run_warm(warm.x, lam_w, rho_w, S["tabs"], k_max=warm_k_max)
                res, tri = self._finish(c, S, "warm", dt, tri, certify)
                attempts.append(dict(start="warm", ok=res.ok, c0=res.c0_local, t=dt, outer=res.info["outer"],
                                     stopped=res.info["stopped"], v=res.info["v"], linked=res.linked))
                n_tried += 1
                if reset or warm_k_max is not None or not warm_fallback:     # phase 2 (B2) only
                    res.info["warm_opts"] = dict(
                        rho_reset=reset, rho_in=rho_w, rho_inherited=float(wi.get("rho", self.cfg.rho0)),
                        k_max=int(self.cfg.warm_max_outer if warm_k_max is None else warm_k_max),
                        fallback=bool(warm_fallback))
                if res.ok or not warm_fallback:
                    res.info.update(attempts=attempts, t_tables=t_tab, t_total=time.perf_counter() - t_all,
                                    mode="warm")
                    return res
                last = res
            if reuse and ce is None:
                ce = self._cold = dict(key=key, states={}, results={}, tri=None, sizes={})
            if ce is not None and tri is None:
                tri = ce["tri"]
            primary = self.cfg.start.rsplit("_", 1)[-1] if "_" in self.cfg.start else "L85"
            labels = [primary] + [lb for lb in self.cfg.fallback_starts if lb != primary]
            if self.cfg.seed_coils and not _SEED_USED[0]:
                labels = ["seed"] + labels
                _SEED_USED[0] = True
            best, cache, t_st, n_solved = None, ({} if ce is None else ce["sizes"]), 0.0, 0
            aborted, n_replayed = False, 0
            for lb in labels:
                if n_solved >= max(1, int(self.cfg.max_starts)):
                    break
                if should_abort is not None and n_tried > 0 and self._call_abort(should_abort):
                    aborted = True
                    break
                st = ce["states"].get(lb) if ce is not None else None
                if st is not None and st[0] == "error":
                    attempts.append(dict(start=lb, ok=False, error=st[1]))
                    continue
                if st is None:
                    t0 = time.perf_counter()
                    try:
                        x0 = self.seed_start() if lb == "seed" else self.start(S["b"], lb, cache)
                    except Exception as e:  # noqa: BLE001
                        err = f"{type(e).__name__}: {e}"
                        attempts.append(dict(start=lb, ok=False, error=err))
                        t_st += time.perf_counter() - t0
                        if ce is not None:
                            ce["states"][lb] = ("error", err)
                        continue
                    t_st += time.perf_counter() - t0
                    n_solved += 1
                    n_tried += 1
                    c, dt = self._run_cold(x0, S["tabs"])
                    if ce is not None:
                        ce["states"][lb] = ("state", c, dt)
                else:
                    _, c, dt = st
                    n_solved += 1
                    n_tried += 1
                    n_replayed += 1
                res, tri = self._finish(c, S, lb, dt, tri, certify)
                attempts.append(dict(start=lb, ok=res.ok, c0=res.c0_local, t=dt, outer=res.info["outer"],
                                     stopped=res.info["stopped"], v=res.info["v"], linked=res.linked))
                if best is None or (res.ok and (not best.ok or res.c0_local < best.c0_local)) or \
                        (not best.ok and not res.ok and res.info["v"] < best.info["v"]):
                    best = res
                if res.ok:
                    break
            if ce is not None:
                ce["tri"] = tri if tri is not None else ce["tri"]
            if aborted:
                out = best if best is not None else last
                extra = dict(aborted=True, attempts=attempts, t_tables=t_tab, t_starts=t_st,
                             t_total=time.perf_counter() - t_all)
                if out is None:
                    return self._fail(dict(error="aborted by should_abort", **extra))
                out.ok = False
                out.info.update(mode=("cold" if best is not None else "warm"), **extra)
                return out
            if best is None:
                return self._fail(dict(error="no usable start", attempts=attempts, t_starts=t_st,
                                       t_total=time.perf_counter() - t_all))
            best.info.update(attempts=attempts, t_tables=t_tab, t_starts=t_st, t_total=time.perf_counter() - t_all,
                             mode="cold")
            if ce is not None:
                best.info["cache_replayed_starts"] = n_replayed
                if warm is None:
                    ce["results"][certify] = copy.deepcopy(best)
            return best
        except Exception as e:  # noqa: BLE001  (contract: never raise)
            info_fail.update(error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc(limit=6),
                             t_total=time.perf_counter() - t_all)
            return self._fail(info_fail)

    def polish(self, boundary: dict, result: CoilResult, k_max=40, ftol=1e-10, viol_tol=None) -> CoilResult:
        """Continue the AL from result (warm, ParetoSolver.correct) with a tight objective-stall
        tolerance: a more stationary point for envelope / finite-difference checks. viol_tol: the
        stop rule's violation tolerance on the solver rows (None: cfg.viol_tol)."""
        S = self.surface(boundary)
        c, dt = self._run_warm(result.x, np.asarray(result.info.get("lam_al", result.lam)),
                               float(result.info.get("rho", self.cfg.rho0)), S["tabs"], k_max=k_max, ftol=ftol,
                               viol_tol=viol_tol)
        res, _ = self._finish(c, S, "polish", dt)
        res.info["mode"] = "polish"
        return res

    # ---- final coil stage (R1 / G4) -------------------------------------------------------------
    _CERT_BLOCKS = ("L", "curv", "msc", "cc", "cs")      # order of the certification's g vector

    @property
    def _cert_blocks(self):
        return (self._CERT_BLOCKS + (("force",) if self.P.force_on else ())
                + (("lpcurv",) if getattr(self.P, "lp_on", False) else ()))

    def _tightened(self, margins):
        """Solver whose SOLVER-ROW limits are tightened by the relative margins[block] (length,
        curvature, MSC caps scaled by 1 - m; coil-coil / coil-surface minima by 1 + m). Same x and
        multiplier layout and the same surface tables; one compiled warm program per margin set."""
        key = tuple(round(float(margins.get(k, 0.0)), 12) for k in self._cert_blocks)
        if not any(key):
            return self
        if key not in self._aux:
            km = dict(zip(self._cert_blocks, key))
            dL, dk, dm, dcc, dcs = (km[b] for b in self._CERT_BLOCKS)
            dF, dP = km.get("force", 0.0), km.get("lpcurv", 0.0)
            cfg = self.cfg
            cfg2 = replace(cfg, per_coil_length=cfg.per_coil_length * (1.0 - dL), max_force=cfg.max_force * (1.0 - dF),
                           lp_curvature_max=cfg.lp_curvature_max * (1.0 - dP),
                           total_length=cfg.total_length * (1.0 - dL),
                           max_curvature=cfg.max_curvature * (1.0 - dk), max_msc=cfg.max_msc * (1.0 - dm),
                           min_cc=cfg.min_cc * (1.0 + dcc), min_cs=cfg.min_cs * (1.0 + dcs))
            self._aux[key] = CoilSolver(cfg2)
        return self._aux[key]

    @staticmethod
    def _summary(res, family=None):
        i = res.info
        return dict(start=i.get("start"), family=family, ok=bool(res.ok), soft_ok=i.get("soft_ok"),
                    certified=bool(res.certified),
                    linked=bool(res.linked), c0=float(res.c0_local), c0_lagr=float(i.get("c0_lagr", np.nan)),
                    v=float(i.get("v", np.nan)), v_soft=float(i.get("v_soft", np.nan)), outer=i.get("outer"),
                    stopped=i.get("stopped"), kkt=i.get("kkt"), t=i.get("t_solve"),
                    cert_g=(i.get("cert") or {}).get("g"))

    def _polish_to_target(self, start, S, tri, *, v_target, ftol, k_max, max_rounds, viol_tol, max_margin):
        """Step 3 of final_coil_stage for ONE start: polish (warm AL from `start`, ftol) until the
        CERTIFIED true-limit violation v <= v_target. Round 0 is the plain warm polish with the
        solver-row tolerance viol_tol; the next rounds re-polish with the solver-row limits of the
        offending blocks tightened by the measured row/certificate mismatch,
        margin_j = max(margin_j, g_cert_j - g_row_j - v_target/2) rounded up to 2 significant
        digits (capped at max_margin). Stops at the first round that meets the target (else when
        neither a new margin nor progress), returning the lowest-Lagrangian round that meets it or
        else the lowest-v round: (result, met, rounds, margins, tri, t)."""
        cfg = self.cfg
        margins = {}
        cur, rounds, final, t_pol = start, [], None, 0.0
        for k in range(int(max_rounds)):
            sol = self._tightened(margins)
            t0 = time.perf_counter()
            c, dt = sol._run_warm(cur.x, np.asarray(cur.info.get("lam_al", cur.lam)),
                                  float(cur.info.get("rho", cfg.rho0)), S["tabs"], k_max=k_max, ftol=ftol,
                                  viol_tol=viol_tol)
            res, tri = self._finish(c, S, f"polish{k}", dt, tri, certify=True)
            t_pol += time.perf_counter() - t0
            g_cert = dict(zip(self._cert_blocks, res.info["cert"]["g"]))
            g_row = {j: res.info["g_block_max"].get(j, -np.inf) for j in self._cert_blocks}
            met = bool(res.info["v"] <= v_target and not res.linked and np.isfinite(res.c0_local))
            rounds.append(dict(round=k, margins={j: float(m) for j, m in margins.items()}, c0=float(res.c0_local),
                               c0_lagr=float(res.info["c0_lagr"]), v=float(res.info["v"]), g_cert=g_cert,
                               g_row=g_row, outer=res.info["outer"], stopped=res.info["stopped"],
                               kkt=res.info["kkt"], linked=bool(res.linked), t=dt, met=met))
            if final is None or (met and not final[1]) or (met == final[1] and (
                    res.info["c0_lagr"] < final[0].info["c0_lagr"] if met else res.info["v"] < final[0].info["v"])):
                final = (res, met)
            if met:
                break
            grew = False
            for j in self._cert_blocks:
                if g_cert[j] > v_target:
                    need = _ceil_sig(g_cert[j] - g_row[j] - 0.5 * v_target)   # 2 significant digits, up:
                    mj = min(float(max_margin), max(margins.get(j, 0.0), need))  # fewer distinct programs
                    if mj > margins.get(j, 0.0):
                        margins[j] = mj
                        grew = True
            cur = res
            if not grew and k > 0 and not (res.info["v"] < rounds[-2]["v"]):
                break                                    # neither a new margin nor progress
        res, met = final
        return res, met, rounds, margins, tri, t_pol

    def final_coil_stage(self, boundary: dict, labels=None, families=None, v_target=1e-5, ftol=1e-10, k_max=40,
                         max_rounds=4, polish_viol_tol=None, unfolded_grid=(128, 128), check_grid=(200, 200),
                         grid_rtol=1e-3, refine_grid=(400, 400), max_margin=0.05, max_polish=None) -> CoilResult:
        """Publication coil stage on a fixed boundary (roadmap R1 / G4).

        1. Cold solves from every supported start: labels = [primary] + cfg.fallback_starts (the
           six starts of solve()) of the cfg.start family; `families` adds other start families
           (offset_3d / offset_planar / circle) - all certified.
        2. Polish pool: every ok (certified true-limit violation v <= cfg.ok_viol), unlinked
           result with a finite c0 / c0_lagr (fallback if there is none: every unlinked finite
           result), in the order of the cold Lagrangian c0_lagr = c0 + c0_ref lam.g; max_polish
           (None = all) keeps only the first max_polish of them. Certified cold results are NOT
           preferred: the cold certification fails mostly through the curvature-sampling gap the
           polish below removes, so an uncertified start can polish to a lower certified optimum
           (s200_exact_frac0.7 step00_iter05 + 3 f-evals: L70, the lowest certified cold start,
           polishes to 8.49e-3; L85_tilt-8, cold v 3.1e-3, to 7.92e-3 with certified v 7.5e-10).
        3. Polish every pool member (warm AL from it, ftol) until the CERTIFIED v <= v_target
           (_polish_to_target): round 0 is CoilSolver.polish with viol_tol = polish_viol_tol
           (default 0.1 v_target) on the solver rows. The solver rows sample the curvature at the
           nq quadrature points and the distances by soft-mins, so where those constraints are
           active a certified excess remains (e.g. max curvature between quadrature points); the
           next rounds re-polish with the solver-row limits of the offending blocks tightened by
           the measured row/certificate mismatch (true limits are unchanged: certification and
           all metrics use this solver's programs).
           The result is the LOWEST POLISHED c0_lagr among the polished results that meet the
           target (unlinked, finite); if none does, the lowest certified v.
        4. Metrics: c0 / bn_ratio_int / area-weighted mean / unweighted mean / max |B.n|/|B| on the
           folded 50x50 PHR grid (info) and on unfolded grids with a convergence check
           (flux_metrics_unfolded: unfolded_grid vs check_grid, one refinement to refine_grid if
           they differ by more than grid_rtol), c0_lagr, elasticities, certified v.

        Returns the polished CoilResult; info['final_stage'] holds the chosen start (and its rank
        in the cold order), every cold candidate, every polished candidate ('polished'), the
        chosen candidate's polish rounds and margins, the default-solve value Jf (what solve()
        returns on this boundary), the chosen cold value and the polished value;
        info['flux_unfolded'] the unfolded-grid metrics. Unlike solve() this may raise on a
        broken boundary."""
        cfg, P = self.cfg, self.P
        t_all = time.perf_counter()
        S = self.surface(boundary)
        tri = self._triangulation(S["b"])
        fam0 = cfg.start.rsplit("_", 1)[0] if "_" in cfg.start else cfg.start
        fams = [fam0] + [f for f in (families or ()) if f != fam0]
        primary = cfg.start.rsplit("_", 1)[-1] if "_" in cfg.start else "L85"
        if labels is None:
            labels = [primary] + [lb for lb in cfg.fallback_starts if lb != primary]
        # ---- 1. every start, certified
        cands, rows, t_st = [], [], 0.0
        for fam in fams:
            sizes = {}
            for lb in labels:
                name = lb if fam == fam0 else f"{fam}:{lb}"
                t0 = time.perf_counter()
                try:
                    x0 = self.start(S["b"], lb, sizes, family=fam)
                except Exception as e:  # noqa: BLE001
                    t_st += time.perf_counter() - t0
                    cands.append((fam, name, None))
                    rows.append(dict(start=name, family=fam, error=f"{type(e).__name__}: {e}"))
                    continue
                t_st += time.perf_counter() - t0
                c, dt = self._run_cold(x0, S["tabs"])
                res, tri = self._finish(c, S, name, dt, tri, certify=True)
                cands.append((fam, name, res))
                rows.append(self._summary(res, fam))
        # what solve(boundary) returns (same starts, same order, same selection rule; the cold solve
        # from a given start is deterministic)
        dflt, n = None, 0
        for fam, name, res in cands:
            if fam != fam0 or res is None:
                continue
            if n >= max(1, int(cfg.max_starts)):
                break
            n += 1
            if dflt is None or (res.ok and (not dflt.ok or res.c0_local < dflt.c0_local)) or \
                    (not dflt.ok and not res.ok and res.info["v"] < dflt.info["v"]):
                dflt = res
            if res.ok:
                break
        # ---- 2. polish pool (cold Lagrangian order)
        results = [r for _, _, r in cands if r is not None]

        def finite(r):
            return bool(np.isfinite(r.c0_local) and np.isfinite(r.info.get("c0_lagr", np.nan)))
        tiers = (("ok_unlinked", [r for r in results if r.ok and not r.linked and finite(r)]),
                 ("unlinked_finite", [r for r in results if not r.linked and finite(r)]))
        tier, pool = next(((t, p) for t, p in tiers if p), (None, []))
        if not pool:
            return self._fail(dict(error="final_coil_stage: no usable start", attempts=rows, mode="final_stage",
                                   t_total=time.perf_counter() - t_all))
        pool = sorted(pool, key=lambda r: r.info["c0_lagr"])          # stable: start order on ties
        n_pool = len(pool)
        if max_polish is not None:
            pool = pool[:max(1, int(max_polish))]
        # ---- 3. polish every pool member to certified v <= v_target; rank AFTER the polish
        pvt = 0.1 * v_target if polish_viol_tol is None else float(polish_viol_tol)
        polished, t_pol = [], 0.0
        for rank, cold in enumerate(pool):
            pres, pmet, prounds, pmargins, tri, tp = self._polish_to_target(
                cold, S, tri, v_target=v_target, ftol=ftol, k_max=k_max, max_rounds=max_rounds, viol_tol=pvt,
                max_margin=max_margin)
            t_pol += tp
            polished.append(dict(rank_cold=rank, cold=cold, res=pres, met=pmet, rounds=prounds, margins=pmargins,
                                 t=tp))
        met_pool = [q for q in polished if q["met"]]
        if met_pool:
            best = min(met_pool, key=lambda q: q["res"].info["c0_lagr"])
        else:
            best = min(polished, key=lambda q: q["res"].info["v"])
        chosen, res, met = best["cold"], best["res"], best["met"]
        rounds, margins = best["rounds"], best["margins"]
        prow = [dict(start=q["cold"].info["start"], rank_cold=q["rank_cold"], c0_cold=float(q["cold"].c0_local),
                     c0_lagr_cold=float(q["cold"].info["c0_lagr"]), v_cold=float(q["cold"].info["v"]),
                     certified_cold=bool(q["cold"].certified), Jf_polished=float(q["res"].c0_local),
                     c0_lagr=float(q["res"].info["c0_lagr"]), v=float(q["res"].info["v"]), met=q["met"],
                     linked=bool(q["res"].linked), n_rounds=len(q["rounds"]),
                     margins={j: float(m) for j, m in q["margins"].items()}, t=q["t"],
                     chosen=q is best) for q in polished]
        # ---- 4. metrics
        fu = self.flux_metrics_unfolded(boundary, res.x[:P.nc], res.currents, grid=unfolded_grid,
                                        check_grid=check_grid, rtol=grid_rtol, refine=refine_grid)
        i = res.info
        c0d = float(dflt.c0_local) if dflt is not None else float("nan")
        fs = dict(chosen_start=chosen.info["start"], chosen_tier=tier, chosen_rank_cold=best["rank_cold"],
                  default_start=(dflt.info["start"] if dflt else None),
                  candidates=rows, polished=prow, n_pool=n_pool, n_polished=len(polished),
                  n_met=len(met_pool), max_polish=(None if max_polish is None else int(max_polish)),
                  Jf_default_cold=c0d, Jf_cold_chosen=float(chosen.c0_local),
                  c0_lagr_cold_chosen=float(chosen.info["c0_lagr"]), v_cold_chosen=float(chosen.info["v"]),
                  certified_cold_chosen=bool(chosen.certified),
                  Jf_polished=float(res.c0_local), c0_lagr=float(i["c0_lagr"]), v=float(i["v"]),
                  v_target=float(v_target), v_target_met=met, certified=bool(res.certified), linked=bool(res.linked),
                  rel_change_vs_default=(float(res.c0_local) / c0d - 1.0 if np.isfinite(c0d) else float("nan")),
                  rel_change_vs_chosen=float(res.c0_local) / float(chosen.c0_local) - 1.0,
                  polish_rounds=rounds, margins={j: float(m) for j, m in margins.items()},
                  polish_viol_tol=pvt, ftol=float(ftol), k_max=int(k_max),
                  flux_50x50=dict(c0=float(res.c0_local), bn_ratio_int=i["bn_ratio_int"],
                                  mean_area=i["mean_bn_area"], mean_unweighted=i["mean_bn_unweighted"],
                                  max=i["max_bn"]),
                  flux_unfolded=dict(value_grid=fu["value_grid"], converged=fu["converged"],
                                     rel_diff=fu["rel_diff"],
                                     **{k: fu["value"][k] for k in ("c0", "bn_ratio_int", "mean_area",
                                                                    "mean_unweighted", "max_bn")},
                                     at_grid={k: fu["at_grid"][k] for k in ("grid", "c0", "bn_ratio_int",
                                                                            "mean_area", "mean_unweighted",
                                                                            "max_bn")}),
                  elasticity=dict(i["elasticity"]), t_starts=t_st, t_polish=t_pol,
                  t_total=time.perf_counter() - t_all)
        i.update(mode="final_stage", chosen_start=fs["chosen_start"], attempts=rows, flux_unfolded=fu, final_stage=fs,
                 t_total=fs["t_total"])
        return res

    # ---- envelope -------------------------------------------------------------------------------
    def envelope_grad(self, boundary: dict, result: CoilResult, names: list) -> np.ndarray:
        """d c0_local* / d boundary[names] by the envelope theorem at (result.x, result.lam):
            d/ds [ c0(x*, s) + c0_ref lam*^T g(x*, s) ]   (JAX, FP64; coils and multipliers fixed).
        Names not present in `boundary` are treated as coefficients with value 0."""
        S = self.surface(boundary, names)
        smap = S["map"]
        if len(S["tabs"]) > 5:
            G = np.asarray(smap._lag_grad_bp(jnp.asarray(S["coef"]), jnp.asarray(result.x, F64),
                                             jnp.asarray(result.lam, F64), S["tabs"][4], S["tabs"][5]))
        else:
            G = np.asarray(smap._lag_grad(jnp.asarray(S["coef"]), jnp.asarray(result.x, F64),
                                          jnp.asarray(result.lam, F64), S["tabs"][4]))
        out = np.zeros(len(names))
        for i, nm_ in enumerate(names):
            out[i] = G[smap.index[parse_name(nm_)]]
        return out

    def set_plasma(self, provider, sign=None):
        """Finite beta (opt-in): provider(boundary, TH, PH) -> Cartesian plasma field (N, 3) on the boundary
        at angles (TH, PH) [T], or None (vacuum, the default).  Enters every flux residual and metric as
        B_coil + sign * B_plasma; the plasma field does not depend on the coils.

        sign orients the plasma field (computed by virtual casing in VMEX's field direction) with the
        coils' field: +1 if the coils produce +B_VMEX, -1 if they produce -B_VMEX (the vacuum problem is
        blind to this, (B.n)^2/|B|^2 being even in B).  Default: profile phr.fb_field_sign, else +1;
        check it with plasma_orientation() on actual coils."""
        s = float(sign if sign is not None else _PROB.get("phr", "fb_field_sign", 1.0))
        if s not in (1.0, -1.0):
            raise ValueError(f"plasma field sign must be +1 or -1, got {s}")
        self.plasma_sign = s
        self.plasma = None if provider is None else (lambda b, TH, PH: s * np.asarray(provider(b, TH, PH)))

    def plasma_orientation(self, x_curves, currents, X, B_vmex):
        """Mean cosine between the coils' field and VMEX's field B_vmex (N, 3) at points X (N, 3), and
        whether its sign agrees with the plasma-field sign in use."""
        Bc = np.asarray(self._field_chunked(np.ravel(x_curves), currents, np.asarray(X)))
        Bv = np.asarray(B_vmex)
        cos = np.sum(Bc * Bv, 1) / np.linalg.norm(Bc, axis=1) / np.linalg.norm(Bv, axis=1)
        return float(np.mean(cos)), bool(np.sign(np.mean(cos)) == getattr(self, "plasma_sign", 1.0))

    def bp_cotangent(self, boundary: dict, result: CoilResult) -> np.ndarray:
        """d c0_local / d B_plasma (Nf, 3) on the flux grid at fixed coils: the cotangent that, pulled
        back through the plasma field's dependence on the boundary, gives the field term of the
        envelope gradient (the constraint rows do not depend on B_plasma)."""
        S = self.surface(boundary)
        if len(S["tabs"]) <= 5:
            raise ValueError("bp_cotangent needs a plasma provider (set_plasma)")
        cot = np.asarray(S["map"]._bp_cot(jnp.asarray(S["coef"]), jnp.asarray(result.x, F64), S["tabs"][5]))
        return getattr(self, "plasma_sign", 1.0) * cot      # cotangent of the provider's (unsigned) field

    def lagrangian_parts(self, boundary: dict, x, names=()):
        """(c0_local(x, s), g(x, s)) in FP64 at fixed coils x (used for FD checks)."""
        S = self.surface(boundary, names)
        c0, g = S["map"]._parts(jnp.asarray(S["coef"]), jnp.asarray(x, F64), S["tabs"][4])
        return float(c0), np.asarray(g)

    # ---- evaluation of given coils (e.g. the PHR coils) -----------------------------------------
    def evaluate_coils(self, boundary: dict, x_curves, currents, certify=True):
        """c0_local and flux / geometry metrics of explicit coils: base-curve DOFs (C*3(2*order+1),
        simsopt CurveXYZFourier order) and per-base-coil currents [A]."""
        S = self.surface(boundary)
        P = self.P
        Xf, nf, wf, Sd = S["tabs"][:4]
        Bp = S["tabs"][5] if len(S["tabs"]) > 5 else None
        xc = jnp.asarray(np.ravel(x_curves), F64)
        I = jnp.asarray(currents, F64)
        c0, rho = P._c0_of(xc, I, Xf, nf, wf, Bp)
        ar = np.abs(np.asarray(rho))
        out = dict(c0_local=float(c0), mean_bn_unweighted=float(np.sum(P.flux_mult * ar) / P.n_full),
                   mean_bn_area=float(np.sum(np.asarray(wf) * ar) / float(jnp.sum(wf))),
                   max_bn=float(ar.max()), area=float(jnp.sum(wf)))
        Bc = self._field_chunked(xc, I, np.asarray(Xf))
        Bm = np.linalg.norm(Bc if Bp is None else Bc + np.asarray(Bp), axis=1)
        out["bn_ratio_int"] = float(np.sum(np.asarray(wf) * ar * Bm) / np.sum(np.asarray(wf) * Bm))
        y0 = (I[:-1] / P.I0 - 1.0) if (P.force_on and P.ny) else jnp.zeros(P.ny)   # force rows need the currents
        g = np.asarray(P._g64(jnp.concatenate([xc, y0]), Sd, S["tabs"][4]))
        out["g_block_max"] = {k: float(np.max(g[s])) for k, s in P.blocks.items() if s.stop > s.start}
        if certify:
            ce = jax.tree_util.tree_map(np.asarray, P._cert(xc, *self._triangulation(S["b"]), I))
            out["cert"] = dict(L=ce["L"].tolist(), max_kappa=float(ce["max_kappa"]), msc=ce["msc"].tolist(),
                               d_cc=float(ce["d_cc"]), d_cs=float(ce["d_cs"]), v=float(ce["v"]),
                               linked=bool(ce["linked"]), certified=bool(ce["certified"]))
            if P.force_on:
                out["cert"].update(max_force=ce["max_force"].tolist(), max_force_ref_MN=ce["max_force_ref_MN"].tolist())
            if P.lp_on:
                out["cert"].update(lp_curvature=ce["lp_curvature"].tolist())
        return out

    def _full_grid_geometry(self, boundary: dict, nphi, ntheta, offset=False, chunk=None):
        """Points X, unit normals n and area weights w (sum w = area) of an UNFOLDED full-torus grid
        phi_j = (j [+ 1/2]) 2 pi / nphi, theta_i = 2 pi i / ntheta (numpy, FP64); chunk: points per
        block for large grids (memory), None = one block (c0_full_grid: unchanged arithmetic)."""
        modes = boundary_modes(boundary)
        smap = self._map_for(modes)
        coef = smap.coef(modes)
        nm = smap.nm
        rc, zs = coef[:nm], coef[nm:]
        ph0 = (np.arange(nphi) + (0.5 if offset else 0.0)) * 2 * np.pi / nphi
        PH, TH = np.meshgrid(ph0, 2 * np.pi * np.arange(ntheta) / ntheta, indexing="ij")
        TH, PH = TH.ravel(), PH.ravel()
        if chunk and TH.size > chunk:
            parts = [self._grid_geometry_points(smap, rc, zs, TH[i:i + chunk], PH[i:i + chunk], nphi, ntheta)
                     for i in range(0, TH.size, chunk)]
            return tuple(np.concatenate([p[k] for p in parts]) for k in range(3))
        return self._grid_geometry_points(smap, rc, zs, TH, PH, nphi, ntheta)

    _GEOM_CHUNK = 65536

    def _grid_geometry_points(self, smap, rc, zs, TH, PH, nphi, ntheta):
        cfg = self.cfg
        a = smap.xm[None] * TH[:, None] - cfg.nfp * smap.xn[None] * PH[:, None]
        c, s = np.cos(a), np.sin(a)
        R, Z = c @ rc, s @ zs
        Rt, Zt = (-s * smap.xm) @ rc, (c * smap.xm) @ zs
        Rp, Zp = (s * cfg.nfp * smap.xn) @ rc, (-c * cfg.nfp * smap.xn) @ zs
        cp, sp_ = np.cos(PH), np.sin(PH)
        X = np.stack([R * cp, R * sp_, Z], -1)
        Xt = np.stack([Rt * cp, Rt * sp_, Zt], -1)
        Xp = np.stack([Rp * cp - R * sp_, Rp * sp_ + R * cp, Zp], -1)
        N = np.cross(Xp, Xt)
        nN = np.linalg.norm(N, axis=1)
        w = nN * (2 * np.pi / nphi) * (2 * np.pi / ntheta)
        return X, N / nN[:, None], w

    def c0_full_grid(self, boundary: dict, x_curves, currents, nphi=None, ntheta=None, offset=False):
        """c0_local on an UNFOLDED full-torus grid (validation of the symmetry folding); offset=True
        uses the half-cell phi offset (XCoil 'full' kind)."""
        cfg = self.cfg
        nphi = nphi or cfg.phr_grid[0]
        ntheta = ntheta or cfg.phr_grid[1]
        X, n, w = self._full_grid_geometry(boundary, nphi, ntheta, offset)
        c0 = self.P._c0_explicit(jnp.asarray(np.ravel(x_curves), F64), jnp.asarray(currents, F64), jnp.asarray(X),
                                 jnp.asarray(n), jnp.asarray(w))
        return float(c0)

    _CHUNK = 8192

    def _field_chunked(self, xc, I, X):
        """FP64 B (N, 3) of coils (xc, per-base-coil currents I) at points X, in fixed-size chunks
        (one compile for every grid)."""
        X = np.asarray(X, float)
        N, k = X.shape[0], self._CHUNK
        Xp = np.concatenate([X, np.repeat(X[-1:], (-N) % k, 0)]) if N % k else X
        xc, I = jnp.asarray(xc, F64), jnp.asarray(I, F64)
        out = [np.asarray(self.P._B64(xc, I, jnp.asarray(Xp[i:i + k]))) for i in range(0, Xp.shape[0], k)]
        return np.concatenate(out)[:N]

    def flux_metrics_full_grid(self, boundary: dict, x_curves, currents, nphi=128, ntheta=128, offset=False):
        """|B.n|/|B| metrics of explicit coils on an UNFOLDED full-torus grid (FP64):
        c0 (PHR local squared flux), bn_ratio_int = sum w|B.n| / sum w|B|
        (bn_over_b_avg), mean_area = sum w|B.n|/|B| / sum w, mean_unweighted (grid-point mean of
        |B.n|/|B|; PHR's avg_Bn_over_B at 50x50) and max_bn."""
        X, n, w = self._full_grid_geometry(boundary, int(nphi), int(ntheta), offset, chunk=self._GEOM_CHUNK)
        B = self._field_chunked(np.ravel(x_curves), currents, X)
        if self.plasma is not None:          # finite beta: total vacuum-side field
            ph0 = (np.arange(int(nphi)) + (0.5 if offset else 0.0)) * 2 * np.pi / int(nphi)
            PH, TH = np.meshgrid(ph0, 2 * np.pi * np.arange(int(ntheta)) / int(ntheta), indexing="ij")
            B = B + np.asarray(self.plasma(boundary, TH.ravel(), PH.ravel()))
        Bm = np.linalg.norm(B, axis=1)
        bn = np.sum(B * n, axis=1)
        rho = bn / Bm
        ar = np.abs(rho)
        return dict(grid=[int(nphi), int(ntheta)], offset=bool(offset), c0=float(0.5 * np.sum(w * rho ** 2)),
                    bn_ratio_int=float(np.sum(w * np.abs(bn)) / np.sum(w * Bm)),
                    mean_area=float(np.sum(w * ar) / np.sum(w)), mean_unweighted=float(np.mean(ar)),
                    max_bn=float(np.max(ar)), area=float(np.sum(w)))

    def flux_metrics_unfolded(self, boundary: dict, x_curves, currents, grid=(128, 128), check_grid=(200, 200),
                              rtol=1e-3, refine=(400, 400)):
        """flux_metrics_full_grid on the unfolded `grid` with a convergence check against
        `check_grid`; if that pair differs by more than rtol, one refinement step adds `refine`
        (None: no refinement). converged <=> the relative differences of c0, bn_ratio_int and
        mean_area between the LAST two grids are <= rtol (|B.n| has kinks on the B.n = 0 lines, so
        the |B.n| integrals converge only ~h^2, c0 spectrally; the max of |B.n|/|B| sits in a narrow
        peak and is reported, not tested). 'value' = metrics on the finest grid evaluated."""
        keys = ("c0", "bn_ratio_int", "mean_area", "mean_unweighted", "max_bn")

        def reldiff(a, b):
            return {k: abs(a[k] - b[k]) / max(abs(b[k]), 1e-300) for k in keys}

        def conv(rel):
            return bool(all(rel[k] <= rtol for k in ("c0", "bn_ratio_int", "mean_area")))
        ladder = [self.flux_metrics_full_grid(boundary, x_curves, currents, *grid),
                  self.flux_metrics_full_grid(boundary, x_curves, currents, *check_grid)]
        rel = reldiff(ladder[0], ladder[1])
        rel_check = rel
        if not conv(rel) and refine is not None and tuple(refine) != tuple(check_grid):
            ladder.append(self.flux_metrics_full_grid(boundary, x_curves, currents, *refine))
            rel = reldiff(ladder[-2], ladder[-1])
        return dict(grid=list(grid), check_grid=list(check_grid), at_grid=ladder[0], at_check=ladder[1],
                    rel_diff_grid_vs_check=rel_check, ladder=ladder, rel_diff=rel, rtol=rtol, converged=conv(rel),
                    value=ladder[-1], value_grid=ladder[-1]["grid"])

__all__ = ["CoilConfig", "CoilResult", "CoilSolver", "parse_name", "boundary_modes"]
