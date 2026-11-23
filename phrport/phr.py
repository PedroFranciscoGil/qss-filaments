"""phrport.phr -- port of the OUTER loop of qss_script_jax_qa_phr_envelope.py.

Reproduces, with injected equilibrium / parallel-equilibrium / coil components:

  * the continuation (max_mode = step + 1, VMEC resolution mpol = ntor = 3 + step),
  * the PHR inequality AL1 loop (update_dual_phr / update_rho_conditioned, verbatim),
  * the AL residual builder (simsopt LeastSquaresProblem built by
    qss_helpers.build_al_subproblem_phr, INCLUDING its failure semantics:
    ObjectiveFailure -> raise on an instance's first evaluation, otherwise a
    weighted sentinel 1e12*sqrt(w) of the full length),
  * simsopt least_squares_mpi_solve_qss (scipy least_squares TRF with defaults,
    hybrid Jacobian = forward FD of the equilibrium residuals with
    s_j = max(|x_j| rel, abs) + analytic last row grad_J2 / (2 max(sqrt(max(J2,1e-20)),1e-10)),
    1e12 sentinels, the MPIFiniteDifference wrong-length coercion, per-MPI-group
    first-evaluation state emulated by round-robin column -> group assignment),
  * the coil penalty J2 closure (compute_J2_and_grad): coil warm-start decision
    (max |dsurf| > 1e-5 -> cold reinit), adaptive flux target (frac x the first
    finite coil solve of each step), phantom-flux guard, envelope gradient,
    NaN path, metrics-only refresh at the accepted iterate before the dual update,
  * dual update order (eq dual, coil dual, rho_eq, rho_coil, history, last-good,
    early exit), graceful stop on an equilibrium failure in the dual update,
  * PHR-format checkpoints / phr_history.json / j1_j2_history.json /
    intermediate_state.json / initial_metrics.json / final_metrics.json,
  * the final (cold) coil stage.

The equilibrium evaluator, the parallel evaluator and the coil solver are
INJECTED (see PHRRunner), so everything can be exercised with cheap stubs.

Interfaces expected (phrport.equilibrium / phrport.vmex_pool / phrport.coils):
  evaluate(boundary: dict, step: int) -> EqResult             (never raises)
  evaluate_many(items: list[(dict, int)]) -> list[EqResult]    (order preserved)
  coil_solver.solve(boundary, warm=CoilResult|None) -> CoilResult
  coil_solver.envelope_grad(boundary, result, names) -> ndarray  (d c0_local / d boundary[names])
  free_dof_names(step) -> list[str]
  seed_boundary -> dict

Phantom-guard inputs (PHR quantities the coil interface does not name):
  min coil-surface distance   : result.info['min_cs_dist'] / ... / info['cert']['d_cs'] or
                                coil_solver.phantom_metrics(boundary, result)[0]
  honest mean (B.n/|B|)^2      : result.info['mean_bn_over_b_sq'] / 'mean_bn2_unweighted' or
                                coil_solver.phantom_metrics(boundary, result)[1]
  (unweighted mean over PHR's 50x50 full-torus surface grid; if missing, the guard is skipped)

Coil failures (phrport.coils never raises):
  c0_local non-finite      -> coil_nonfinite_policy 'sentinel' (default): PHR exception path,
                              J2 = 1e12, zero grad, warm-start state untouched;
                              'phr_nan': PHR NaN-metric path (J2 = 0, zero grad, cold restart).
  ok=False, finite c0      -> coil_fail_policy 'use' (default, PHR uses its AL iterate) | 'sentinel'.

Determinism assumption (port knobs that do not change results): equilibria are reused at
bitwise-identical (step, x) [reuse_eq_cache] and the f-eval equilibrium runs in the pool while
the coils solve [overlap_eq_and_coils]; PHR recomputes them, which is identical for a
deterministic evaluator (VmexPool documents bitwise pool == serial).

Roadmap phase 0 (ROADMAP_improvements.md).  Every item that can change iterates is OPT-IN; with
the defaults every pre-existing output value is bit-identical to the pre-phase-0 code.  Pure
reporting additions (new files / new keys only) are on by default and switched off together by
PHRConfig.phase0_reporting=False (--no-phase0-reporting):
  B0  al1_checkpoint_final.json after the PHR loop (reporting); --resume-keep-target (mid-mode
      resume keeping the captured flux target, lam_c / rho_c and the rho-growth history);
      --start-step K (start the continuation at step K keeping the checkpoint's lam / rho);
      --flux-target-floor T (A1: target = S max(frac c0, T)).
  B1  KKT certificate after every outer (feasibility, complementarity, BVLS stationarity; copied
      from the earlier SQP prototype's kkt, without its cvxpy import) in port_outer_log / phr_history / final_metrics
      (reporting; active set = g_i >= -kkt_active_tol OR the PHR hinge active, i.e. lambda_i > 0
      after the dual update); --final-mode-kkt-exit (final mode: exit iff feasible and
      stationary, else up to --kkt-extra-outers more outers with lambda kept).
  R1  --final-coil-stage: CoilSolver.final_coil_stage (all starts, every ok unlinked one polished
      to certified v <= 1e-5, lowest polished Lagrangian) and its metrics in final_metrics; coil export final_coils.json / .npz
      (simsopt CurveXYZFourier + coils_via_symmetries) (reporting).
  G5  provenance.json; eval_x.npz (every f-eval x) + eval_id / qs_total / coil v in eval_log
      (reporting).
  S-a --abort-doomed-coil-solves: the coil solve of a trial TRF certainly rejects (VMEX failed or
      0.5||r_J1||^2 > accepted cost, the skip_coil_on_rejected test) stops before its fallback
      starts; the aborted solve is treated like a skipped one (warm-start state reconstructed on
      demand), so the accepted iterates are unchanged.
  S-b --no-certify-trials: trial coil solves skip certification (soft ok); refresh / final
      solves are certified and the accepted point's coils are certified afterwards (logged).
  S-c --reuse-vmex-state: f-eval equilibria carry their converged VMEX state, which the exact
      Jacobian reuses instead of a cold re-solve (bitwise-identical Jacobian).

Roadmap phase 2 (opt-in; the defaults leave every output bit-identical to the phase-1 code):
  B2  --coil-oracle anchored (default 'phr' = PHR's warm/reset rule): the coil result at TRF's
      ACCEPTED point is kept as the anchor (_coil_acc; set when TRF forms its Jacobian at the point
      of the last coil solve); every coil solve warm-starts from the anchor only (never from a
      rejected trial), with a window on d = max|x - x_anchor| (anchored_coil_solve):
        d <= 1e-3: warm with the anchor's (lam_al, rho), k_max = 2 outers, no certification;
                   accepted iff soft ok and NNLS kkt <= max(3 kkt_anchor, 5e-3), else ->
        d <= 1e-2 (or the guard failed): warm with rho = rho0 and lam_al kept (rho reset); the
                   coil solver falls back to its cold starts when that result is not ok;
        d > 1e-2 / no anchor: cold.
      The post-LS metrics-only refresh becomes a per-outer COLD AUDIT at the accepted iterate that
      keeps (as anchor and for the coil dual update) the lower coil Lagrangian c0 + c0_ref lam.g of
      {anchor at x, cold solve}.  Decisions go to eval_log ('oracle' per f-eval; one
      kind='oracle_audit' line per outer, with an eval_id / eval_x row) and port_outer_log.
      Reset guard (part of the anchored rule since the B2 gate review; --no-oracle-reset-guard = the
      roadmap's literal rule): a rho-reset result that ran out of outers (stop test not met) or fails the
      kkt threshold is replaced by a cold solve.  Without it the offline gate fails on the trajectory
      directions (58/60: two h = 3e-3 rho-reset results that used all 15 outers, err_cold +0.44 % and
      +2.52 % at NNLS kkt 0.40, still counted ok); with it 60/60 (analysis_phrport/B2_STATUS.md).
      --oracle-d-floor D0 (noise floor, opt-in; 0 = off): for d <= D0 the k_max = 2 inherited solve
      (relative c0 noise ~3e-4 at d ~ 1e-6) is replaced by --oracle-floor-rule: 'linear' (default)
      returns the anchor's coils with c0 = c0_anchor + G.dx (G = the anchor's envelope gradient, as
      in TRF's model; no solve, result uncertified; the per-outer audit prefers its cold solve over
      such an anchor) or 'converge' (the inherited solve with --oracle-floor-k-max outers).
  B5  --carry-flux-multiplier: lam_c / rho_c are not reset at the start of a mode (the flux target
      is still re-captured).

Roadmap phase 3 (A4; opt-in, a separate entry point; run() is unchanged):
  The PHR outer body is PHRRunner._outer_iteration (shared by run() and sweep(); run() output is
  bit-identical to the pre-A4 code).  PHRRunner.sweep(SweepConfig) / CLI --sweep --resume-from SEED:
  from a seed (checkpoint, run directory or front point) at its mode, for eps_k = r^k Jf0
  (--sweep-ratio r = 0.85; Jf0 = a cold coil solve at the seed, or --sweep-jf0) the absolute flux
  target S eps_k is set, lam_eq / rho_eq and lam_c are carried, rho_c is chosen so that J2 ~ QS at
  the start point (sweep_rho_c), at most --sweep-outers (4) outers run with the B1 KKT exit, and
  front_point_{k:02d}.json is written (boundary, QS at the PHR resolution, coils of the final coil
  stage (polished), KKT certificate, lam_c, wall times; + coil export, wout, al1_checkpoint_front{k}).
  Stops at --sweep-max-points, below --sweep-min-target, when rho_c reaches rho_max, after 2
  (--sweep-infeasible-stop) consecutive points missing their target, or on a VMEC failure.
  analysis_phrport/front_collect.py gathers the points into the non-dominated front.
  al1_checkpoint_final.json of a sweep carries the sweep fields (sweep_k = the last completed point,
  or the failed one with sweep_point_done False after a VMEC failure; sweep_jf0, sweep_ratio,
  sweep_eps), so --sweep --resume-from <sweep dir> continues at the next k with the same Jf0 (a
  directory whose final checkpoint lacks them seeds from its highest al1_checkpoint_frontKK.json).
"""
# Parts of this module (the PHR helpers update_dual_phr / update_rho_conditioned, finite_difference_steps and the
# port of least_squares_mpi_solve_qss) are derived from simsopt (https://github.com/hiddenSymmetries/simsopt),
# MIT License, Copyright (c) 2023 Simsopt contributors; see THIRD_PARTY_NOTICES.md.
import os as _os
import sys as _sys
# same rule as phrport/__init__.py (which already ran when imported as phrport.phr); this line only
# matters when the file is executed by path: $PHRPORT_PYCACHE, else a set prefix, else <repo>/.pycache
_sys.pycache_prefix = (_os.environ.get("PHRPORT_PYCACHE") or _sys.pycache_prefix
                       or _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), ".pycache"))

import argparse
import collections
import copy
import datetime
import glob
import hashlib
import inspect
import json
import math
import os
import platform
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict

import numpy as np
from phrport import problem as _PROB
from scipy.optimize import least_squares

__all__ = [
    "update_dual_phr", "update_rho_conditioned", "finite_difference_steps",
    "PHRConfig", "ALSubproblem", "EquilibriumFailure", "CoilSolveFailure", "CoilSolveAborted",
    "eq_constraint_labels", "eq_constraint_values", "least_squares_qss",
    "LSOutcome", "PHRRunner", "run_self_test", "surface_area_volume",
    "bvls_multipliers", "kkt_certificate", "export_coils", "collect_provenance",
    "surface_distance", "coil_lagrangian", "anchored_coil_solve", "SweepConfig", "sweep_rho_c",
]

SEP = "=" * 60
SEP_THIN = "-" * 60

# ---------------------------------------------------------------------------
# PHR constants (qss_script_jax_qa_phr_envelope.py)
# ---------------------------------------------------------------------------
RHO_MAX = 1e4
V_TOL = 1e-4
LAM_MAX = 1e3
SENTINEL = 1.0e12                 # LeastSquaresProblem.fail / mpi.py sentinel
J2_FLOOR = 1e-20                  # sqrt(max(J2, 1e-20))
SQRT_J2_FLOOR = 1e-10             # max(sqrt(...), 1e-10) in jac_hybrid
SKIP_REL_MARGIN = 1e-10           # skip_coil_on_rejected: 0.5||r_eq||^2 > C (1 + margin); the
                                  # margin covers dot-product rounding (m * eps << 1e-10)
FLUX_PHANTOM_EPS = 1e-7
CS_THRESHOLD = 0.15
SURFACE_DOF_RESET_THRESHOLD = 1e-5

# keys may be nested with '.', e.g. 'cert.d_cs' (phrport.coils certification block)
_MIN_CS_KEYS = ("min_cs_dist", "min_cs", "min_cs_distance", "cs_min", "shortest_cs_distance",
                "cert.d_cs")
_HONEST_KEYS = ("mean_bn_over_b_sq", "mean_bn2_unweighted", "honest_c0", "bn_over_b_sq_mean")


# ---------------------------------------------------------------------------
# PHR helpers: VERBATIM copies of qss_helpers.update_dual_phr / update_rho_conditioned
# ---------------------------------------------------------------------------
def update_dual_phr(lam, rho, g_vals, lam_max):
    """PHR multiplier update for inequality constraints g_i <= 0:

        lam_i <- clip(lam_i + rho_i * g_i, 0, lam_max)

    Returns (lam_new, violations) with violations = max(0, g).
    """
    g = np.asarray(g_vals, dtype=np.float64)
    lam_new = np.clip(lam + rho * g, 0.0, lam_max)
    return lam_new, np.maximum(0.0, g)


def update_rho_conditioned(rho, v, v_prev, theta=0.25, tau=2.0,
                           rho_max=1e4, v_tol=1e-4):
    """LANCELOT-style conditioned per-constraint penalty growth.

    Grow rho_i (factor tau, capped at rho_max) only where v_i > v_tol AND
    v_i > theta * v_prev_i. On the first call (v_prev is None) rho is
    returned unchanged. Returns (rho_new, grew).
    """
    rho = np.asarray(rho, dtype=np.float64).copy()
    v = np.asarray(v, dtype=np.float64)
    if v_prev is None:
        return rho, np.zeros(rho.shape, dtype=bool)
    v_prev = np.asarray(v_prev, dtype=np.float64)
    grew = (v > v_tol) & (v > theta * v_prev)
    rho[grew] = np.minimum(rho[grew] * tau, rho_max)
    return rho, grew


def finite_difference_steps(x, abs_step=1.0e-7, rel_step=0.0):
    """Verbatim simsopt._core.util.finite_difference_steps."""
    if abs_step < 0:
        raise ValueError('abs_step must be >= 0')
    if rel_step < 0:
        raise ValueError('rel_step must be >= 0')
    steps = np.max((np.abs(x) * rel_step, np.full(len(x), abs_step)), axis=0)
    if np.any(steps == 0.0):
        raise ValueError('Finite difference step size cannot be 0. '
                         'Increase abs_step.')
    return steps


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class PHRConfig:
    """Defaults = the PHR script's CLI defaults. The reference run
    AB_with_coils_A4p5_scale100 used flux_metric_scale = 100 (all else default)."""
    n_continuation_steps: int = 4
    al_outer_iter: int = 6
    max_nfev: int = 20
    rho_init: float = 1.0
    rho_tau: float = 2.0
    rho_theta: float = 0.25
    rho_max: float = RHO_MAX
    v_tol: float = V_TOL
    lam_max: float = LAM_MAX
    # equilibrium inequality targets (hard-coded in the PHR script)
    aspect_target: float = float(_PROB.get("phr", "aspect_target", 4.5))
    iota_mean_target: float = float(_PROB.get("phr", "iota_mean_target", 0.43))
    elongation_threshold: float = float(_PROB.get("phr", "elongation_threshold", 4.0))
    vacuum_well: bool = bool(_PROB.get("phr", "vacuum_well", False))
    well_threshold: float = float(_PROB.get("phr", "well_threshold", 0.015))
    # problem profiles (phrport.problem): aspect as an equality band |A - A*| <= aspect_tol (two rows)
    # instead of PHR's A <= A*, and an iota band |iota - iota*| <= iota_tol (PHR: 0, an equality)
    aspect_two_sided: bool = bool(_PROB.get("phr", "aspect_two_sided", False))
    aspect_tol: float = float(_PROB.get("phr", "aspect_tol", 0.0))
    iota_tol: float = float(_PROB.get("phr", "iota_tol", 0.0))
    # size constraints (problem profiles): plasma volume and minor radius as equality bands
    # |V - V*| <= volume_tol V*, |a - a*| <= aminor_tol a* (0 target = off); aspect_off drops the
    # aspect rows (implied by V and a, since V = 2 pi^2 R a^2)
    aspect_off: bool = bool(_PROB.get("phr", "aspect_off", False))
    # finite beta (opt-in; profile phr.finite_beta): virtual-casing plasma field in the coil problem,
    # context at Jacobian points, field term of the envelope gradient (phrport.finite_beta)
    finite_beta: bool = bool(_PROB.get("phr", "finite_beta", False))
    fb_n: int = int(_PROB.get("phr", "fb_n", 32))
    fb_digits_value: int = int(_PROB.get("phr", "fb_digits_value", 6))
    fb_digits_grad: int = int(_PROB.get("phr", "fb_digits_grad", 4))
    # QI profiles (equilibrium.physics = "qi"): mirror ratio Delta <= mirror_max at s = 1/51
    # (Goodman JPP Eq. 17-18; 0 = off), iota bands |iota(0) - t| <= iota_end_tol and
    # |iota(1) - t| <= iota_end_tol (PRX Energy f_iota; target 0 = off), and iota_mean_off to drop
    # PHR's mean-iota pair.  Defaults leave PHR unchanged.
    mirror_max: float = float(_PROB.get("phr", "mirror_max", 0.0))
    # opt-in lower bound Delta >= mirror_min (8 Nov, seed sensitivity: aim for a mirror ratio; 0 = off)
    mirror_min: float = float(_PROB.get("phr", "mirror_min", 0.0))
    # max-J row g = LSE(slope - target) <= 0 (target and pitches in equilibrium.qi_options.maxj_*)
    maxj: bool = bool(_PROB.get("phr", "maxj", False))
    # max-J row active from continuation step K (0-based) on (profile phr.maxj_from_step; 0 = every step).
    # Before step K the row is held at g = -1 (no hinge, no gradient), so its multiplier stays 0 and the row
    # enters at step K with lam = 0; the constraint count never changes, so the multipliers carry across
    # stages (the paper's max-J-late schedule inside one call)
    maxj_from_step: int = int(_PROB.get("phr", "maxj_from_step", 0))
    # closure ramp (profile phr.maxj_ramp_outers N, 0 = off): the row is relaxed to g = maxj - off_k with
    # off_k = g0 max(0, 1 - (k + 1) / N), g0 = max(0, maxj) at the point where the row switches on and k the outers
    # since then, i.e. the target is lowered linearly from the switch-on value to the profile target over N outers
    maxj_ramp_outers: int = int(_PROB.get("phr", "maxj_ramp_outers", 0))
    # trust radius at the start of every continuation step from maxj_from_step on (profile phr.maxj_switch_delta,
    # 0 = PHRPORT_TRUST_RESET_STEP behaviour): replaces scipy's initial Delta = ||x|| after the step reset
    maxj_switch_delta: float = float(_PROB.get("phr", "maxj_switch_delta", 0.0))
    # closure tracking (profile phr.maxj_track_frac r, 0 = off; replaces the linear ramp): at the start of every
    # outer from switch-on the row is relaxed to g = maxj - (1 - r) maxj_k, maxj_k the closure value at the current
    # iterate, i.e. each outer asks for a relative decrease r of the closure residual; once maxj_k <= maxj_track_floor
    # the relaxation is 0 and the row is the true closure constraint
    maxj_track_frac: float = float(_PROB.get("phr", "maxj_track_frac", 0.0))
    maxj_track_floor: float = float(_PROB.get("phr", "maxj_track_floor", 0.5))
    # field-line B_max spread row g = LSE(spread - target) <= 0 (target, temperatures in
    # equilibrium.qi_options.bmaxsp_*), active from continuation step phr.bmaxsp_from_step (0-based; -1 before)
    bmaxsp: bool = bool(_PROB.get("phr", "bmaxsp", False))
    bmaxsp_from_step: int = int(_PROB.get("phr", "bmaxsp_from_step", 0))
    iota_axis_target: float = float(_PROB.get("phr", "iota_axis_target", 0.0))
    iota_edge_target: float = float(_PROB.get("phr", "iota_edge_target", 0.0))
    iota_end_tol: float = float(_PROB.get("phr", "iota_end_tol", 0.01))
    # Mercier row (mercier_margin - merc) / mercier_scale <= 0, merc = smooth min of s DMerc
    # (phrport.mercier; equilibrium.mercier_options must be set)
    mercier: bool = bool(_PROB.get("phr", "mercier", False))
    mercier_margin: float = float(_PROB.get("phr", "mercier_margin", 0.0))
    mercier_scale: float = float(_PROB.get("phr", "mercier_scale", 1.0))
    iota_edge_tol: float = float(_PROB.get("phr", "iota_edge_tol", 0.0))   # lowA_ncsx: edge band only (0 = iota_end_tol)
    iota_mean_off: bool = bool(_PROB.get("phr", "iota_mean_off", False))
    volume_target: float = float(_PROB.get("phr", "volume_target", 0.0))
    volume_tol: float = float(_PROB.get("phr", "volume_tol", 1e-3))
    aminor_target: float = float(_PROB.get("phr", "aminor_target", 0.0))
    aminor_tol: float = float(_PROB.get("phr", "aminor_tol", 1e-3))
    # QS ceiling (opt-in): row g = QS_total / qs_max - 1 <= 0, appended after every other row (0 = off), so a
    # tight coil flux bound can trade QS up to this cap (exact gradient 2 r^T dr/dx / qs_max)
    qs_max: float = float(_PROB.get("phr", "qs_max", 0.0))
    # coils / promoted squared flux (ENABLE_FLUX_FIT is forced ON in the envelope script)
    enable_coils: bool = True
    flux_metric_scale: float = 1.0
    flux_phr_target: float = 0.0
    flux_target_frac: float = 0.5
    flux_phantom_eps: float = FLUX_PHANTOM_EPS
    cs_threshold: float = float(_PROB.get("phr", "cs_threshold", CS_THRESHOLD))   # CS_FLUX_SAFE_MIN = 0.5 * this
    surface_dof_reset_threshold: float = SURFACE_DOF_RESET_THRESHOLD
    reinit_coils_each_alm: bool = True
    # surface least squares (least_squares_mpi_solve_qss)
    fd_abs_step: float = 1e-5
    fd_rel_step: float = 1e-5
    diff_method: str = "forward"
    ls_kwargs: dict = field(default_factory=dict)   # extra scipy kwargs (PHR passes none)
    ls_verbose: int = 0                             # PHR used verbose=2 (print only)
    fd_ngroups: int = 8                             # PHR: mpirun -n 8 -> 8 groups
    # port knobs that do not change results (given a deterministic evaluator)
    overlap_eq_and_coils: bool = True    # run the f-eval equilibrium in the pool while coils solve
    reuse_eq_cache: bool = True          # reuse EqResults at bitwise-identical (step, x): f-evals,
                                         # FD points, jac probe, dual update (PHR recomputes them)
    eq_cache_size: int = int(_PROB.get("phr", "eq_cache_size", 256))
    # resolution of the initial (seed) metrics run: input.nfp2_QA mpol=ntor=5 == 3 + 2
    initial_eval_step: int = 2
    # CoilResult.ok == False with a FINITE c0 (linked / certified violation): 'use' (PHR: the AL
    # coil solve always returns its iterate) or 'sentinel' (failed coil evaluation -> J2 = 1e12)
    coil_fail_policy: str = "use"
    # NON-FINITE c0 (phrport.coils never raises; a build/solve failure returns c0=NaN):
    # 'sentinel' = PHR's exception path (J2 = 1e12, zero grad: the trial is rejected) or
    # 'phr_nan' = PHR's NaN-metric path (J2 = 0, zero grad, cold coil restart next call)
    coil_nonfinite_policy: str = "sentinel"
    skip_surface_opt: bool = False
    # OPT-IN (default 'fd' = PHR): 'exact' replaces the forward-FD equilibrium block of the
    # Jacobian by phrport.exact_jacobian (implicit VMEX tangents, gauge-continued map; falls
    # back to FD at any failure). exact_jac_gauge: 'sqrt' (default) | 'frozen'.
    eq_jacobian: str = "fd"
    exact_jac_gauge: str = "sqrt"
    # OPT-IN speedups (defaults = PHR exactly; see least_squares_qss / phrport.trf_port):
    #  skip_coil_on_rejected: run VMEX first; when the trial is certain to be rejected by TRF
    #    (VMEX failed / non-finite residuals, or 0.5||r_eq||^2 alone > the accepted cost) skip the
    #    coil solve. Exact: same accepted iterates (coil warm-start state is reconstructed lazily).
    #  stall_stop K > 0: stop an inner least_squares after K consecutive rejected trials once the
    #    trust radius < stall_delta_factor * xtol * (xtol + ||x||) (returns the current iterate).
    #    Default factor 10 (~5e-8 at ||x|| ~ 0.5): TRF was seen to recover from a collapsed radius
    #    and make real progress at radii ~1e-6 (the speed-up runs), while the
    #    only acceptances below ~1e-8 in runs_phrport/scale200 were noise (||step|| ~ 2e-9).
    #    Cross-outer caveat: the post-LS metrics-only coil refresh is warm-started from the last
    #    evaluated trial, so later outers are not bit-identical to the default.
    #  carry_trust_radius: start each least_squares (after the first) with the trust radius
    #    Delta0 = min(largest accepted step of the previous call, ||x0||) instead of ||x0||.
    #    CHANGES the iterates.
    skip_coil_on_rejected: bool = False
    skip_coil_scope: str = "head"         # 'head' | 'all' (which f-evals are sequenced; timing only)
    stall_stop: int = 0
    stall_delta_factor: float = 10.0      # see the stall_stop note above / trf_port
    carry_trust_radius: bool = False
    spectral_scale: float = 0.0          # opt-in (29 Oct): TRF x_scale_i = (1 + m^2 + n^2)^(-p/2) per free DOF (0 = off)
    # ---- roadmap phase 0 (see the module docstring); every default reproduces the previous code
    flux_target_floor: float = 0.0        # A1: captured target = S max(frac c0, floor) (0 = PHR)
    final_mode_flux_target: float = 0.0   # A5: final mode captures the absolute target S*E (0 = off)
    # coil-fit flux bound stated as <|B.n|>/<|B|> (opt-in, profile phr.flux_target_bn, 0 = off): at each capture the
    # c0 target is c0 (b*/b)^2, b = the coil solution's <|B.n|>/<|B|> (bn_ratio_int); c0 is quadratic in the normal
    # field, so the bound asks for <|B.n|>/<|B|> <= b* (never above the frac / floor target)
    flux_target_bn: float = float(_PROB.get("phr", "flux_target_bn", 0.0))
    resume_keep_target: bool = False      # B0: resume keeps the checkpoint's target / lam_c / rho_c
    start_step: int = None                # B0: first continuation step (None = checkpoint step / 0)
    kkt_active_tol: float = 1e-3          # B1: g_i >= -tol enters the BVLS estimate (BVLS multiplier estimate)
    kkt_phr_active: bool = True           # B1: ... and so does every constraint PHR treats as active
                                          #     (lambda_i > 0 after the dual update)
    kkt_grad_floor: float = 1e-3          # B1: stationarity = ||res|| / max(floor, ||grad QS||)
    final_mode_kkt_exit: bool = False     # B1: final-mode exit iff feasibility < v_tol and
    kkt_stationarity_tol: float = 0.1     #     stationarity < this, else up to kkt_extra_outers
    kkt_extra_outers: int = 3             #     more outers (lambda kept)
    final_coil_stage: bool = False        # R1/G4: CoilSolver.final_coil_stage for the final coils
    full_outer_budget: bool = False       # every mode runs all al_outer_iter outers (no PHR exit
                                          # when all violations are below v_tol)
    abort_doomed_coil_solves: bool = False   # S-a
    certify_trials: bool = True           # S-b: False = trial coil solves are not certified
    reuse_vmex_state: bool = False        # S-c: exact Jacobian reuses the f-eval VMEX state
    phase0_reporting: bool = True         # provenance / eval x sidecar / KKT logging / final
                                          # checkpoint / coil export (new files and keys only)
    # ---- roadmap phase 2 (see the module docstring); every default reproduces the previous code
    coil_oracle: str = "phr"              # B2: 'phr' (PHR warm/reset rule) | 'anchored'
    oracle_d_inherit: float = 1e-3        # B2: d <= this: warm, inherited (lam, rho), k_max outers
    oracle_d_reset: float = 1e-2          # B2: d <= this (or guard failed): warm with rho = rho0
    oracle_k_max: int = 2                 # B2: outer budget of the inherited warm solve
    oracle_kkt_factor: float = 3.0        # B2: guard kkt <= max(factor * kkt_anchor, floor)
    oracle_kkt_floor: float = 5e-3
    oracle_audit: bool = True             # B2: per-outer cold audit (False: plain refresh rule)
    oracle_reset_guard: bool = True       # B2: a rho-reset result that ran out of outers or fails the kkt
                                          #    threshold -> cold.  Part of the anchored rule (the offline gate
                                          #    fails without it); False = the roadmap's literal rule.  Read
                                          #    only with coil_oracle='anchored' (no effect on 'phr').
    oracle_d_floor: float = 0.0           # B2 noise floor (opt-in, 0 = off): d <= this uses oracle_floor_rule
    oracle_floor_rule: str = "linear"     #    'linear': c0_anchor + G.dx on the anchor's coils (no solve);
                                          #    'converge': the inherited warm solve with oracle_floor_k_max outers
    oracle_floor_k_max: int = 15          #    (stop test decides; 15 = CoilConfig.warm_max_outer)
    carry_flux_multiplier: bool = False   # B5: keep lam_c / rho_c at the start of a mode
    carry_flux_from_mode: int = 0         # B5: carry only into modes >= this (0 = every mode)
    # ---- 2026-11 single-run speed-ups (P1 / P5 OFF by default; P7 on, finite beta only)
    phase_timers: bool = False            # P1: per-eval / per-Jacobian phase table (phase_timers.jsonl)
    fb_reuse_tol: float = 0.0             # P5: finite beta: keep the plasma-field context / field term while
    fb_reuse_every: int = 4               #     max |dx| since the refresh < tol, refresh at least every k Jacobians
    fb_jit: str = "vjp"                   # P7: finite beta: jitted virtual casing (vjp | off = eager | check);
                                          #     falls back to eager for the rest of a mode if jit raises

    @property
    def cs_flux_safe_min(self):
        return 0.5 * self.cs_threshold

    @classmethod
    def reference_scale100(cls, **kw):
        return cls(flux_metric_scale=100.0, **kw)


class EquilibriumFailure(Exception):
    """simsopt ObjectiveFailure raised by LeastSquaresProblem on an instance's first evaluation."""


class CoilSolveFailure(Exception):
    """A coil solve that produced no usable result."""


class CoilSolveAborted(Exception):
    """(phase 0, S-a) A trial coil solve stopped by should_abort: the trial is certainly rejected by
    TRF, so its J2 is never used; the runner treats it like a skipped coil solve."""


# ---------------------------------------------------------------------------
# Constraint metrics and the AL subproblem
# ---------------------------------------------------------------------------
def eq_constraint_labels(cfg):
    labels = [] if cfg.aspect_off else (["aspect"] + (["aspect_lo"] if cfg.aspect_two_sided else []))
    labels += ([] if cfg.iota_mean_off else ["iota_mean_hi", "iota_mean_lo"]) + ["elong"]
    if cfg.vacuum_well:
        labels.append("well")
    if cfg.mirror_max > 0.0:
        labels.append("mirror")
    if cfg.mirror_min > 0.0:
        labels.append("mirror_lo")
    if cfg.maxj:
        labels.append("maxj")
    if cfg.bmaxsp:
        labels.append("bmaxsp")
    if cfg.iota_axis_target != 0.0:
        labels += ["iota_axis_hi", "iota_axis_lo"]
    if cfg.iota_edge_target != 0.0:
        labels += ["iota_edge_hi", "iota_edge_lo"]
    if cfg.volume_target > 0.0:
        labels += ["volume_hi", "volume_lo"]
    if cfg.aminor_target > 0.0:
        labels += ["aminor_hi", "aminor_lo"]
    if cfg.mercier:
        labels.append("merc")
    if cfg.qs_max > 0.0:
        labels.append("qs_cap")
    return labels


# continuation step the runner is in (set with PHRRunner._step), read for PHRConfig.maxj_from_step, and the max-J
# closure relaxation off_k (PHRConfig.maxj_ramp_outers / maxj_track_frac), set per outer by the runner.  The runner
# executes as __main__ (python -m phrport.phr) while exact_jacobian.phr_rows_jacobian imports phrport.phr, a second
# module instance: both share these lists through the package module, so the Jacobian sees the same g as the residuals
import sys as _sys_state
_pkg_state = _sys_state.modules.get(__package__) if __package__ else None
_shared_state = getattr(_pkg_state, "_phr_maxj_state", None) if _pkg_state is not None else None
if _shared_state is None or len(_shared_state) < 2:
    _shared_state = ([None], [0.0])
    if _pkg_state is not None:
        _pkg_state._phr_maxj_state = _shared_state
_MAXJ_STEP, _MAXJ_OFFSET = _shared_state[0], _shared_state[1]


def maxj_active(cfg):
    """True when the max-J row counts at the current continuation step (always for maxj_from_step <= 0)."""
    return cfg.maxj_from_step <= 0 or (_MAXJ_STEP[0] is not None and int(_MAXJ_STEP[0]) >= cfg.maxj_from_step)


def bmaxsp_active(cfg):
    """True when the B_max spread row counts at the current continuation step (always for bmaxsp_from_step <= 0)."""
    return cfg.bmaxsp_from_step <= 0 or (_MAXJ_STEP[0] is not None and int(_MAXJ_STEP[0]) >= cfg.bmaxsp_from_step)


def eq_constraint_values(eq, cfg):
    """Raw signed g <= 0 (same expressions as _aspect_g/_iota_mean_hi/_lo/_elong_g/_well_g)."""
    if cfg.aspect_off:
        g = []
    elif cfg.aspect_two_sided:
        g = [float(eq.aspect) - (cfg.aspect_target + cfg.aspect_tol),
             (cfg.aspect_target - cfg.aspect_tol) - float(eq.aspect)]
    else:
        g = [float(eq.aspect) - cfg.aspect_target]
    if not cfg.iota_mean_off:
        g += [float(eq.mean_iota) - (cfg.iota_mean_target + cfg.iota_tol),
              (cfg.iota_mean_target - cfg.iota_tol) - float(eq.mean_iota)]
    g += [float(eq.elongation) - cfg.elongation_threshold]
    if cfg.vacuum_well:
        g.append(cfg.well_threshold - float(eq.well))
    if cfg.mirror_max > 0.0:
        g.append(float(getattr(eq, "mirror", float("nan"))) - cfg.mirror_max)
    if cfg.mirror_min > 0.0:
        g.append(cfg.mirror_min - float(getattr(eq, "mirror", float("nan"))))
    if cfg.maxj:
        g.append(float(getattr(eq, "maxj", float("nan"))) - _MAXJ_OFFSET[0] if maxj_active(cfg) else -1.0)
    if cfg.bmaxsp:
        g.append(float(getattr(eq, "bmaxsp", float("nan"))) if bmaxsp_active(cfg) else -1.0)
    for kk, tgt in (("iota_axis", cfg.iota_axis_target), ("iota_edge", cfg.iota_edge_target)):
        if tgt != 0.0:
            v = float(getattr(eq, kk))
            tol = cfg.iota_edge_tol if (kk == "iota_edge" and cfg.iota_edge_tol > 0.0) else cfg.iota_end_tol
            g += [v - (tgt + tol), (tgt - tol) - v]
    # size rows are relative (g in units of the target), like the other rows' O(1) scale
    if cfg.volume_target > 0.0:
        v = float(getattr(eq, "volume", float("nan"))) / cfg.volume_target
        g += [v - (1.0 + cfg.volume_tol), (1.0 - cfg.volume_tol) - v]
    if cfg.aminor_target > 0.0:
        r = float(getattr(eq, "aminor", float("nan"))) / cfg.aminor_target
        g += [r - (1.0 + cfg.aminor_tol), (1.0 - cfg.aminor_tol) - r]
    if cfg.mercier:
        g.append((cfg.mercier_margin - float(getattr(eq, "merc", float("nan")))) / cfg.mercier_scale)
    if cfg.qs_max > 0.0:
        g.append(float(eq.qs_total) / cfg.qs_max - 1.0)
    return np.array(g, dtype=np.float64)


def exact_constraint_grads(jac, cfg):
    """d g_i / d x for every eq constraint label, from an exact-Jacobian dict (rows of g <= 0)."""
    dg = {"aspect": jac["aspect"], "aspect_lo": -np.asarray(jac["aspect"]),
          "iota_mean_hi": jac["mean_iota"], "iota_mean_lo": -np.asarray(jac["mean_iota"]),
          "elong": jac["elongation"]}
    if "well" in jac:
        dg["well"] = -np.asarray(jac["well"])
    if "mirror" in jac:
        dg["mirror"] = np.asarray(jac["mirror"]); dg["mirror_lo"] = -np.asarray(jac["mirror"])
    if "maxj" in jac:
        dg["maxj"] = np.asarray(jac["maxj"])
    if "bmaxsp" in jac:
        dg["bmaxsp"] = np.asarray(jac["bmaxsp"])
    for kk in ("iota_axis", "iota_edge"):
        if kk in jac:
            dg[kk + "_hi"] = np.asarray(jac[kk]); dg[kk + "_lo"] = -np.asarray(jac[kk])
    for kk, tgt in (("volume", cfg.volume_target), ("aminor", cfg.aminor_target)):
        if kk in jac and tgt > 0.0:
            dg[kk + "_hi"] = np.asarray(jac[kk]) / tgt
            dg[kk + "_lo"] = -np.asarray(jac[kk]) / tgt
    if "merc" in jac and cfg.mercier:
        dg["merc"] = -np.asarray(jac["merc"]) / cfg.mercier_scale
    if cfg.qs_max > 0.0 and "qs_total" in jac:
        dg["qs_cap"] = np.asarray(jac["qs_total"]) / cfg.qs_max
    return dg


def align_eq_multipliers(ck, labels, rho_init, log=None):
    """(lam_eq, rho_eq) of a checkpoint laid out on ``labels``. The checkpoint's own row order comes from its
    ``eq_labels`` or, for checkpoints written before rows were added, from PHRPORT_RESUME_EQ_LABELS (comma list);
    rows new to ``labels`` start at lam = 0, rho = rho_init. Checkpoints without eq_labels that are exactly one row
    short of ``labels`` ending in "merc" (a Mercier stage resumed from a stage without the row) get that row
    appended; any other length mismatch with no label list raises instead of silently misaligning the multipliers."""
    lam = np.asarray(ck["lam_eq"], dtype=np.float64); rho = np.asarray(ck["rho_eq"], dtype=np.float64)
    old = ck.get("eq_labels") or [t for t in os.environ.get("PHRPORT_RESUME_EQ_LABELS", "").split(",") if t]
    if not old:
        if len(lam) == len(labels):
            return lam, rho
        if len(lam) == len(labels) - 1 and labels[-1] in ("merc", "qs_cap"):
            old = list(labels[:-1])
        else:
            raise RuntimeError(f"checkpoint has {len(lam)} eq multipliers for {len(labels)} constraint rows "
                               f"{labels}; set PHRPORT_RESUME_EQ_LABELS to the checkpoint's row labels")
    if len(old) != len(lam):
        raise RuntimeError(f"checkpoint eq labels {old} do not match its {len(lam)} multipliers")
    if list(old) == list(labels):
        return lam, rho
    lo, ro = dict(zip(old, lam)), dict(zip(old, rho))
    lam_n = np.array([lo.get(t, 0.0) for t in labels]); rho_n = np.array([ro.get(t, rho_init) for t in labels])
    gone = [t for t in old if t not in labels]
    if gone and log:
        log(f"  [WARNING] checkpoint rows not in this problem: {gone}; their multipliers are dropped")
    if log:
        log(f"  eq multipliers remapped {list(old)} -> {list(labels)}; new rows "
            f"{[t for t in labels if t not in lo]} start at lam 0, rho {rho_init}; dropped {[t for t in old if t not in labels]}")
    return lam_n, rho_n


class ALSubproblem:
    """Port of build_al_subproblem_phr(OBJECTIVES, INEQ_CONSTRAINTS, lam, rho):

        LeastSquaresProblem.from_tuples([(qs.residuals, 0, 1)] +
                                        [(hinge_i, 0, rho_i/2) for i])
        hinge_i = max(0, g_i + lam_i/rho_i)

    residuals = (f - goal) * sqrt(weights). Failure semantics of
    LeastSquaresProblem.unweighted_residuals (fail=1e12) are reproduced PER MPI
    GROUP: an ObjectiveFailure on a group's first evaluation of this instance is
    re-raised (EquilibriumFailure); afterwards it returns
    full(nvals, 1e12) * sqrt(weights_of_last_success).
    One instance per outer iteration (PHR rebuilds al_prob every outer iter)."""

    def __init__(self, lam, rho, cfg, n_groups=1, constraint_fn=None):
        self.cfg = cfg
        self.constraint_fn = constraint_fn or (lambda eq: eq_constraint_values(eq, cfg))
        self.lam = np.asarray(lam, dtype=np.float64).copy()
        self.rho = np.asarray(rho, dtype=np.float64).copy()
        self.n_con = len(self.lam)
        self.shift = [float(self.lam[i] / self.rho[i]) for i in range(self.n_con)]
        self.inp_weights = [1.0] + [float(self.rho[i]) / 2.0 for i in range(self.n_con)]
        self.n_terms = 1 + self.n_con             # parent_return_fns_no
        self.n_groups = max(1, int(n_groups))
        self.first_eval = [True] * self.n_groups
        self.nvals = None                         # residual length (set by the first success)
        self.weights = None                       # per-residual weights of the last success
        self.n_fail = 0

    def unweighted(self, eq):
        outputs, new_weights = [], []
        qs = np.asarray(eq.qs_residuals, dtype=np.float64)
        out = np.array([qs]) if not np.ndim(qs) else np.asarray(qs)
        outputs.append(out - 0.0)
        new_weights += [self.inp_weights[0]] * len(out)
        g = self.constraint_fn(eq)
        for i in range(self.n_con):
            h = max(0.0, float(g[i]) + self.shift[i])
            outputs.append(np.array([h]) - 0.0)
            new_weights += [self.inp_weights[i + 1]]
        return np.concatenate(outputs), np.asarray(new_weights)

    def residuals(self, eq, group=0):
        group = int(group) % self.n_groups
        if eq is None or not bool(eq.converged):
            self.n_fail += 1
            if self.first_eval[group]:
                raise EquilibriumFailure(getattr(eq, "message", "equilibrium failure"))
            return np.full(self.nvals, SENTINEL) * np.sqrt(self.weights)
        u, w = self.unweighted(eq)
        if self.first_eval[group]:
            if self.nvals is None:
                self.nvals = len(u)
            self.first_eval[group] = False
        self.weights = w
        return u * np.sqrt(w)

    def peek_residuals(self, eq, group=0):
        """(phase 0, S-a) What residuals(eq, group) would return, WITHOUT its side effects
        (first-evaluation flags, stored length / weights, failure count); None where it would
        raise EquilibriumFailure (a failure on the group's first evaluation)."""
        group = int(group) % self.n_groups
        if eq is None or not bool(eq.converged):
            if self.first_eval[group]:
                return None
            return np.full(self.nvals, SENTINEL) * np.sqrt(self.weights)
        u, w = self.unweighted(eq)
        return u * np.sqrt(w)

    def hinges(self, eq):
        g = self.constraint_fn(eq)
        return np.array([max(0.0, float(g[i]) + self.shift[i]) for i in range(self.n_con)])


# ---------------------------------------------------------------------------
# least_squares_mpi_solve_qss port
# ---------------------------------------------------------------------------
@dataclass
class LSOutcome:
    x: np.ndarray
    x0: np.ndarray
    result: object = None
    failed: bool = False          # least_squares raised -> x0 returned ("Failure on proc0_world")
    error: str = ""
    nfev_total: int = 0           # f evaluations (as counted by the driver)
    njev: int = 0
    n_fd_points: int = 0
    last_f_x: np.ndarray = None
    f_log: list = field(default_factory=list)       # per f-eval: dict(x, cost, J2, r_fail, j2_fail)
    jac_log: list = field(default_factory=list)     # per jac: dict(x, jac (optional))
    time_f: float = 0.0
    time_jac: float = 0.0
    jac_x_mismatch: int = 0
    n_exact_jac: int = 0          # Jacobians whose equilibrium block came from eq_jac (opt-in)
    # opt-in speedups (zero / None on the default path)
    n_coil_skipped: int = 0       # f-evals whose coil solve was skipped (certain TRF rejection)
    used_port: bool = False       # phrport.trf_port instead of scipy.optimize.least_squares
    stall_stopped: bool = False
    Delta0: float = None          # initial trust radius passed to the port (None = scipy's)
    Delta_final: float = None
    max_accepted_step_h: float = None
    # phase 0
    n_coil_aborted: int = 0       # S-a: trial coil solves stopped by should_abort
    last_jac: object = None       # B1: the full Jacobian of the last jac call (a reference)
    last_jac_x: np.ndarray = None


def E_NFP():
    from . import equilibrium as _E
    return int(_E.NFP)


def least_squares_qss(x0, residuals_fn, residuals_many_fn, compute_J2_and_grad, n_terms, *,
                      abs_step=1.0e-7, rel_step=0.0, diff_method="forward", n_groups=1,
                      residuals_prefetch=None, iteration_callback=None, jac_callback=None,
                      log=None, keep_jacobians=False, verbose=0, eq_jac=None, j2_grad_hook=None,
                      skip_coil_on_rejected=False, skip_gate=None, on_coil_skipped=None,
                      skip_coil_scope="head", stall_stop=0, stall_delta_factor=10.0, Delta0=None,
                      use_trf_port=False, abort_doomed=False, abort_gate=None, phase_hook=None, **kwargs):
    """Port of simsopt.solve.mpi.least_squares_mpi_solve_qss (proc0 view).

    residuals_fn(x) -> r_J1 (group 0; may raise -> sentinel full(last_len or n_terms, 1e12))
    residuals_many_fn(list_of_x, list_of_groups) -> list of r_J1 or Exception instances,
        evaluated in column order (the MPIFiniteDifference leader loop order).
    compute_J2_and_grad(x) -> (J2, grad_J2); exceptions -> J2 = 1e12, zero grad.
        None -> plain least_squares_mpi_solve (no sqrt(J2) row).
    residuals_prefetch(x) -> thunk; if given, the equilibrium is started before the
        coil solve and collected after it (results identical; overlap only).
    kwargs -> scipy.optimize.least_squares (PHR: max_nfev only).
    eq_jac(x) -> equilibrium Jacobian block (rows of residuals_fn) or None (OPT-IN; None or a
        wrong shape falls back to the FD block; default None = PHR's FD).

    OPT-IN speedups (all off by default; the default path calls scipy exactly as before):
    skip_coil_on_rejected: after the first Jacobian (i.e. once TRF has an accepted cost C),
        each f-eval runs the equilibrium FIRST (no overlap) and skips compute_J2_and_grad when
        TRF is certain to reject the trial whatever J2 is (J2 >= 0, row sqrt(max(J2,1e-20))):
          * r_J1 non-finite -> TRF's non-finite branch (Delta = 0.25*||step_h||, no ratio /
            termination test) is taken for any J2, so the returned vector keeps r_J1 as is;
          * 0.5*||r_J1||^2 > C*(1+SKIP_REL_MARGIN) (this covers the 1e12 VMEX sentinels): the
            true cost_new > C, so actual_reduction < 0 -> ratio <= 0 (pred > 0) or 0
            (pred <= 0) -> Delta = 0.25*||step_h|| in update_tr_radius, check_termination's
            ftol test is false (ratio <= 0.25) and its xtol test does not involve f, the trial
            is rejected and alpha/Delta are updated exactly as with the true J2.
        The returned vector is [r_J1, sqrt(1e-20)] (J2 replaced by its lower bound 0), whose
        cost is > C as well.  FD points / the Jacobian probe never go through _f; the first
        f-eval (x0) is never skipped (no accepted cost yet).  skip_gate() -> bool lets the
        caller veto a skip; on_coil_skipped(x) is told about every skipped coil solve.
        Sequencing loses the VMEX/coil overlap (~VMEX time per f-eval), so skip_coil_scope
        chooses which f-evals are sequenced (and may skip): 'head' (default) = the first trial
        of the call (radius ||x0||: 19 of 23 skippable on runs_phrport/scale200) and every trial
        right after a skipped one; 'all' = every f-eval after the first Jacobian.  The scope
        only moves wall time; results are identical either way.
    stall_stop / stall_delta_factor / Delta0 / use_trf_port: run phrport.trf_port (verbatim port
        of scipy's TRF, bit-identical with the hooks off) with the stall-stop hook and/or the
        initial trust radius Delta0; LSOutcome then carries Delta_final / max_accepted_step_h.
    abort_doomed (phase 0, S-a; OPT-IN): once TRF has an accepted cost C (after the first
        Jacobian), compute_J2_and_grad(x, should_abort=pred) is called; pred() (called by the
        coil solver before each FALLBACK start) is True when the trial is certainly rejected
        whatever J2 is -- the skip_coil_on_rejected test above on the equilibrium residuals of
        the trial: VMEX failed (1e12 sentinels) / non-finite / 0.5||r_J1||^2 > C(1+margin).
        In the overlap path the residuals are read without side effects and without blocking
        (residuals_prefetch(x).peek(); unknown yet -> False); abort_gate() -> bool can veto.
        compute_J2_and_grad raises CoilSolveAborted after an aborted solve: the f-eval then
        returns [r_J1, sqrt(1e-20)] like a skipped coil solve (rejected all the same).
    LSOutcome.last_jac / last_jac_x: the full Jacobian of the last jac call (B1 certificate).
    phase_hook(kind, info) (2026-11 speedups, OPT-IN; passive): timing hook, kind in ('f_begin',
        'f_end', 'jac_begin', 'jac_end'); info carries the wall times measured here (t_eq_jac,
        t_fd, t_hook for the Jacobian).  It never changes what is evaluated.
    """
    log = log or (lambda *a, **k: None)
    x0 = np.array(x0, dtype=np.float64, copy=True)
    n = x0.size
    with_j2 = compute_J2_and_grad is not None
    st = dict(nevals=0, last_f_nevals=0, last_J2=None, last_grad_J2=None,
              last_r_J1_len=None, jac_size=None, x_cache=None, jac_cache=None)
    out = LSOutcome(x=x0.copy(), x0=x0.copy())
    skip_on = bool(skip_coil_on_rejected) and with_j2
    abort_on = bool(abort_doomed) and with_j2
    track_acc = skip_on or abort_on            # TRF's accepted cost is needed
    if skip_on:
        if skip_coil_scope not in ("head", "all"):
            raise ValueError(f"skip_coil_scope must be 'head' or 'all', got {skip_coil_scope!r}")
        st.update(acc_cost=None, last_res=None, prev_skipped=False)
    elif abort_on:
        st.update(acc_cost=None, last_res=None)

    def _sentinel_len():
        # length of the 1e12 vector _f substitutes for a raising residuals call
        return st["last_r_J1_len"] if (with_j2 and st["last_r_J1_len"] is not None) else n_terms

    def _abort_predicate(thunk, r_known):
        """S-a: certain TRF rejection of the current trial, from its equilibrium part alone."""
        acc = st["acc_cost"]

        def pred():
            if abort_gate is not None and not bool(abort_gate()):
                return False
            if thunk is not None:
                peek = getattr(thunk, "peek", None)
                if peek is None:
                    return False
                state, r = peek()
                if state == "pending":
                    return False
                if state == "raise":
                    r = np.full(_sentinel_len(), 1.0e12)
            else:
                r = r_known
            r = np.atleast_1d(np.asarray(r, dtype=np.float64))
            return (not bool(np.all(np.isfinite(r)))
                    or 0.5 * float(np.dot(r, r)) > acc * (1.0 + SKIP_REL_MARGIN))
        return pred

    def _j2_eval(x, abort_pred=None):
        try:
            if abort_pred is None:
                J2, grad_J2 = compute_J2_and_grad(x)
            else:
                J2, grad_J2 = compute_J2_and_grad(x, should_abort=abort_pred)
            J2 = float(J2)
            grad_J2 = np.asarray(grad_J2, dtype=np.float64).ravel()
            j2_fail = False
        except CoilSolveAborted:
            st["last_J2"] = None          # never used: an aborted trial is never accepted
            st["last_grad_J2"] = None
            out.n_coil_aborted += 1
            return None, False
        except Exception as e:  # noqa: BLE001  (mpi.py catches everything)
            log(f"  [LS] compute_J2_and_grad failed ({type(e).__name__}: {e}); "
                f"using J2=1e12, zero grad.")
            J2 = 1.0e12
            grad_J2 = np.zeros(n)
            j2_fail = True
        st["last_J2"] = J2
        st["last_grad_J2"] = grad_J2
        return J2, j2_fail

    def _f_sequenced(x):
        """skip_coil_on_rejected path: equilibrium first, coils only if TRF could accept."""
        r_fail = False
        try:
            r_J1 = np.atleast_1d(residuals_fn(x))
            st["last_r_J1_len"] = r_J1.size
        except Exception:
            r_fail = True
        if r_fail:
            _n = st["last_r_J1_len"] if st["last_r_J1_len"] is not None else n_terms
            r_J1 = np.full(_n, 1.0e12)
        certain_reject = (not bool(np.all(np.isfinite(r_J1)))
                          or 0.5 * float(np.dot(r_J1, r_J1)) > st["acc_cost"] * (1.0 + SKIP_REL_MARGIN))
        if certain_reject and (skip_gate is None or bool(skip_gate())):
            st["last_J2"] = None          # never used: a skipped trial is never accepted
            st["last_grad_J2"] = None
            out.n_coil_skipped += 1
            if on_coil_skipped is not None:
                on_coil_skipped(np.array(x, dtype=np.float64, copy=True))
            return r_J1, r_fail, 0.0, False, True
        J2, j2_fail = _j2_eval(x)
        return r_J1, r_fail, J2, j2_fail, False

    def _f(x):
        t0 = time.time()
        x = np.asarray(x)
        if phase_hook is not None:
            phase_hook("f_begin", dict(x=x, nevals=st["nevals"] + 1))
        skipped = False
        aborted = False
        if skip_on and st["acc_cost"] is not None and (
                skip_coil_scope == "all" or st["nevals"] == 1 or st["prev_skipped"]):
            r_J1, r_fail, J2, j2_fail, skipped = _f_sequenced(x)
        else:
            thunk = residuals_prefetch(x) if residuals_prefetch is not None else None
            r_fail = False
            if thunk is None:
                try:
                    r_J1 = np.atleast_1d(residuals_fn(x))
                    st["last_r_J1_len"] = r_J1.size
                except Exception:
                    r_fail = True
            j2_fail = False
            if with_j2:
                if abort_on and st["acc_cost"] is not None:
                    r_known = None
                    if thunk is None:
                        r_known = np.full(_sentinel_len(), 1.0e12) if r_fail else r_J1
                    J2, j2_fail = _j2_eval(x, _abort_predicate(thunk, r_known))
                    if J2 is None:
                        aborted, J2 = True, 0.0     # lower bound: the trial is rejected anyway
                else:
                    J2, j2_fail = _j2_eval(x)
            if thunk is not None:
                try:
                    r_J1 = np.atleast_1d(thunk())
                    st["last_r_J1_len"] = r_J1.size
                except Exception:
                    r_fail = True
            if r_fail:
                # least_squares_mpi_solve_qss: cached successful length, else parent_return_fns_no;
                # least_squares_mpi_solve (no J2): always parent_return_fns_no
                _n = (st["last_r_J1_len"] if (with_j2 and st["last_r_J1_len"] is not None)
                      else n_terms)
                r_J1 = np.full(_n, 1.0e12)
        if with_j2:
            sqrt_J2 = np.sqrt(max(J2, J2_FLOOR))
            residuals = np.concatenate([r_J1, [sqrt_J2]])
            objective_val = float(np.sum(r_J1 ** 2)) + J2
        else:
            residuals = r_J1
            objective_val = float(np.sum(r_J1 ** 2))
        if skip_on:
            st["last_res"] = residuals
            st["prev_skipped"] = skipped
        elif abort_on:
            st["last_res"] = residuals
        st["nevals"] += 1
        st["last_f_nevals"] = st["nevals"]
        out.last_f_x = np.array(x, copy=True)
        rec = dict(x=np.array(x, copy=True), objective=objective_val,
                   J2=(J2 if with_j2 else None), r_fail=r_fail, j2_fail=j2_fail,
                   n_res=int(residuals.size))
        if skipped:
            rec.update(J2=None, coil_skipped=True)
        if aborted:
            rec.update(J2=None, coil_aborted=True)
        out.f_log.append(rec)
        out.time_f += time.time() - t0
        if phase_hook is not None:
            phase_hook("f_end", dict(rec, wall=time.time() - t0, nevals=st["nevals"],
                                     skipped=skipped, aborted=aborted))
        if iteration_callback is not None:
            try:
                iteration_callback(st["nevals"], x, residuals, objective_val)
            except Exception as e:  # noqa: BLE001
                log(f"  [LS] iteration_callback raised: {e}")
        return residuals

    def _safe_many(xs, groups):
        res = residuals_many_fn(xs, groups)
        fixed = []
        for r in res:
            if isinstance(r, BaseException) or r is None:
                r = np.full(n_terms, 1.0e12)          # _safe_eq_residuals except-branch
            fixed.append(np.atleast_1d(np.asarray(r, dtype=np.float64)))
        return fixed

    def _fd_jac(x):
        # MPIFiniteDifference.jac
        if st["x_cache"] is not None and np.all(x == st["x_cache"]) and st["jac_cache"] is not None:
            return st["jac_cache"]
        if st["jac_size"] is None:          # one evaluation of the code at x (proc0)
            probe = _safe_many([np.array(x, copy=True)], [0])[0]
            st["jac_size"] = (len(probe), n)
        x0_ = np.copy(x)
        steps = finite_difference_steps(x0_, abs_step=abs_step, rel_step=rel_step)
        if diff_method == "centered":
            nevals_jac = 2 * n
            xs = np.zeros((n, nevals_jac))
            for j in range(n):
                xs[:, 2 * j] = x0_[:]
                xs[j, 2 * j] = x0_[j] + steps[j]
                xs[:, 2 * j + 1] = x0_[:]
                xs[j, 2 * j + 1] = x0_[j] - steps[j]
        elif diff_method == "forward":
            nevals_jac = n + 1
            xs = np.zeros((n, nevals_jac))
            xs[:, 0] = x0_[:]
            for j in range(n):
                xs[:, j + 1] = x0_[:]
                xs[j, j + 1] = x0_[j] + steps[j]
        else:
            raise ValueError(f"Finite difference method {diff_method} not implemented.")
        cols = [np.array(xs[:, j], copy=True) for j in range(nevals_jac)]
        groups = [j % max(1, n_groups) for j in range(nevals_jac)]
        res = _safe_many(cols, groups)
        m = st["jac_size"][0]
        evals = np.zeros((m, nevals_jac))
        for j, r in enumerate(res):
            if r.shape[0] != m:                 # MPIFiniteDifference coercion
                r = np.full(m, 1.0e12)
            evals[:, j] = r
        jac = np.zeros(st["jac_size"])
        if diff_method == "centered":
            for j in range(n):
                jac[:, j] = (evals[:, 2 * j] - evals[:, 2 * j + 1]) / (2 * steps[j])
        else:
            for j in range(n):
                jac[:, j] = (evals[:, j + 1] - evals[:, 0]) / steps[j]
        out.n_fd_points += nevals_jac
        st["x_cache"] = np.array(x, copy=True)
        st["jac_cache"] = jac
        return jac

    def _jac(x, *a):
        t0 = time.time()
        x = np.asarray(x)
        tj = {}
        if phase_hook is not None:
            phase_hook("jac_begin", dict(nevals=st["last_f_nevals"]))
        if out.last_f_x is None or not np.array_equal(out.last_f_x, x):
            out.jac_x_mismatch += 1     # never expected with scipy TRF (diagnostic only)
            if track_acc:
                st["acc_cost"] = None   # cannot tell TRF's accepted cost: stop skipping
        elif track_acc:
            # TRF calls jac right after accepting the last f-eval (or at x0): its cost is
            # 0.5 * np.dot(f, f) of exactly the vector _f returned
            f_acc = st["last_res"]
            st["acc_cost"] = 0.5 * np.dot(f_acc, f_acc)
        J_J1 = None
        if eq_jac is not None:
            _t = time.time()
            try:
                J_J1 = eq_jac(np.array(x, copy=True))
            except Exception as e:  # noqa: BLE001
                log(f"  [LS] eq_jac raised ({type(e).__name__}: {e}); using FD.")
                J_J1 = None
            if J_J1 is not None:
                J_J1 = np.asarray(J_J1, dtype=np.float64)
                n_rows = st["last_r_J1_len"]
                if J_J1.ndim != 2 or J_J1.shape[1] != n or (n_rows is not None and J_J1.shape[0] != n_rows):
                    log(f"  [LS] eq_jac shape {J_J1.shape} != ({n_rows}, {n}); using FD.")
                    J_J1 = None
                else:
                    out.n_exact_jac += 1
            tj["t_eq_jac"] = time.time() - _t
        if J_J1 is None:
            _t = time.time()
            J_J1 = np.asarray(_fd_jac(x))
            tj["t_fd"] = time.time() - _t
        if with_j2:
            J2, grad_J2 = st["last_J2"], st["last_grad_J2"]
            if j2_grad_hook is not None and J2 is not None and grad_J2 is not None:
                _t = time.time()
                try:                       # finite beta: exact plasma field + field term at x
                    grad_J2 = np.asarray(j2_grad_hook(np.array(x, copy=True), J2, grad_J2), dtype=np.float64)
                    st["last_grad_J2"] = grad_J2
                except Exception as e:  # noqa: BLE001
                    log(f"  [LS] j2_grad_hook raised ({type(e).__name__}: {e}); explicit coil gradient only.")
                tj["t_hook"] = time.time() - _t
            if J2 is None or grad_J2 is None:
                log("  [LS] jac_hybrid: no stored J2/grad_J2; using zero last row.")
                last_row = np.zeros(n)
            else:
                sqrt_J2 = max(np.sqrt(max(J2, J2_FLOOR)), SQRT_J2_FLOOR)
                last_row = (1.0 / (2.0 * sqrt_J2)) * np.asarray(grad_J2).ravel()
            last_row = np.atleast_2d(last_row)
            if J_J1.ndim == 1:
                J_J1 = J_J1.reshape(1, -1)
            full_jac = np.vstack([J_J1, last_row])
        else:
            full_jac = J_J1
        out.njev += 1
        out.last_jac = full_jac                     # reference only (B1 certificate)
        out.last_jac_x = np.array(x, copy=True)
        out.jac_log.append(dict(x=np.array(x, copy=True),
                                jac=(np.array(full_jac, copy=True) if keep_jacobians else None)))
        out.time_jac += time.time() - t0
        if jac_callback is not None:
            _t = time.time()
            try:
                jac_callback(st["last_f_nevals"], x, full_jac)
            except Exception as e:  # noqa: BLE001
                log(f"  [LS] jac_callback raised: {e}")
            tj["t_jac_callback"] = time.time() - _t
        if phase_hook is not None:
            phase_hook("jac_end", dict(tj, wall=time.time() - t0, nevals=st["last_f_nevals"]))
        return full_jac

    use_port = bool(use_trf_port) or (int(stall_stop or 0) > 0) or (Delta0 is not None)
    try:
        if use_port:
            from .trf_port import least_squares_trf
            out.used_port = True
            out.Delta0 = None if Delta0 is None else float(Delta0)
            result = least_squares_trf(_f, x0, _jac, verbose=verbose, Delta0=Delta0,
                                       stall_stop=int(stall_stop or 0),
                                       stall_delta_factor=stall_delta_factor, **kwargs)
            out.stall_stopped = bool(result.stall_stopped)
            out.Delta_final = float(result.Delta_final)
            out.max_accepted_step_h = float(result.max_accepted_step_h)
        else:
            result = least_squares(_f, x0, jac=_jac, verbose=verbose, **kwargs)
        out.result = result
        out.x = np.array(result.x, dtype=np.float64, copy=True)
    except Exception as e:  # noqa: BLE001  ("Failure on proc0_world in least_squares_mpi_solve_qss")
        log(f"  [LS] Failure on proc0_world in least_squares_mpi_solve_qss: "
            f"{type(e).__name__}: {e}")
        out.failed = True
        out.error = f"{type(e).__name__}: {e}"
        out.x = x0.copy()
    out.nfev_total = st["nevals"]
    return out


# ---------------------------------------------------------------------------
# Geometry helper (final/initial metrics: surf.area(), surf.volume())
# ---------------------------------------------------------------------------
def _parse_name(name):
    pre, rest = name.split("(")
    m, n = rest.rstrip(")").split(",")
    return pre, int(m), int(n)


def surface_area_volume(boundary, nfp, ntheta=96, nphi_per_period=64):
    """Area and volume of the stellarator-symmetric RZ-Fourier boundary
    (simsopt convention: angle = m*theta - n*nfp*phi). Trapezoidal rule on a full-torus
    grid: spectrally accurate for these smooth periodic integrands."""
    th = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
    ph = np.linspace(0, 2 * np.pi, nphi_per_period * nfp, endpoint=False)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    R = np.zeros_like(TH); Z = np.zeros_like(TH)
    Rt = np.zeros_like(TH); Zt = np.zeros_like(TH)
    Rp = np.zeros_like(TH); Zp = np.zeros_like(TH)
    for name, v in boundary.items():
        v = float(v)
        if v == 0.0:
            continue
        pre, m, nn = _parse_name(name)
        a = m * TH - nn * nfp * PH
        if pre == "rc":
            R += v * np.cos(a); Rt += -m * v * np.sin(a); Rp += nn * nfp * v * np.sin(a)
        elif pre == "zs":
            Z += v * np.sin(a); Zt += m * v * np.cos(a); Zp += -nn * nfp * v * np.cos(a)
    cp, sp = np.cos(PH), np.sin(PH)
    X_t = np.stack([Rt * cp, Rt * sp, Zt], -1)
    X_p = np.stack([Rp * cp - R * sp, Rp * sp + R * cp, Zp], -1)
    N = np.cross(X_p, X_t)
    dA = (2 * np.pi / ntheta) * (2 * np.pi / PH.shape[1])
    area = float(np.sum(np.linalg.norm(N, axis=-1)) * dA)
    X = np.stack([R * cp, R * sp, Z], -1)
    volume = float(abs(np.sum(np.einsum("ijk,ijk->ij", X, N))) * dA / 3.0)
    return area, volume


# ---------------------------------------------------------------------------
# Phase 0 (B1): KKT certificate -- bounded least-squares multipliers, as in the earlier SQP prototype
# (that module imports cvxpy at the top, which .venv-vmex does not have)
# ---------------------------------------------------------------------------
def bvls_multipliers(grad, cols, lower):
    """min ||grad + A lam||_2 s.t. lam >= lower, A = [cols] (scipy lsq_linear BVLS,
    tol 1e-14). Returns (lam, residual = grad + A lam)."""
    from scipy.optimize import lsq_linear
    grad = np.asarray(grad, dtype=np.float64).ravel()
    if not cols:
        return np.zeros(0), grad.copy()
    A = np.array(cols, dtype=np.float64).T
    lo = np.asarray(lower, dtype=np.float64)
    sol = lsq_linear(A, -grad, bounds=(lo, np.full(lo.size, np.inf)), method="bvls", tol=1e-14)
    lam = np.asarray(sol.x, dtype=np.float64)
    return lam, grad + A @ lam


def _rss_mb():
    """Resident set size of this process in MB (Linux /proc; 0 when unavailable)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 0


def kkt_certificate(grad_f, g, lam, grads, *, active_tol=1e-3, grad_floor=1e-3,
                    iota_pair=("iota_mean_hi", "iota_mean_lo"), phr_active=True):
    """First-order certificate of  min f  s.t.  g_i <= 0  at a point (roadmap B1).

    grad_f : gradient of f (here QS = sum of the QS residuals squared);
    g, lam : {label: constraint value g_i, PHR multiplier lambda_i} (same labels, PHR units:
             g_coil = scale c0 - target); lambda_i is the multiplier AFTER the PHR dual update
             at this point, lambda_i = clip(lambda_i^old + rho_i g_i, 0, lam_max), so
             lambda_i > 0  <=>  the hinge g_i + lambda_i^old / rho_i of the outer's subproblem
             is positive (PHR's own active set);
    grads  : {label: gradient of g_i or None (not available)}.
    The iota pair (g_hi = iota - t, g_lo = t - iota) is an equality: when both are active it
    enters the bounded least squares as ONE free-sign column (lambda_net = lambda_hi - lambda_lo).

      feasibility     = max(max_i g_i^+, |g_iota|)
      complementarity = max_i |min(-g_i, lambda_i)|   (PHR multipliers; *_ls: BVLS multipliers)
      stationarity    = ||grad_f + A^T lambda_ls||_2 / max(grad_floor, ||grad_f||_2), lambda_ls >= 0
                        from BVLS over the active set
                            {g_i >= -active_tol}  (BVLS multiplier estimate)
                          U {lambda_i > 0}        (phr_active=True: what PHR itself treats as
                                                   active at this point)
                        (stationarity_inf: the same in the max norm; stationarity_phr: with the
                        PHR multipliers instead of lambda_ls, when every needed gradient exists).
    One absolute tolerance alone cannot serve rows whose scales differ by orders of magnitude:
    on runs_phrport/scale200_exactjac the scaled flux row sits at g = -1.5e-3 (-2.5 % of its
    target) with lambda = 2.8e-3; without it the certificate reads 0.93, with it 0.031 (the
    roadmap value). A multiplier the fit puts on a row that is not tight shows up in
    complementarity_ls. 'active_phr_only' lists the rows that entered only through lambda_i > 0.
    Active constraints without a gradient are listed in 'missing' (complete = False)."""
    grad_f = np.asarray(grad_f, dtype=np.float64).ravel()
    labels = list(g.keys())
    gv = {k: float(g[k]) for k in labels}
    lv = {k: float(lam.get(k, 0.0)) for k in labels}
    fin = [k for k in labels if np.isfinite(gv[k])]
    hi, lo = iota_pair
    feas = max([max(0.0, gv[k]) for k in fin] + [0.0])
    if hi in fin:
        feas = max(feas, abs(gv[hi]))
    elif lo in fin:
        feas = max(feas, abs(gv[lo]))
    compl = max([abs(min(-gv[k], lv[k])) for k in fin] + [0.0])
    by_tol = [k for k in fin if gv[k] >= -active_tol]
    active = [k for k in fin if gv[k] >= -active_tol or (phr_active and lv[k] > 0.0)]

    def grad_of(k):
        d = grads.get(k)
        if d is None and k in (hi, lo):         # the pair shares one gradient up to sign
            o = lo if k == hi else hi
            if grads.get(o) is not None:
                d = -np.asarray(grads[o], dtype=np.float64)
        if d is None:
            return None
        d = np.asarray(d, dtype=np.float64).ravel()
        return d if (d.size == grad_f.size and np.all(np.isfinite(d))) else None

    merged = hi in active and lo in active
    cols, lower, names, missing = [], [], [], []
    for k in active:
        if merged and k == lo:
            continue
        d = grad_of(k)
        if d is None:
            missing.extend([hi, lo] if (merged and k == hi) else [k])
            continue
        cols.append(d)
        lower.append(-np.inf if (merged and k == hi) else 0.0)
        names.append("iota_net" if (merged and k == hi) else k)
    lam_bv, res = bvls_multipliers(grad_f, cols, lower)
    lam_ls = {k: 0.0 for k in labels}
    for nm_, l_ in zip(names, lam_bv):
        if nm_ == "iota_net":
            lam_ls[hi], lam_ls[lo] = max(float(l_), 0.0), max(-float(l_), 0.0)
        else:
            lam_ls[nm_] = float(l_)
    gn = float(np.linalg.norm(grad_f))
    gi = float(np.max(np.abs(grad_f))) if grad_f.size else 0.0
    rn = float(np.linalg.norm(res))
    ri = float(np.max(np.abs(res))) if res.size else 0.0
    stat_phr, missing_phr = None, []
    r_phr = grad_f.copy()
    for k in fin:
        if lv[k] > 0.0:
            d = grad_of(k)
            if d is None:
                missing_phr.append(k)
            else:
                r_phr = r_phr + lv[k] * d
    if not missing_phr:
        stat_phr = float(np.linalg.norm(r_phr)) / max(grad_floor, gn)
    compl_ls = max([abs(min(-gv[k], lam_ls[k])) for k in fin] + [0.0])
    return dict(
        feasibility=feas, complementarity=compl, complementarity_ls=compl_ls,
        stationarity=rn / max(grad_floor, gn), stationarity_inf=ri / max(grad_floor, gi),
        stationarity_phr=stat_phr, grad_norm=gn, residual_norm=rn,
        active=active, active_phr_only=[k for k in active if k not in by_tol], phr_active=bool(phr_active),
        missing=sorted(set(missing)), complete=not missing,
        lam_ls=lam_ls, lam_phr=lv, g=gv, iota_merged=merged,
        lambda_iota_net_ls=lam_ls.get(hi, 0.0) - lam_ls.get(lo, 0.0),
        active_tol=float(active_tol), grad_floor=float(grad_floor))


# ---------------------------------------------------------------------------
# Phase 0 (R1): coil export (simsopt CurveXYZFourier + coils_via_symmetries)
# ---------------------------------------------------------------------------
_COIL_LOADER = """\
import json, numpy as np
from simsopt.geo import CurveXYZFourier
from simsopt.field import Current, coils_via_symmetries, BiotSavart
d = json.load(open(PATH))
curves = []
for dofs in d['base_curve_dofs']:
    c = CurveXYZFourier(d['quadpoints'], d['order']); c.x = np.asarray(dofs); curves.append(c)
coils = coils_via_symmetries(curves, [Current(I) for I in d['base_currents_A']], d['nfp'], d['stellsym'])
bs = BiotSavart(coils)"""


def export_coils(path_stem, coil_solver, result, boundary, *, source="", extra=None):
    """Write <stem>.json and <stem>.npz with the base-coil curve DOFs (simsopt CurveXYZFourier.x
    order, one row per base coil) and per-base-coil currents [A] of a phrport.coils result;
    the full coil set is coils_via_symmetries(curves, currents, nfp, stellsym=True) (the JSON
    'loader' field is a ready simsopt snippet). Returns the JSON path, or None when the solver
    does not expose the layout (cfg.order / n_coils / nfp / nq, P.nc)."""
    cfg = getattr(coil_solver, "cfg", None)
    Pp = getattr(coil_solver, "P", None)
    if cfg is None or Pp is None or result is None or getattr(result, "x", None) is None:
        return None
    order, C, nfp, nq = int(cfg.order), int(cfg.n_coils), int(cfg.nfp), int(cfg.nq)
    nc = int(Pp.nc)
    x = np.asarray(result.x, dtype=np.float64).ravel()
    if x.size < nc or nc != C * 3 * (2 * order + 1):
        return None
    dofs = x[:nc].reshape(C, 3 * (2 * order + 1))
    cur = np.asarray(result.currents, dtype=np.float64).ravel()
    info = getattr(result, "info", None) or {}
    js = dict(
        format="phrport-coils-v1", source=source,
        description="base coils (one row per coil) for simsopt: CurveXYZFourier(quadpoints, order).x "
                    "= base_curve_dofs[c]; Current(base_currents_A[c]); full set = "
                    "coils_via_symmetries(curves, currents, nfp, stellsym)",
        curve_type="CurveXYZFourier", order=order, quadpoints=nq, n_base_coils=C, nfp=nfp,
        stellsym=True, base_curve_dofs=dofs.tolist(), base_currents_A=cur.tolist(),
        total_current_A=float(np.sum(cur)), solver_x=x.tolist(),
        c0_local=float(result.c0_local), ok=bool(result.ok),
        certified=bool(getattr(result, "certified", False)),
        linked=bool(getattr(result, "linked", False)),
        certified_violation=_to_jsonable(_info_lookup(info, "v")),
        slot=_to_jsonable(_info_lookup(info, "slot")),
        ordering_axis=_to_jsonable(coil_solver.axis_for(boundary)) if hasattr(coil_solver, "axis_for") else None,
        boundary={k: float(v) for k, v in boundary.items()},
        coil_config=_to_jsonable(asdict(cfg)) if hasattr(cfg, "__dataclass_fields__") else None,
        loader=_COIL_LOADER.replace("PATH", repr(os.path.basename(path_stem) + ".json")))
    if extra:
        js.update(_to_jsonable(extra))
    _json_dump(_to_jsonable(js), path_stem + ".json")
    names = np.array(list(boundary.keys()))
    vals = np.array([float(v) for v in boundary.values()])
    tmp = path_stem + ".tmp.npz"
    np.savez(tmp, base_curve_dofs=dofs, base_currents_A=cur, solver_x=x, order=order, quadpoints=nq,
             n_base_coils=C, nfp=nfp, stellsym=True, boundary_names=names, boundary_values=vals,
             c0_local=float(result.c0_local))
    os.replace(tmp, path_stem + ".npz")
    return path_stem + ".json"


# ---------------------------------------------------------------------------
# Phase 0 (G5): provenance
# ---------------------------------------------------------------------------
_PHRPORT_DIR = os.path.dirname(os.path.abspath(__file__))
_SANDBOX_ROOT = os.path.dirname(_PHRPORT_DIR)
_PROV_ENV_EXACT = ("XLA_FLAGS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                   "VECLIB_MAXIMUM_THREADS", "CUDA_VISIBLE_DEVICES", "PYTHONPYCACHEPREFIX")
_PROV_PACKAGES = ("numpy", "scipy", "jax", "jaxlib", "vmex", "netCDF4", "equinox", "solvax",
                  "booz_xform_jax", "h5py")


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _sha256_tree(root):
    """Hash of a directory tree: sorted relative paths + contents (__pycache__, *.pyc and
    .DS_Store excluded)."""
    h = hashlib.sha256()
    n = 0
    for dp, dns, fns in os.walk(root):
        dns[:] = sorted(d for d in dns if d != "__pycache__")
        for fn in sorted(fns):
            if fn.endswith(".pyc") or fn == ".DS_Store":
                continue
            p = os.path.join(dp, fn)
            rel = os.path.relpath(p, root).replace(os.sep, "/")
            h.update(rel.encode() + b"\0" + _sha256_file(p).encode() + b"\n")
            n += 1
    return h.hexdigest(), n


def _git_head(repo):
    """HEAD commit of a git checkout, read from .git without running git (nothing is written)."""
    gd = os.path.join(repo, ".git")
    if not os.path.isdir(gd):
        return None
    with open(os.path.join(gd, "HEAD")) as f:
        head = f.read().strip()
    if not head.startswith("ref:"):
        return dict(head=head, ref=None)
    ref = head.split(None, 1)[1].strip()
    p = os.path.join(gd, ref)
    if os.path.exists(p):
        with open(p) as f:
            return dict(head=f.read().strip(), ref=ref)
    pk = os.path.join(gd, "packed-refs")
    if os.path.exists(pk):
        with open(pk) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2 and parts[1] == ref:
                    return dict(head=parts[0], ref=ref)
    return dict(head=None, ref=ref)


def collect_provenance(*, argv=None, cfg=None, coil_solver=None, input_deck=None, extra_files=None):
    """Run provenance (roadmap G5): package versions, sha256 of phrport/*.py, of the vendored
    XCoil tree and of the input deck, the VMEX git HEAD, jax devices (only when jax is already
    imported), PHRPORT_* / JAX_* / XLA_FLAGS (and thread) environment variables, argv, host.
    Every part is best effort (a failing part is recorded as an error string)."""
    from importlib import metadata

    def safe(fn):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            return f"error: {type(e).__name__}: {e}"

    def pkgs():
        out = {}
        for p in _PROV_PACKAGES:
            try:
                out[p] = metadata.version(p)
            except Exception:  # noqa: BLE001
                out[p] = None
        return out

    def phr_files():
        return {fn: _sha256_file(os.path.join(_PHRPORT_DIR, fn))
                for fn in sorted(os.listdir(_PHRPORT_DIR)) if fn.endswith(".py")}

    def xcoil_info():
        mod = _sys.modules.get("xcoil")
        if mod is None:
            return None
        fc = os.path.dirname(os.path.abspath(mod.__file__))
        tree, nf = _sha256_tree(fc)
        out = dict(path=fc, tree_sha256=tree, n_files=nf, backend=None, version=None, direct_url=None)
        try:
            from phrport.backend import select_xcoil
            out["backend"] = select_xcoil().describe()
        except Exception:  # noqa: BLE001
            pass
        try:
            from importlib import metadata
            d = metadata.distribution("xcoil")
            out["version"] = d.version
            out["direct_url"] = d.read_text("direct_url.json")   # pip's record of the git URL and commit
        except Exception:  # noqa: BLE001
            pass
        return out

    def vmex_git():
        mod = _sys.modules.get("vmex")
        cands = []
        if mod is not None and getattr(mod, "__file__", None):
            cands.append(os.path.dirname(os.path.dirname(os.path.abspath(mod.__file__))))
        cands.append(os.path.join(_SANDBOX_ROOT, "external", "VMEX"))
        for c in cands:
            hd = _git_head(c)
            if hd is not None:
                return dict(repo=c, **hd)
        return None

    def jax_info():
        jx = _sys.modules.get("jax")
        if jx is None:
            return "jax not imported by this process"
        x64 = getattr(jx.config, "jax_enable_x64", None)
        return dict(devices=[str(d) for d in jx.devices()], default_backend=jx.default_backend(),
                    x64=(None if x64 is None else bool(x64)))

    def deck():
        if not input_deck or not os.path.isfile(str(input_deck)):
            return dict(path=input_deck, sha256=None)
        return dict(path=os.path.abspath(str(input_deck)), sha256=_sha256_file(str(input_deck)))

    def files():
        out = {}
        for k, p in (extra_files or {}).items():
            out[k] = (dict(path=os.path.abspath(p), sha256=_sha256_file(p))
                      if p and os.path.isfile(p) else dict(path=p, sha256=None))
        return out

    env = {k: v for k, v in sorted(os.environ.items())
           if k.startswith("PHRPORT_") or k.startswith("JAX_") or k in _PROV_ENV_EXACT}
    ccfg = getattr(coil_solver, "cfg", None)
    return dict(
        schema="phrport-provenance-v1",
        created=datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        argv=list(argv if argv is not None else _sys.argv), cwd=os.getcwd(),
        host=safe(socket.gethostname), platform=safe(platform.platform),
        python=_sys.version, executable=_sys.executable, pid=os.getpid(),
        packages=safe(pkgs), phrport_sha256=safe(phr_files), xcoil=safe(xcoil_info),
        input_deck=safe(deck), files=safe(files), vmex_git=safe(vmex_git), jax=safe(jax_info),
        env=env, config=(_to_jsonable(asdict(cfg)) if cfg is not None else None),
        coil_config=(_to_jsonable(asdict(ccfg)) if hasattr(ccfg, "__dataclass_fields__") else None))


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------
def _json_dump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _to_jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, dict):
        return {str(k): _to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_jsonable(v) for v in o]
    return o


def _info_lookup(info, key):
    cur = info
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur or cur[part] is None:
            return None
        cur = cur[part]
    return cur


def _info_get(res, keys):
    info = getattr(res, "info", None) or {}
    for k in keys:
        v = _info_lookup(info, k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return None


# ---------------------------------------------------------------------------
# Roadmap phase 2, B2: the anchored coil oracle (opt-in, --coil-oracle anchored)
# ---------------------------------------------------------------------------
def surface_distance(b1, b2):
    """max |b1[k] - b2[k]| over the union of the boundary coefficient names (a missing name counts
    as 0). Within a continuation step (same fixed DOFs) this is max |x - x'| of the free DOFs; across
    steps it is the distance of the two surfaces."""
    keys = set(b1) | set(b2)
    if not keys:
        return 0.0
    return float(max(abs(float(b1.get(k, 0.0)) - float(b2.get(k, 0.0))) for k in keys))


def coil_lagrangian(res):
    """The coil subproblem's Lagrangian value c0 + c0_ref lam.g at its NNLS multipliers
    (info['c0_lagr'] of phrport.coils); c0_local when the solver does not report it."""
    v = _info_get(res, ("c0_lagr",))
    return float(res.c0_local) if v is None else float(v)


def _finite_ok(res):
    return (res is not None and bool(getattr(res, "ok", False))
            and bool(np.isfinite(float(getattr(res, "c0_local", np.nan)))))


def _safe_pred(fn):
    try:
        return bool(fn())
    except Exception:  # noqa: BLE001  (S-a contract: a failing predicate never aborts)
        return False


def _linear_coil_result(a_res, dc0, anchor_seq=None):
    """B2 noise floor, rule 'linear': the anchor's coil result moved to first order, c0 = c0_anchor +
    G.dx (and the same increment on its Lagrangian; G = the anchor's envelope gradient, the TRF
    model's coil gradient).  The coils, multipliers and kkt are the anchor's; the result is marked
    uncertified (certified / info['certify'] False) since no solve ran on this surface."""
    r = copy.copy(a_res)
    ai = getattr(a_res, "info", None) or {}
    c0a = float(a_res.c0_local)
    la = coil_lagrangian(a_res)
    r.c0_local = c0a + float(dc0)
    info = dict(ai)
    info.update(mode="linear", c0_lagr=la + float(dc0), certify=False, attempts=[],
                linear=dict(c0_anchor=c0a, lagr_anchor=la, dc0=float(dc0), anchor_seq=anchor_seq,
                            anchor_mode=ai.get("mode")))
    r.info = info
    if hasattr(r, "certified"):
        r.certified = False
    return r


def anchored_coil_solve(coil_solver, boundary, anchor, *, d=None, d_inherit=1e-3, d_reset=1e-2, k_max=2,
                        kkt_factor=3.0, kkt_floor=5e-3, certify=None, should_abort=None, supports=None,
                        reset_guard=False, d_floor=0.0, floor_rule="linear", floor_k_max=15, linear=None):
    """Roadmap B2: one coil evaluation of the anchored oracle on `boundary`.

    anchor: None, or dict(boundary=<full boundary dict>, res=<finite, ok CoilResult>, seq=...) -- the
        coils at TRF's accepted point.  d = surface_distance(boundary, anchor['boundary']) unless given.
      d <= d_inherit: warm from the anchor with its own (lam_al, rho), at most k_max outers, without
          certification and without the cold fallback (solve(..., certify=False, rho_reset=False,
          warm_k_max=k_max, warm_fallback=False)); accepted iff c0 is finite, soft ok and the NNLS
          stationarity info['kkt'] <= max(kkt_factor * kkt_anchor, kkt_floor) (a missing kkt passes);
          otherwise (guard failed) the rho-reset solve below;
      d <= d_reset (or the guard failed): warm from the anchor with rho = rho0 and lam_al kept
          (solve(..., rho_reset=True)); the coil solver falls back to its cold starts when that result
          is not ok;
      d > d_reset, or no anchor: cold (solve(boundary, warm=None)).
    certify / should_abort (phase 0, S-b / S-a) are passed to the rho-reset and cold solves;
    should_abort is also checked before the rho-reset retry after a failed guard (a fallback start
    as well): an abort returns the guard-failed result with ok=False and info['aborted'] = True.
    supports(name) -> bool says which keyword arguments coil_solver.solve accepts (others are
    dropped; None = all).
    reset_guard (here default off = the roadmap's literal rule; the runner passes PHRConfig.oracle_reset_guard,
    default on): a rho-reset warm result must also have met
    its stop test (info['stopped'], i.e. not run out of outers) and pass the same kkt threshold,
    otherwise a cold solve follows (stage reset_guard_cold; should_abort is checked first).
    Noise floor (opt-in, d_floor > 0; default 0 = off): for 0 <= d <= min(d_floor, d_inherit),
      floor_rule 'linear': no solve; linear() -> G.dx (the anchor's first-order c0 change) and the
          result is _linear_coil_result(anchor, G.dx) (stage 'linear'); when linear is None, raises
          or is not finite, the inherited solve below runs (decision floor.fallback);
      floor_rule 'converge': the inherited warm solve gets floor_k_max outers instead of k_max (its
          stop test decides; stage 'inherit_conv' when the guard passes, otherwise as above).
      Reason: k_max = 2 inherited solves carry ~3e-4 relative c0 noise at d ~ 1e-6 (offline gate
      err_lin), larger than the f-decrease TRF predicts for such steps -> tiny-step rejections.
    Returns (result, decision) with decision = dict(plan, stage, d, anchor_seq, guard[, reset_guard,
    floor]); stage in linear | inherit | inherit_conv | reset | reset_cold | reset_guard_cold | cold |
    aborted (reset_cold: the rho-reset warm result was not ok and the solver's cold fallback answered)."""
    sup = supports if supports is not None else (lambda name: True)

    def call(warm, **kw):
        kw = {k: v for k, v in kw.items() if v is not None and sup(k)}
        return coil_solver.solve(boundary, warm=warm, **kw)

    def mode_of(r):
        return (getattr(r, "info", None) or {}).get("mode")

    dec = dict(plan=None, stage=None, d=None, anchor_seq=None, guard=None)
    a_res = None
    thr = float(kkt_floor)
    if anchor is not None and anchor.get("res") is not None:
        a_res = anchor["res"]
        kkt_a = (getattr(a_res, "info", None) or {}).get("kkt")
        kkt_a = None if kkt_a is None else float(kkt_a)
        if kkt_a is not None and np.isfinite(kkt_a):
            thr = max(float(kkt_factor) * kkt_a, float(kkt_floor))
        if d is None:
            d = surface_distance(boundary, anchor["boundary"])
        d = float(d)
        dec.update(d=d, anchor_seq=anchor.get("seq"))
        plan = "inherit" if d <= d_inherit else ("reset" if d <= d_reset else "cold")
    else:
        plan = "cold"
    dec["plan"] = plan
    res, stage = None, None
    km, conv = int(k_max), False
    if plan == "inherit" and float(d_floor) > 0.0 and d <= float(d_floor):
        fl = dict(rule=str(floor_rule), d_floor=float(d_floor))
        dec["floor"] = fl
        if floor_rule == "linear":
            dc0, err = None, None
            if linear is not None:
                try:
                    dc0 = linear()
                except Exception as e:  # noqa: BLE001  (no prediction -> the inherited solve)
                    err = f"{type(e).__name__}: {e}"
            if dc0 is not None and np.isfinite(float(dc0)) and np.isfinite(float(a_res.c0_local)):
                res = _linear_coil_result(a_res, float(dc0), anchor_seq=dec["anchor_seq"])
                fl.update(dc0=float(dc0), c0=float(res.c0_local))
                dec["stage"] = "linear"
                return res, dec
            fl.update(fallback="inherit", error=err)
        elif floor_rule == "converge":
            km, conv = int(floor_k_max), True
            fl["k_max"] = km
        else:
            raise ValueError(f"floor_rule must be 'linear' or 'converge', got {floor_rule!r}")
    if plan == "inherit":
        r = call(a_res, certify=False, rho_reset=False, warm_k_max=km, warm_fallback=False)
        ri = getattr(r, "info", None) or {}
        kkt = None if ri.get("kkt") is None else float(ri["kkt"])
        soft = bool(ri.get("soft_ok", getattr(r, "ok", False)))
        c0 = float(getattr(r, "c0_local", np.nan))
        kkt_ok = kkt is None or bool(np.isfinite(kkt) and kkt <= thr)
        g_ok = bool(np.isfinite(c0) and soft and kkt_ok)
        dec["guard"] = dict(ok=g_ok, kkt=kkt, kkt_anchor=kkt_a, thr=thr, soft_ok=soft, c0=c0,
                            outer=ri.get("outer"), mode=mode_of(r))
        if g_ok:
            res, stage = r, (("inherit_conv" if conv else "inherit") if mode_of(r) != "cold" else "inherit_cold")
        else:
            if should_abort is not None and _safe_pred(should_abort):
                r.ok = False
                r.info = dict(ri, aborted=True)
                dec["stage"] = "aborted"
                return r, dec
            plan = "reset"
    if res is None and plan == "reset":
        res = call(a_res, certify=certify, should_abort=should_abort, rho_reset=True)
        if (getattr(res, "info", None) or {}).get("aborted"):
            stage = "aborted"
        else:
            stage = "reset_cold" if mode_of(res) == "cold" else "reset"
        if reset_guard and stage == "reset":
            ri = getattr(res, "info", None) or {}
            kkt = None if ri.get("kkt") is None else float(ri["kkt"])
            c0 = float(getattr(res, "c0_local", np.nan))
            stopped = bool(ri.get("stopped", True))
            rg_ok = bool(np.isfinite(c0) and bool(getattr(res, "ok", False)) and stopped
                         and (kkt is None or (np.isfinite(kkt) and kkt <= thr)))
            dec["reset_guard"] = dict(ok=rg_ok, kkt=kkt, thr=thr, stopped=stopped, outer=ri.get("outer"), c0=c0)
            if not rg_ok:
                if should_abort is not None and _safe_pred(should_abort):
                    res.ok = False
                    res.info = dict(ri, aborted=True)
                    dec["stage"] = "aborted"
                    return res, dec
                res = call(None, certify=certify, should_abort=should_abort)
                stage = "aborted" if (getattr(res, "info", None) or {}).get("aborted") else "reset_guard_cold"
    if res is None:
        res = call(None, certify=certify, should_abort=should_abort)
        stage = "aborted" if (getattr(res, "info", None) or {}).get("aborted") else "cold"
    dec["stage"] = stage
    return res, dec


# ---------------------------------------------------------------------------
# Roadmap A4: front sweep (PHRRunner.sweep)
# ---------------------------------------------------------------------------
@dataclass
class SweepConfig:
    """A4 front sweep (PHRRunner.sweep; CLI --sweep --resume-from SEED).  Point k has the absolute
    flux target eps_k = ratio^k Jf0 (raw c0 units; the scaled target is flux_metric_scale eps_k)."""
    ratio: float = 0.85             # eps_k = ratio^k Jf0
    max_points: int = 10            # points run by this sweep (the seed point not counted)
    min_target: float = 0.0         # stop before a point whose eps_k < this (raw)
    max_qs: float = 0.0             # stop after a point whose QS exceeds this (0 = off)
    outers: int = 4                 # outer budget per point (B1 KKT exit; no extra outers)
    jf0: float = None               # raw Jf0 (None: a front-point seed's Jf0, else a cold coil solve
                                    # at the seed surface)
    k0: int = None                  # first k (None: 1, or the next point after a sweep checkpoint)
    rho_balance: float = 1.0        # rho_c at the start of a point: J2 = rho_balance * QS (sweep_rho_c)
    infeasible_stop: int = 2        # stop after this many consecutive points missing their target
    infeasible_tol: float = None    # target missed iff scaled g_coil > this (None: cfg.v_tol)
    seed_point: bool = False        # also write front_point_seed.json (seed surface, no outers)
    polish: bool = True             # point coils from CoilSolver.final_coil_stage (False: cold solve)


def sweep_rho_c(qs, g, lam, *, rho_carried, rho_min, rho_max, balance=1.0):
    """A4: the coil penalty for a new sweep target.  With the scaled violation g = S (c0 - eps) at the
    start point and the carried multiplier lam, PHR's J2(rho) = 0.5 rho max(0, g + lam/rho)^2 is set
    equal to J = balance * QS:

        g > 0, lam = 0:      rho = 2 J / g^2
        g > 0, J > 2 lam g:  rho = u^2, u = (sqrt(2J) + sqrt(2J - 4 g lam)) / (2 g)   (J2 increasing in rho)
        g > 0, J <= 2 lam g: rho = lam / g  (min_rho J2 = 2 lam g >= J: the multiplier alone outweighs QS)
        g <= 0 / no data:    the carried rho (the target is already met)

    clipped to [rho_min, rho_max].  -> (rho, rule)."""
    rho_carried = float(rho_carried)
    J = float(balance) * float(qs) if qs is not None else float("nan")
    g = float("nan") if g is None else float(g)
    lam = max(0.0, float(lam))
    if not (np.isfinite(J) and J > 0.0 and np.isfinite(g) and g > 0.0):
        rho, rule = rho_carried, ("carried (target already met)" if np.isfinite(g) and g <= 0.0
                                  else "carried (no QS / coil value)")
    elif lam == 0.0:
        rho, rule = 2.0 * J / g ** 2, "J2 = balance QS (lam_c = 0)"
    elif J > 2.0 * lam * g:
        u = (math.sqrt(2.0 * J) + math.sqrt(2.0 * J - 4.0 * g * lam)) / (2.0 * g)
        rho, rule = u * u, "J2 = balance QS"
    else:
        rho, rule = lam / g, "lam_c-dominated (2 lam_c g >= balance QS): rho = lam_c / g minimises J2"
    lo, hi = float(rho_min), float(rho_max)
    if rho >= hi:
        rho, rule = hi, rule + "; capped at rho_max"
    elif rho < lo:
        rho, rule = lo, rule + "; raised to rho_min"
    return float(rho), rule


def _sweep_meta(ck):
    """Sweep fields of a checkpoint (A4): {} for a plain PHR checkpoint."""
    if "sweep_k" not in ck:
        return {}
    return dict(k=int(ck["sweep_k"]), jf0=ck.get("sweep_jf0"), ratio=ck.get("sweep_ratio"),
                eps=ck.get("sweep_eps"), done=bool(ck.get("sweep_point_done", False)))


class _CoilCtx:
    """Per-outer-iteration closure state of compute_J2_and_grad (_rho_c, _lam_c, _targets_c, ...)."""

    def __init__(self, rho_c, lam_c, targets_c, al_iter, max_mode, step, names):
        self.rho = np.array(rho_c, dtype=np.float64, copy=True)
        self.lam = np.array(lam_c, dtype=np.float64, copy=True)
        self.targets = np.array(targets_c, dtype=np.float64, copy=True)
        self.al_iter, self.max_mode, self.step = int(al_iter), int(max_mode), int(step)
        self.names = list(names)


class _LoopState:
    """Mutable state of the PHR outer loop shared by PHRRunner.run / sweep and _outer_iteration (A4)."""
    __slots__ = ("x", "boundary", "lam_eq", "rho_eq", "lam_coil", "rho_coil", "v_prev_eq", "v_prev_coil",
                 "last_good", "last_good_step", "aborted", "fin_state")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.pop(k))
        if kw:
            raise TypeError(f"unknown loop-state fields {sorted(kw)}")


class _PhaseProf:
    """P1 (opt-in, PHRConfig.phase_timers): per f-eval and per Jacobian phase timers (passive).

    Every f-eval / Jacobian gets one line in the log ('[phase] ...') and one record in
    <run-dir>/phase_timers.jsonl.  The split comes from timers that already exist (runner timers, the
    VMEX worker's t_solve / t_metrics / t_state / t_worker, the exact-Jacobian report, the finite-beta
    coupler stats) plus wall clocks around each call; every one of them ends at a host value
    (numpy result / device_get), so no extra block_until_ready is needed and nothing evaluated changes.
    The f-eval VMEX solve runs in the worker process while the main process solves the coils (overlap),
    so 'other' is the wall time not covered by max(VMEX round trip, coil solve + envelope)."""

    def __init__(self, runner):
        self.r = runner
        self.path = os.path.join(runner.out_dir, "phase_timers.jsonl")
        self.where = {}
        self._s0 = None
        self._j0 = None

    def _snap(self):
        r = self.r
        fb = dict(r._fb.stats) if r._fb is not None else {}
        return dict(t=time.time(), coil=float(r.timers["coil_solve"]), env=float(r.timers["envelope"]),
                    eq_many=float(r.timers["eq_many"]), eq_single=float(r.timers["eq_single"]),
                    fb_ctx=float(fb.get("t_context", 0.0)), fb_field=float(fb.get("t_field_grad", 0.0)))

    def _emit(self, rec):
        rec.update(self.where)
        try:
            with open(self.path, "a") as fh:
                fh.write(json.dumps(_to_jsonable(rec)) + "\n")
        except Exception:  # noqa: BLE001
            pass

    def hook(self, kind, info):
        try:
            self._hook(kind, info)
        except Exception as e:  # noqa: BLE001  (timing never stops a run)
            self.r._warn_once("phase_prof", f"  [WARNING] phase timers failed: {type(e).__name__}: {e}")

    def _hook(self, kind, info):
        r = self.r
        if kind == "f_begin":
            r._prof_eq = None
            self._s0 = self._snap()
        elif kind == "f_end":
            s0, s1 = self._s0, self._snap()
            d = {k: s1[k] - s0[k] for k in s1}
            eqi, rt, hit, conv = r._prof_eq if r._prof_eq is not None else ({}, None, None, None)
            if hit:            # cache hit: the solve's own timers belong to an earlier wall interval
                eqi = dict(iterations=eqi.get("iterations"), fsqr=eqi.get("fsqr"), fsqz=eqi.get("fsqz"),
                           fsql=eqi.get("fsql"))
            wall = float(info.get("wall", d["t"]))
            vm = rt if rt is not None else 0.0
            path = max(vm, d["coil"] + d["env"])
            rec = dict(kind="f", nevals=int(info.get("nevals", 0)), wall=wall, vmex_roundtrip=rt,
                       eq_hit=hit, t_worker=eqi.get("t_worker"), t_solve=eqi.get("t_solve"),
                       iterations=eqi.get("iterations"), ier_flag=eqi.get("ier_flag"),
                       fsqr=eqi.get("fsqr"), fsqz=eqi.get("fsqz"), fsql=eqi.get("fsql"),
                       t_metrics=eqi.get("t_metrics"), t_state=eqi.get("t_state"),
                       ipc=(None if (rt is None or eqi.get("t_worker") is None) else rt - float(eqi["t_worker"])),
                       coil=d["coil"], envelope=d["env"],
                       other=wall - path, objective=info.get("objective"),
                       skipped=bool(info.get("skipped")), aborted=bool(info.get("aborted")),
                       converged=conv)
            self._emit(rec)
            f = lambda v: "-" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))  # noqa: E731
            r.log(f"  [phase] f{rec['nevals']} wall {wall:.2f}s | vmex rt {f(rt)} (worker {f(rec['t_worker'])}: "
                  f"solve {f(rec['t_solve'])} it {f(rec['iterations'])} conv {f(conv)}"
                  f" fsq {('-' if rec['fsqr'] is None else '%.1e/%.1e/%.1e' % (rec['fsqr'], rec['fsqz'], rec['fsql']))}"
                  f" metrics {f(rec['t_metrics'])} state {f(rec['t_state'])}; ipc {f(rec['ipc'])}"
                  f"{'; cache hit' if hit else ''}) | coil {d['coil']:.2f} env {d['env']:.2f}"
                  f" | other {rec['other']:.2f}")
        elif kind == "jac_begin":
            self._j0 = self._snap()
        elif kind == "jac_end":
            s0, s1 = self._j0, self._snap()
            d = {k: s1[k] - s0[k] for k in s1}
            ex = getattr(r._exact_jac, "last", None) if r._exact_jac is not None else None
            rep = dict(getattr(ex, "report", None) or {}) if ex is not None else {}
            wall = float(info.get("wall", d["t"]))
            acc = sum(float(info.get(k) or 0.0) for k in ("t_eq_jac", "t_fd", "t_hook", "t_jac_callback"))
            rec = dict(kind="jac", nevals=int(info.get("nevals", 0)), wall=wall, t_eq_jac=info.get("t_eq_jac"),
                       t_fd=info.get("t_fd"), ej_state_load=rep.get("t_state_load"),
                       ej_metrics_numpy=rep.get("t_metrics_numpy"), ej_tangent_push=rep.get("t_tangent_push"),
                       ej_elongation=rep.get("t_elongation"), ej_solve=rep.get("t_solve"),
                       t_hook=info.get("t_hook"), fb_context=d["fb_ctx"], fb_field_term=d["fb_field"],
                       hook_coil_resolve=d["coil"], hook_envelope=d["env"], t_jac_callback=info.get("t_jac_callback"),
                       other=wall - acc)
            self._emit(rec)
            f = lambda v: "-" if v is None else f"{float(v):.2f}"  # noqa: E731
            r.log(f"  [phase] jac@f{rec['nevals']} wall {wall:.2f}s | eq-jac {f(rec['t_eq_jac'])} (state-load "
                  f"{f(rec['ej_state_load'])} metrics {f(rec['ej_metrics_numpy'])} tangents {f(rec['ej_tangent_push'])}"
                  f" elong {f(rec['ej_elongation'])}) fd {f(rec['t_fd'])} | fb-hook {f(rec['t_hook'])} (context "
                  f"{d['fb_ctx']:.2f} coil {d['coil']:.2f} env {d['env']:.2f} field {d['fb_field']:.2f}) | "
                  f"callback {f(rec['t_jac_callback'])} | other {rec['other']:.2f}")


class PHRRunner:
    """Continuation + PHR AL1 loop + final coil stage (proc0 view of the PHR script).

    evaluate(boundary, step) -> EqResult; evaluate_many(items) -> list[EqResult];
    coil_solver: .solve(boundary, warm) / .envelope_grad(boundary, result, names);
    free_dof_names(step) -> list[str]; seed_boundary: dict (PHR: input + rotating-ellipse seed).
    """

    def __init__(self, cfg, *, evaluate, evaluate_many, coil_solver, free_dof_names,
                 seed_boundary, out_dir, vmec_input="input.nfp2_QA", nfp=2, stellsym=True,
                 log_to_stdout=True, resume_from=None, init_surface=None,
                 write_wout=None, keep_jacobians=False, argv=None, input_deck=None):
        if cfg.coil_oracle not in ("phr", "anchored"):
            raise ValueError(f"coil_oracle must be 'phr' or 'anchored', got {cfg.coil_oracle!r}")
        self.cfg = cfg
        self.evaluate = evaluate
        self.evaluate_many = evaluate_many
        self.coil_solver = coil_solver
        self.free_dof_names = free_dof_names
        self.seed = collections.OrderedDict((k, float(v)) for k, v in seed_boundary.items())
        self.out_dir = os.path.abspath(out_dir)
        os.makedirs(self.out_dir, exist_ok=True)
        self.vmec_input = str(vmec_input)
        self.nfp, self.stellsym = int(nfp), bool(stellsym)
        self.log_to_stdout = log_to_stdout
        self.resume_from, self.init_surface = resume_from, init_surface
        self.write_wout = write_wout
        self.keep_jacobians = keep_jacobians
        self._logf = open(os.path.join(self.out_dir, "phr_port.log"), "a")
        self._executor = ThreadPoolExecutor(max_workers=1) if cfg.overlap_eq_and_coils else None
        self.eq_labels = eq_constraint_labels(cfg)
        self.n_eq_con = len(self.eq_labels)
        self.coil_names = ["Jf"]
        self.n_coil_con = 1
        # histories (PHR format)
        self.J1_J2_history = []
        self.phr_history = []
        self.outer_log = []          # port extras: one dict per outer iteration
        self.eval_log_path = os.path.join(self.out_dir, "eval_log.jsonl")
        # PHRPORT_FRAMES=1: frames.jsonl, one line per f-eval / accepted point / oracle audit with the full
        # boundary and the coil state (solver_x, currents, c0, B.n metrics) for iteration-by-iteration movies
        self._frames_path = (os.path.join(self.out_dir, "frames.jsonl")
                             if _os.environ.get("PHRPORT_FRAMES", "0") not in ("", "0") else None)
        self._frame_coil = None
        self._frame_ids = collections.OrderedDict()   # x bytes -> eval_id of the latest f-eval there
        # coil-side state (bs persistence, alm_warmstart_state, _last_* holders)
        self._coil_res = None
        self._warm_prev_x = None
        self._last_coil_vals = np.full(self.n_coil_con, np.nan)
        self._last_coil_x = None
        self._j2_holder = [None]
        self._flux_captured = [False]
        self._coil_scales = np.array([cfg.flux_metric_scale], dtype=np.float64)
        self._coil_targets = np.array([cfg.flux_phr_target * cfg.flux_metric_scale],
                                      dtype=np.float64)
        self._eq_cache = collections.OrderedDict()
        self._sub = None
        self._base = None       # full boundary dict at the current step (fixed DOFs)
        self._names = None
        self._step = None
        self._last_eval_info = {}
        self.counters = collections.Counter()
        self.timers = collections.Counter()
        self._warned = set()
        self._exact_jac = None      # phrport.exact_jacobian.PHRExactJacobian (cfg.eq_jacobian='exact')
        self._fb = None             # phrport.finite_beta.FiniteBetaCoupler (cfg.finite_beta)
        # opt-in speedups: coil solves skipped by skip_coil_on_rejected whose (unknown) effect on
        # the coil warm-start state has not been resolved yet; see _resolve_pending_skips
        self._pending_skips = []
        self._last_solve_cold = None    # was the last successful coil solve a cold one?
        self._trust_carry = None        # carry_trust_radius: largest accepted step of last LS
        self._maxj_g0 = None            # closure ramp: max(0, maxj) where the row switched on
        self._maxj_k = 0                # closure ramp: outers since the row switched on
        # a resume continues with the interrupted run's carried trust radius (env PHRPORT_TRUST_CARRY0: its last
        # '[speedups] ... max_accepted_step')
        if os.environ.get("PHRPORT_TRUST_CARRY0"):
            self._trust_carry = float(os.environ["PHRPORT_TRUST_CARRY0"])
        # ---- phase 0
        self.argv = argv
        self.input_deck = input_deck if input_deck is not None else (
            vmec_input if os.path.isfile(str(vmec_input)) else None)
        self._trial_certify = None if cfg.certify_trials else False     # S-b: trial f-evals
        self._kkt_on = bool(cfg.phase0_reporting or cfg.final_mode_kkt_exit)
        self._kkt_cg = collections.OrderedDict()   # B1: x.tobytes() -> scaled envelope grad
        self._exact_last_x = None       # B1: x of the last successful exact Jacobian
        self._eq_state_kw = None        # S-c: evaluate / evaluate_many accept return_state
        self.kkt_log = []
        self._eval_x = None             # G5: per-f-eval x sidecar (eval_x.npz)
        self._eval_x_path = os.path.join(self.out_dir, "eval_x.npz")
        self._eval_id_base = 0
        self._eval_x_new = 0
        if cfg.phase0_reporting:
            self._init_eval_x()
        # ---- phase 2 (B2 anchored coil oracle)
        self._anchored = bool(cfg.coil_oracle == "anchored" and cfg.enable_coils)
        self._coil_acc = None           # the anchor: coils at TRF's accepted point (see _set_anchor)
        self._anchor_seq = 0
        self._last_eval_id = None       # eval_id of the latest f-eval (anchor provenance)
        self._last_oracle_stage = None
        self._solve_kw_ok = {}
        # ---- 2026-11 speed-ups (opt-in; nothing below is used on the default path)
        self._prof = _PhaseProf(self) if cfg.phase_timers else None
        self._prof_eq = None                 # P1: (EqResult info, round-trip wall, cache hit, converged) of the last f-eval
        self._fb_ref = None                  # P5: dict(step, x, gf, n) at the last field-term refresh

    # ------------------------------------------------------------------ utils
    def log(self, msg=""):
        if self.log_to_stdout:
            print(msg, flush=True)
        self._logf.write(msg + "\n")
        self._logf.flush()

    def _warn_once(self, key, msg):
        if key not in self._warned:
            self._warned.add(key)
            self.log(msg)

    def close(self):
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        try:
            self._logf.close()
        except Exception:  # noqa: BLE001
            pass

    def _boundary_with(self, x):
        b = collections.OrderedDict(self._base)
        for k, v in zip(self._names, np.asarray(x, dtype=np.float64)):
            b[k] = float(v)
        return b

    def _xvec(self, boundary, names):
        return np.array([float(boundary[k]) for k in names], dtype=np.float64)

    def _cache_key(self, step, x):
        return (int(step), np.asarray(x, dtype=np.float64).tobytes())

    def _cache_put(self, step, x, eq):
        k = self._cache_key(step, x)
        self._eq_cache[k] = eq
        self._eq_cache.move_to_end(k)
        while len(self._eq_cache) > max(1, self.cfg.eq_cache_size):
            self._eq_cache.popitem(last=False)

    def _cache_get(self, step, x):
        return self._eq_cache.get(self._cache_key(step, x))

    def _eval_single(self, boundary, step):
        t0 = time.time()
        try:
            eq = self.evaluate(boundary, step)
        except Exception as e:  # noqa: BLE001  (interface says never raises; be safe)
            eq = _FailedEq(f"evaluate raised {type(e).__name__}: {e}")
        self.timers["eq_single"] += time.time() - t0
        self.counters["eq_single"] += 1
        return eq

    def _eval_many(self, boundaries, step, return_state=False):
        if not boundaries:
            return []
        t0 = time.time()
        try:
            if return_state:      # S-c (opt-in): only f-eval points carry the converged state
                res = list(self.evaluate_many([(b, step) for b in boundaries], return_state=True))
            else:
                res = list(self.evaluate_many([(b, step) for b in boundaries]))
        except Exception as e:  # noqa: BLE001
            self.log(f"  [WARNING] evaluate_many raised {type(e).__name__}: {e}; "
                     f"marking {len(boundaries)} points failed.")
            res = [_FailedEq(str(e))] * len(boundaries)
        self.timers["eq_many"] += time.time() - t0
        self.counters["eq_many_points"] += len(boundaries)
        return res

    # ------------------------------------------------- LS-driver callables
    @staticmethod
    def _accepts_kw(fn, name):
        try:
            ps = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            return False
        return name in ps or any(p.kind == p.VAR_KEYWORD for p in ps.values())

    def _accepts_return_state(self, fn):
        return self._accepts_kw(fn, "return_state")

    def _state_wanted(self):
        """S-c: request converged VMEX states at f-eval points (needs the eq cache and evaluators
        that accept return_state; otherwise disabled with a warning)."""
        if not self.cfg.reuse_vmex_state:
            return False
        if self._eq_state_kw is None:
            fn = self.evaluate_many if self.cfg.overlap_eq_and_coils else self.evaluate
            ok = bool(self.cfg.reuse_eq_cache) and self._accepts_return_state(fn)
            if not ok:
                self.log("  [WARNING] reuse_vmex_state needs reuse_eq_cache=True and an evaluator "
                         "accepting return_state; disabled.")
            self._eq_state_kw = ok
        return self._eq_state_kw

    def _f_eq(self, x):
        """Equilibrium at an f-eval point (cached for the dual update / FD base)."""
        step = self._step
        t_rt = time.time()
        eq = self._cache_get(step, x) if self.cfg.reuse_eq_cache else None
        hit = eq is not None
        if eq is not None:
            self.counters["eq_cache_hits"] += 1
        else:
            b = self._boundary_with(x)
            if self.cfg.reuse_vmex_state and self._state_wanted():
                if self.cfg.overlap_eq_and_coils:
                    eq = self._eval_many([b], step, return_state=True)[0]
                else:
                    t0 = time.time()
                    try:
                        eq = self.evaluate(b, step, return_state=True)
                    except Exception as e:  # noqa: BLE001
                        eq = _FailedEq(f"evaluate raised {type(e).__name__}: {e}")
                    self.timers["eq_single"] += time.time() - t0
                    self.counters["eq_single"] += 1
            elif self.cfg.overlap_eq_and_coils:
                eq = self._eval_many([b], step)[0]
            else:
                eq = self._eval_single(b, step)
            if self.cfg.reuse_eq_cache:
                self._cache_put(step, x, eq)
        if self._prof is not None:          # P1 (opt-in): (EqResult info, round-trip wall, cache hit, converged)
            self._prof_eq = (dict(getattr(eq, "info", None) or {}), time.time() - t_rt, hit, bool(eq.converged))
        self._last_eval_info["eq_converged"] = bool(eq.converged)
        if self.cfg.phase0_reporting:
            self._last_eval_info["qs_total"] = (float(eq.qs_total) if bool(eq.converged) else None)
        return eq

    def _residuals_fn(self, x):
        return self._sub.residuals(self._f_eq(np.asarray(x, dtype=np.float64)), group=0)

    def _residuals_prefetch(self, x):
        x = np.array(x, dtype=np.float64, copy=True)
        fut = self._executor.submit(self._f_eq, x)

        def thunk():
            return self._sub.residuals(fut.result(), group=0)

        def peek():
            """S-a: ('pending', None) | ('raise', None) | ('ok', r_J1); no side effects, no wait."""
            if not fut.done():
                return "pending", None
            if fut.exception() is not None:
                return "raise", None
            r = self._sub.peek_residuals(fut.result(), group=0)
            return ("raise", None) if r is None else ("ok", r)
        thunk.peek = peek
        return thunk

    def _residuals_many(self, xs, groups):
        step = self._step
        eqs = [None] * len(xs)
        todo, idx = [], []
        for i, x in enumerate(xs):
            eq = self._cache_get(step, x) if self.cfg.reuse_eq_cache else None
            if eq is not None:
                eqs[i] = eq
                self.counters["eq_cache_hits"] += 1
            else:
                todo.append(self._boundary_with(x)); idx.append(i)
        for i, eq in zip(idx, self._eval_many(todo, step)):
            eqs[i] = eq
            if self.cfg.reuse_eq_cache:
                self._cache_put(step, xs[i], eq)
        out = []
        for eq, g in zip(eqs, groups):       # column order == per-group leader order
            try:
                out.append(self._sub.residuals(eq, group=g))
            except EquilibriumFailure as e:
                out.append(e)
        self.counters["fd_failed_points"] += sum(1 for eq in eqs if eq is None or not bool(eq.converged))
        return out

    # ------------------------------------------- compute_J2_and_grad (port)
    def _phantom_inputs(self, boundary, res):
        min_cs = _info_get(res, _MIN_CS_KEYS)
        honest = _info_get(res, _HONEST_KEYS)
        if (min_cs is None or honest is None) and hasattr(self.coil_solver, "phantom_metrics"):
            try:
                pm = self.coil_solver.phantom_metrics(boundary, res)
                min_cs = float(pm[0]) if min_cs is None and pm[0] is not None else min_cs
                honest = float(pm[1]) if honest is None and pm[1] is not None else honest
            except Exception as e:  # noqa: BLE001
                self._warn_once("phantom_metrics", f"  [WARNING] phantom_metrics failed: {e}")
        return min_cs, honest

    # ---------------- skip_coil_on_rejected: lazy reconstruction of the coil warm-start state
    # A skipped coil solve at a rejected trial point x_s would (PHR) have set _warm_prev_x = x_s,
    # _coil_res, _last_coil_x = x_s, _last_coil_vals -- unless it failed, which cannot be known
    # without running it.  With reinit_coils_each_alm (required by the skip gate) a cold solve
    # does not depend on that state, so the next real solve at x is unaffected iff it is cold for
    # EVERY possible predecessor (the real _warm_prev_x or any pending x_s).  Otherwise the
    # pending solves are run ("materialised", metrics only) in order, which reproduces PHR's state
    # exactly.  A successful real solve overwrites the whole state and clears the pending list.
    def _skip_gate(self):
        cfg = self.cfg
        return bool(cfg.reinit_coils_each_alm and
                    (cfg.flux_target_frac <= 0.0 or self._flux_captured[0]))

    def _on_coil_skipped(self, ctx, x):
        x = np.array(x, dtype=np.float64, copy=True).ravel()
        self._pending_skips.append(dict(x=x, boundary=self._boundary_with(x), ctx=ctx,
                                        certify=self._trial_certify))
        self._last_eval_info.update(coil_mode="skipped", coil_skipped=True)
        self.counters["coil_skipped"] += 1
        self.log(f"  [skip-coil] trial certainly rejected by TRF (equilibrium part alone "
                 f">= accepted cost, or VMEX failed): coil solve skipped "
                 f"({len(self._pending_skips)} pending).")

    def _on_coil_aborted(self, ctx, x, boundary, certify):
        """S-a: an aborted trial solve is a skipped one whose result is unknown (the warm-start
        state is left untouched and reconstructed on demand, exactly as for skipped solves)."""
        self._pending_skips.append(dict(x=np.array(x, dtype=np.float64, copy=True).ravel(),
                                        boundary=boundary, ctx=ctx, certify=certify))
        self._last_eval_info.update(coil_aborted=True)
        self.counters["coil_aborted"] += 1
        self.log(f"  [abort-coil] trial certainly rejected by TRF (VMEX failed or equilibrium part "
                 f"alone >= accepted cost): coil solve stopped before its fallback starts "
                 f"({len(self._pending_skips)} pending).")

    def _would_be_warm(self, x, prev):
        if prev is None or np.shape(prev) != np.shape(x):
            return False
        return float(np.max(np.abs(x - prev))) <= self.cfg.surface_dof_reset_threshold

    def _materialize_pending(self, why):
        pend, self._pending_skips = self._pending_skips, []
        if not pend:
            return
        self.log(f"  [skip-coil] {why}: running {len(pend)} skipped coil solve(s) to "
                 f"reconstruct the warm-start state.")
        saved_info, saved_flux = self._last_eval_info, self._flux_captured[0]
        for p in pend:
            self._last_eval_info = {}
            self._flux_captured[0] = True     # skips only happen once the target is captured
            self.counters["coil_materialized"] += 1
            try:
                self._compute_J2_and_grad(p["ctx"], p["x"], compute_grad=False,
                                          boundary=p["boundary"], certify=p.get("certify"))
            except Exception as e:  # noqa: BLE001  (PHR: failure -> state untouched)
                self.log(f"  [skip-coil] materialised solve failed ({e}); state untouched.")
        self._last_eval_info, self._flux_captured[0] = saved_info, saved_flux

    def _resolve_pending_skips(self, x):
        """Before a real coil solve at x."""
        if not self._pending_skips:
            return
        if self._anchored:
            # B2: warm starts read the anchor only; a skipped / aborted trial is never accepted, so it
            # can neither be nor change the anchor: nothing to reconstruct
            self._pending_skips = []
            return
        cands = [self._warm_prev_x] + [p["x"] for p in self._pending_skips]
        if any(self._would_be_warm(x, c) for c in cands):
            self._materialize_pending("next coil solve could be a warm start")

    def _refresh_elidable(self, x):
        """After LS, real _last_coil_x == x (accepted point) but solves were skipped since: PHR
        refreshes iff one of them succeeded.  If the solve at x was cold and a refresh at x would
        be cold after every pending point, the refresh reproduces the same deterministic cold
        solve at x, i.e. the state already held: the refresh can be elided exactly."""
        return bool(self._last_solve_cold and
                    not any(self._would_be_warm(x, p["x"]) for p in self._pending_skips))

    # ---------------- phase 2, B2: the anchored coil oracle
    def _solve_supports(self, name):
        """Does coil_solver.solve accept keyword `name`? (cached; a one-time warning for the B2 ones)"""
        if name not in self._solve_kw_ok:
            ok = self._accepts_kw(getattr(self.coil_solver, "solve", None), name)
            self._solve_kw_ok[name] = ok
            if not ok and name in ("rho_reset", "warm_k_max", "warm_fallback"):
                self.log(f"  [WARNING] coil_solver.solve has no {name} argument: the anchored coil "
                         f"oracle runs without it.")
        return self._solve_kw_ok[name]

    def _seed_anchor(self, x, boundary):
        """j9ctl (opt-in, env PHRPORT_SEED_ANCHOR=<json>): the parent run's last anchor (solver_x, currents from its
        frames.jsonl) re-converged at this boundary by a warm solve with rho reset (its multipliers were not saved), and
        set as the anchor, so the first coil solve inherits like the parent run's did. Used once."""
        from .coils import CoilResult
        path = os.environ.pop("PHRPORT_SEED_ANCHOR")
        sd = json.load(open(path))
        db = max((abs(float(boundary[k]) - float(v)) for k, v in sd["boundary"].items() if k in boundary), default=float("nan"))
        xs = np.asarray(sd["solver_x"], dtype=np.float64)
        m = int(self.coil_solver.P.m)
        pseudo = CoilResult(ok=True, c0_local=float(sd["c0"]), flux_F=float("nan"), x=xs,
                            currents=np.asarray(sd["currents"], dtype=np.float64), lam=np.zeros(m),
                            certified=False, linked=False, info={})
        t0 = time.time()
        res = self.coil_solver.solve(boundary, warm=pseudo, rho_reset=True)
        ri = getattr(res, "info", None) or {}
        self.log(f"  [seed-anchor] {path}: max |boundary - seed boundary| = {db:.3e}; seed c0 {float(sd['c0']):.9e} -> "
                 f"re-converged c0 {float(res.c0_local):.9e} ok={bool(getattr(res, 'ok', True))} mode={ri.get('mode')} "
                 f"max |dx| {float(np.max(np.abs(np.asarray(res.x) - xs))):.3e} kkt={ri.get('kkt')} [{time.time() - t0:.1f}s]")
        self._set_anchor(x, boundary, res, "seed")

    def _set_anchor(self, x, boundary, res, source, eval_id=None):
        """B2: the coil result at TRF's accepted point (or kept by the per-outer audit) becomes the
        anchor every later coil solve warm-starts from."""
        self._anchor_seq += 1
        ri = getattr(res, "info", None) or {}
        self._coil_acc = dict(x=np.array(x, dtype=np.float64, copy=True).ravel(),
                              boundary=collections.OrderedDict(boundary), res=res, step=self._step,
                              seq=self._anchor_seq, source=source, eval_id=eval_id,
                              c0=float(res.c0_local), lagr=coil_lagrangian(res),
                              kkt=(None if ri.get("kkt") is None else float(ri["kkt"])))
        self.counters["oracle_anchor_set"] += 1
        kk = self._coil_acc["kkt"]
        self.log(f"  [oracle] anchor #{self._anchor_seq} <- {source}: c0={float(res.c0_local):.6e} "
                 f"Lagrangian={self._coil_acc['lagr']:.6e} kkt={('%.3e' % kk) if kk is not None else 'n/a'}")

    def _anchor_from_jac(self, nevals, x):
        """B2: TRF forms its Jacobian at x0 and right after accepting a trial, i.e. at an accepted
        point; when the last coil solve was at that x, its result becomes the anchor.  A not-ok /
        non-finite result (or none at x: failed solve) leaves the previous anchor in place."""
        x = np.asarray(x, dtype=np.float64).ravel()
        res = self._coil_res
        if res is None or self._last_coil_x is None or not np.array_equal(self._last_coil_x, x):
            self.counters["oracle_anchor_missing"] += 1
            self.log("  [oracle] Jacobian point without a coil result at it (failed coil solve): "
                     "anchor kept.")
            return
        if self._coil_acc is not None and self._coil_acc["res"] is res:
            return
        if not _finite_ok(res):
            self.counters["oracle_anchor_notok"] += 1
            self.log("  [oracle] coil result at the accepted point is not ok / not finite: anchor kept.")
            return
        self._set_anchor(x, self._boundary_with(x), res,
                         source=f"f-eval {int(nevals)} ({self._last_oracle_stage})",
                         eval_id=self._last_eval_id)

    def _oracle_floor_kw(self, x):
        """B2 noise floor: the anchored_coil_solve keywords (none when cfg.oracle_d_floor = 0, so the
        call is the previous one).  linear() -> G.dx with G = the anchor's envelope gradient over the
        free DOFs of this step (computed once per anchor, cached in it); None when the anchor belongs
        to another step / DOF set."""
        cfg = self.cfg
        if not cfg.oracle_d_floor > 0.0:
            return {}
        x = np.asarray(x, dtype=np.float64).ravel()

        def linear():
            acc = self._coil_acc
            if acc is None or acc.get("step") != self._step or acc["x"].shape != x.shape:
                return None
            G = acc.get("grad")
            if G is None:
                t0 = time.time()
                G = np.asarray(self.coil_solver.envelope_grad(acc["boundary"], acc["res"], list(self._names)),
                               dtype=np.float64).ravel()
                self.timers["oracle_floor_grad"] += time.time() - t0
                self.counters["oracle_floor_grad"] += 1
                acc["grad"] = G
            if G.shape != x.shape or not np.all(np.isfinite(G)):
                return None
            return float(G @ (x - acc["x"]))
        return dict(d_floor=float(cfg.oracle_d_floor), floor_rule=cfg.oracle_floor_rule,
                    floor_k_max=int(cfg.oracle_floor_k_max), linear=linear)

    def _oracle_counts(self):
        return {k: int(v) for k, v in self.counters.items()
                if k.startswith("coil_oracle_") or k.startswith("oracle_")}

    def _oracle_audit_solve(self, boundary, certify=None):
        """B2 per-outer cold audit at the accepted iterate: a cold solve, compared with the anchor when
        the anchor sits at this surface; the lower coil Lagrangian (coil_lagrangian) among the finite
        ok ones is kept (ties: the anchor), else the cold result (PHR's refresh)."""
        acc = self._coil_acc
        d = None if acc is None else surface_distance(boundary, acc["boundary"])
        a_res = acc["res"] if (acc is not None and d == 0.0) else None
        kw = {} if certify is None else dict(certify=certify)
        cold = self.coil_solver.solve(boundary, warm=None, **kw)
        cand = []
        a_lin = a_res is not None and (getattr(a_res, "info", None) or {}).get("mode") == "linear"
        # B2 noise floor: a 'linear' anchor holds a first-order model value, not a solve on this
        # surface; it competes only when the cold solve is not usable
        if _finite_ok(a_res) and not (a_lin and _finite_ok(cold)):
            cand.append(("anchor", a_res))
        if _finite_ok(cold):
            cand.append(("cold", cold))
        if cand:
            kept, res = min(cand, key=lambda t: coil_lagrangian(t[1]))
        else:
            kept, res = "cold", cold
        ai = (getattr(a_res, "info", None) or {}) if a_res is not None else {}
        ci = getattr(cold, "info", None) or {}
        c0c = float(getattr(cold, "c0_local", np.nan))
        la = None if a_res is None else coil_lagrangian(a_res)
        lc = coil_lagrangian(cold) if np.isfinite(c0c) else None
        audit = dict(kept=kept, anchor_at_x=a_res is not None, d_anchor=d,
                     anchor_seq=(None if acc is None else acc.get("seq")),
                     anchor_source=(None if acc is None else acc.get("source")),
                     c0_anchor=(None if a_res is None else float(a_res.c0_local)), lagr_anchor=la,
                     kkt_anchor=ai.get("kkt"), ok_anchor=(None if a_res is None else bool(a_res.ok)),
                     c0_cold=c0c, lagr_cold=lc, kkt_cold=ci.get("kkt"), ok_cold=bool(getattr(cold, "ok", False)),
                     start_cold=ci.get("start"), v_cold=ci.get("v"),
                     rel_lagr_cold_vs_anchor=((lc - la) / abs(la) if (la is not None and lc is not None and la != 0.0)
                                              else None))
        if a_lin:
            audit["anchor_linear"] = True
        return res, dict(plan="audit", stage="audit", d=d, anchor_seq=audit["anchor_seq"], guard=None,
                         audit=audit)

    def _oracle_audit(self, ctx, x, step, al_iter):
        """B2: the post-LS refresh as a per-outer cold audit (metrics only; the kept result is the
        anchor and feeds the coil dual update); one kind='oracle_audit' line in eval_log.jsonl (with
        an eval_id / eval_x.npz row when phase-0 reporting is on, so eval_id stays the line index)."""
        self.log("  [oracle] per-outer cold audit at the accepted iterate (the lower coil Lagrangian "
                 "of {anchor, cold} is kept).")
        self._last_eval_info = {}
        seq0 = self._anchor_seq
        try:
            self._compute_J2_and_grad(ctx, x, compute_grad=False, oracle_audit=True)
        except Exception as e:  # noqa: BLE001  (as the PHR refresh: the coil dual update is skipped)
            self.log(f"  [WARNING] oracle audit failed ({e}); coil dual update skipped this iteration.")
            self._last_coil_vals[:] = np.nan
        ent, self._last_eval_info = self._last_eval_info, {}
        audit = (ent.get("oracle") or {}).get("audit")
        if audit is not None:
            fmt = lambda v: "n/a" if v is None else f"{v:.6e}"  # noqa: E731
            self.log(f"  [oracle] audit kept the {audit['kept']} result (anchor at x: {audit['anchor_at_x']}; "
                     f"Lagrangian anchor {fmt(audit['lagr_anchor'])} cold {fmt(audit['lagr_cold'])}; "
                     f"cold start {audit['start_cold']} ok={audit['ok_cold']})")
            self.counters[f"oracle_audit_kept_{audit['kept']}"] += 1
        rec = dict(kind="oracle_audit", step=int(step), al_iter=int(al_iter), nevals=None)
        rec.update(ent)
        if self.cfg.phase0_reporting:
            try:
                rec["eval_id"] = self._record_eval_x(step, al_iter, -1, x)
                if self._anchor_seq != seq0 and self._coil_acc is not None:
                    self._coil_acc["eval_id"] = rec["eval_id"]
            except Exception as e:  # noqa: BLE001
                self._warn_once("eval_x", f"  [WARNING] eval_x sidecar failed: {e}")
        with open(self.eval_log_path, "a") as f:
            f.write(json.dumps(_to_jsonable(rec)) + "\n")
        if self._frames_path is not None:
            if self._coil_res is not None:
                self._frame_coil = self._frame_coil_state(self._coil_res, "audit")
            self._write_frame("audit", step, al_iter, -1, x, rec, objective=None)
        return audit

    # ------------------------------------------------- movie frames (PHRPORT_FRAMES)
    def _frame_coil_state(self, res, mode):
        try:
            return self._frame_coil_state_(res, mode)
        except Exception as e:  # noqa: BLE001
            self._warn_once("frames_coil", f"  [WARNING] frame coil state failed: {e}")
            return None

    def _frame_coil_state_(self, res, mode):
        ri = getattr(res, "info", None) or {}
        return dict(mode=mode, ok=bool(getattr(res, "ok", True)), c0=float(res.c0_local),
                    certified=bool(getattr(res, "certified", False)), v=_info_get(res, ("v", "cert.v")),
                    bn_ratio_int=ri.get("bn_ratio_int"), mean_bn_area=ri.get("mean_bn_area"),
                    mean_bn_unweighted=ri.get("mean_bn_unweighted"),
                    solver_x=np.asarray(res.x, dtype=np.float64).ravel().tolist(),
                    currents=np.asarray(res.currents, dtype=np.float64).ravel().tolist())

    def _append_frame(self, rec):
        try:
            with open(self._frames_path, "a") as f:
                f.write(json.dumps(_to_jsonable(rec)) + "\n")
        except Exception as e:  # noqa: BLE001
            self._warn_once("frames", f"  [WARNING] frames.jsonl write failed: {e}")

    def _write_frame(self, kind, step, al_iter, nevals, x, rec, objective=None):
        try:
            self._write_frame_(kind, step, al_iter, nevals, x, rec, objective)
        except Exception as e:  # noqa: BLE001
            self._warn_once("frames_write", f"  [WARNING] frame record failed: {e}")
        self._frame_coil = None

    def _write_frame_(self, kind, step, al_iter, nevals, x, rec, objective=None):
        x = np.asarray(x, dtype=np.float64).ravel()
        eid = rec.get("eval_id")
        if eid is not None:
            self._frame_ids[x.tobytes()] = eid
            while len(self._frame_ids) > 4096:
                self._frame_ids.popitem(last=False)
        try:
            bnd = {k: float(v) for k, v in self._boundary_with(x).items()}
        except Exception:  # noqa: BLE001
            bnd = None
        keep = ("qs_total", "eq_converged", "J2", "c0", "coil_mode", "phantom", "min_cs", "aspect", "mean_iota",
                "iota_axis", "iota_edge", "g_eq")
        out = dict(kind=kind, step=int(step), al_iter=int(al_iter), nevals=int(nevals), eval_id=eid,
                   time=time.time(), objective=objective, boundary=bnd, coil=self._frame_coil,
                   **{k: rec[k] for k in keep if k in rec})
        self._frame_coil = None
        self._append_frame(out)

    def _compute_J2_and_grad(self, ctx, x, compute_grad=True, boundary=None, should_abort=None,
                             certify=None, oracle_audit=False):
        """PHR compute_J2_and_grad. Phase 0 (opt-in; None = the previous call exactly):
        should_abort (S-a) and certify (S-b) are passed on to coil_solver.solve; an aborted
        solve raises CoilSolveAborted after registering a pending (skipped) solve.
        Phase 2 (B2, cfg.coil_oracle='anchored'): the coil solve is anchored_coil_solve from the
        anchor (oracle_audit=True: the per-outer cold audit, whose kept result becomes the anchor)."""
        cfg = self.cfg
        x = np.array(x, dtype=np.float64, copy=True).ravel()
        n_surf = x.size
        if self._pending_skips:
            self._resolve_pending_skips(x)
        boundary = self._boundary_with(x) if boundary is None else boundary
        info = self._last_eval_info
        skw = {}
        if should_abort is not None:
            skw["should_abort"] = should_abort
        if certify is not None:
            skw["certify"] = certify
        if self._anchored and self._coil_acc is None and not oracle_audit and os.environ.get("PHRPORT_SEED_ANCHOR"):
            self._seed_anchor(x, boundary)          # j9ctl: restore the parent run's anchor before the first solve
        if self._anchored:
            acc = self._coil_acc
            if oracle_audit:
                self.log("  AL2 init [oracle]: cold audit.")
            else:
                d_a = None if acc is None else surface_distance(boundary, acc["boundary"])
                self.log("  AL2 init [oracle]: " + ("no anchor -> cold." if acc is None else
                                                    f"anchor #{acc['seq']} at max |dsurf|={d_a:.3e}."))
            t0 = time.time()
            try:
                if oracle_audit:
                    res, dec = self._oracle_audit_solve(boundary, certify)
                else:
                    res, dec = anchored_coil_solve(
                        self.coil_solver, boundary, acc, d_inherit=cfg.oracle_d_inherit,
                        d_reset=cfg.oracle_d_reset, k_max=cfg.oracle_k_max,
                        kkt_factor=cfg.oracle_kkt_factor, kkt_floor=cfg.oracle_kkt_floor,
                        certify=certify, should_abort=should_abort, supports=self._solve_supports,
                        reset_guard=cfg.oracle_reset_guard, **self._oracle_floor_kw(x))
            finally:
                dt = time.time() - t0
                self.timers["coil_solve"] += dt
                info["coil_wall"] = dt
            if dec.get("floor") is not None and acc is not None:
                dec["floor"]["anchor_eval_id"] = acc.get("eval_id")     # the anchor's eval_x row
            mode, delta = dec["stage"], dec.get("d")
            self._last_oracle_stage = mode
            info.update(coil_mode=mode, dsurf=delta, oracle=dec)
            self.counters[f"coil_oracle_{mode}"] += 1
            if dec.get("guard") is not None and not dec["guard"]["ok"]:
                self.counters["oracle_guard_fail"] += 1
            if dec.get("reset_guard") is not None and not dec["reset_guard"]["ok"]:
                self.counters["oracle_reset_guard_fail"] += 1
            warm = None if (getattr(res, "info", None) or {}).get("mode", "cold") == "cold" else res
        else:
            # alm_reset_decision_from_surface_change
            prev = self._warm_prev_x
            if prev is None or np.shape(prev) != np.shape(x):
                use_initial, delta = True, None
            else:
                delta = float(np.max(np.abs(x - prev)))
                use_initial = delta > cfg.surface_dof_reset_threshold
            if use_initial:
                msg = ("initial coils/multipliers (first iter)." if delta is None else
                       f"reset (max |dsurf|={delta:.3e} > {cfg.surface_dof_reset_threshold:.3e}).")
                mode = "initial" if delta is None else "reset"
            else:
                msg = f"warm-start (max |dsurf|={delta:.3e})."
                mode = "warm"
            self.log(f"  AL2 init: {msg}")
            warm = None if (use_initial and cfg.reinit_coils_each_alm) else self._coil_res
            info.update(coil_mode=mode, dsurf=delta)
            self.counters[f"coil_{mode}"] += 1
            t0 = time.time()
            try:
                res = self.coil_solver.solve(boundary, warm=warm, **skw)
            finally:
                dt = time.time() - t0
                self.timers["coil_solve"] += dt
                info["coil_wall"] = dt
        if should_abort is not None and bool((getattr(res, "info", None) or {}).get("aborted")):
            self.timers["coil_solve_aborted"] += dt
            self._on_coil_aborted(ctx, x, boundary, certify)
            raise CoilSolveAborted("coil solve aborted (trial certainly rejected)")
        ok = bool(getattr(res, "ok", True))
        info["coil_ok"] = ok
        if cfg.phase0_reporting:
            ri = getattr(res, "info", None) or {}
            info.update(coil_v=_info_get(res, ("v", "cert.v")), coil_v_soft=_info_get(res, ("v_soft",)),
                        coil_certified=bool(getattr(res, "certified", False)),
                        coil_certify=(ri.get("certify") if isinstance(ri, dict) else None))
        # failed coil evaluations: PHR's exception path leaves alm_warmstart_state untouched
        # (update_alm_warmstart_state is after the solve) and keeps the previous coils (bs)
        if not np.isfinite(float(res.c0_local)) and cfg.coil_nonfinite_policy == "sentinel":
            self.counters["coil_failed"] += 1
            raise CoilSolveFailure(f"coil solve failed (c0_local={res.c0_local}, ok={ok})")
        if not ok:
            self.counters["coil_not_ok"] += 1
            if cfg.coil_fail_policy == "sentinel":
                raise CoilSolveFailure("coil solve returned ok=False")
        # update_alm_warmstart_state(state, current_surface_x, None, None); bs keeps the coils
        self._warm_prev_x = x.copy()
        self._last_solve_cold = warm is None
        if self._pending_skips:
            self._pending_skips = []      # state fully determined by this solve
        self._coil_res = res
        if self._frames_path is not None:
            self._frame_coil = self._frame_coil_state(res, mode)
        self.log(f"  [coil-opt] {mode}: wall={dt:.2f}s ok={ok} c0_local={float(res.c0_local):.6e}")
        if self._anchored and oracle_audit and _finite_ok(res):
            aud = (info.get("oracle") or {}).get("audit") or {}
            self._set_anchor(x, boundary, res, source=f"audit (kept {aud.get('kept')})")

        raw_c0 = float(res.c0_local)
        coil_vals = np.array([raw_c0], dtype=np.float64) * self._coil_scales
        # phantom-flux guard
        flux_phantom = False
        min_cs, honest = self._phantom_inputs(boundary, res)
        if min_cs is None:
            self._warn_once("min_cs", "  [WARNING] coil result has no min coil-surface distance; "
                                      "phantom guard checks c0 <= eps only.")
        if raw_c0 <= cfg.flux_phantom_eps or (min_cs is not None and min_cs < cfg.cs_flux_safe_min):
            if honest is None:
                self._warn_once("honest", "  [WARNING] phantom condition met but no honest "
                                          "mean (B.n/|B|)^2 available; guard skipped.")
            elif honest > cfg.flux_phantom_eps:
                flux_phantom = True
                coil_vals[0] = honest * self._coil_scales[0]
                self.counters["phantom"] += 1
                self.log(f"  [flux guard] phantom c0={raw_c0:.2e} (min CS dist "
                         f"{(min_cs if min_cs is not None else float('nan')):.2e}): using honest "
                         f"mean (B.n/|B|)^2={honest:.3e}, dropping flux gradient.")
        info.update(c0=raw_c0, phantom=flux_phantom, min_cs=min_cs)
        self._last_coil_vals[:] = coil_vals
        self._last_coil_x = x.copy()

        if not np.all(np.isfinite(coil_vals)):
            self.log(f"  [WARNING] Coil metric values contain NaN/inf: {coil_vals}. "
                     f"Returning J2=0, zero grad.")
            self._warm_prev_x = None
            self.counters["coil_nan"] += 1
            return 0.0, np.zeros(n_surf)

        # adaptive flux target (first finite, non-phantom coil solve of the step)
        if cfg.flux_target_frac > 0.0 and not self._flux_captured[0] and not flux_phantom:
            if cfg.flux_target_floor > 0.0:
                # roadmap A1 (opt-in): S * max(frac * c0, floor) stops the frac^k compounding
                raw_tgt = max(cfg.flux_target_frac * raw_c0, float(cfg.flux_target_floor))
                tgt = raw_tgt * self._coil_scales[0]
            else:
                raw_tgt = None
                tgt = cfg.flux_target_frac * raw_c0 * self._coil_scales[0]
            bn_now = _info_lookup(getattr(res, "info", None) or {}, "bn_ratio_int")
            if cfg.flux_target_bn > 0.0 and bn_now is not None and np.isfinite(float(bn_now)) and float(bn_now) > 0.0:
                raw_bn = raw_c0 * (cfg.flux_target_bn / float(bn_now)) ** 2
                tgt = min(tgt, raw_bn * self._coil_scales[0])
                self.log(f"  [flux-target-bn] <|B.n|>/<|B|> = {float(bn_now):.4e} -> bound {cfg.flux_target_bn:.4e}: "
                         f"c0 target {raw_bn:.3e} raw (scaled {tgt:.3e})")
            if cfg.final_mode_flux_target > 0.0 and self._step == cfg.n_continuation_steps - 1:
                # roadmap A5 (opt-in): standalone front point -- the final mode solves the same problem
                # as a sweep point, min QS s.t. c0 <= E with E absolute (modes 1..n-1 keep the frac schedule)
                tgt = float(cfg.final_mode_flux_target) * self._coil_scales[0]
                self.log(f"  [final-mode-flux-target] target = {cfg.final_mode_flux_target:.6e} raw "
                         f"(scaled {tgt:.3e}) instead of the frac/floor value")
            self._coil_targets[0] = tgt
            ctx.targets[0] = tgt
            self._flux_captured[0] = True
            self.log(f"  [Option B adaptive] flux target set to {cfg.flux_target_frac:g} x "
                     f"achieved c0={raw_c0:.3e} = {cfg.flux_target_frac * raw_c0:.3e} raw "
                     f"(scaled {tgt:.3e})")
            if raw_tgt is not None:
                self.log(f"  [flux-target-floor] target = max({cfg.flux_target_frac * raw_c0:.3e}, "
                         f"floor {cfg.flux_target_floor:.3e}) = {raw_tgt:.3e} raw")

        if not compute_grad:
            shifted = np.maximum(0.0, coil_vals - ctx.targets + ctx.lam / ctx.rho)
            J2 = 0.5 * float(np.sum(ctx.rho * shifted ** 2))
            self._j2_holder[0] = float(J2)
            self.log(f"  AL1 coil penalty J2 = {J2:.6e}  (metrics-only refresh, gradient skipped)")
            return J2, np.zeros(n_surf)

        t1 = time.time()
        try:
            cg = np.asarray(self.coil_solver.envelope_grad(boundary, res, ctx.names),
                            dtype=np.float64).reshape(1, n_surf)
            cg = cg * self._coil_scales[:, None]
            if not np.all(np.isfinite(cg)):
                self.log("  [WARNING] Envelope produced non-finite gradients; zeroing.")
                cg = np.zeros((self.n_coil_con, n_surf))
        except Exception as e:  # noqa: BLE001
            self.log(f"  [WARNING] Envelope gradient failed: {e}; zeroing the coil gradient.")
            cg = np.zeros((self.n_coil_con, n_surf))
        self.timers["envelope"] += time.time() - t1
        if flux_phantom:
            cg[0, :] = 0.0
        if self._kkt_on:          # B1: the coil-constraint gradient at this point (reference)
            self._kkt_cg[x.tobytes()] = (cg[0].copy(), bool(flux_phantom))
            while len(self._kkt_cg) > 256:
                self._kkt_cg.popitem(last=False)
        shifted = np.maximum(0.0, coil_vals - ctx.targets + ctx.lam / ctx.rho)
        J2 = 0.5 * float(np.sum(ctx.rho * shifted ** 2))
        grad_J2 = np.zeros_like(cg[0])
        for k in range(self.n_coil_con):
            grad_J2 += ctx.rho[k] * shifted[k] * cg[k]
        if not np.isfinite(J2) or not np.all(np.isfinite(grad_J2)):
            self.log(f"  [WARNING] Non-finite J2={J2} or grad_J2; zeroing out.")
            J2 = 0.0
            grad_J2 = np.zeros_like(grad_J2)
        self.log(f"  AL1 coil penalty J2 = {J2:.6e}  ||grad J2|| = {np.linalg.norm(grad_J2):.6e}"
                 f"  (value={coil_vals[0]:.6e} target<={ctx.targets[0]:.6e} "
                 f"||dc/ds||={np.linalg.norm(cg[0]):.3e})")
        self._j2_holder[0] = float(J2)
        return J2, grad_J2

    # -------------------------------------------------- callbacks (PHR fmt)
    def _make_callbacks(self, step, al_iter, max_mode, ctx_ref):
        cfg = self.cfg
        last_residuals = [None]
        is_qss = cfg.enable_coils

        def iteration_callback(nevals, x, residuals, objective):
            last_residuals[0] = np.asarray(residuals)
            J1 = float(objective)
            J2_record = 0.0
            if is_qss and self._j2_holder[0] is not None:
                J2_record = float(self._j2_holder[0])
            self.log(SEP_THIN)
            self.log(f"  [continuation step {step+1}/{cfg.n_continuation_steps}, AL1 iter "
                     f"{al_iter+1}/{cfg.al_outer_iter}, max_mode={max_mode}]")
            if is_qss and self._j2_holder[0] is not None:
                self.log(f"  Surface iter {nevals} (max_mode={max_mode})  J1 = {J1:.6e}  "
                         f"J2 = {J2_record:.6e}")
            else:
                self.log(f"  Surface iter {nevals} (max_mode={max_mode})  J1 = {J1:.6e}")
            # PHRPORT_CLEAR_CACHES_EVERY=N: drop JAX's compiled-function caches every N surface evaluations
            # (long finite-beta / QI runs otherwise grow by GBs per hour and get OOM-killed)
            _ncc = int(_os.environ.get("PHRPORT_CLEAR_CACHES_EVERY", "0") or 0)
            # PHRPORT_CLEAR_CACHES_RSS_MB=M: drop them only once the process RSS exceeds M MB (keeps the
            # exact-Jacobian / QI / coil programs compiled between evaluations; ~100 s per recompile)
            _rss_cap = int(_os.environ.get("PHRPORT_CLEAR_CACHES_RSS_MB", "0") or 0)
            _rss = _rss_mb() if _rss_cap > 0 else 0
            if (_ncc > 0 and nevals % _ncc == 0) or (_rss_cap > 0 and _rss > _rss_cap):
                import gc, jax as _jax
                _jax.clear_caches(); gc.collect()
                if _rss_cap > 0:
                    self.log(f"  [caches] RSS {_rss} MB > {_rss_cap} MB: JAX caches cleared "
                             f"(now {_rss_mb()} MB)")
            self.J1_J2_history.append({
                "J1": J1, "J2": J2_record, "grad_J1": None, "grad_J2": None,
                "step": step, "al_iter": al_iter, "max_mode": max_mode, "nevals": int(nevals),
            })
            rec = dict(step=step, al_iter=al_iter, nevals=int(nevals), objective=J1,
                       J2=(float(residuals[-1]) ** 2 if is_qss else None))
            rec.update({k: v for k, v in self._last_eval_info.items()})
            self._last_eval_info = {}
            if cfg.phase0_reporting:          # G5: x of every f-eval -> eval_x.npz (row eval_id)
                try:
                    rec["eval_id"] = self._record_eval_x(step, al_iter, int(nevals), x)
                except Exception as e:  # noqa: BLE001
                    self._warn_once("eval_x", f"  [WARNING] eval_x sidecar failed: {e}")
            if self._anchored:
                self._last_eval_id = rec.get("eval_id")
            with open(self.eval_log_path, "a") as f:
                f.write(json.dumps(_to_jsonable(rec)) + "\n")
            if self._frames_path is not None:
                self._write_frame("eval", step, al_iter, int(nevals), x, rec, objective=J1)

        def jac_callback(nevals, x, jac):
            if self._anchored:                # B2: TRF's Jacobian point = an accepted point
                self._anchor_from_jac(nevals, x)
            if self._frames_path is not None:
                xb = np.asarray(x, dtype=np.float64).ravel().tobytes()
                # the coils at the accepted point after TRF's Jacobian evaluation: at finite beta that re-solve uses
                # the plasma field context refreshed at this point (the f-eval's coils used the previous context)
                self._append_frame(dict(kind="accepted", step=int(step), al_iter=int(al_iter), nevals=int(nevals),
                                        eval_id=self._frame_ids.get(xb), time=time.time(),
                                        coil=(self._frame_coil_state(self._coil_res, "accepted")
                                              if self._coil_res is not None else None)))
            r = last_residuals[0]
            if r is None:
                return
            jac = np.asarray(jac)
            r = np.atleast_1d(r)
            grad = 2.0 * (r @ jac) if jac.shape[0] == len(r) else 2.0 * (jac @ r)
            n1 = float(np.linalg.norm(grad))
            self.log(f"  ||grad J|| = {n1:.6e}")
            if self.J1_J2_history:
                self.J1_J2_history[-1]["grad_J1"] = n1

        return iteration_callback, jac_callback

    # ------------------------------------------------------- checkpoints
    def _freeze_ordering_axis(self, step, boundary, ck=None):
        """Freeze the reference axis of the coil ordering (slot) rows for continuation step `step`: the checkpoint's
        axis when `ck` (a checkpoint of this same step) carries one, else the geometric axis (m = 0 modes) of the
        boundary at the start of the step. No-op for coil solvers without ordering rows / that axis."""
        cs = self.coil_solver
        if not hasattr(cs, "set_ordering_axis") or getattr(getattr(cs, "cfg", None), "ordering_axis", "") != "frozen_step":
            return None
        if ck is not None and ck.get("ordering_axis") is not None:
            ax = cs.set_ordering_axis(axis=ck["ordering_axis"])
            src = "the checkpoint"
        else:
            ax = cs.set_ordering_axis(boundary=boundary)
            src = "the boundary at the start of the step"
        if getattr(cs.cfg, "slot_rows", False):
            self.log(f"  [ordering-axis] step {step + 1}: reference axis frozen from {src} "
                     f"(R0 {ax[0][0]:.6f}, {len(ax[0])} harmonics)")
        return ax

    def _checkpoint(self, step, al_iter, max_mode, lam_eq, rho_eq, lam_coil, rho_coil, x, tag=None,
                    extra=None):
        """PHR checkpoint al1_checkpoint_step{SS}_iter{II}.json (+ _latest).  A4 (sweep only): tag ->
        al1_checkpoint_{tag}_iter{II}.json instead, extra = additional keys."""
        ck = {
            "alm_mode": "phr",
            "step": int(step),
            "al_iter": int(al_iter),
            "max_mode": int(max_mode),
            "rho_eq": rho_eq.tolist(),
            "rho_coil": rho_coil.tolist(),
            "lam_eq": lam_eq.tolist(),
            "lam_coil": lam_coil.tolist(),
            "eq_labels": list(self.eq_labels),
            "surface_dofs": np.asarray(x).tolist(),
            "vmec_dofs": np.asarray(x).tolist(),
            "vmec_input": self.vmec_input,
            # port extras (ignored by the PHR resume path)
            "surface_dof_names": list(self._names),
            "coil_targets_scaled": self._coil_targets.tolist(),
            "flux_target_captured": bool(self._flux_captured[0]),
            # frozen reference axis of the coil ordering rows in this step (None: not frozen)
            "ordering_axis": getattr(self.coil_solver, "ordering_axis", None),
        }
        if extra:
            ck.update(_to_jsonable(extra))
        name = (f"al1_checkpoint_step{step:02d}_iter{al_iter:02d}.json" if tag is None
                else f"al1_checkpoint_{tag}_iter{al_iter:02d}.json")
        _json_dump(ck, os.path.join(self.out_dir, name))
        _json_dump(ck, os.path.join(self.out_dir, "al1_checkpoint_latest.json"))

    # ------------------------------------------------- phase 0: reporting helpers
    def _init_eval_x(self):
        """G5 sidecar: one row per f-eval (eval_id = line index of eval_log.jsonl, counting the
        lines already there); rows of an existing eval_x.npz in out_dir are kept."""
        ev = dict(eval_id=[], step=[], al_iter=[], nevals=[], x=[])
        names = {}
        if os.path.exists(self.eval_log_path):
            with open(self.eval_log_path) as f:
                self._eval_id_base = sum(1 for _ in f)
        if os.path.exists(self._eval_x_path):
            try:
                with np.load(self._eval_x_path, allow_pickle=False) as d:
                    for i in range(d["eval_id"].size):
                        n = int(d["n_free"][i])
                        ev["x"].append(np.array(d["x"][i, :n]))
                        for k in ("eval_id", "step", "al_iter", "nevals"):
                            ev[k].append(int(d[k][i]))
                    for k in d.files:
                        if k.startswith("names_step"):
                            names[k] = [str(s) for s in d[k]]
            except Exception as e:  # noqa: BLE001
                self.log(f"  [WARNING] could not read the existing eval_x.npz ({e}); starting afresh.")
                ev = dict(eval_id=[], step=[], al_iter=[], nevals=[], x=[])
        self._eval_x = dict(ev, names=names)

    def _record_eval_x(self, step, al_iter, nevals, x):
        ev = self._eval_x
        eid = self._eval_id_base + self._eval_x_new
        self._eval_x_new += 1
        ev["eval_id"].append(eid); ev["step"].append(int(step)); ev["al_iter"].append(int(al_iter))
        ev["nevals"].append(int(nevals)); ev["x"].append(np.array(x, dtype=np.float64, copy=True).ravel())
        ev["names"][f"names_step{int(step):02d}"] = list(self._names or [])
        self._write_eval_x()
        return eid

    def _write_eval_x(self):
        ev = self._eval_x
        if ev is None or not ev["eval_id"]:
            return
        nmax = max(r.size for r in ev["x"])
        X = np.full((len(ev["x"]), nmax), np.nan)
        for i, r in enumerate(ev["x"]):
            X[i, :r.size] = r
        tmp = self._eval_x_path + ".tmp.npz"
        np.savez(tmp, eval_id=np.asarray(ev["eval_id"], dtype=np.int64),
                 step=np.asarray(ev["step"], dtype=np.int64),
                 al_iter=np.asarray(ev["al_iter"], dtype=np.int64),
                 nevals=np.asarray(ev["nevals"], dtype=np.int64),
                 n_free=np.asarray([r.size for r in ev["x"]], dtype=np.int64), x=X,
                 **{k: np.asarray(v, dtype=str) for k, v in ev["names"].items()})
        os.replace(tmp, self._eval_x_path)

    def _write_provenance(self, status, wall=None, extra=None):
        if not self.cfg.phase0_reporting:
            return
        path = os.path.join(self.out_dir, "provenance.json")
        try:
            if status == "started" or not os.path.exists(path):
                files = {}
                if self.resume_from or self.init_surface:
                    src = self.resume_from or self.init_surface
                    files["resume_or_init"] = (os.path.join(src, "al1_checkpoint_latest.json")
                                               if os.path.isdir(src) else src)
                self._prov = collect_provenance(argv=self.argv, cfg=self.cfg,
                                                coil_solver=self.coil_solver,
                                                input_deck=self.input_deck, extra_files=files)
                self._prov.update(out_dir=self.out_dir, resume_from=self.resume_from,
                                  init_surface=self.init_surface)
            prov = dict(getattr(self, "_prov", {}) or {})
            prov["status"] = status
            if status != "started":
                prov["finished"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
                prov["wall_time_s"] = wall
            if extra:
                prov.update(extra)
            _json_dump(_to_jsonable(prov), path)
        except Exception as e:  # noqa: BLE001
            self.log(f"  [WARNING] provenance.json failed: {e}")

    def _write_final_checkpoint(self, st, lam_eq, rho_eq, lam_coil, rho_coil, x, extra=None):
        """B0: the state after the PHR loop in checkpoint format (al1_checkpoint_final.json; the
        latest / per-outer checkpoints are unchanged).  step / al_iter = where a resume continues
        (al_iter = the next outer; after a graceful stop, the outer that failed), plus the extras
        a --resume-keep-target resume restores exactly (target, captured flag, v_prev)."""
        if not self.cfg.phase0_reporting or st is None:
            return None
        ck = {
            "alm_mode": "phr", "step": int(st["step"]), "al_iter": int(st["al_iter"]),
            "max_mode": int(st["step"]) + 1,
            "rho_eq": np.asarray(rho_eq).tolist(), "rho_coil": np.asarray(rho_coil).tolist(),
            "lam_eq": np.asarray(lam_eq).tolist(), "lam_coil": np.asarray(lam_coil).tolist(),
            "eq_labels": list(self.eq_labels),
            "surface_dofs": np.asarray(x).tolist(), "vmec_dofs": np.asarray(x).tolist(),
            "vmec_input": self.vmec_input,
            "surface_dof_names": list(self._names or []),
            "eq_labels": list(self.eq_labels),
            "coil_targets_scaled": self._coil_targets.tolist(),
            "flux_target_captured": bool(self._flux_captured[0]),
            # phase-0 extras
            "final": True, "completed": not st.get("aborted", False),
            "aborted": bool(st.get("aborted", False)),
            "v_prev_eq": (None if st.get("v_prev_eq") is None else np.asarray(st["v_prev_eq"]).tolist()),
            "v_prev_coil": (None if st.get("v_prev_coil") is None
                            else np.asarray(st["v_prev_coil"]).tolist()),
            "outers_done_in_step": int(st.get("done", 0)),
            "ordering_axis": getattr(self.coil_solver, "ordering_axis", None),
        }
        if extra:                       # A4: the sweep fields (a sweep directory resumes as a seed)
            ck.update(extra)
        path = os.path.join(self.out_dir, "al1_checkpoint_final.json")
        _json_dump(_to_jsonable(ck), path)
        return path

    def _v_prev_from_history(self, ck_path, step, al_iter):
        """--resume-keep-target: the rho-growth history (v_prev_eq, v_prev_coil) at the start of
        outer al_iter of `step`, from phr_history.json next to the checkpoint."""
        if al_iter <= 0:
            return None, None, "outer 0 of the step (v_prev = None)"
        hp = os.path.join(os.path.dirname(os.path.abspath(ck_path)), "phr_history.json")
        if not os.path.exists(hp):
            return None, None, f"no {hp}"
        with open(hp) as f:
            hist = json.load(f)
        ent = [e for e in hist if int(e["step"]) == int(step) and int(e["al_iter"]) < int(al_iter)]
        if not ent:
            return None, None, "no matching phr_history entries"
        # the run that wrote the checkpoint may have been resumed itself: keep the last entry
        # per al_iter and require the outer just before this one
        by_it = {}
        for e in ent:
            by_it[int(e["al_iter"])] = e
        if (int(al_iter) - 1) not in by_it:
            return None, None, f"phr_history lacks outer {al_iter - 1} of step {step}"
        v_eq = np.asarray(by_it[int(al_iter) - 1]["v_eq"], dtype=np.float64)
        v_coil = None
        for it in sorted(by_it):
            gc = by_it[it].get("g_coil") or []
            if gc and all(v is not None and np.isfinite(v) for v in gc):
                v_coil = np.maximum(0.0, np.asarray(gc, dtype=np.float64))
        return v_eq, v_coil, "phr_history.json"

    def _kkt_after_outer(self, ls, x, eq_acc, g_eq, g_coil, lam_eq, lam_coil, sub):
        """B1 certificate at the accepted x from the Jacobian TRF formed there (no extra solve):
        grad QS = 2 J_qs^T r_qs; equilibrium-constraint gradients from the exact Jacobian
        (cfg.eq_jacobian='exact') or from the hinge rows of the LS Jacobian where the hinge is
        active (J_i / sqrt(rho_i/2)); coil gradient = the scaled envelope gradient of the f-eval
        at x.  Never raises (an error string instead)."""
        cfg = self.cfg
        try:
            J, Jx = ls.last_jac, ls.last_jac_x
            if J is None or Jx is None or not np.array_equal(Jx, x):
                return dict(error="no LS Jacobian at the accepted point")
            J = np.asarray(J)
            r_qs = np.asarray(eq_acc.qs_residuals, dtype=np.float64).ravel()
            n_qs = r_qs.size
            n_rows = n_qs + self.n_eq_con + (1 if cfg.enable_coils else 0)
            if J.ndim != 2 or J.shape != (n_rows, x.size):
                return dict(error=f"LS Jacobian shape {J.shape} != {(n_rows, x.size)}")
            grad_qs = 2.0 * (J[:n_qs].T @ r_qs)
            grads, src_eq = {}, "hinge_rows"
            ex = None
            if (self._exact_jac is not None and self._exact_last_x is not None
                    and np.array_equal(self._exact_last_x, x)):
                ex = getattr(self._exact_jac, "last", None)
            if ex is not None:
                src_eq = "exact"
                dg = exact_constraint_grads(ex.jac, cfg)
                for lab in self.eq_labels:
                    grads[lab] = None if lab not in dg else np.asarray(dg[lab], dtype=np.float64)
            else:
                for i, lab in enumerate(self.eq_labels):
                    h = float(g_eq[i]) + float(sub.lam[i] / sub.rho[i])
                    w = math.sqrt(float(sub.rho[i]) / 2.0)
                    grads[lab] = (np.asarray(J[n_qs + i], dtype=np.float64) / w) if h > 0.0 else None
            g = {lab: float(g_eq[i]) for i, lab in enumerate(self.eq_labels)}
            lam = {lab: float(lam_eq[i]) for i, lab in enumerate(self.eq_labels)}
            src_coil = None
            if cfg.enable_coils:
                g["Jf"] = float(g_coil[0]) if np.size(g_coil) else float("nan")
                lam["Jf"] = float(lam_coil[0])
                ent = self._kkt_cg.get(np.asarray(x, dtype=np.float64).tobytes())
                grads["Jf"] = None if ent is None else ent[0]
                src_coil = (None if ent is None else ("envelope(phantom: zeroed)" if ent[1] else "envelope"))
            # lam_eq / lam_coil are the multipliers after this outer's dual update: lambda_i > 0
            # exactly where the PHR hinge g_i + lambda_i^old / rho_i was positive at x
            cert = kkt_certificate(grad_qs, g, lam, grads, active_tol=cfg.kkt_active_tol,
                                   grad_floor=cfg.kkt_grad_floor, phr_active=cfg.kkt_phr_active)
            hi, lo = "iota_mean_hi", "iota_mean_lo"
            cert.update(source_eq=src_eq, source_coil=src_coil, qs_total=float(eq_acc.qs_total),
                        lambda_iota_net=float(lam.get(hi, 0.0) - lam.get(lo, 0.0)),
                        kkt_ok=bool(cert["feasibility"] < cfg.v_tol
                                    and cert["stationarity"] < cfg.kkt_stationarity_tol))
            return cert
        except Exception as e:  # noqa: BLE001
            return dict(error=f"{type(e).__name__}: {e}")

    def _certify_accepted(self, x):
        """S-b: when the accepted point's coils come from an uncertified (trial) solve, certify
        those coils (no re-solve) and compare soft ok with the certified ok rule."""
        res = self._coil_res
        if res is None or self._last_coil_x is None or not np.array_equal(self._last_coil_x, x):
            return dict(error="no coil result at the accepted point")
        info = getattr(res, "info", None) or {}
        if info.get("certify", True):
            return dict(source="certified solve", v=_info_get(res, ("v", "cert.v")),
                        ok=bool(getattr(res, "ok", False)), soft_ok=info.get("soft_ok"),
                        agree=(None if info.get("soft_ok") is None
                               else bool(info.get("soft_ok")) == bool(getattr(res, "ok", False))))
        ev = getattr(self.coil_solver, "evaluate_coils", None)
        if ev is None:
            return dict(error="coil solver has no evaluate_coils")
        t0 = time.time()
        Pp = getattr(self.coil_solver, "P", None)
        nc = int(Pp.nc) if Pp is not None else np.size(res.x)
        out = ev(self._boundary_with(x), np.asarray(res.x)[:nc], np.asarray(res.currents), certify=True)
        self.timers["coil_cert_accepted"] += time.time() - t0
        ce = out.get("cert") or {}
        ok_viol = float(getattr(getattr(self.coil_solver, "cfg", None), "ok_viol", 2e-2))
        ok_cert = bool(np.isfinite(float(res.c0_local)) and not bool(ce.get("linked", False))
                       and float(ce.get("v", np.inf)) <= ok_viol)
        soft = info.get("soft_ok")
        return dict(source="certified afterwards (evaluate_coils)", v=ce.get("v"),
                    certified=ce.get("certified"), linked=ce.get("linked"), ok=ok_cert,
                    soft_ok=soft, v_soft=info.get("v_soft"),
                    agree=(None if soft is None else bool(soft) == ok_cert), t=time.time() - t0)

    def _write_histories(self):
        _json_dump(self.phr_history, os.path.join(self.out_dir, "phr_history.json"))
        _json_dump(self.J1_J2_history, os.path.join(self.out_dir, "j1_j2_history.json"))
        _json_dump(_to_jsonable(self.outer_log), os.path.join(self.out_dir, "port_outer_log.json"))

    def _maybe_write_wout(self, boundary, step, name):
        if self.write_wout is None:
            return
        try:
            self.write_wout(boundary, step, os.path.join(self.out_dir, name))
            self.log(f"  Saved {name}")
        except Exception as e:  # noqa: BLE001
            self.log(f"  [WARNING] could not write {name}: {e}")

    def _loop_flags(self):
        """The opt-in speed / phase-2 switches of the outer loop (validation + their log lines);
        -> (speedups_on, p0_speed, abort_on).  Shared by run() and sweep()."""
        cfg = self.cfg
        speedups_on = bool(cfg.skip_coil_on_rejected or cfg.stall_stop or cfg.carry_trust_radius)
        if speedups_on:
            if int(cfg.stall_stop) < 0 or not cfg.stall_delta_factor > 0:
                raise ValueError("stall_stop must be >= 0 and stall_delta_factor > 0")
            if (cfg.stall_stop or cfg.carry_trust_radius) and (
                    cfg.ls_kwargs.get("method", "trf") != "trf"
                    or cfg.ls_kwargs.get("loss", "linear") != "linear"
                    or "bounds" in cfg.ls_kwargs):
                raise ValueError("stall_stop / carry_trust_radius need TRF, linear loss, no bounds")
            if cfg.skip_coil_on_rejected and not cfg.reinit_coils_each_alm:
                self.log("  [WARNING] skip_coil_on_rejected needs reinit_coils_each_alm=True "
                         "(otherwise every coil solve depends on the skipped ones); disabled.")
            self.log(f"  [speedups] skip_coil_on_rejected={cfg.skip_coil_on_rejected} "
                     f"(scope {cfg.skip_coil_scope}) "
                     f"stall_stop={cfg.stall_stop} (delta factor {cfg.stall_delta_factor:g}) "
                     f"carry_trust_radius={cfg.carry_trust_radius}")
        p0_speed = bool(cfg.abort_doomed_coil_solves or not cfg.certify_trials or cfg.reuse_vmex_state)
        abort_on = bool(cfg.abort_doomed_coil_solves and cfg.enable_coils)
        if p0_speed:
            solve_fn = getattr(self.coil_solver, "solve", None)
            if abort_on and not self._accepts_kw(solve_fn, "should_abort"):
                self.log("  [WARNING] coil_solver.solve has no should_abort argument; "
                         "abort_doomed_coil_solves disabled.")
                abort_on = False
            if self._trial_certify is not None and not self._accepts_kw(solve_fn, "certify"):
                self.log("  [WARNING] coil_solver.solve has no certify argument; trial solves "
                         "stay certified.")
                self._trial_certify = None
            if cfg.abort_doomed_coil_solves and not cfg.reinit_coils_each_alm:
                self.log("  [WARNING] abort_doomed_coil_solves needs reinit_coils_each_alm=True; "
                         "no solve will be aborted.")
            if cfg.reuse_vmex_state and cfg.eq_jacobian != "exact":
                self.log("  [WARNING] reuse_vmex_state only acts on the exact Jacobian "
                         "(eq_jacobian='exact'); no effect.")
            self.log(f"  [phase0-speed] abort_doomed_coil_solves={cfg.abort_doomed_coil_solves} "
                     f"certify_trials={cfg.certify_trials} reuse_vmex_state={cfg.reuse_vmex_state}")
        phase2_on = bool(cfg.coil_oracle != "phr" or cfg.carry_flux_multiplier)
        if phase2_on:
            if cfg.coil_oracle == "anchored" and not cfg.enable_coils:
                self.log("  [WARNING] coil_oracle='anchored' without coils: no effect.")
            if self._anchored and not (0.0 <= cfg.oracle_d_inherit <= cfg.oracle_d_reset):
                raise ValueError("oracle windows need 0 <= oracle_d_inherit <= oracle_d_reset")
            if self._anchored and cfg.oracle_d_floor > 0.0:
                if cfg.oracle_floor_rule not in ("linear", "converge"):
                    raise ValueError(f"oracle_floor_rule must be 'linear' or 'converge', got {cfg.oracle_floor_rule!r}")
                if cfg.oracle_d_floor > cfg.oracle_d_inherit:
                    raise ValueError("the oracle noise floor needs oracle_d_floor <= oracle_d_inherit")
            self.log(f"  [phase2] coil_oracle={cfg.coil_oracle} (d_inherit={cfg.oracle_d_inherit:g}, "
                     f"d_reset={cfg.oracle_d_reset:g}, k_max={cfg.oracle_k_max}, guard kkt <= "
                     f"max({cfg.oracle_kkt_factor:g} kkt_anchor, {cfg.oracle_kkt_floor:g}), "
                     f"audit={cfg.oracle_audit}, reset_guard={cfg.oracle_reset_guard}"
                     + (f", noise floor d <= {cfg.oracle_d_floor:g}: {cfg.oracle_floor_rule}"
                        + (f" (k_max {cfg.oracle_floor_k_max})" if cfg.oracle_floor_rule == "converge" else "")
                        if cfg.oracle_d_floor > 0.0 else "")
                     + f") carry_flux_multiplier={cfg.carry_flux_multiplier}")
        return speedups_on, p0_speed, abort_on

    def _outer_iteration(self, st, step, al_iter, max_mode, *, flags, kkt_exit_mode, n_budget,
                         base_budget, ck_tag=None, ck_extra=None, rec_extra=None):
        """One PHR AL1 outer iteration (checkpoint, least squares, refresh / audit, dual updates, rho
        growth, KKT certificate, histories, early-exit test) on the loop state st (_LoopState; updated
        in place).  -> None (continue), "exit" (early exit) or "abort" (graceful stop after a VMEC
        failure in the dual update; st.aborted is set).  base_budget = the regular outer budget (the
        default cfg.al_outer_iter; extra KKT outers beyond it up to n_budget).  A4 (sweep only):
        ck_tag / ck_extra name the per-outer checkpoint, rec_extra is merged into the phr_history /
        port_outer_log records."""
        cfg = self.cfg
        speedups_on, p0_speed, abort_on = flags
        x, boundary = st.x, st.boundary
        lam_eq, rho_eq, lam_coil, rho_coil = st.lam_eq, st.rho_eq, st.lam_coil, st.rho_coil
        v_prev_eq, v_prev_coil = st.v_prev_eq, st.v_prev_coil
        last_good, last_good_step, aborted = st.last_good, st.last_good_step, st.aborted
        fin_state = st.fin_state
        try:
            t_outer = time.time()
            self.log(SEP_THIN)
            self.log(f"  PHR AL1 outer iter {al_iter+1}/{base_budget}  (max_mode={max_mode})"
                     + ("  [extra outer: KKT exit test not met]"
                        if al_iter >= base_budget else ""))
            self.log(SEP_THIN)
            self._checkpoint(step, al_iter, max_mode, lam_eq, rho_eq, lam_coil, rho_coil, x,
                             tag=ck_tag, extra=ck_extra)
            if cfg.maxj and cfg.maxj_track_frac > 0.0 and maxj_active(cfg):
                eqs = self._cache_get(step, x)
                if eqs is None:
                    eqs = self._eval_single(boundary, step)
                mj = float(getattr(eqs, "maxj", float("nan"))) if bool(getattr(eqs, "converged", False)) else float("nan")
                if math.isfinite(mj):
                    _MAXJ_OFFSET[0] = (1.0 - cfg.maxj_track_frac) * mj if mj > cfg.maxj_track_floor else 0.0
                self.log(f"  [maxj-track] outer {self._maxj_k + 1} since switch-on: maxj = {mj:.4e}, "
                         f"relaxation {_MAXJ_OFFSET[0]:.4e} (r = {cfg.maxj_track_frac:g})")
                self._maxj_k += 1
            elif cfg.maxj and cfg.maxj_ramp_outers > 0 and maxj_active(cfg):
                if self._maxj_g0 is None:
                    eqs = self._eval_single(boundary, step)
                    g0 = float(getattr(eqs, "maxj", float("nan"))) if bool(getattr(eqs, "converged", False)) else float("nan")
                    self._maxj_g0 = max(0.0, g0) if math.isfinite(g0) else 0.0
                    self.log(f"  [maxj-ramp] closure row on: maxj = {g0:.4e} at the switch-on point; target lowered "
                             f"over {cfg.maxj_ramp_outers} outers")
                _MAXJ_OFFSET[0] = self._maxj_g0 * max(0.0, 1.0 - (self._maxj_k + 1) / cfg.maxj_ramp_outers)
                self.log(f"  [maxj-ramp] outer {self._maxj_k + 1} since switch-on: relaxation {_MAXJ_OFFSET[0]:.4e}")
                self._maxj_k += 1
            self._sub = ALSubproblem(lam_eq, rho_eq, cfg, n_groups=cfg.fd_ngroups)
            ctx = _CoilCtx(rho_coil, lam_coil, self._coil_targets, al_iter, max_mode, step,
                           self._names)
            iter_cb, jac_cb = self._make_callbacks(step, al_iter, max_mode, ctx)
            ls_kwargs = dict(cfg.ls_kwargs)
            ls_kwargs["max_nfev"] = cfg.max_nfev
            if cfg.spectral_scale > 0.0 and "x_scale" not in ls_kwargs:
                # spectral scaling: trust region and step in coordinates where high (m, n) modes move less
                mn = [tuple(int(v) for v in nm[nm.index("(") + 1:-1].split(",")) for nm in self._names]
                ls_kwargs["x_scale"] = np.array([(1.0 + m * m + n * n) ** (-0.5 * cfg.spectral_scale)
                                                 for m, n in mn], dtype=np.float64)
            j2fn = ((lambda xx, should_abort=None, _c=ctx, _tc=self._trial_certify:
                     self._compute_J2_and_grad(_c, xx, should_abort=should_abort, certify=_tc))
                    if cfg.enable_coils else None)
            prefetch = (self._residuals_prefetch
                        if (cfg.overlap_eq_and_coils and cfg.enable_coils) else None)
            eq_jac = None
            if cfg.eq_jacobian == "exact":      # opt-in; default 'fd' leaves PHR unchanged
                if self._exact_jac is None:
                    from .exact_jacobian import PHRExactJacobian
                    if cfg.reuse_vmex_state and self._state_wanted():
                        # S-c (opt-in): the f-eval cache holds EqResults carrying their state
                        self._exact_jac = PHRExactJacobian(gauge=cfg.exact_jac_gauge, log=self.log,
                                                           state_lookup=self._cache_get)
                    else:
                        self._exact_jac = PHRExactJacobian(gauge=cfg.exact_jac_gauge, log=self.log)

                if cfg.finite_beta:
                    self._exact_jac.want_state_tangents = True

                def eq_jac(xx, _s=step, _sub=self._sub):
                    if cfg.finite_beta and cfg.fb_reuse_tol > 0:     # P5: tangents only for a refresh
                        self._exact_jac.want_state_tangents = not self._fb_reuse_ok(np.asarray(xx), _s)
                    J_ = self._exact_jac(xx, step=_s, names=self._names,
                                         boundary_fn=self._boundary_with, sub=_sub, cfg=cfg)
                    # B1: remember where the exact block (raw constraint gradients) is from
                    self._exact_last_x = None if J_ is None else np.array(xx, dtype=np.float64)
                    return J_
            elif cfg.eq_jacobian != "fd":
                raise ValueError(f"eq_jacobian must be 'fd' or 'exact', got {cfg.eq_jacobian!r}")
            speed_kw = {}
            if abort_on:                                             # S-a (opt-in)
                speed_kw.update(abort_doomed=True, abort_gate=self._skip_gate)
            if speedups_on:       # opt-in only: the default call is unchanged
                Delta0 = None
                if cfg.carry_trust_radius and self._trust_carry is not None:
                    xs = ls_kwargs.get("x_scale", 1.0)
                    xs = 1.0 if (xs is None or isinstance(xs, str)) else np.asarray(xs, float)
                    d_default = float(np.linalg.norm(x / xs)) or 1.0   # scipy's initial Delta
                    Delta0 = min(float(self._trust_carry), d_default)
                    # opt-in floor (env PHRPORT_TRUST_FLOOR, absolute in scaled units): a carry that collapsed
                    # in one subproblem (no step accepted) would otherwise end every later one at xtol
                    floor = float(os.environ.get("PHRPORT_TRUST_FLOOR", 0) or 0)
                    if floor > 0:
                        Delta0 = max(Delta0, min(floor, d_default))
                speed_kw.update(
                    skip_coil_on_rejected=bool(cfg.skip_coil_on_rejected and cfg.enable_coils),
                    skip_gate=self._skip_gate, skip_coil_scope=cfg.skip_coil_scope,
                    on_coil_skipped=(lambda xx, _c=ctx: self._on_coil_skipped(_c, xx)),
                    stall_stop=int(cfg.stall_stop), stall_delta_factor=cfg.stall_delta_factor,
                    Delta0=Delta0,
                    use_trf_port=bool(cfg.carry_trust_radius or cfg.stall_stop > 0))
            if self._prof is not None:                               # P1 (opt-in, passive)
                speed_kw["phase_hook"] = self._prof.hook
                self._prof.where = dict(step=int(step), al_iter=int(al_iter))
            oc0 = self._oracle_counts() if self._anchored else None
            if cfg.finite_beta:
                self._fb_set_at(x, step)       # exact plasma field at the subproblem's start point
                speed_kw["j2_grad_hook"] = (lambda xx, J2_, g_, _c=ctx, _s=step: self._fb_hook(_c, xx, J2_, g_, _s))
            ls = least_squares_qss(
                x, self._residuals_fn, self._residuals_many, j2fn, self._sub.n_terms,
                abs_step=cfg.fd_abs_step, rel_step=cfg.fd_rel_step,
                diff_method=cfg.diff_method, n_groups=cfg.fd_ngroups,
                residuals_prefetch=prefetch, iteration_callback=iter_cb,
                jac_callback=jac_cb, log=self.log, keep_jacobians=self.keep_jacobians,
                verbose=cfg.ls_verbose, eq_jac=eq_jac, **speed_kw, **ls_kwargs)
            x = ls.x.copy()
            boundary = self._boundary_with(x)
            res = ls.result
            self.log(f"  LS: {'FAILED -> x0 (' + ls.error + ')' if ls.failed else res.message} "
                     f"nfev={ls.nfev_total} njev={ls.njev} fd_points={ls.n_fd_points} "
                     f"cost={(res.cost if res is not None else float('nan')):.6e}")
            if speedups_on:
                if cfg.carry_trust_radius and ls.max_accepted_step_h:
                    self._trust_carry = float(ls.max_accepted_step_h)
                self.log(f"  [speedups] coil solves skipped={ls.n_coil_skipped} "
                         f"stall_stopped={ls.stall_stopped} Delta0={ls.Delta0} "
                         f"Delta_final={ls.Delta_final} "
                         f"max_accepted_step={ls.max_accepted_step_h}")
            if p0_speed:
                self.log(f"  [phase0-speed] coil solves aborted={ls.n_coil_aborted}"
                         + (f" exact-jac stats={dict(self._exact_jac.stats)}"
                            if self._exact_jac is not None else ""))

            refreshed = False
            refresh_elided = False
            audit = None
            if self._anchored:
                # B2: no warm start depends on a skipped / aborted (rejected) trial
                self._pending_skips = []
            if cfg.enable_coils and self._pending_skips and self._last_coil_x is not None \
                    and np.array_equal(self._last_coil_x, x):
                # PHR refreshes iff a skipped (rejected-trial) solve would have succeeded
                if self._refresh_elidable(x):
                    self.log(f"  [skip-coil] metrics-only refresh elided: the accepted iterate's "
                             f"cold coil solve is what PHR's refresh would recompute.")
                    self._pending_skips = []
                    refresh_elided = True
                    self.counters["refresh_elided"] += 1
                else:
                    self._materialize_pending("refresh decision depends on skipped solves")
            if cfg.enable_coils:
                if self._anchored and cfg.oracle_audit:      # B2: per-outer cold audit
                    refreshed = True
                    if cfg.finite_beta and cfg.fb_reuse_tol > 0:
                        self._fb_set_at(x, step)       # P5: the audit sees the exact plasma field at x
                    audit = self._oracle_audit(ctx, x, step, al_iter)
                elif (self._last_coil_x is None or not np.array_equal(self._last_coil_x, x)
                      or (cfg.finite_beta and self._fb is not None
                          and self._fb.key != (int(step), np.asarray(x, dtype=np.float64).tobytes()))):
                    self.log("  Last coil metrics were from a non-accepted trial point (or a lagged plasma "
                             "field); re-evaluating at the accepted iterate for the dual update.")
                    refreshed = True
                    try:
                        if cfg.finite_beta:
                            self._fb_set_at(x, step)
                        self._compute_J2_and_grad(ctx, x, compute_grad=False)
                    except Exception as e:  # noqa: BLE001  (PHR would abort the job here)
                        self.log(f"  [WARNING] metrics-only refresh failed ({e}); coil dual "
                                 f"update skipped this iteration.")
                        self._last_coil_vals[:] = np.nan
                    self._last_eval_info = {}
            acc_cert = None
            if cfg.enable_coils and self._trial_certify is False:   # S-b: certify accepted coils
                try:
                    acc_cert = self._certify_accepted(x)
                except Exception as e:  # noqa: BLE001
                    acc_cert = dict(error=f"{type(e).__name__}: {e}")
                self.log(f"  [no-certify-trials] accepted point: {acc_cert}")

            # 2. PHR dual update for the equilibrium constraints
            eq_acc = self._cache_get(step, x) if cfg.reuse_eq_cache else None
            if eq_acc is None:
                eq_acc = self._eval_single(boundary, step)
            if not bool(eq_acc.converged):
                self.log(f"  [WARNING] VMEC failed during the PHR AL1 dual update: {eq_acc.message}")
                self.log("  Graceful stop: restoring the last converged surface and "
                         "proceeding to the final coil optimisation.")
                boundary = collections.OrderedDict(last_good)
                x = self._xvec(boundary, self._names)
                aborted = True
                self.outer_log.append(dict(step=step, al_iter=al_iter, aborted=True,
                                           ls_message=(res.message if res is not None else ls.error)))
                fin_state = dict(step=step, al_iter=al_iter, v_prev_eq=v_prev_eq,
                                 v_prev_coil=v_prev_coil, done=fin_state["done"], aborted=True)
                return "abort"
            g_eq = eq_constraint_values(eq_acc, cfg)
            lam_eq, v_eq = update_dual_phr(lam_eq, rho_eq, g_eq, cfg.lam_max)

            # 3. coil dual update
            if cfg.enable_coils:
                if np.all(np.isfinite(self._last_coil_vals)):
                    g_coil = self._last_coil_vals - self._coil_targets
                    lam_coil, v_coil = update_dual_phr(lam_coil, rho_coil, g_coil, cfg.lam_max)
                    coil_ok = True
                else:
                    self.log("  [WARNING] Coil metrics NaN -- skipping coil dual update.")
                    g_coil = np.full(self.n_coil_con, np.nan)
                    v_coil = np.full(self.n_coil_con, np.nan)
                    coil_ok = False
            else:
                coil_ok = False
                g_coil = np.zeros(0)
                v_coil = np.zeros(0)

            # 4. conditioned rho growth
            rho_eq, grew_eq = update_rho_conditioned(
                rho_eq, v_eq, v_prev_eq, theta=cfg.rho_theta, tau=cfg.rho_tau,
                rho_max=cfg.rho_max, v_tol=cfg.v_tol)
            if np.any(grew_eq):
                self.log(f"  rho_eq grew (x{cfg.rho_tau:g}) for: "
                         f"{[self.eq_labels[i] for i in np.where(grew_eq)[0]]}")
            v_prev_eq = v_eq.copy()
            if cfg.enable_coils and coil_ok:
                rho_coil, grew_coil = update_rho_conditioned(
                    rho_coil, v_coil, v_prev_coil, theta=cfg.rho_theta, tau=cfg.rho_tau,
                    rho_max=cfg.rho_max, v_tol=cfg.v_tol)
                if np.any(grew_coil):
                    self.log(f"  rho_coil grew (x{cfg.rho_tau:g}) for: ['Jf']")
                v_prev_coil = v_coil.copy()

            # B1: KKT certificate at the accepted point (reporting; no extra evaluation)
            kkt = None
            if self._kkt_on:
                kkt = _to_jsonable(self._kkt_after_outer(ls, x, eq_acc, g_eq, g_coil, lam_eq,
                                                         lam_coil, self._sub))
                self.kkt_log.append(dict(step=int(step), al_iter=int(al_iter), **kkt))
                if "error" in kkt:
                    self.log(f"  [KKT] certificate unavailable: {kkt['error']}")
                else:
                    sp = kkt["stationarity_phr"]
                    self.log(f"  [KKT] feasibility={kkt['feasibility']:.3e} "
                             f"complementarity={kkt['complementarity']:.3e} "
                             f"stationarity={kkt['stationarity']:.3e} (inf {kkt['stationarity_inf']:.3e}"
                             f", PHR lambda {('%.3e' % sp) if sp is not None else 'n/a'}) "
                             f"|grad QS|={kkt['grad_norm']:.3e} active={kkt['active']}"
                             + (f" (by lambda>0 only: {kkt['active_phr_only']})"
                                if kkt.get("active_phr_only") else "")
                             + (f" missing={kkt['missing']}" if kkt["missing"] else "")
                             + f" [{kkt['source_eq']}]")

            # 5. diagnostics + history
            qs_rms = float(np.sqrt(np.mean(np.asarray(eq_acc.qs_residuals) ** 2)))
            self.log(f"  PHR iter {al_iter+1}/{base_budget}:  QS rms={qs_rms:.2e}")
            labels = self.eq_labels + (self.coil_names if cfg.enable_coils else [])
            gs = list(g_eq) + (list(g_coil) if cfg.enable_coils else [])
            lams = list(lam_eq) + (list(lam_coil) if cfg.enable_coils else [])
            rhos = list(rho_eq) + (list(rho_coil) if cfg.enable_coils else [])
            for lab, gi, li, ri in zip(labels, gs, lams, rhos):
                if not np.isfinite(gi):
                    self.log(f"    {lab:12s}  g=NaN (update skipped)")
                    continue
                vi = max(0.0, gi)
                status = "VIOLATED" if vi > 0 else ("active*" if li > 0 else "inactive")
                self.log(f"    {lab:12s}  g={gi:+.3e}  v={vi:.3e}  lam={li:.3e}  rho={ri:.4g}  {status}")
            self.phr_history.append({
                "step": int(step), "al_iter": int(al_iter), "max_mode": int(max_mode),
                "g_eq": g_eq.tolist(), "v_eq": v_eq.tolist(),
                "lam_eq": lam_eq.tolist(), "rho_eq": rho_eq.tolist(),
                "g_coil": (g_coil.tolist() if cfg.enable_coils else []),
                "lam_coil": lam_coil.tolist(), "rho_coil": rho_coil.tolist(),
            })
            if kkt is not None:
                self.phr_history[-1]["kkt"] = kkt
            last_good = collections.OrderedDict(boundary)
            last_good_step = step
            self.outer_log.append(dict(
                step=step, al_iter=al_iter, max_mode=max_mode, n_free=len(x),
                ls_status=(int(res.status) if res is not None else None),
                ls_message=(res.message if res is not None else ls.error),
                ls_failed=ls.failed, nfev=ls.nfev_total, njev=ls.njev,
                n_fd_points=ls.n_fd_points,
                n_accepted=(ls.njev - 1 if ls.njev > 0 else 0),
                n_rejected=(ls.nfev_total - ls.njev if ls.njev > 0 else ls.nfev_total),
                cost=(float(res.cost) if res is not None else None),
                refreshed=refreshed, qs_total=float(eq_acc.qs_total),
                maxj_raw=float(getattr(eq_acc, "maxj", float("nan"))), maxj_offset=float(_MAXJ_OFFSET[0]),
                coil_target_scaled=float(self._coil_targets[0]),
                c0_scaled=float(self._last_coil_vals[0]),
                wall=time.time() - t_outer, time_f=ls.time_f, time_jac=ls.time_jac))
            if speedups_on:
                self.outer_log[-1].update(
                    n_coil_skipped=ls.n_coil_skipped, refresh_elided=refresh_elided,
                    stall_stopped=ls.stall_stopped, Delta0=ls.Delta0,
                    Delta_final=ls.Delta_final, max_accepted_step_h=ls.max_accepted_step_h)
            if p0_speed:
                self.outer_log[-1].update(
                    n_coil_aborted=ls.n_coil_aborted, refresh_elided=refresh_elided,
                    accepted_coil_cert=acc_cert,
                    exact_jac_stats=(dict(self._exact_jac.stats)
                                     if self._exact_jac is not None else None))
            if kkt is not None:
                self.outer_log[-1]["kkt"] = kkt
            if self._anchored:            # B2: this outer's oracle decisions and its audit
                oc1 = self._oracle_counts()
                acc = self._coil_acc or {}
                self.outer_log[-1]["oracle"] = dict(
                    counts={k: v - oc0.get(k, 0) for k, v in oc1.items() if v != oc0.get(k, 0)},
                    audit=audit, anchor_seq=acc.get("seq"), anchor_source=acc.get("source"),
                    anchor_c0=acc.get("c0"), anchor_eval_id=acc.get("eval_id"))
            if rec_extra:                     # A4 sweep: point index / target in the records
                self.phr_history[-1].update(rec_extra)
                self.outer_log[-1].update(rec_extra)
            fin_state = dict(step=step, al_iter=al_iter + 1, v_prev_eq=v_prev_eq,
                             v_prev_coil=v_prev_coil, done=fin_state["done"] + 1)
            self._write_histories()

            # 6. early exit
            v_all = np.concatenate([v_eq, v_coil]) if cfg.enable_coils else v_eq
            phr_exit = bool(np.all(np.isfinite(v_all)) and float(np.max(v_all)) < cfg.v_tol)
            if kkt_exit_mode:          # B1 (opt-in), final mode only
                have = kkt is not None and "error" not in kkt
                if have and kkt["feasibility"] < cfg.v_tol \
                        and kkt["stationarity"] < cfg.kkt_stationarity_tol:
                    self.log(f"  [KKT exit] feasibility {kkt['feasibility']:.3e} < {cfg.v_tol:.1e} "
                             f"and stationarity {kkt['stationarity']:.3e} < "
                             f"{cfg.kkt_stationarity_tol:g} -- exiting the final mode.")
                    self.outer_log[-1].update(early_exit=True, kkt_exit=True)
                    return "exit"
                if not have:
                    # no certificate: PHR's rule, no extra outers
                    if phr_exit or al_iter + 1 >= base_budget:
                        self.log("  [KKT exit] no certificate at this outer: PHR exit rule"
                                 + (" (all violations below v_tol)." if phr_exit else
                                    " (outer budget reached; no extra outers)."))
                        if phr_exit:
                            self.outer_log[-1]["early_exit"] = True
                        return "exit"
                else:
                    self.outer_log[-1]["kkt_continue"] = True
                    if al_iter + 1 < n_budget:
                        self.log(f"  [KKT exit] not met (feasibility {kkt['feasibility']:.3e}, "
                                 f"stationarity {kkt['stationarity']:.3e})"
                                 + (" although all violations are below v_tol" if phr_exit else "")
                                 + " -- continuing with lambda kept"
                                 + (" (extra outer)." if al_iter + 1 >= base_budget else "."))
                    else:
                        self.log(f"  [KKT exit] not met after {n_budget - base_budget} "
                                 f"extra outers; leaving the final mode.")
            elif phr_exit and cfg.full_outer_budget:
                self.log(f"  All inequality violations below {cfg.v_tol:.1e} -- "
                         f"continuing (full outer budget).")
            elif phr_exit:
                self.log(f"  All inequality violations below {cfg.v_tol:.1e} -- "
                         f"exiting PHR AL1 loop early.")
                self.outer_log[-1]["early_exit"] = True
                return "exit"
            return None
        finally:
            st.x, st.boundary = x, boundary
            st.lam_eq, st.rho_eq, st.lam_coil, st.rho_coil = lam_eq, rho_eq, lam_coil, rho_coil
            st.v_prev_eq, st.v_prev_coil = v_prev_eq, v_prev_coil
            st.last_good, st.last_good_step, st.aborted = last_good, last_good_step, aborted
            st.fin_state = fin_state


    # ---------------------------------------------------------------- run
    def run(self):
        cfg = self.cfg
        wall0 = time.time()
        self.log("")
        self.log(SEP)
        self.log("  QSS QA Optimization -- PHR inequality AL1 [phrport: VMEX + XCoil]")
        self.log(SEP)
        self.log(f"  config: {json.dumps(_to_jsonable(asdict(cfg)))}")
        self._write_provenance("started")
        boundary = collections.OrderedDict(self.seed)

        # ---- initial metrics (PHR: vmec.run() at the input resolution)
        eq0 = self._eval_single(boundary, cfg.initial_eval_step)
        if eq0.converged:
            g0 = eq_constraint_values(eq0, cfg)
            area0, vol0 = surface_area_volume(boundary, self.nfp)
            init = {
                "alm_mode": "phr", "aspect_ratio": float(eq0.aspect),
                "mean_iota": float(eq0.mean_iota), "iota_axis": float(eq0.iota_axis),
                "iota_edge": float(eq0.iota_edge), "qs_residual_sum_sq": float(eq0.qs_total),
                "g_eq": {lab: float(g0[i]) for i, lab in enumerate(self.eq_labels)},
                "vacuum_well": float(eq0.well), "beta_total": 0.0, "finite_beta": False,
                "pres_scale": 0.0, "area": area0, "volume": vol0, "nfp": self.nfp,
                "stellsym": self.stellsym, "vmec_input": self.vmec_input,
            }
            _json_dump(init, os.path.join(self.out_dir, "initial_metrics.json"))
            self.log(f"  Initial: aspect={eq0.aspect:.6f} mean_iota={eq0.mean_iota:.6f} "
                     f"QS={eq0.qs_total:.6e} g={np.array2string(g0, precision=6)}")
            self._maybe_write_wout(boundary, cfg.initial_eval_step, "wout_initial.nc")
        elif self.resume_from or self.init_surface:
            # lowA_ncsx: a resumed stage of a profile ramp (e.g. beta) may use a deck whose own seed has no equilibrium;
            # the seed solve only feeds initial_metrics.json, and the run continues from the checkpoint boundary.
            self.log(f"  [WARN] seed equilibrium of this deck failed ({eq0.message}); continuing from the resume / "
                     f"init surface (no initial_metrics.json)")
        else:
            # PHR: the initial vmec.run() would raise and abort the job.
            self.log(f"  [ERROR] initial equilibrium failed: {eq0.message}")
            raise RuntimeError(f"initial equilibrium failed: {eq0.message}")

        # ---- PHR state
        lam_eq = np.zeros(self.n_eq_con)
        lam_coil = np.zeros(self.n_coil_con)
        rho_eq = np.full(self.n_eq_con, cfg.rho_init)
        rho_coil = np.full(self.n_coil_con, cfg.rho_init)
        last_good = collections.OrderedDict(boundary)
        last_good_step = cfg.initial_eval_step
        aborted = False
        resume_step, resume_al_iter = 0, 0
        is_resume = bool(self.resume_from)
        if self.resume_from or self.init_surface:
            src = self.resume_from or self.init_surface
            src_dir = src if os.path.isdir(src) else None
            cp = os.path.join(src_dir, "al1_checkpoint_latest.json") if src_dir else src
            with open(cp) as f:
                ck = json.load(f)
            if not is_resume and src_dir is not None:
                fin = os.path.join(src_dir, "intermediate_state.json")
                if os.path.exists(fin):
                    with open(fin) as f:
                        fin_dofs = json.load(f)["surface_dofs"]
                    if len(fin_dofs) == len(ck["surface_dofs"]):
                        ck["surface_dofs"] = fin_dofs
            resume_step = int(ck["step"])
            self._resume_ck = ck if is_resume else None
            names = self.free_dof_names(resume_step)
            sd = np.asarray(ck["surface_dofs"], dtype=np.float64)
            if len(names) != len(sd):
                raise RuntimeError(f"warm-start DOF mismatch: checkpoint has {len(sd)} free DOFs, "
                                   f"free_dof_names({resume_step}) gives {len(names)}.")
            for k, v in zip(names, sd):
                boundary[k] = float(v)
            if is_resume:
                resume_al_iter = int(ck["al_iter"])
                lam_eq, rho_eq = align_eq_multipliers(ck, self.eq_labels, cfg.rho_init, self.log)
                lam_coil = np.asarray(ck["lam_coil"], dtype=np.float64)
                rho_coil = np.asarray(ck["rho_coil"], dtype=np.float64)
            last_good = collections.OrderedDict(boundary)
            last_good_step = resume_step
            self.log(f"  {'RESUME' if is_resume else 'INIT-SURFACE'} from {cp}: step={resume_step} "
                     f"al_iter={resume_al_iter} lam_eq={lam_eq} rho_eq={rho_eq} "
                     f"lam_coil={lam_coil} rho_coil={rho_coil}")

        # ---- phase 0 (B0): --start-step / --resume-keep-target (opt-in)
        first_step = resume_step
        keep_first = False             # skip the step-top target re-capture / lam_c reset once
        v_prev_resume = (None, None)
        if cfg.start_step is not None:
            first_step = int(cfg.start_step)
            if not 0 <= first_step < max(1, cfg.n_continuation_steps):
                raise ValueError(f"start_step {first_step} outside [0, {cfg.n_continuation_steps})")
            self.log(f"  [start-step] continuation starts at step {first_step} (outer 0) with "
                     f"lam/rho from {'the checkpoint' if is_resume else 'the defaults'}.")
        if cfg.resume_keep_target:
            if not is_resume:
                raise ValueError("resume_keep_target needs resume_from")
            if "coil_targets_scaled" not in ck or "flux_target_captured" not in ck:
                raise ValueError(f"{cp} has no coil_targets_scaled / flux_target_captured "
                                 f"(PHR-format checkpoint): cannot keep the flux target")
            keep_first = True
            explicit = cfg.flux_target_frac <= 0.0 and cfg.flux_phr_target > 0.0
            if explicit:
                self.log(f"  [resume-keep-target] explicit absolute target flux_phr_target="
                         f"{cfg.flux_phr_target:.6e} kept (frac 0); lam_c / rho_c from the checkpoint.")
            else:
                self._coil_targets[:] = np.asarray(ck["coil_targets_scaled"], dtype=np.float64)
                self._flux_captured[0] = bool(ck["flux_target_captured"])
            if cfg.start_step is None and resume_al_iter > 0:
                if ck.get("v_prev_eq") is not None or ck.get("final"):
                    v_prev_resume = (None if ck.get("v_prev_eq") is None
                                     else np.asarray(ck["v_prev_eq"], dtype=np.float64),
                                     None if ck.get("v_prev_coil") is None
                                     else np.asarray(ck["v_prev_coil"], dtype=np.float64))
                    why = "checkpoint"
                else:
                    ve, vc, why = self._v_prev_from_history(cp, resume_step, resume_al_iter)
                    v_prev_resume = (ve, vc)
                self.log(f"  [resume-keep-target] v_prev_eq={v_prev_resume[0]} "
                         f"v_prev_coil={v_prev_resume[1]} (from {why})")
            self.log(f"  [resume-keep-target] coil target (scaled)={self._coil_targets} "
                     f"captured={bool(self._flux_captured[0])}; no re-capture / lam_c reset at "
                     f"step {first_step + 1}.")

        self.log(f"  Initial lam_eq={lam_eq} lam_coil={lam_coil} rho_eq={rho_eq} rho_coil={rho_coil}")
        flags = self._loop_flags()     # (speedups_on, p0_speed, abort_on)
        n_steps_run = 0 if (cfg.skip_surface_opt and not (self.resume_from or self.init_surface)) \
            else cfg.n_continuation_steps
        step = None
        al_iter = None
        # loop state (A4 refactor: the outer body is _outer_iteration, shared with sweep())
        st = _LoopState(x=None, boundary=boundary, lam_eq=lam_eq, rho_eq=rho_eq, lam_coil=lam_coil,
                        rho_coil=rho_coil, v_prev_eq=None, v_prev_coil=None, last_good=last_good,
                        last_good_step=last_good_step, aborted=aborted,
                        fin_state=None)       # B0: fin_state = where a resume of this run continues
        for step in range(first_step, n_steps_run):
            t_step = time.time()
            max_mode = step + 1
            st.v_prev_eq = None
            st.v_prev_coil = None
            keep_now = keep_first and step == first_step
            if keep_now:
                st.v_prev_eq, st.v_prev_coil = v_prev_resume
            elif cfg.flux_target_frac > 0.0:
                self._flux_captured[0] = False
                if cfg.carry_flux_multiplier and max_mode >= cfg.carry_flux_from_mode:     # B5 (opt-in): lam_c / rho_c carried into the mode
                    self.log(f"  [carry-flux-multiplier] lam_coil={st.lam_coil} rho_coil={st.rho_coil} kept "
                             f"at the start of step {step + 1} (flux target re-captured).")
                else:
                    st.lam_coil[0] = 0.0
                    st.rho_coil[0] = cfg.rho_init
            self._step = step
            _MAXJ_STEP[0] = step
            if cfg.maxj and cfg.maxj_from_step > 0:
                self.log(f"  [maxj-from-step] max-J row {'active' if maxj_active(cfg) else 'inactive (g = -1)'} "
                         f"at step {step + 1} (active from step {cfg.maxj_from_step + 1})")
            if cfg.bmaxsp and cfg.bmaxsp_from_step > 0:
                self.log(f"  [bmaxsp-from-step] B_max spread row {'active' if bmaxsp_active(cfg) else 'inactive (g = -1)'} "
                         f"at step {step + 1} (active from step {cfg.bmaxsp_from_step + 1})")
            self._names = list(self.free_dof_names(step))
            self._base = collections.OrderedDict(st.boundary)
            st.x = self._xvec(st.boundary, self._names)
            # ordering rows: the reference axis is frozen at the start of every step (a resume inside the same step
            # reuses the checkpoint's axis)
            same_step = bool(is_resume and step == resume_step and cfg.start_step is None)
            self._freeze_ordering_axis(step, st.boundary, getattr(self, "_resume_ck", None) if same_step else None)
            self._eq_cache.clear()
            self.log("")
            self.log(SEP)
            self.log(f"  Continuation step {step+1}/{cfg.n_continuation_steps} -- max_mode={max_mode}, "
                     f"mpol=ntor={3 + step}  ({len(st.x)} free surface DOFs)")
            self.log(f"  lam_eq={st.lam_eq}, lam_coil={st.lam_coil}")
            self.log(f"  rho_eq={st.rho_eq}, rho_coil={st.rho_coil}")
            self.log(SEP)
            # opt-in (env PHRPORT_TRUST_RESET_STEP=1): a new continuation step frees new modes, so the trust
            # radius carried from the previous step's last accepted step does not apply; start from scipy's
            if os.environ.get("PHRPORT_TRUST_RESET_STEP") == "1" and self._trust_carry is not None:
                self.log(f"  [speedups] trust radius carry reset at the new step (was {self._trust_carry:.3e})")
                self._trust_carry = None
            if cfg.maxj and cfg.maxj_switch_delta > 0.0 and step >= cfg.maxj_from_step:
                self._trust_carry = float(cfg.maxj_switch_delta)
                self.log(f"  [maxj-switch-delta] trust radius {self._trust_carry:.3e} at the start of step {step + 1}")

            al_start = resume_al_iter if (is_resume and step == resume_step
                                          and cfg.start_step is None) else 0
            # B1 (opt-in): in the final mode the outer budget grows by kkt_extra_outers, used
            # only while the KKT exit test fails (see "6. early exit")
            kkt_exit_mode = bool(cfg.final_mode_kkt_exit and step == cfg.n_continuation_steps - 1)
            n_budget = cfg.al_outer_iter + (max(0, int(cfg.kkt_extra_outers)) if kkt_exit_mode else 0)
            st.fin_state = dict(step=step, al_iter=al_start, v_prev_eq=st.v_prev_eq,
                                v_prev_coil=st.v_prev_coil, done=0)
            for al_iter in range(al_start, n_budget):
                if self._outer_iteration(st, step, al_iter, max_mode, flags=flags,
                                         kkt_exit_mode=kkt_exit_mode, n_budget=n_budget,
                                         base_budget=cfg.al_outer_iter) is not None:
                    break
            if st.aborted:
                self.log(f"  Continuation halted at step {step+1}/{cfg.n_continuation_steps} "
                         f"(max_mode={max_mode}) after a VMEC convergence failure.")
                break
            self.log(f"  Completed max_mode = {max_mode}.  [step wall={time.time() - t_step:.1f}s, "
                     f"elapsed={time.time() - wall0:.1f}s]")
        boundary, lam_eq, rho_eq, lam_coil, rho_coil = (st.boundary, st.lam_eq, st.rho_eq, st.lam_coil,
                                                        st.rho_coil)
        aborted, last_good_step, fin_state = st.aborted, st.last_good_step, st.fin_state

        # ---- intermediate state
        x_final_free = (self._xvec(boundary, self._names) if self._names is not None
                        else np.zeros(0))
        final_ck = None
        if fin_state is not None:      # B0 (reporting): the post-loop state as a checkpoint
            try:
                final_ck = self._write_final_checkpoint(fin_state, lam_eq, rho_eq, lam_coil, rho_coil,
                                                        x_final_free)
            except Exception as e:  # noqa: BLE001  (reporting never stops a run)
                self.log(f"  [WARNING] al1_checkpoint_final.json failed: {type(e).__name__}: {e}")
        _json_dump({
            "alm_mode": "phr", "surface_dofs": x_final_free.tolist(),
            "vmec_dofs": x_final_free.tolist(), "lam_eq": lam_eq.tolist(),
            "lam_coil": lam_coil.tolist(), "rho_eq": rho_eq.tolist(), "rho_coil": rho_coil.tolist(),
            "boundary": dict(boundary),
        }, os.path.join(self.out_dir, "intermediate_state.json"))
        self._write_histories()

        if aborted:
            final_step = last_good_step
        elif step is not None:
            final_step = step
        else:
            final_step = cfg.initial_eval_step
        summary = dict(boundary=dict(boundary), lam_eq=lam_eq, rho_eq=rho_eq,
                       lam_coil=lam_coil, rho_coil=rho_coil, aborted=aborted,
                       final_step=final_step, phr_history=self.phr_history,
                       outer_log=self.outer_log, final_metrics=None)
        if final_ck is not None:
            summary["final_checkpoint"] = final_ck

        # ---- final VMEC run
        eqf = self._eval_single(boundary, final_step)
        if not bool(eqf.converged):
            self.log(f"  [WARNING] VMEC failed at the final surface: {eqf.message}; "
                     f"skipping the final coil optimisation and metrics.")
            summary["final_eq_failed"] = True
            self._finish(wall0)
            return summary
        self._maybe_write_wout(boundary, final_step, "wout_optimized.nc")

        # ---- final coil stage: cold (reinit) solve, same settings
        self.log("")
        self.log(SEP)
        self.log("  Final coil optimisation (for metrics)")
        self.log(SEP)
        coil_final = None
        if True:   # PHR runs the final coil stage regardless of --no-coils
            t0 = time.time()
            if cfg.final_coil_stage:       # R1 / G4 (opt-in); None -> the cold solve below
                coil_final = self._run_final_coil_stage(boundary)
            if coil_final is None:
                try:
                    coil_final = self.coil_solver.solve(boundary, warm=None)
                except Exception as e:  # noqa: BLE001
                    self.log(f"  [WARNING] final coil solve failed: {e}")
            self.timers["coil_final"] += time.time() - t0
        g_final = eq_constraint_values(eqf, cfg)
        area, vol = surface_area_volume(boundary, self.nfp)
        cinfo = (getattr(coil_final, "info", None) or {}) if coil_final is not None else {}

        def ci(*keys):
            for k in keys:
                v = _info_lookup(cinfo, k)
                if v is not None:
                    return _to_jsonable(v)
            return None
        lengths = ci("individual_lengths", "lengths", "cert.L")
        msc = ci("individual_msc", "msc", "cert.msc")
        wall_total = time.time() - wall0
        from . import equilibrium as _Eqm   # qh_funk: report the profile's helicity
        final_metrics = {
            "surface": {
                "aspect_ratio": float(eqf.aspect), "mean_iota": float(eqf.mean_iota),
                "iota_axis": float(eqf.iota_axis), "iota_edge": float(eqf.iota_edge),
                "qs_residual_sum_sq": float(eqf.qs_total),
                "g_eq": {lab: float(g_final[i]) for i, lab in enumerate(self.eq_labels)},
                "vacuum_well": float(eqf.well), "beta_total": 0.0,
                "mercier_min_interior": None, "mercier_unstable_frac": None,
                "mercier_soft_min_sDMerc": _to_jsonable(getattr(eqf, "merc", None)),
                "area": area, "volume": vol, "n_dofs_free": 0, "nfp": self.nfp,
            },
            "coils": {
                "squared_flux_Jf": (float(coil_final.c0_local) if coil_final is not None else None),
                "squared_flux_F": (float(getattr(coil_final, "flux_F", float("nan")))
                                   if coil_final is not None else None),
                "ok": (bool(coil_final.ok) if coil_final is not None else None),
                "certified": (bool(getattr(coil_final, "certified", False))
                              if coil_final is not None else None),
                "linked": (bool(getattr(coil_final, "linked", False))
                           if coil_final is not None else None),
                "avg_Bn_over_B": ci("avg_Bn_over_B", "avg_bn_over_b", "mean_bn_unweighted"),
                "avg_Bn_over_B_area_weighted": ci("mean_bn_area"),
                "max_Bn_over_B": ci("max_Bn_over_B", "max_bn_over_b", "max_bn"),
                "total_length": (ci("total_length") if ci("total_length") is not None
                                 else (float(np.sum(lengths)) if lengths is not None else None)),
                "individual_lengths": lengths,
                "cc_distance_penalty": ci("cc_distance_penalty"),
                "min_cc_distance": ci("min_cc_distance", "min_cc_dist", "min_cc", "cert.d_cc"),
                "cs_distance_penalty": ci("cs_distance_penalty"),
                "min_cs_distance": ci(*_MIN_CS_KEYS),
                "max_curvature": ci("max_curvature", "cert.max_kappa"),
                "individual_max_curvatures": ci("individual_max_curvatures"),
                "mean_squared_curvature_avg": (ci("mean_squared_curvature_avg") if
                                               ci("mean_squared_curvature_avg") is not None else
                                               (float(np.mean(msc)) if msc is not None else None)),
                "individual_msc": msc,
                "linking_number": ci("linking_number"),
                "certified_violation": ci("v", "cert.v"),
                "currents": (np.asarray(coil_final.currents).tolist()
                             if coil_final is not None and getattr(coil_final, "currents", None) is not None
                             else None),
            },
            "optimization": {
                "alm_mode": "phr", "hessian_method": "envelope(port)",
                "n_continuation_steps": cfg.n_continuation_steps,
                "al1_outer_iters": cfg.al_outer_iter, "rho_init": cfg.rho_init,
                "rho_tau": cfg.rho_tau, "rho_theta": cfg.rho_theta, "rho_max": cfg.rho_max,
                "v_tol": cfg.v_tol, "lam_max": cfg.lam_max,
                "rho_eq_final": rho_eq.tolist(), "rho_coil_final": rho_coil.tolist(),
                "lam_eq_final": lam_eq.tolist(), "lam_coil_final": lam_coil.tolist(),
                "flux_threshold": 0.0, "flux_fit_enabled": True,
                "cs_flux_safe_min": cfg.cs_flux_safe_min, "flux_phantom_eps": cfg.flux_phantom_eps,
                "flux_metric_scale": cfg.flux_metric_scale, "flux_phr_target": cfg.flux_phr_target,
                "flux_target_frac": cfg.flux_target_frac,
                "coil_scales": self._coil_scales.tolist(),
                "coil_targets_scaled": self._coil_targets.tolist(),
                "aspect_target": cfg.aspect_target, "iota_mean_target": cfg.iota_mean_target,
                "elongation_threshold": cfg.elongation_threshold,
                "vacuum_well_target": cfg.well_threshold, "finite_beta": False,
                "pres_scale": 0.0, "mercier_margin": None,
                "aborted": aborted, "final_step": final_step,
                "wall_time_total_s": wall_total,
                "coil_solve_wall_s": float(self.timers["coil_solve"]),
                "port_timers_s": dict(self.timers), "port_counters": dict(self.counters),
            },
            "qs_parameters": {"helicity_m": _Eqm.QS_HELICITY_M, "helicity_n": _Eqm.QS_HELICITY_N,
                              "surfaces": np.arange(0, 1.01, 0.1).tolist()},
        }
        if cfg.phase0_reporting or cfg.final_coil_stage or cfg.final_mode_kkt_exit:
            try:
                self._final_metrics_phase0(final_metrics, coil_final, cinfo, boundary, lam_eq)
            except Exception as e:  # noqa: BLE001  (reporting never stops a run)
                self.log(f"  [WARNING] phase-0 final metrics failed: {type(e).__name__}: {e}")
        if cfg.coil_oracle != "phr" or cfg.carry_flux_multiplier:      # phase 2 (opt-in only)
            final_metrics["optimization"]["phase2"] = dict(
                coil_oracle=cfg.coil_oracle, oracle_d_inherit=cfg.oracle_d_inherit,
                oracle_d_reset=cfg.oracle_d_reset, oracle_k_max=cfg.oracle_k_max,
                oracle_kkt_factor=cfg.oracle_kkt_factor, oracle_kkt_floor=cfg.oracle_kkt_floor,
                oracle_audit=cfg.oracle_audit, oracle_reset_guard=cfg.oracle_reset_guard,
                carry_flux_multiplier=cfg.carry_flux_multiplier, oracle_counters=self._oracle_counts())
            if cfg.oracle_d_floor > 0.0:
                final_metrics["optimization"]["phase2"].update(
                    oracle_d_floor=cfg.oracle_d_floor, oracle_floor_rule=cfg.oracle_floor_rule,
                    oracle_floor_k_max=cfg.oracle_floor_k_max)
        _json_dump(_to_jsonable(final_metrics), os.path.join(self.out_dir, "final_metrics.json"))
        if coil_final is not None and getattr(coil_final, "x", None) is not None:
            np.save(os.path.join(self.out_dir, "final_coils_x.npy"), np.asarray(coil_final.x))
        self.log(f"  Final: QS={eqf.qs_total:.6e} aspect={eqf.aspect:.6f} iota={eqf.mean_iota:.6f} "
                 f"c0_local={final_metrics['coils']['squared_flux_Jf']}")
        summary["final_metrics"] = final_metrics
        summary["coil_final"] = coil_final
        self._finish(wall0)
        return summary

    # ------------------------------------------------- roadmap A4: front sweep
    def _load_sweep_seed(self, src):
        """A4 seed -> (ck, path, meta).  src: a PHR checkpoint file; a front_point_*.json (its
        'checkpoint' block); or a run directory: al1_checkpoint_final.json (B0), else
        al1_checkpoint_latest.json with the post-loop surface and multipliers of
        intermediate_state.json (pre-phase-0 runs; the latest checkpoint is the state BEFORE the last
        outer).  meta = the sweep fields of a sweep checkpoint (k, jf0, ratio, done)."""
        note = None
        if os.path.isdir(src):
            fin = os.path.join(src, "al1_checkpoint_final.json")
            fronts = sorted(glob.glob(os.path.join(src, "al1_checkpoint_front[0-9][0-9].json")))
            ck = None
            if os.path.exists(fin):
                path = fin
                with open(path) as f:
                    ck = json.load(f)
                if fronts and "sweep_k" not in ck:
                    # a sweep directory written before the final checkpoint carried the sweep fields:
                    # the last end-of-point checkpoint (k, Jf0, done) seeds the continuation
                    ck = None
            if ck is None and fronts:
                path = fronts[-1]
                with open(path) as f:
                    ck = json.load(f)
                note = (f"sweep directory: last end-of-point checkpoint {os.path.basename(path)}"
                        + (" (al1_checkpoint_final.json has no sweep fields)" if os.path.exists(fin) else ""))
            elif ck is None:
                path = os.path.join(src, "al1_checkpoint_latest.json")
                with open(path) as f:
                    ck = json.load(f)
                ist = os.path.join(src, "intermediate_state.json")
                if os.path.exists(ist):
                    with open(ist) as f:
                        im = json.load(f)
                    if len(im.get("surface_dofs", [])) == len(ck["surface_dofs"]):
                        for key in ("surface_dofs", "vmec_dofs", "lam_eq", "rho_eq", "lam_coil", "rho_coil"):
                            if key in im:
                                ck[key] = im[key]
                        note = f"surface / multipliers from {ist} (post-loop state)"
                    else:
                        note = f"{ist} has another DOF count: latest checkpoint used as is"
        else:
            path = src
            with open(path) as f:
                ck = json.load(f)
            if ck.get("kind") == "front_point":
                ck = ck["checkpoint"]
                note = "checkpoint block of a front point"
        return ck, path, _sweep_meta(ck), note

    def _sweep_eq(self, x):
        """Equilibrium at x for the sweep's bookkeeping (eq cache first; cached afterwards)."""
        eq = self._cache_get(self._step, x) if self.cfg.reuse_eq_cache else None
        if eq is None:
            eq = self._eval_single(self._boundary_with(x), self._step)
            if self.cfg.reuse_eq_cache:
                self._cache_put(self._step, x, eq)
        return eq

    def _next_eval_id(self):
        return (self._eval_id_base + self._eval_x_new) if self._eval_x is not None else None

    def _sweep_checkpoint(self, st, step, k, eps, jf0, ratio, done):
        """A4: the state at the end of point k in PHR checkpoint format (+ the sweep fields); resumable
        with --resume-from (run) and as a sweep seed (continues with k + 1 when done)."""
        return {"alm_mode": "phr", "step": int(step), "al_iter": 0, "max_mode": int(step) + 1,
                "rho_eq": np.asarray(st.rho_eq).tolist(), "rho_coil": np.asarray(st.rho_coil).tolist(),
                "lam_eq": np.asarray(st.lam_eq).tolist(), "lam_coil": np.asarray(st.lam_coil).tolist(),
                "surface_dofs": np.asarray(st.x).tolist(), "vmec_dofs": np.asarray(st.x).tolist(),
                "vmec_input": self.vmec_input, "surface_dof_names": list(self._names),
                "coil_targets_scaled": self._coil_targets.tolist(), "flux_target_captured": True,
                "sweep_k": int(k), "sweep_eps": float(eps), "sweep_jf0": float(jf0),
                "sweep_ratio": float(ratio), "sweep_point_done": bool(done)}

    def _sweep_front_point(self, scfg, st, *, k, eps, jf0, ratio, step, is_seed, t_pt, wall0, kkt_n0,
                           ph_n0, ol_n0, eid0, rho_start, rho_rule, seed_info):
        """A4: write front_point_{k:02d}.json (front_point_seed.json for the seed): boundary, QS at the
        PHR resolution, coils of the final coil stage (polished; R1), the point's last KKT certificate,
        multipliers, target status and wall times; plus the coil export, wout (when a writer is
        injected) and the end-of-point checkpoint al1_checkpoint_front{k:02d}.json."""
        cfg = self.cfg
        S = float(self._coil_scales[0])
        x, boundary = np.asarray(st.x, dtype=np.float64), st.boundary
        stem = "front_point_seed" if is_seed else f"front_point_{k:02d}"
        eq = self._sweep_eq(x)
        g_eq = eq_constraint_values(eq, cfg) if bool(eq.converged) else np.full(self.n_eq_con, np.nan)
        jf_loop = (float(self._last_coil_vals[0]) / S
                   if (self._last_coil_x is not None and np.array_equal(self._last_coil_x, x)
                       and np.isfinite(self._last_coil_vals[0])) else None)
        # coils: R1 final coil stage (polished), else a cold solve
        t0 = time.time()
        coils = self._run_final_coil_stage(boundary) if scfg.polish else None
        # length front: also polish warm from the coupled solve's own coils; keep the lower c0
        if getattr(self, "_coil_res", None) is not None:
            try:
                cw = self.coil_solver.solve(boundary, warm=self._coil_res)
                c_w = float(getattr(cw, "c0_local", np.nan))
                c_f = float(getattr(coils, "c0_local", np.nan)) if coils is not None else np.inf
                self.log(f"  [sweep-point] warm polish from coupled-solve coils: c0 {c_w:.6e} (final stage {c_f:.6e})")
                if np.isfinite(c_w) and bool(getattr(cw, "ok", True)) and not (c_w >= c_f):
                    coils = cw
            except Exception as e:  # noqa: BLE001
                self.log(f"  [WARNING] warm polish failed: {e}")
        if coils is None:
            try:
                coils = self.coil_solver.solve(boundary, warm=None)
            except Exception as e:  # noqa: BLE001
                self.log(f"  [WARNING] sweep point coil solve failed: {e}")
                coils = None
        t_coil = time.time() - t0
        self.timers["sweep_coil_stage"] += t_coil
        ci = (getattr(coils, "info", None) or {}) if coils is not None else {}
        fs = ci.get("final_stage") or {}
        fu = fs.get("flux_unfolded") or {}

        def gi(key):
            v = _info_lookup(ci, key)
            return None if v is None else _to_jsonable(v)
        c0 = float(getattr(coils, "c0_local", np.nan)) if coils is not None else float("nan")
        export = None
        if coils is not None and np.isfinite(c0):
            try:
                export = export_coils(os.path.join(self.out_dir, stem + "_coils"), self.coil_solver, coils,
                                      boundary, source=("final_coil_stage" if fs else "cold solve"),
                                      extra=dict(sweep_k=int(k), sweep_eps=float(eps)))
                export = os.path.basename(export) if export else None
            except Exception as e:  # noqa: BLE001
                self.log(f"  [WARNING] coil export failed: {e}")
        self._maybe_write_wout(boundary, step, f"wout_{stem}.nc")
        hist = self.phr_history[ph_n0:]
        olog = self.outer_log[ol_n0:]
        g_coil = (hist[-1]["g_coil"][0] if hist and hist[-1].get("g_coil") else None)
        tol = cfg.v_tol if scfg.infeasible_tol is None else float(scfg.infeasible_tol)
        target_met = (None if is_seed else
                      bool(g_coil is not None and np.isfinite(g_coil) and g_coil <= tol))
        kkts = self.kkt_log[kkt_n0:]
        kkt = kkts[-1] if kkts else None
        ck = self._sweep_checkpoint(st, step, k, eps, jf0, ratio, done=True)
        rec = dict(
            kind="front_point", k=int(k), is_seed=bool(is_seed), eps=float(eps), target_scaled=S * float(eps),
            jf0=float(jf0), ratio=float(ratio), flux_metric_scale=S, step=int(step), max_mode=int(step) + 1,
            surface_dof_names=list(self._names), surface_dofs=x.tolist(), boundary=dict(boundary),
            qs_phr=float(eq.qs_total), eq_converged=bool(eq.converged), aspect=float(eq.aspect),
            mean_iota=float(eq.mean_iota), elongation=float(eq.elongation),
            g_eq={lab: float(g_eq[i]) for i, lab in enumerate(self.eq_labels)},
            max_g_eq=float(np.max(g_eq)) if np.all(np.isfinite(g_eq)) else None,
            Jf=(c0 if np.isfinite(c0) else None), Jf_loop=jf_loop,
            coils=dict(source=("final_coil_stage" if fs else ("cold solve" if coils is not None else None)),
                       ok=(None if coils is None else bool(getattr(coils, "ok", False))),
                       certified=(None if coils is None else bool(getattr(coils, "certified", False))),
                       Jf_polished=fs.get("Jf_polished"), Jf_cold=fs.get("Jf_default_cold"),
                       c0_lagr=gi("c0_lagr"), certified_v=gi("v") if gi("v") is not None else gi("cert.v"),
                       v_target_met=fs.get("v_target_met"), chosen_start=fs.get("chosen_start"),
                       bn_ratio_int=gi("bn_ratio_int"), bn_ratio_int_converged=fu.get("bn_ratio_int"),
                       mean_bn_area=(fu.get("mean_area") if fu.get("mean_area") is not None
                                     else gi("mean_bn_area")),
                       max_bn=gi("max_bn"), elasticity=_to_jsonable(fs.get("elasticity", gi("elasticity"))),
                       export=export),
            g_coil_scaled=g_coil, target_met=target_met, target_tol_scaled=tol,
            kkt=kkt, kkt_exit=bool(olog and olog[-1].get("kkt_exit")),
            lambda_c=float(st.lam_coil[0]), lam_eq=np.asarray(st.lam_eq).tolist(),
            rho_eq=np.asarray(st.rho_eq).tolist(), lam_coil=np.asarray(st.lam_coil).tolist(),
            rho_coil=np.asarray(st.rho_coil).tolist(), rho_coil_start=rho_start, rho_rule=rho_rule,
            n_outers=len(hist),
            outers=[dict(al_iter=o.get("al_iter"), qs=o.get("qs_total"), c0_scaled=o.get("c0_scaled"),
                         ls_message=o.get("ls_message"), nfev=o.get("nfev"),
                         feasibility=(o.get("kkt") or {}).get("feasibility"),
                         stationarity=(o.get("kkt") or {}).get("stationarity")) for o in olog],
            eval_id_range=[eid0, self._next_eval_id()],
            wall_point_s=time.time() - t_pt, wall_coil_stage_s=t_coil, wall_since_start_s=time.time() - wall0,
            seed=seed_info, checkpoint=ck)
        _json_dump(_to_jsonable(rec), os.path.join(self.out_dir, stem + ".json"))
        if not is_seed:
            _json_dump(_to_jsonable(ck), os.path.join(self.out_dir, f"al1_checkpoint_front{k:02d}.json"))
        self.log(f"  [sweep] {stem}: QS={float(eq.qs_total):.6e} Jf={c0:.6e} (loop "
                 f"{('%.6e' % jf_loop) if jf_loop is not None else 'n/a'}, target {eps:.6e}, "
                 f"met={target_met}) bn_ratio_int={rec['coils']['bn_ratio_int']} "
                 f"lam_c={float(st.lam_coil[0]):.4e} rho_c={float(st.rho_coil[0]):.4g} "
                 f"stationarity={(kkt or {}).get('stationarity')} [point {rec['wall_point_s']:.1f}s, "
                 f"coil stage {t_coil:.1f}s]")
        return rec

    def sweep(self, scfg=None):
        """Roadmap A4: front sweep from a seed (resume_from: checkpoint file, front point or run dir).

        For k = k0, k0 + 1, ...: the absolute flux target eps_k = ratio^k Jf0 (scaled S eps_k; no
        per-mode capture), lam_eq / rho_eq and lam_c carried, rho_c from sweep_rho_c (J2 ~ QS at the
        start point), at most scfg.outers PHR outers (_outer_iteration) with the B1 KKT exit, then
        front_point_{k:02d}.json (_sweep_front_point).  Stops after max_points points, before a target
        below min_target, when rho_c reaches rho_max, after infeasible_stop consecutive points that
        missed their target, or on a VMEC failure in a dual update (graceful stop: the last good
        surface is kept).  The mode is the seed's step (or cfg.start_step).  -> summary dict
        (also sweep_summary.json)."""
        cfg = self.cfg
        scfg = scfg if scfg is not None else SweepConfig()
        if not (0.0 < scfg.ratio < 1.0):
            raise ValueError(f"sweep ratio must be in (0, 1), got {scfg.ratio}")
        if int(scfg.outers) < 1 or int(scfg.max_points) < 0:
            raise ValueError("sweep needs outers >= 1 and max_points >= 0")
        if not cfg.enable_coils:
            raise ValueError("the front sweep needs coils (enable_coils=True)")
        if not self.resume_from:
            raise ValueError("the front sweep needs a seed (resume_from)")
        wall0 = time.time()
        self.log("")
        self.log(SEP)
        self.log("  QSS QA front sweep (roadmap A4) -- PHR AL1 [phrport: VMEX + XCoil]")
        self.log(SEP)
        self.log(f"  config: {json.dumps(_to_jsonable(asdict(cfg)))}")
        self.log(f"  sweep: {json.dumps(_to_jsonable(asdict(scfg)))}")
        self._write_provenance("started", extra=dict(mode="sweep", sweep=asdict(scfg)))
        if getattr(self, "_prov", None) is not None:
            self._prov.update(mode="sweep", sweep=_to_jsonable(asdict(scfg)))

        ck, cp, meta, note = self._load_sweep_seed(self.resume_from)
        ck_step = int(ck["step"])
        ck_names = list(self.free_dof_names(ck_step))
        sd = np.asarray(ck["surface_dofs"], dtype=np.float64)
        if len(ck_names) != len(sd):
            raise RuntimeError(f"seed DOF mismatch: {cp} has {len(sd)} free DOFs, free_dof_names({ck_step}) "
                               f"gives {len(ck_names)}.")
        if ck.get("surface_dof_names") and list(ck["surface_dof_names"]) != ck_names:
            raise RuntimeError(f"seed DOF names of {cp} differ from free_dof_names({ck_step})")
        boundary = collections.OrderedDict(self.seed)
        for name, v in zip(ck_names, sd):
            boundary[name] = float(v)
        step = ck_step if cfg.start_step is None else int(cfg.start_step)
        if not 0 <= step < max(1, cfg.n_continuation_steps):
            raise ValueError(f"sweep step {step} outside [0, {cfg.n_continuation_steps})")
        lam_eq, rho_eq = align_eq_multipliers(ck, self.eq_labels, self.cfg.rho_init, self.log)
        lam_coil = np.asarray(ck["lam_coil"], dtype=np.float64)
        rho_coil = np.asarray(ck["rho_coil"], dtype=np.float64)
        self.log(f"  SEED {cp} (step {ck_step}, al_iter {ck.get('al_iter')})" + (f" [{note}]" if note else "")
                 + f": lam_eq={lam_eq} rho_eq={rho_eq} lam_coil={lam_coil} rho_coil={rho_coil}")
        seed_info = dict(path=cp, step=ck_step, al_iter=ck.get("al_iter"), note=note, sweep=meta or None)

        max_mode = step + 1
        self._step = step
        _MAXJ_STEP[0] = step
        self._names = list(self.free_dof_names(step))
        self._base = collections.OrderedDict(boundary)
        x = self._xvec(boundary, self._names)
        self._freeze_ordering_axis(step, boundary, ck if step == ck_step else None)   # the seed's axis in its step
        self._eq_cache.clear()
        self._kkt_on = True                 # the B1 certificate drives the per-point exit
        self._flux_captured[0] = True       # absolute targets: no per-mode capture
        S = float(self._coil_scales[0])
        flags = self._loop_flags()
        eq0 = self._sweep_eq(x)
        if cfg.finite_beta:
            self._fb_set_at(x, step, boundary)
        if not bool(eq0.converged):
            raise RuntimeError(f"seed equilibrium failed: {eq0.message}")
        self.log(f"  Seed: step {step} (max_mode {max_mode}, {len(x)} free DOFs) QS={eq0.qs_total:.6e} "
                 f"aspect={eq0.aspect:.6f} mean_iota={eq0.mean_iota:.6f} flux_metric_scale={S:g}")

        c_seed = None
        if scfg.jf0 is not None:
            jf0, jf0_src = float(scfg.jf0), "sweep config (jf0)"
        elif meta.get("jf0") is not None:
            jf0, jf0_src = float(meta["jf0"]), f"sweep seed {cp}"
        else:
            t0 = time.time()
            r0 = self.coil_solver.solve(boundary, warm=None)
            self.timers["coil_solve"] += time.time() - t0
            jf0 = float(r0.c0_local)
            if not np.isfinite(jf0) or jf0 <= 0.0:
                raise RuntimeError(f"cold coil solve at the seed gave c0={jf0}: give SweepConfig.jf0")
            c_seed, jf0_src = S * jf0, f"cold coil solve at the seed (ok={bool(getattr(r0, 'ok', True))})"
        ratio = float(scfg.ratio)
        if meta.get("ratio") is not None and abs(float(meta["ratio"]) - ratio) > 1e-12:
            self.log(f"  [WARNING] the seed sweep used ratio {meta['ratio']}; this sweep uses {ratio}.")
        if scfg.k0 is not None:
            k = int(scfg.k0)
        elif meta:
            k = meta["k"] + 1 if meta.get("done") else meta["k"]
        else:
            k = 1
        self.log(f"  Jf0 = {jf0:.6e} ({jf0_src}); eps_k = {ratio:g}^k Jf0 from k = {k}; up to "
                 f"{scfg.max_points} points, {scfg.outers} outers each (B1 KKT exit)")

        st = _LoopState(x=x, boundary=boundary, lam_eq=lam_eq, rho_eq=rho_eq, lam_coil=lam_coil,
                        rho_coil=rho_coil, v_prev_eq=None, v_prev_coil=None,
                        last_good=collections.OrderedDict(boundary), last_good_step=step, aborted=False,
                        fin_state=dict(step=step, al_iter=0, v_prev_eq=None, v_prev_coil=None, done=0))
        points, n_done, n_infeas, stop = [], 0, 0, None
        k_last_done = k - 1                 # the last completed point (k0 - 1: none yet)
        summary_path = os.path.join(self.out_dir, "sweep_summary.json")

        def write_summary(final=False):
            _json_dump(_to_jsonable(dict(
                kind="sweep_summary", seed=seed_info, jf0=jf0, jf0_source=jf0_src, ratio=ratio, step=step,
                flux_metric_scale=S, sweep=asdict(scfg), stop=stop, finished=bool(final),
                wall_s=time.time() - wall0,
                points=[dict(k=p["k"], is_seed=p["is_seed"], eps=p["eps"], qs_phr=p["qs_phr"], Jf=p["Jf"],
                             Jf_loop=p["Jf_loop"], bn_ratio_int=p["coils"]["bn_ratio_int"],
                             target_met=p["target_met"], max_g_eq=p["max_g_eq"],
                             stationarity=(p["kkt"] or {}).get("stationarity"),
                             feasibility=(p["kkt"] or {}).get("feasibility"), kkt_exit=p["kkt_exit"],
                             lambda_c=p["lambda_c"], rho_c=p["rho_coil"][0], n_outers=p["n_outers"],
                             wall_point_s=p["wall_point_s"])
                        for p in points])), summary_path)

        if scfg.seed_point:
            points.append(self._sweep_front_point(
                scfg, st, k=k - 1, eps=jf0 * ratio ** (k - 1), jf0=jf0, ratio=ratio, step=step, is_seed=True,
                t_pt=time.time(), wall0=wall0, kkt_n0=len(self.kkt_log), ph_n0=len(self.phr_history),
                ol_n0=len(self.outer_log), eid0=self._next_eval_id(), rho_start=float(st.rho_coil[0]),
                rho_rule="seed", seed_info=seed_info))
            write_summary()

        while True:
            if n_done >= int(scfg.max_points):
                stop = "max_points"
                break
            if float(scfg.max_qs) > 0.0 and points and not points[-1]["is_seed"] \
                    and float(points[-1]["qs_phr"]) > float(scfg.max_qs):
                stop = "max_qs"
                self.log(f"  [sweep] last point QS {points[-1]['qs_phr']:.4e} > max_qs {scfg.max_qs:.4e}: stop.")
                break
            eps = jf0 * ratio ** k
            if eps < float(scfg.min_target):
                stop = "min_target"
                self.log(f"  [sweep] next target {eps:.6e} < min_target {scfg.min_target:.6e}: stop.")
                break
            t_pt = time.time()
            tgt = S * eps
            self._coil_targets[0] = tgt
            eq_x = self._sweep_eq(st.x)
            if (self._last_coil_x is not None and np.array_equal(self._last_coil_x, st.x)
                    and np.isfinite(self._last_coil_vals[0])):
                c_cur = float(self._last_coil_vals[0])
            else:
                c_cur = c_seed
            qs_cur = float(eq_x.qs_total) if bool(eq_x.converged) else None
            g0 = None if c_cur is None else c_cur - tgt
            rho_new, rho_rule = sweep_rho_c(qs_cur, g0, st.lam_coil[0], rho_carried=st.rho_coil[0],
                                            rho_min=cfg.rho_init, rho_max=cfg.rho_max,
                                            balance=scfg.rho_balance)
            st.rho_coil = np.array(st.rho_coil, dtype=np.float64, copy=True)
            st.rho_coil[0] = rho_new
            st.v_prev_coil = None             # new target: the coil violation history restarts
            self.log("")
            self.log(SEP)
            self.log(f"  Sweep point k={k} ({n_done + 1}/{scfg.max_points}): target eps={eps:.6e} raw "
                     f"(scaled {tgt:.6e}); start QS={qs_cur} c0={'n/a' if c_cur is None else '%.6e' % (c_cur / S)}"
                     f" raw; rho_c={rho_new:.6g} [{rho_rule}] lam_c={float(st.lam_coil[0]):.6e}")
            self.log(f"  lam_eq={st.lam_eq}, rho_eq={st.rho_eq}")
            self.log(SEP)
            kkt_n0, ph_n0, ol_n0, eid0 = len(self.kkt_log), len(self.phr_history), len(self.outer_log), \
                self._next_eval_id()
            st.fin_state = dict(step=step, al_iter=0, v_prev_eq=st.v_prev_eq, v_prev_coil=None, done=0)
            tag = f"sweep{k:02d}"
            extra = dict(sweep_k=k, sweep_eps=eps, sweep_jf0=jf0, sweep_ratio=ratio, sweep_point_done=False)
            for al_iter in range(int(scfg.outers)):
                if self._outer_iteration(st, step, al_iter, max_mode, flags=flags, kkt_exit_mode=True,
                                         n_budget=int(scfg.outers), base_budget=int(scfg.outers),
                                         ck_tag=tag, ck_extra=extra,
                                         rec_extra=dict(sweep_k=k, sweep_eps=eps)) is not None:
                    break
            if st.aborted:
                stop = "vmec_failure"
                self.log(f"  [sweep] VMEC failure in a dual update at point k={k}: stop (last good surface "
                         f"kept; no front point for k={k}).")
                break
            p = self._sweep_front_point(scfg, st, k=k, eps=eps, jf0=jf0, ratio=ratio, step=step, is_seed=False,
                                        t_pt=t_pt, wall0=wall0, kkt_n0=kkt_n0, ph_n0=ph_n0, ol_n0=ol_n0,
                                        eid0=eid0, rho_start=rho_new, rho_rule=rho_rule, seed_info=seed_info)
            points.append(p)
            n_done += 1
            k_last_done = k
            n_infeas = 0 if p["target_met"] else n_infeas + 1
            write_summary()
            if float(st.rho_coil[0]) >= cfg.rho_max:
                stop = "rho_cap"
                self.log(f"  [sweep] rho_c = {float(st.rho_coil[0]):.4g} reached rho_max: stop.")
                break
            if n_infeas >= int(scfg.infeasible_stop):
                stop = "infeasible"
                self.log(f"  [sweep] target missed at {n_infeas} consecutive points: stop.")
                break
            k += 1

        self.log(f"  [sweep] stopped ({stop}) after {n_done} point(s); elapsed {time.time() - wall0:.1f}s")
        if st.fin_state is not None:
            # the sweep fields: a resume from this directory continues after the last completed point
            # (or repeats point k after a VMEC failure in it), with this sweep's Jf0 and ratio
            k_fin, done_fin = (k, False) if stop == "vmec_failure" else (k_last_done, True)
            fin_extra = dict(sweep_k=int(k_fin), sweep_eps=float(jf0 * ratio ** k_fin), sweep_jf0=float(jf0),
                             sweep_ratio=float(ratio), sweep_point_done=bool(done_fin))
            try:
                self._write_final_checkpoint(st.fin_state, st.lam_eq, st.rho_eq, st.lam_coil, st.rho_coil,
                                             self._xvec(st.boundary, self._names), extra=fin_extra)
            except Exception as e:  # noqa: BLE001
                self.log(f"  [WARNING] al1_checkpoint_final.json failed: {type(e).__name__}: {e}")
        self._write_histories()
        write_summary(final=True)
        with open(summary_path) as f:
            out = json.load(f)
        out["points_full"] = points
        self._finish(wall0)
        return out

    # ------------------------------------------------- phase 0: final stage / metrics
    # -------------------------------------------------- finite beta (opt-in)
    # ------------------------------------------- 2026-11 speed-ups (opt-in helpers)
    def _fb_reuse_ok(self, x, step):
        """P5: the field term (and context) from the last refresh may be reused at x."""
        ref = self._fb_ref
        return bool(self.cfg.finite_beta and self.cfg.fb_reuse_tol > 0 and ref is not None
                    and ref["step"] == step and ref["n"] + 1 < max(1, int(self.cfg.fb_reuse_every))
                    and np.shape(ref["x"]) == np.shape(x)
                    and float(np.max(np.abs(np.asarray(x) - ref["x"]))) < self.cfg.fb_reuse_tol)

    def _fb_init(self):
        if not self.cfg.finite_beta or self._fb is not None:
            return
        from .finite_beta import FiniteBetaCoupler
        self._fb = FiniteBetaCoupler(E_NFP(), n=self.cfg.fb_n, digits_value=self.cfg.fb_digits_value,
                                     digits_grad=self.cfg.fb_digits_grad, log=self.log, jit=self.cfg.fb_jit,
                                     check_path=os.path.join(self.out_dir, "fb_jit_check.jsonl"))
        self.coil_solver.set_plasma(self._fb.provider)
        self.log(f"  [finite-beta] plasma field in the coil problem (virtual casing, {self.cfg.fb_n}^2 grid, "
                 f"{self.cfg.fb_digits_value} digits values / {self.cfg.fb_digits_grad} digits field term; "
                 f"jit {self.cfg.fb_jit})")

    def _fb_imp(self, step):
        from .exact_jacobian import ExactEquilibriumJacobian
        if self._exact_jac is not None:
            ej = self._exact_jac.get(step)
        else:
            if not hasattr(self, "_fb_ej"):
                self._fb_ej = {}
            ej = self._fb_ej.get(step)
            if ej is None:
                ej = ExactEquilibriumJacobian(step)
                self._fb_ej[step] = ej
        return ej.ctx.inp, ej.cfg

    def _fb_set_at(self, x, step, boundary=None):
        """Plasma-field context = the converged VMEX state of x (f-eval cache, else a solve)."""
        if not self.cfg.finite_beta:
            return
        self._fb_init()
        from vmex.core import implicit as imp
        from . import equilibrium as _E
        x = np.asarray(x, dtype=np.float64)
        key = (int(step), x.tobytes())
        if key == self._fb.key:
            return
        boundary = self._boundary_with(x) if boundary is None else boundary
        eq = None
        try:
            eq = self._cache_get(step, x)
        except Exception:  # noqa: BLE001
            eq = None
        vs = getattr(eq, "vmex_state", None) if eq is not None else None
        if vs is None:
            eq = _E.evaluate(boundary, step, return_state=True)
            vs = getattr(eq, "vmex_state", None)
            if vs is None:
                raise RuntimeError(f"finite-beta: no converged state at the context point ({eq.message})")
        inp_static, icfg = self._fb_imp(step)
        with _E.vmex_device_ctx():
            state = vs.spectral_state()
            params = imp.params_from_input(_E.vmec_input(boundary, step))
        self._fb.set_context(inp_static, icfg, state, params, key)

    def _fb_hook(self, ctx, x, J2, grad_J2, step):
        """At a Jacobian point: exact plasma field at x, warm coil re-solve, explicit + field term.
        P5 (opt-in fb_reuse_tol > 0): while max|x - x_refresh| < tol (at most fb_reuse_every - 1 times in a
        row) the context is kept (no re-solve: the f-eval's coils at x were solved under it) and the field
        term of the last refresh is reused."""
        x = np.asarray(x, dtype=np.float64)
        if self.cfg.fb_reuse_tol > 0 and self._fb_reuse_ok(x, step):
            ref = self._fb_ref
            ref["n"] += 1
            self.counters["fb_reuse"] += 1
            gf = ref["gf"]
            shifted = np.maximum(0.0, self._last_coil_vals - ctx.targets + ctx.lam / ctx.rho)
            add = float(ctx.rho[0] * shifted[0]) * self._coil_scales[0] * gf
            if self._kkt_on and x.tobytes() in self._kkt_cg:
                cg0, ph = self._kkt_cg[x.tobytes()]
                self._kkt_cg[x.tobytes()] = (cg0 + self._coil_scales[0] * gf, ph)
            self.log(f"  [fb-reuse] context and field term kept (max|dx| since refresh "
                     f"{float(np.max(np.abs(x - ref['x']))):.2e} < {self.cfg.fb_reuse_tol:g}, "
                     f"reuse {ref['n']}/{max(1, int(self.cfg.fb_reuse_every)) - 1})")
            return np.asarray(grad_J2, dtype=np.float64) + add
        ex = None
        if (self._exact_jac is not None and self._exact_last_x is not None
                and np.array_equal(self._exact_last_x, x)):
            ex = getattr(self._exact_jac, "last", None)
        inp_static, icfg = self._fb_imp(step)
        if ex is not None and getattr(ex, "state", None) is not None:
            self._fb.set_context(inp_static, icfg, ex.state, ex.params, (int(step), x.tobytes()))
        else:
            self._fb_set_at(x, step)
        J2n, gradn = self._compute_J2_and_grad(ctx, x)          # warm re-solve with the exact plasma field
        if self.cfg.fb_reuse_tol > 0:                           # P5 check: error of the lagged context
            g0 = np.asarray(grad_J2, dtype=np.float64)
            self._fb_check = dict(rel_explicit=float(np.linalg.norm(gradn - g0) / max(np.linalg.norm(gradn), 1e-300)))
        if ex is None or getattr(ex, "state_tangents", None) is None:
            self.log("  [finite-beta] no exact-Jacobian state tangents at this point: explicit coil gradient only")
            if self.cfg.fb_reuse_tol > 0:
                self._fb_ref = None
            return gradn
        boundary = self._boundary_with(x)
        cot = self.coil_solver.bp_cotangent(boundary, self._coil_res)
        gf = self._fb.field_grad(inp_static, icfg, ex, cot, self.coil_solver.P.flux_TH, self.coil_solver.P.flux_PH)
        coil_vals = self._last_coil_vals
        shifted = np.maximum(0.0, coil_vals - ctx.targets + ctx.lam / ctx.rho)
        add = float(ctx.rho[0] * shifted[0]) * self._coil_scales[0] * gf
        if self.cfg.fb_jit == "check" and self._fb.last_check_g is not None:   # P7: coil-row error of the jit term
            row = np.asarray(gradn) + add
            nr = max(float(np.linalg.norm(row)), 1e-300)
            rj = np.asarray(gradn) + float(ctx.rho[0] * shifted[0]) * self._coil_scales[0] * self._fb.last_check_g
            self._fb._check_write(dict(kind="row", step=int(step), norm_row=nr, norm_add=float(np.linalg.norm(add)),
                                       rel_row_vjp=float(np.linalg.norm(rj - row) / nr)))
        if self._kkt_on and x.tobytes() in self._kkt_cg:
            cg0, ph = self._kkt_cg[x.tobytes()]
            self._kkt_cg[x.tobytes()] = (cg0 + self._coil_scales[0] * gf, ph)
        if self.cfg.fb_reuse_tol > 0:                           # P5: refresh + Taylor-type check of a reuse
            ref = self._fb_ref
            chk = dict(getattr(self, "_fb_check", None) or {})
            if ref is not None and ref["step"] == step and np.shape(ref["x"]) == np.shape(x):
                dx = float(np.max(np.abs(x - ref["x"])))
                ng = max(float(np.linalg.norm(gf)), 1e-300)
                chk.update(dx=dx, rel_field=float(np.linalg.norm(gf - ref["gf"]) / ng),
                           cos_field=float(np.dot(gf, ref["gf"]) / (ng * max(float(np.linalg.norm(ref["gf"])), 1e-300))),
                           rel_row=float(np.linalg.norm((np.asarray(gradn) + add) - (np.asarray(grad_J2) + float(
                               ctx.rho[0] * shifted[0]) * self._coil_scales[0] * ref["gf"]))
                               / max(float(np.linalg.norm(np.asarray(gradn) + add)), 1e-300)))
                self.log(f"  [fb-reuse] refresh: max|dx| {dx:.2e}; a reuse here would have had field-term error "
                         f"{chk['rel_field']:.2e} (cos {chk['cos_field']:.6f}), explicit-gradient error "
                         f"{chk.get('rel_explicit', float('nan')):.2e}, coil-row error {chk['rel_row']:.2e}")
                with open(os.path.join(self.out_dir, "fb_reuse_check.jsonl"), "a") as fh:
                    fh.write(json.dumps(dict(step=int(step), **chk)) + "\n")
            self._fb_ref = dict(step=int(step), x=x.copy(), gf=np.array(gf, copy=True), n=0)
            self.counters["fb_refresh"] += 1
        return np.asarray(gradn) + add

    def _run_final_coil_stage(self, boundary):
        """R1: CoilSolver.final_coil_stage (all starts, every ok unlinked one polished to certified
        v <= 1e-5, lowest polished Lagrangian). None (-> the plain cold solve) when unavailable or
        not finite."""
        if self.cfg.finite_beta and self._step is not None:     # the boundary's own exact plasma field
            self._fb_set_at(self._xvec(boundary, list(self.free_dof_names(self._step))), self._step, boundary)
        fn = getattr(self.coil_solver, "final_coil_stage", None)
        if fn is None:
            self.log("  [final-coil-stage] coil solver has no final_coil_stage; plain cold solve.")
            return None
        t0 = time.time()
        try:
            res = fn(boundary)
        except Exception as e:  # noqa: BLE001
            self.log(f"  [WARNING] final_coil_stage failed ({type(e).__name__}: {e}); plain cold solve.")
            return None
        finally:
            self.timers["coil_final_stage"] += time.time() - t0
        if res is None or not np.isfinite(float(getattr(res, "c0_local", np.nan))):
            self.log("  [WARNING] final_coil_stage returned no finite result; plain cold solve.")
            return None
        fs = (getattr(res, "info", None) or {}).get("final_stage") or {}
        self.log(f"  [final-coil-stage] chosen start {fs.get('chosen_start')} (default "
                 f"{fs.get('default_start')}; {fs.get('n_met')} of {fs.get('n_polished')} polished starts "
                 f"met the target): Jf cold default={fs.get('Jf_default_cold')} "
                 f"chosen={fs.get('Jf_cold_chosen')} polished={fs.get('Jf_polished')} "
                 f"certified v={fs.get('v')} (target met: {fs.get('v_target_met')}) "
                 f"[{time.time() - t0:.1f}s]")
        return res

    def _final_metrics_phase0(self, fm, coil_final, cinfo, boundary, lam_eq):
        """R1 / B1 additions to final_metrics (new keys only) and the coil export."""
        cfg = self.cfg

        def ci(k):
            v = _info_lookup(cinfo, k)
            return None if v is None else _to_jsonable(v)
        fmc = fm["coils"]
        fmc.update(c0_lagr=ci("c0_lagr"), bn_ratio_int=ci("bn_ratio_int"),
                   avg_Bn_over_B_unweighted=ci("mean_bn_unweighted"), elasticity=ci("elasticity"),
                   lam_block_sum=ci("lam_block_sum"), certified_violation_soft=ci("v_soft"),
                   coil_start=ci("start"), coil_mode=ci("mode"), slot=ci("slot"))
        fmc["flux_metrics"] = dict(grid_50x50=dict(
            c0=fmc["squared_flux_Jf"], bn_ratio_int=ci("bn_ratio_int"), mean_area=ci("mean_bn_area"),
            mean_unweighted=ci("mean_bn_unweighted"), max=ci("max_bn")))
        fs = cinfo.get("final_stage") if isinstance(cinfo, dict) else None
        if fs:
            keys = ("chosen_start", "chosen_tier", "chosen_rank_cold", "default_start", "Jf_default_cold",
                    "Jf_cold_chosen", "c0_lagr_cold_chosen", "v_cold_chosen", "certified_cold_chosen",
                    "n_pool", "n_polished", "n_met", "max_polish", "polished",
                    "Jf_polished", "c0_lagr", "v", "v_target",
                    "v_target_met", "certified", "linked", "rel_change_vs_default", "rel_change_vs_chosen",
                    "margins", "polish_viol_tol", "ftol", "k_max", "t_starts", "t_polish", "t_total",
                    "polish_rounds", "candidates")
            st = {k: _to_jsonable(fs.get(k)) for k in keys}
            st["Jf_cold"] = st.pop("Jf_default_cold")
            st["certified_v"] = st.pop("v")
            fmc["final_stage"] = st
            fmc["flux_metrics"]["grid_50x50"] = _to_jsonable(fs.get("flux_50x50"))
            fmc["flux_metrics"]["converged_grid"] = _to_jsonable(fs.get("flux_unfolded"))
            fmc["elasticity"] = _to_jsonable(fs.get("elasticity", fmc.get("elasticity")))
        opt = fm["optimization"]
        if "iota_mean_hi" in self.eq_labels and "iota_mean_lo" in self.eq_labels:
            opt["lambda_iota_net"] = float(lam_eq[self.eq_labels.index("iota_mean_hi")]
                                           - lam_eq[self.eq_labels.index("iota_mean_lo")])
        opt["kkt_final"] = _to_jsonable(self.kkt_log[-1]) if self.kkt_log else None
        opt["phase0"] = dict(final_coil_stage=bool(cfg.final_coil_stage),
                             final_mode_kkt_exit=bool(cfg.final_mode_kkt_exit),
                             flux_target_floor=cfg.flux_target_floor,
                             final_mode_flux_target=cfg.final_mode_flux_target,
                             resume_keep_target=bool(cfg.resume_keep_target), start_step=cfg.start_step,
                             abort_doomed_coil_solves=bool(cfg.abort_doomed_coil_solves),
                             certify_trials=bool(cfg.certify_trials),
                             reuse_vmex_state=bool(cfg.reuse_vmex_state))
        if self._exact_jac is not None:
            opt["exact_jac_stats"] = _to_jsonable(dict(self._exact_jac.stats))
        if coil_final is not None and np.isfinite(float(getattr(coil_final, "c0_local", np.nan))):
            try:
                p = export_coils(os.path.join(self.out_dir, "final_coils"), self.coil_solver, coil_final,
                                 boundary, source=("final_coil_stage" if fs else "cold solve"),
                                 extra=dict(final_stage_chosen_start=(fs or {}).get("chosen_start")))
                fmc["export"] = (os.path.basename(p) if p else None)
                if p:
                    self.log(f"  Coils exported: {p} (+ .npz)")
            except Exception as e:  # noqa: BLE001
                self.log(f"  [WARNING] coil export failed: {e}")

    def _finish(self, wall0):
        w = time.time() - wall0
        c = self.timers["coil_solve"]
        self.log(SEP)
        self.log(f"  WALL TIME: total = {w:.1f} s; coil solves = {c:.1f} s "
                 f"({100 * c / max(w, 1e-9):.0f}%); eq f-evals+FD = "
                 f"{self.timers['eq_many'] + self.timers['eq_single']:.1f} s")
        self.log(f"  counters: {dict(self.counters)}")
        self.log(SEP)
        self._write_provenance("finished", wall=w, extra=dict(counters=dict(self.counters),
                                                              timers=dict(self.timers)))
        self.close()


class _FailedEq:
    """Minimal failed EqResult (used only if an evaluator raises)."""
    converged = False
    qs_residuals = np.zeros(0)
    qs_total = float("nan")
    aspect = mean_iota = iota_axis = iota_edge = elongation = well = float("nan")

    def __init__(self, message=""):
        self.message = message
        self.info = {}


# ---------------------------------------------------------------------------
# Self-test (port of run_self_test in the PHR script)
# ---------------------------------------------------------------------------
def run_self_test(verbose=True, rho_init=1.0, rho_theta=0.25, rho_tau=2.0):
    """KKT check of update_dual_phr + update_rho_conditioned (PHR run_self_test).
    Returns (ok, dict(x, lam, rho, v))."""
    def g(x):
        return np.array([x[0] + x[1] - 2.0, x[0] - 10.0])

    lam = np.array([0.0, 5.0])
    rho = np.full(2, rho_init)
    v_prev = None
    x = np.zeros(2)
    for _ in range(25):
        lam_c, rho_c = lam.copy(), rho.copy()

        def resid(x, lam_c=lam_c, rho_c=rho_c):
            hinge = np.maximum(0.0, g(x) + lam_c / rho_c)
            return np.concatenate([[x[0] - 2.0, x[1] - 1.0], np.sqrt(rho_c / 2.0) * hinge])

        x = least_squares(resid, x).x
        lam, v = update_dual_phr(lam, rho, g(x), lam_max=LAM_MAX)
        rho, _grew = update_rho_conditioned(rho, v, v_prev, theta=rho_theta, tau=rho_tau,
                                            rho_max=RHO_MAX, v_tol=1e-10)
        v_prev = v
    checks = [
        ("x converges to KKT point (1.5, 0.5)", bool(np.allclose(x, [1.5, 0.5], atol=1e-5))),
        ("active-constraint multiplier lam1 -> 1 (KKT value)", bool(abs(lam[0] - 1.0) < 1e-4)),
        ("inactive-constraint multiplier lam2 decays 5 -> 0", bool(lam[1] == 0.0)),
        ("inactive-constraint rho2 never grew", bool(rho[1] == rho_init)),
        ("final violation ~ 0", bool(v_prev[0] < 1e-6)),
    ]
    ok = all(p for _, p in checks)
    if verbose:
        print(f"  x={x} lam={lam} rho={rho} v={v_prev}")
        for label, passed in checks:
            print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        print(f"  Self-test {'PASSED' if ok else 'FAILED'}")
    return ok, dict(x=x, lam=lam, rho=rho, v=v_prev, checks=checks)


# ---------------------------------------------------------------------------
# CLI: wire the real components (phrport.equilibrium / vmex_pool / coils)
# ---------------------------------------------------------------------------
def build_default_components(n_workers=8, threads_per_worker=1, coil_overrides=None,
                             cold_cache=False):
    """Real components. cold_cache (S-b, opt-in): CoilSolver(cold_cache=True) reuses the most
    recent cold solve of an identical surface (bit-identical; used with --no-certify-trials).
    S-c needs nothing here: the runner requests states per f-eval (evaluate_many(...,
    return_state=True) / evaluate(..., return_state=True))."""
    from phrport import equilibrium as eqmod
    from phrport.vmex_pool import VmexPool
    from phrport.coils import CoilSolver, CoilConfig
    pool = VmexPool(n_workers=n_workers, threads_per_worker=threads_per_worker)
    ccfg = CoilConfig(**(coil_overrides or {}))
    coil = CoilSolver(ccfg, cold_cache=True) if cold_cache else CoilSolver(ccfg)
    return dict(evaluate=eqmod.evaluate, evaluate_many=pool.evaluate_many, coil_solver=coil,
                free_dof_names=eqmod.free_dof_names, seed_boundary=eqmod.seed_boundary(),
                vmec_input=str(getattr(eqmod, "INPUT_FILE", "input.nfp2_QA")),
                write_wout=getattr(eqmod, "write_wout", None), closers=[pool.close])


def build_parser():
    """The argparse parser of python -m phrport.phr / qss-run (also read by tools/gen_reference.py)."""
    p = argparse.ArgumentParser(description="phrport: PHR AL1 + envelope coils on VMEX + XCoil")
    p.add_argument("--run-dir", default=None)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--no-coils", dest="enable_coils", action="store_false", default=True)
    p.add_argument("--n-continuation-steps", type=int, default=4)
    p.add_argument("--al-outer-iter", type=int, default=6)
    p.add_argument("--max-nfev", type=int, default=20)
    p.add_argument("--rho-init", type=float, default=1.0)
    p.add_argument("--rho-tau", type=float, default=2.0)
    p.add_argument("--rho-theta", type=float, default=0.25)
    p.add_argument("--rho-max", type=float, default=RHO_MAX, help="cap on the PHR penalty parameters")
    p.add_argument("--vacuum-well", action="store_true", default=PHRConfig.vacuum_well)
    p.add_argument("--well-threshold", type=float, default=PHRConfig.well_threshold)
    p.add_argument("--flux-metric-scale", type=float, default=1.0)
    p.add_argument("--flux-phr-target", type=float, default=0.0)
    p.add_argument("--flux-target-frac", type=float, default=0.5)
    p.add_argument("--resume-from", type=str, default=None)
    p.add_argument("--init-surface", type=str, default=None)
    p.add_argument("--skip-surface-opt", action="store_true", default=False)
    p.add_argument("--n-workers", type=int, default=8)
    p.add_argument("--fd-ngroups", type=int, default=8)
    p.add_argument("--coil-fail-policy", choices=["use", "sentinel"], default="use")
    p.add_argument("--coil-nonfinite-policy", choices=["sentinel", "phr_nan"], default="sentinel")
    p.add_argument("--ls-verbose", type=int, default=2)
    p.add_argument("--no-overlap", dest="overlap", action="store_false", default=True)
    p.add_argument("--eq-jacobian", choices=["fd", "exact"], default="fd",
                   help="equilibrium Jacobian block: PHR forward FD (default) or opt-in exact")
    p.add_argument("--exact-jac-gauge", choices=["sqrt", "frozen"], default="sqrt")
    # opt-in speedups (defaults reproduce PHR exactly)
    p.add_argument("--skip-coil-on-rejected", action="store_true", default=False,
                   help="VMEX first; skip the coil solve of trials TRF certainly rejects "
                        "(exact: same accepted iterates)")
    p.add_argument("--skip-coil-scope", choices=["head", "all"], default="head",
                   help="f-evals sequenced VMEX->coils (and skippable): first trial of each "
                        "least_squares + trials after a skip (head), or all")
    p.add_argument("--stall-stop", type=int, default=0, metavar="K",
                   help="stop an inner least_squares after K consecutive rejected trials once the "
                        "trust radius < F*xtol*(xtol+||x||) (0 = off)")
    p.add_argument("--stall-delta-factor", type=float, default=10.0, metavar="F")
    p.add_argument("--surface-dof-reset-threshold", type=float, default=SURFACE_DOF_RESET_THRESHOLD,
                   help="coil warm-start if max|dsurf| <= this, else cold reset (PHR default 1e-5)")
    p.add_argument("--spectral-scale", type=float, default=0.0, metavar="P",
                   help="opt-in: TRF x_scale (1 + m^2 + n^2)^(-P/2) on the free boundary DOFs (0 = off)")
    p.add_argument("--carry-trust-radius", action="store_true", default=False,
                   help="start each least_squares with the previous call's largest accepted step "
                        "as trust radius (changes iterates)")
    p.add_argument("--no-carry-trust-radius", dest="carry_trust_radius", action="store_false",
                   help="undo --carry-trust-radius (e.g. after --preset production; the QI tokamak start ran without it)")
    # roadmap phase 0 (all opt-in; defaults reproduce the previous code)
    g0 = p.add_argument_group("roadmap phase 0")
    g0.add_argument("--flux-target-floor", type=float, default=0.0, metavar="T",
                    help="A1: captured flux target = scale * max(frac * c0, T) (0 = PHR)")
    g0.add_argument("--final-mode-flux-target", type=float, default=0.0, metavar="E",
                    help="A5: in the final continuation mode the flux target is the absolute E (raw c0 "
                         "units, like a sweep point's eps) instead of frac * c0 (0 = off)")
    g0.add_argument("--flux-target-bn", type=float, default=PHRConfig.flux_target_bn, metavar="B",
                    help="coil-fit flux bound as <|B.n|>/<|B|> <= B: each captured c0 target is c0 (B/b)^2, b the "
                         "coil solution's <|B.n|>/<|B|> (0 = off; profile phr.flux_target_bn)")
    g0.add_argument("--resume-keep-target", action="store_true", default=False,
                    help="B0: --resume-from keeps the checkpoint's flux target, lam_c / rho_c and "
                         "rho-growth history (no step-top re-capture / reset in the resumed step)")
    g0.add_argument("--start-step", type=int, default=None, metavar="K",
                    help="B0: start the continuation at step K (outer 0), keeping the "
                         "checkpoint's lam / rho")
    g0.add_argument("--final-mode-kkt-exit", action="store_true", default=False,
                    help="B1: final mode exits iff KKT feasibility < v_tol and stationarity < "
                         "--kkt-stationarity-tol; otherwise up to --kkt-extra-outers more outers")
    g0.add_argument("--kkt-stationarity-tol", type=float, default=0.1)
    g0.add_argument("--full-outer-budget", action="store_true", default=False,
                    help="run all --al-outer-iter outers in every mode, even when every violation "
                         "is below v_tol")
    g0.add_argument("--kkt-extra-outers", type=int, default=3)
    g0.add_argument("--kkt-active-tol", type=float, default=1e-3,
                    help="B1: constraints with g >= -tol enter the certificate's multiplier fit "
                         "(as do the constraints PHR treats as active, lambda > 0)")
    g0.add_argument("--kkt-no-phr-active", dest="kkt_phr_active", action="store_false", default=True,
                    help="B1: active set from --kkt-active-tol alone (not also lambda > 0)")
    g0.add_argument("--final-coil-stage", action="store_true", default=False,
                    help="R1/G4: final coils from CoilSolver.final_coil_stage (every start; every "
                         "ok unlinked one polished to certified v <= 1e-5; lowest polished Lagrangian)")
    g0.add_argument("--abort-doomed-coil-solves", action="store_true", default=False,
                    help="S-a: stop a trial's coil solve before its fallback starts once TRF is "
                         "certain to reject the trial (same accepted iterates)")
    g0.add_argument("--no-certify-trials", dest="certify_trials", action="store_false", default=True,
                    help="S-b: trial coil solves without certification (soft ok); refresh / final "
                         "solves certified, accepted coils certified afterwards; cold-solve cache")
    g0.add_argument("--reuse-vmex-state", action="store_true", default=False,
                    help="S-c: the exact Jacobian reuses the f-eval's converged VMEX state")
    g0.add_argument("--no-phase0-reporting", dest="phase0_reporting", action="store_false",
                    default=True,
                    help="no provenance.json / eval_x.npz / KKT logging / final checkpoint / coil "
                         "export (every output file exactly as before phase 0)")
    # roadmap phase 2 (all opt-in; defaults reproduce the previous code)
    g2 = p.add_argument_group("roadmap phase 2")
    g2.add_argument("--coil-oracle", choices=["phr", "anchored"], default="phr",
                    help="B2: coil warm-start rule: PHR's (default) or the anchored oracle (warm starts "
                         "from the coils at TRF's accepted point only, distance window, per-outer cold "
                         "audit keeping the lower coil Lagrangian)")
    g2.add_argument("--oracle-d-inherit", type=float, default=1e-3, metavar="D1",
                    help="B2: d <= D1: warm with the anchor's (lam, rho), --oracle-k-max outers, kkt guard")
    g2.add_argument("--oracle-d-reset", type=float, default=1e-2, metavar="D2",
                    help="B2: d <= D2 (or guard failed): warm with rho = rho0, lam kept; beyond: cold")
    g2.add_argument("--oracle-k-max", type=int, default=2)
    g2.add_argument("--oracle-kkt-factor", type=float, default=3.0,
                    help="B2 guard: NNLS kkt <= max(F * kkt_anchor, --oracle-kkt-floor)")
    g2.add_argument("--oracle-kkt-floor", type=float, default=5e-3)
    g2.add_argument("--no-oracle-audit", dest="oracle_audit", action="store_false", default=True,
                    help="B2: no per-outer cold audit (PHR's refresh rule, through the oracle)")
    g2.add_argument("--oracle-reset-guard", dest="oracle_reset_guard", action="store_true", default=True,
                    help="B2 (default with --coil-oracle anchored): a rho-reset warm result that ran out of "
                         "outers or fails the kkt threshold is replaced by a cold solve")
    g2.add_argument("--no-oracle-reset-guard", dest="oracle_reset_guard", action="store_false",
                    help="B2: the roadmap's literal rule (no reset guard; fails the offline gate)")
    g2.add_argument("--oracle-d-floor", type=float, default=0.0, metavar="D0",
                    help="B2 noise floor (opt-in; 0 = off): d <= D0 (<= D1) uses --oracle-floor-rule "
                         "instead of the k_max inherited solve (suggested 1e-5)")
    g2.add_argument("--oracle-floor-rule", choices=["linear", "converge"], default="linear",
                    help="B2 noise floor: 'linear' = c0_anchor + G.dx on the anchor's coils (no solve; "
                         "the per-outer audit replaces it by a cold solve), 'converge' = the inherited "
                         "warm solve with --oracle-floor-k-max outers (stop test decides)")
    g2.add_argument("--oracle-floor-k-max", type=int, default=15)
    g2.add_argument("--carry-flux-from-mode", type=int, default=0,
                    help="with --carry-flux-multiplier: carry lam_c / rho_c only into modes >= this (0 = every mode)")
    g2.add_argument("--carry-flux-multiplier", action="store_true", default=False,
                    help="B5: do not reset lam_c / rho_c at the start of a mode (target still re-captured)")
    g3 = p.add_argument_group("roadmap A4 front sweep (python -m phrport.phr --sweep --resume-from SEED ...)")
    g3.add_argument("--sweep", action="store_true", default=False,
                    help="A4: front sweep from --resume-from (checkpoint file, run dir or front_point json) "
                         "instead of the continuation run")
    g3.add_argument("--sweep-ratio", type=float, default=0.85, metavar="R", help="eps_k = R^k Jf0")
    g3.add_argument("--sweep-max-points", type=int, default=10)
    g3.add_argument("--target-flux", type=float, default=0.0, metavar="E",
                    help="A5: standalone front point -- after the normal run, ramp the absolute flux target "
                         "(raw c0) down to E in the final mode with ratio <= --sweep-ratio (0 = off)")
    g3.add_argument("--sweep-min-target", type=float, default=0.0, metavar="T",
                    help="stop before a point whose raw target eps_k < T")
    g3.add_argument("--sweep-outers", type=int, default=4, help="outer budget per point (B1 KKT exit)")
    g3.add_argument("--sweep-max-qs", type=float, default=0.0, help="stop after a point with QS above this (0 = off)")
    g3.add_argument("--sweep-jf0", type=float, default=None,
                    help="raw Jf0 (default: the seed sweep's, else a cold coil solve at the seed)")
    g3.add_argument("--sweep-k0", type=int, default=None, help="first k (default 1 / next after a sweep seed)")
    g3.add_argument("--sweep-rho-balance", type=float, default=1.0,
                    help="rho_c at the start of a point so that J2 = this x QS")
    g3.add_argument("--sweep-infeasible-stop", type=int, default=2)
    g3.add_argument("--sweep-infeasible-tol", type=float, default=None,
                    help="target missed iff the scaled coil violation > this (default v_tol)")
    g3.add_argument("--sweep-seed-point", action="store_true", default=False,
                    help="also write front_point_seed.json (seed surface with final-stage coils)")
    g3.add_argument("--sweep-no-polish", dest="sweep_polish", action="store_false", default=True,
                    help="point coils from a cold solve instead of the final coil stage")
    g4 = p.add_argument_group("2026-11 single-run speed-ups")
    g4.add_argument("--phase-timers", action="store_true",
                    default=_os.environ.get("PHRPORT_PHASE_TIMERS", "0") not in ("", "0"),
                    help="P1: per f-eval / Jacobian phase timer table (log + phase_timers.jsonl); passive, "
                         "env PHRPORT_PHASE_TIMERS=1")
    g4.add_argument("--fb-reuse-tol", type=float, default=0.0, metavar="D",
                    help="P5 (finite beta): reuse the plasma-field context and field term while max|dx| < D since "
                         "the last refresh (0 = off; tested 2e-3)")
    g4.add_argument("--fb-reuse-every", type=int, default=4, metavar="K",
                    help="P5: refresh at least every K Jacobians")
    g4.add_argument("--fb-jit", choices=("vjp", "off", "check"), default="vjp",
                    help="P7 (finite beta; default vjp): jitted virtual casing for the context, its interpolation and "
                         "the field term (one reverse pass). Falls back to eager for the rest of a mode if compiling "
                         "or running it fails (e.g. out of GPU memory). off: the eager path (pre-patch runs); check: "
                         "eager results, jitted ones compared in fb_jit_check.jsonl. Vacuum runs are unaffected")
    # undo forms of the on/off options a qss-run preset switches on (so a profile "run" section or the command line
    # can always undo a preset); argparse keeps the last occurrence
    p.add_argument("--no-skip-coil-on-rejected", dest="skip_coil_on_rejected", action="store_false",
                   help="undo --skip-coil-on-rejected")
    p.add_argument("--no-full-outer-budget", dest="full_outer_budget", action="store_false",
                   help="undo --full-outer-budget")
    p.add_argument("--no-abort-doomed-coil-solves", dest="abort_doomed_coil_solves", action="store_false",
                   help="undo --abort-doomed-coil-solves")
    p.add_argument("--certify-trials", dest="certify_trials", action="store_true",
                   help="undo --no-certify-trials (certify every trial coil solve, the default)")
    p.add_argument("--no-reuse-vmex-state", dest="reuse_vmex_state", action="store_false",
                   help="undo --reuse-vmex-state")
    return p


def main(argv=None):
    p = build_parser()
    a = p.parse_args(argv)
    if a.self_test:
        ok, _ = run_self_test()
        return 0 if ok else 1
    if not a.run_dir:
        p.error("--run-dir is required")
    if a.sweep and not a.resume_from:
        p.error("--sweep needs --resume-from SEED")
    cfg = PHRConfig(n_continuation_steps=a.n_continuation_steps, al_outer_iter=a.al_outer_iter,
                    max_nfev=a.max_nfev, rho_init=a.rho_init, rho_max=a.rho_max, rho_tau=a.rho_tau,
                    rho_theta=a.rho_theta, vacuum_well=a.vacuum_well,
                    well_threshold=a.well_threshold, enable_coils=a.enable_coils,
                    flux_metric_scale=a.flux_metric_scale, flux_phr_target=a.flux_phr_target,
                    flux_target_frac=a.flux_target_frac, skip_surface_opt=a.skip_surface_opt,
                    fd_ngroups=a.fd_ngroups, coil_fail_policy=a.coil_fail_policy,
                    coil_nonfinite_policy=a.coil_nonfinite_policy,
                    ls_verbose=a.ls_verbose, overlap_eq_and_coils=a.overlap,
                    eq_jacobian=a.eq_jacobian, exact_jac_gauge=a.exact_jac_gauge,
                    skip_coil_on_rejected=a.skip_coil_on_rejected,
                    skip_coil_scope=a.skip_coil_scope, stall_stop=a.stall_stop,
                    stall_delta_factor=a.stall_delta_factor,
                    carry_trust_radius=a.carry_trust_radius,
                    spectral_scale=a.spectral_scale,
                    surface_dof_reset_threshold=a.surface_dof_reset_threshold,
                    flux_target_floor=a.flux_target_floor, final_mode_flux_target=a.final_mode_flux_target,
                    flux_target_bn=a.flux_target_bn,
                    resume_keep_target=a.resume_keep_target,
                    start_step=a.start_step, final_mode_kkt_exit=a.final_mode_kkt_exit,
                    kkt_stationarity_tol=a.kkt_stationarity_tol, kkt_extra_outers=a.kkt_extra_outers,
                    kkt_active_tol=a.kkt_active_tol, kkt_phr_active=a.kkt_phr_active,
                    final_coil_stage=a.final_coil_stage, full_outer_budget=a.full_outer_budget,
                    abort_doomed_coil_solves=a.abort_doomed_coil_solves,
                    certify_trials=a.certify_trials, reuse_vmex_state=a.reuse_vmex_state,
                    phase0_reporting=a.phase0_reporting,
                    coil_oracle=a.coil_oracle, oracle_d_inherit=a.oracle_d_inherit,
                    oracle_d_reset=a.oracle_d_reset, oracle_k_max=a.oracle_k_max,
                    oracle_kkt_factor=a.oracle_kkt_factor, oracle_kkt_floor=a.oracle_kkt_floor,
                    oracle_audit=a.oracle_audit, oracle_reset_guard=a.oracle_reset_guard,
                    oracle_d_floor=a.oracle_d_floor, oracle_floor_rule=a.oracle_floor_rule,
                    oracle_floor_k_max=a.oracle_floor_k_max,
                    carry_flux_multiplier=a.carry_flux_multiplier, carry_flux_from_mode=a.carry_flux_from_mode,
                    phase_timers=bool(a.phase_timers),
                    fb_reuse_tol=a.fb_reuse_tol, fb_reuse_every=a.fb_reuse_every, fb_jit=a.fb_jit)
    comps = build_default_components(n_workers=a.n_workers, cold_cache=not a.certify_trials)
    closers = comps.pop("closers")
    try:
        r = PHRRunner(cfg, out_dir=a.run_dir, resume_from=a.resume_from,
                      init_surface=a.init_surface,
                      argv=(list(_sys.argv) if argv is None else ["phrport.phr"] + list(argv)),
                      **comps)
        if a.sweep:
            r.sweep(SweepConfig(ratio=a.sweep_ratio, max_points=a.sweep_max_points,
                                min_target=a.sweep_min_target, max_qs=a.sweep_max_qs,
                                outers=a.sweep_outers, jf0=a.sweep_jf0,
                                k0=a.sweep_k0, rho_balance=a.sweep_rho_balance,
                                infeasible_stop=a.sweep_infeasible_stop,
                                infeasible_tol=a.sweep_infeasible_tol, seed_point=a.sweep_seed_point,
                                polish=a.sweep_polish))
        else:
            r.run()
            if a.target_flux > 0.0:
                # roadmap A5 (opt-in): standalone front point.  After the normal run, ramp the absolute
                # flux target geometrically (ratio <= --sweep-ratio) down to E inside the final mode with
                # the A4 sweep machinery, so the run reaches E the way a front sweep does.
                with open(os.path.join(a.run_dir, "final_metrics.json")) as fh:
                    jf_end = float(json.load(fh)["coils"]["squared_flux_Jf"])
                if jf_end > a.target_flux * (1.0 + 1e-3):
                    n_ramp = int(np.ceil(np.log(a.target_flux / jf_end) / np.log(a.sweep_ratio) - 1e-9))
                    ratio = (a.target_flux / jf_end) ** (1.0 / n_ramp)
                    print(f"[target-flux] ramp Jf {jf_end:.6e} -> {a.target_flux:.6e} in {n_ramp} step(s), "
                          f"ratio {ratio:.4f}", flush=True)
                    r2 = PHRRunner(cfg, out_dir=os.path.join(a.run_dir, "ramp"), resume_from=a.run_dir,
                                   init_surface=None,
                                   argv=(list(_sys.argv) if argv is None else ["phrport.phr"] + list(argv)),
                                   **comps)
                    r2.sweep(SweepConfig(ratio=ratio, max_points=n_ramp, outers=a.sweep_outers, jf0=jf_end,
                                         rho_balance=a.sweep_rho_balance,
                                         infeasible_stop=a.sweep_infeasible_stop,
                                         infeasible_tol=a.sweep_infeasible_tol, polish=a.sweep_polish))
                else:
                    print(f"[target-flux] run already at Jf {jf_end:.6e} <= {a.target_flux:.6e}; no ramp",
                          flush=True)
    finally:
        for c in closers:
            try:
                c()
            except Exception:  # noqa: BLE001
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
