"""Every user-facing option of the package, collected from the code, with its documentation.

The inventory is built by introspection (``inventory()``): the profile keys the loader reads (every
``_PROB.get(section, key, default)`` in phrport, plus the dataclass fields of PHRConfig / CoilConfig that read
the profile), the keys of ``equilibrium.qi_options`` / ``mercier_options`` (keyword arguments
of the functions they are passed to), the argparse options of every entry point (qss-run / python -m
phrport.phr, qss-coils, qss-check) and the ``PHRPORT_*`` environment variables. ``DOC`` holds the
hand-written part (group, units, physics, description) for every one of them; ``tools/gen_reference.py`` turns
both into docs/reference.md and tests/test_options_documented.py fails when an option has no ``DOC`` entry.

Keys of ``DOC``:
  "phr.<field>"   PHRConfig field (profile section "phr" and/or a qss-run flag of the same name)
  "coils.<field>" CoilConfig field (profile section "coils")
  "eq.<key>"      profile section "equilibrium"
  "qi.<key>"      equilibrium.qi_options       "merc.<key>" equilibrium.mercier_options
  "run.<dest>"    qss-run / phrport.phr flag without a PHRConfig field
  "coilscli.<dest>" qss-coils flag              "checkcli.<dest>" qss-check flag               "seedcli.<dest>" qss-seed option
  "env.<NAME>"    environment variable          "profile.<key>" top-level profile key
"""
from __future__ import annotations

import dataclasses
import inspect
import os
import re

_ROOT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------------------------------------------
# groups (order of docs/reference.md)
# ---------------------------------------------------------------------------------------------------------------
GROUPS = {
    "problem": "Problem and physics (what is optimized)",
    "eqcon": "Equilibrium constraints",
    "coilcon": "Coil constraints (engineering limits)",
    "fluxbound": "Coil-fit flux bound (the coil row of the outer problem)",
    "coilsolver": "Coil solver / XCoil",
    "finitebeta": "Finite beta (plasma field in the coil problem)",
    "optimizer": "Continuation and optimizer",
    "resolution": "Resolution (equilibrium, QI and coil grids)",
    "perf": "Parallelism and performance",
    "io": "Input / output",
    "resume": "Checkpoints and resume",
    "reporting": "Reporting and diagnostics",
    "sweep": "Front sweeps",
    "experimental": "Experimental / advanced",
    "internal": "Internal constants (fixed in the code, not settable)",
}

ALL, QS, QI, FB = "all", "QS", "QI", "finite beta"


def _d(group, desc, units="-", physics=ALL):
    return dict(group=group, desc=desc, units=units, physics=physics)


DOC = {
    # ------------------------------------------------------------------ profile top level
    "profile.name": _d("io", "Free-form name of the problem (reported in the outputs)."),
    "profile.run": _d("io", "Optional dictionary of qss-run options (flag names without dashes, e.g. "
                      "{\"n_continuation_steps\": 5, \"flux_target_frac\": 0.5}); read by qss-run only. Command-line "
                      "flags override it, it overrides --preset. See docs/profile.md."),
    # ------------------------------------------------------------------ equilibrium section
    "eq.input_file": _d("problem", "VMEC input deck (INDATA namelist): the seed boundary, nfp, pressure / current "
                        "profiles, PHIEDGE. Absolute, relative to the profile's folder or to the repository, or "
                        "__ROOT__/... Default: inputs/input.nfp2_QA."),
    "eq.physics": _d("problem", "Objective: \"qs\" = quasi-symmetry residual (VMEX QuasisymmetryRatioResidual), "
                     "\"qi\" = smooth constructed quasi-isodynamic residual (Goodman et al. 2023)."),
    "eq.nfp": _d("problem", "Number of field periods (must match the deck)."),
    "eq.helicity_m": _d("problem", "QS helicity M (1 for QA and QH).", physics=QS),
    "eq.helicity_n": _d("problem", "QS helicity N (0 for QA, nfp for QH; sign as in simsopt).", physics=QS),
    "eq.qs_surfaces": _d("problem", "Normalized-flux surfaces of the QS residual (default 0, 0.1, ..., 1).",
                         physics=QS),
    "eq.qi_surfaces": _d("problem", "Normalized-flux surfaces of the constructed-QI residual (default 0.1, 0.3, "
                         "0.5, 0.7, 0.9).", physics=QI),
    "eq.qi_mirror_s": _d("eqcon", "Surface of the mirror ratio when qi_options.mirror_all is false (default 1/51).",
                         physics=QI),
    "eq.qi_options": _d("problem", "QI residual, mirror, max-J and well-shaping options (the \"qi.*\" entries).",
                        physics=QI),
    "eq.mercier_options": _d("eqcon", "Mercier metric options (the \"merc.*\" entries); setting it computes the "
                             "metric, phr.mercier adds the row."),
    "eq.mercier_jac_batch": _d("perf", "DOF columns per batched Mercier pushforward in the exact Jacobian (memory "
                               "vs speed)."),
    "eq.surf_mpol": _d("resolution", "Boundary Fourier table: poloidal modes m <= surf_mpol are DOFs (as continuation "
                       "reaches them)."),
    "eq.surf_ntor": _d("resolution", "Boundary Fourier table: toroidal modes |n| <= surf_ntor."),
    "eq.seed_ellipse": _d("problem", "If the deck is exactly axisymmetric, add this (2,1) rotating ellipse to it "
                          "(PHR's start: 0.07); 0 keeps the deck unchanged (docs/seeding.md)."),
    "eq.free_r00": _d("experimental", "Also free rc(0,0) (major radius); PHR keeps it fixed."),
    "eq.vmec_mpol": _d("resolution", "Fixed VMEC resolution mpol = ntor at every step (default: 3 + continuation "
                       "step, PHR)."),
    "eq.vmec_ntor": _d("resolution", "Fixed VMEC ntor different from vmec_mpol (e.g. to resolve ripple modes)."),
    "eq.ns_array": _d("resolution", "VMEC radial ladder: surfaces per stage (PHR 16, 50; paper 16/50 or 25/50)."),
    "eq.ftol_array": _d("resolution", "VMEC force tolerance per stage."),
    "eq.niter_array": _d("resolution", "VMEC iteration cap per stage."),
    "eq.ntor_extra": _d("optimizer", "Continuation step k frees |n| <= k + ntor_extra (capped at surf_ntor) instead of "
                        "|n| <= k: lets the axis torsion / bean rotation in early (tokamak cold starts)."),
    "eq.ripple_band": _d("experimental", "Extra free modes m <= m_max, n_min <= |n| <= n_max from step from_step on "
                         "(dict with those keys), so the boundary can follow the coil ripple."),
    "eq.deck_ntor_max": _d("experimental", "Drop deck boundary modes with |n| > this from the seed."),
    # ------------------------------------------------------------------ qi_options
    "qi.mboz": _d("resolution", "Boozer poloidal resolution of the traceable Boozer transform.", physics=QI),
    "qi.nboz": _d("resolution", "Boozer toroidal resolution.", physics=QI),
    "qi.oversample": _d("resolution", "Boozer transform angular oversampling.", physics=QI),
    "qi.nphi": _d("resolution", "Points per field line (one field period) of the QI residual.", physics=QI),
    "qi.nalpha": _d("resolution", "Field lines per surface of the QI residual.", physics=QI),
    "qi.n_bounce": _d("resolution", "Bounce levels of the constructed-QI target.", physics=QI),
    "qi.phimin": _d("problem", "Toroidal origin of the field-line grid.", physics=QI),
    "qi.soft": _d("problem", "Smoothing temperature of the constructed-QI residual (relative to <B>).", physics=QI),
    "qi.weights": _d("problem", "Per-surface weights of the QI residual.", physics=QI),
    "qi.alpha_weights": _d("problem", "Weight field lines by their bounce structure.", physics=QI),
    "qi.alpha_weight_floor": _d("problem", "Floor of the field-line weights.", physics=QI),
    "qi.mirror_soft": _d("eqcon", "Smoothing of B_max / B_min in the mirror ratio.", physics=QI),
    "qi.mirror_all": _d("eqcon", "Mirror ratio = soft max over every QI surface (+ mirror surface), so mirror_max "
                        "caps all of them.", physics=QI),
    "qi.mirror_tau": _d("eqcon", "Soft-max temperature of mirror_all (alias of mirror_all_tau; paper 0.002).",
                        physics=QI),
    "qi.mirror_all_tau": _d("eqcon", "Soft-max temperature of mirror_all (default 5e-4 when mirror_tau is unset).",
                            physics=QI),
    "qi.maxj_target": _d("eqcon", "Max-J target slope of the constructed field (row: soft max of slope - target "
                         "<= 0); setting it computes the metric, phr.maxj adds the row. Paper: 1.0.", physics=QI),
    "qi.maxj_depths": _d("eqcon", "Trapping depths (fraction of the well) of the constructed max-J condition.",
                         physics=QI),
    "qi.maxj_tau": _d("eqcon", "Soft-max temperature of the max-J row.", physics=QI),
    "qi.jreal_surfaces": _d("eqcon", "Surfaces of the real-field J-contour closure (folded into the max-J row); "
                            "None = off.", physics=QI),
    "qi.jreal_fracs": _d("eqcon", "Trapped-particle levels B* (fractions of the band trapped on every line).",
                         physics=QI),
    "qi.jreal_target": _d("eqcon", "Closure target (scalar or one per level; with measure \"starts\" the allowed "
                          "mean opening).", physics=QI),
    "qi.jreal_nphi": _d("resolution", "Points per field line of the J integrals.", physics=QI),
    "qi.jreal_nalpha": _d("resolution", "Field lines of the J integrals.", physics=QI),
    "qi.jreal_tau": _d("eqcon", "Soft-max temperature over pairs / levels.", physics=QI),
    "qi.jreal_order": _d("eqcon", "\"max\" (max-J ordering) or \"either\" (max-J or min-J ordering counts as "
                         "closed).", physics=QI),
    "qi.jreal_enclose_inner": _d("eqcon", "Number of inner (birth) surfaces whose J range must be enclosed by some "
                                 "outer surface.", physics=QI),
    "qi.jreal_measure": _d("eqcon", "\"range\" (surface-pair J ranges) or \"starts\" (each start scored, like "
                           "a closure count).", physics=QI),
    "qi.jreal_tau_sep": _d("eqcon", "Soft-min temperature over outer surfaces (measure starts).", physics=QI),
    "qi.jreal_tau_alpha": _d("eqcon", "Soft extrema over field lines (relative to <J>).", physics=QI),
    "qi.jreal_width": _d("eqcon", "Softplus width of an open start (measure starts).", physics=QI),
    "qi.jreal_row_weights": _d("eqcon", "Weights of the inner surfaces (measure starts).", physics=QI),
    "qi.bmaxsp_target": _d("eqcon", "B_max spread across field lines, (max - min over alpha of B_max)/B00 <= target "
                           "(row phr.bmaxsp).", physics=QI),
    "qi.bmaxsp_soft": _d("eqcon", "Soft max of B along a line (relative to <B>).", physics=QI),
    "qi.bmaxsp_tau_alpha": _d("eqcon", "Soft extrema over field lines (relative to B00).", physics=QI),
    "qi.bmaxsp_tau": _d("eqcon", "Soft max over surfaces of the spread row.", physics=QI),
    "qi.bmaxsp_smax": _d("eqcon", "Only QI surfaces with s <= this enter the spread row.", physics=QI),
    "qi.bmaxsp_which": _d("eqcon", "\"max\" (B_max spread), \"min\" (B_min spread) or \"both\".", physics=QI),
    "qi.bminslope_target": _d("eqcon", "Also cap d<B_min>/ds / B00 between kept surfaces (same row).", physics=QI),
    "qi.bmin_weight": _d("problem", "> 0: well-bottom flatness rows (B_min alpha-variance) appended to the QI "
                         "residual with this weight.", physics=QI),
    "qi.bmin_surfaces": _d("problem", "Extra surfaces of the well-bottom flatness rows.", physics=QI),
    "qi.bmin_soft": _d("problem", "Soft min along a line of the flatness rows.", physics=QI),
    "qi.bminu_surfaces": _d("experimental", "Surfaces of the field-line B_min uniformity rows appended to the QI "
                            "residual (renamed from the lf tree's bmin_*).", physics=QI),
    "qi.bminu_nalpha": _d("experimental", "Field lines of the B_min uniformity rows.", physics=QI),
    "qi.bminu_nphi": _d("experimental", "Points per line of the B_min uniformity rows.", physics=QI),
    "qi.bminu_tau": _d("experimental", "Soft-min temperature of the B_min uniformity rows.", physics=QI),
    "qi.bminu_weight": _d("experimental", "Weight of the B_min uniformity rows.", physics=QI),
    "qi.djds_surfaces": _d("experimental", "Surfaces of the shallow-well real-field dJ/ds rows (penalize dJ/ds > 0).",
                           physics=QI),
    "qi.djds_depths": _d("experimental", "Well depths of the dJ/ds rows.", physics=QI),
    "qi.djds_nalpha": _d("experimental", "Field lines of the dJ/ds rows.", physics=QI),
    "qi.djds_nphi": _d("experimental", "Points per line of the dJ/ds rows.", physics=QI),
    "qi.djds_tau_g": _d("experimental", "Softplus temperature of the dJ/ds rows.", physics=QI),
    "qi.djds_weight": _d("experimental", "Weight of the dJ/ds rows.", physics=QI),
    "qi.jspread_depths": _d("experimental", "Well depths of the J-spread rows (on djds_surfaces).", physics=QI),
    "qi.jspread_weight": _d("experimental", "Weight of the J-spread rows (None = off).", physics=QI),
    # ------------------------------------------------------------------ mercier options
    "merc.s_min": _d("eqcon", "Inner edge of the Mercier window (normalized flux)."),
    "merc.s_max": _d("eqcon", "Outer edge of the Mercier window."),
    "merc.tau": _d("eqcon", "Soft-min temperature of s DMerc over the window."),
    # ------------------------------------------------------------------ phr: physics rows
    "phr.aspect_target": _d("eqcon", "Aspect ratio A*: row A - A* <= 0 (or the two-sided band)."),
    "phr.aspect_two_sided": _d("eqcon", "Aspect as a band |A - A*| <= aspect_tol (two rows) instead of A <= A*."),
    "phr.aspect_tol": _d("eqcon", "Half width of the two-sided aspect band (0 = equality, paper)."),
    "phr.aspect_off": _d("eqcon", "Drop the aspect rows (when volume and minor radius are constrained)."),
    "phr.iota_mean_target": _d("eqcon", "Mean rotational transform target: band |iota_mean - t| <= iota_tol."),
    "phr.iota_tol": _d("eqcon", "Half width of the mean-iota band (PHR 0)."),
    "phr.iota_mean_off": _d("eqcon", "Drop the mean-iota rows (use axis / edge bands instead)."),
    "phr.iota_axis_target": _d("eqcon", "Axis iota band |iota(0) - t| <= iota_end_tol (0 = off)."),
    "phr.iota_edge_target": _d("eqcon", "Edge iota band |iota(1) - t| <= iota_end_tol, or iota_edge_tol (0 = off)."),
    "phr.iota_end_tol": _d("eqcon", "Half width of the axis / edge iota bands."),
    "phr.iota_edge_tol": _d("eqcon", "Half width of the edge band only (0 = iota_end_tol)."),
    "phr.elongation_threshold": _d("eqcon", "Maximum cross-section elongation (always a row)."),
    "phr.vacuum_well": _d("eqcon", "Add the magnetic-well row W >= well_threshold (also a qss-run flag)."),
    "phr.well_threshold": _d("eqcon", "Minimum magnetic well W = (V'(0) - V'(1))/V'(0) (negative = hill allowed)."),
    "phr.mirror_max": _d("eqcon", "Mirror ratio cap (B_max - B_min)/(B_max + B_min) <= this (0 = off).", physics=QI),
    "phr.mirror_min": _d("eqcon", "Mirror ratio lower bound (0 = off).", physics=QI),
    "phr.maxj": _d("eqcon", "Add the max-J row (qi_options.maxj_target and / or jreal_* must be set).", physics=QI),
    "phr.maxj_from_step": _d("eqcon", "Max-J row active from continuation step K (0-based; held at g = -1 before, "
                             "multipliers carried). Paper QI: 5 (max-J late).", physics=QI),
    "phr.maxj_ramp_outers": _d("experimental", "Lower the max-J target linearly from its switch-on value over N outers "
                               "(0 = off).", physics=QI),
    "phr.maxj_track_frac": _d("experimental", "Each outer asks for a relative max-J decrease r (0 = off; replaces "
                              "the ramp).", physics=QI),
    "phr.maxj_track_floor": _d("experimental", "Tracking stops (true row) once the max-J value <= this.", physics=QI),
    "phr.maxj_switch_delta": _d("experimental", "Trust radius at the start of every step from maxj_from_step on.",
                                physics=QI),
    "phr.bmaxsp": _d("eqcon", "Add the B_max spread row (qi_options.bmaxsp_target).", physics=QI),
    "phr.bmaxsp_from_step": _d("eqcon", "B_max spread row active from step K.", physics=QI),
    "phr.mercier": _d("eqcon", "Add the Mercier row (needs equilibrium.mercier_options)."),
    "phr.mercier_margin": _d("eqcon", "Mercier row (margin - softmin s DMerc)/scale <= 0."),
    "phr.mercier_scale": _d("eqcon", "Normalization of the Mercier row."),
    "phr.volume_target": _d("eqcon", "Plasma volume band |V/V* - 1| <= volume_tol (0 = off).", units="m^3"),
    "phr.volume_tol": _d("eqcon", "Relative half width of the volume band."),
    "phr.aminor_target": _d("eqcon", "Minor radius band |a/a* - 1| <= aminor_tol (0 = off).", units="m"),
    "phr.aminor_tol": _d("eqcon", "Relative half width of the minor-radius band."),
    "phr.qs_max": _d("eqcon", "QS cap: QS_total / qs_max - 1 <= 0 (0 = off); lets a tight coil bound trade QS."),
    # ------------------------------------------------------------------ phr: finite beta
    "phr.finite_beta": _d("finitebeta", "Plasma field (virtual casing on the VMEX state) in the coil residual, with "
                          "its envelope-gradient term (needs --eq-jacobian exact for the field term).", physics=FB),
    "phr.fb_field_sign": _d("finitebeta", "Orientation of the plasma field relative to the coils (+1 / -1; LP QA "
                            "beta: -1).", physics=FB),
    "phr.fb_current_from_G": _d("finitebeta", "Scale the coil currents to the current the equilibrium links "
                                "(lambda_G = pi |G_edge| / (mu0 nfp I)).", physics=FB),
    "phr.fb_n": _d("finitebeta", "Virtual-casing grid n x n per field period.", physics=FB),
    "phr.fb_digits_value": _d("finitebeta", "Virtual-casing accuracy (digits) of the plasma field values.", physics=FB),
    "phr.fb_digits_grad": _d("finitebeta", "Virtual-casing accuracy (digits) of the field term.", physics=FB),
    "phr.fb_jit": _d("perf", "Jitted virtual casing: vjp (default), off (eager), check (compare both).", physics=FB),
    "phr.fb_reuse_tol": _d("perf", "Reuse the plasma field while max|dx| < D since the last refresh (0 = off; tested "
                           "2e-3, 1.5x faster).", physics=FB),
    "phr.fb_reuse_every": _d("perf", "Refresh the reused plasma field at least every K Jacobians.", physics=FB),
    # ------------------------------------------------------------------ phr: flux bound
    "phr.enable_coils": _d("fluxbound", "Coils in the loop (flag --no-coils turns them off: equilibrium only, "
                           "two-stage first stage)."),
    "phr.flux_metric_scale": _d("fluxbound", "Scale S of the coil row S c0 (paper 500)."),
    "phr.flux_target_frac": _d("fluxbound", "Adaptive bound: at the start of each mode the target is frac x the "
                               "achieved c0 (paper 0.5; <= 0 = use flux_phr_target)."),
    "phr.flux_phr_target": _d("fluxbound", "Absolute c0 target when flux_target_frac <= 0."),
    "phr.flux_target_floor": _d("fluxbound", "Captured target = max(frac c0, floor) (stops frac^k compounding)."),
    "phr.final_mode_flux_target": _d("fluxbound", "Absolute c0 target in the final mode (0 = off)."),
    "phr.flux_target_bn": _d("fluxbound", "Bound on <|B.n|>/<|B|>: captured target c0 (b*/b)^2, never above the frac "
                             "target (0 = off)."),
    "phr.cs_threshold": _d("fluxbound", "Phantom-flux guard: below 0.5 x this coil-surface distance (or c0 <= 1e-7) "
                           "the honest mean (B.n/|B|)^2 replaces c0 and the flux gradient is dropped.", units="m"),
    "phr.carry_flux_multiplier": _d("fluxbound", "Keep the coil row's multiplier and penalty across modes (target "
                                    "still re-captured)."),
    "phr.carry_flux_from_mode": _d("fluxbound", "Carry only into modes >= this."),
    # ------------------------------------------------------------------ phr: optimizer
    "phr.n_continuation_steps": _d("optimizer", "Continuation steps: step k frees boundary modes |m|, |n| <= k."),
    "phr.al_outer_iter": _d("optimizer", "Augmented-Lagrangian outers per step (multiplier / penalty updates)."),
    "phr.max_nfev": _d("optimizer", "Function evaluations per inner least-squares solve (paper 30)."),
    "phr.rho_init": _d("optimizer", "Initial PHR penalty of every row."),
    "phr.rho_tau": _d("optimizer", "Penalty growth factor when a violation does not decrease enough."),
    "phr.rho_theta": _d("optimizer", "Required violation decrease ratio before the penalty grows."),
    "phr.rho_max": _d("optimizer", "Penalty cap."),
    "phr.eq_jacobian": _d("optimizer", "Equilibrium Jacobian block: fd (PHR forward differences) or exact (implicit "
                          "VMEX tangents; paper)."),
    "phr.exact_jac_gauge": _d("optimizer", "Gauge of the exact Jacobian map: sqrt (default) or frozen."),
    "phr.surface_dof_reset_threshold": _d("optimizer", "Warm coil start if the boundary moved by <= this, else cold."),
    "phr.coil_fail_policy": _d("optimizer", "Coil result not ok but finite: use it (PHR) or sentinel (reject)."),
    "phr.coil_nonfinite_policy": _d("optimizer", "Non-finite coil result: sentinel (reject trial) or phr_nan."),
    "phr.skip_surface_opt": _d("optimizer", "Skip the boundary optimization (coils / metrics only)."),
    "phr.spectral_scale": _d("experimental", "TRF step scaling (1 + m^2 + n^2)^(-P/2) per boundary DOF (0 = off)."),
    "phr.carry_trust_radius": _d("optimizer", "Start each inner solve with the previous largest accepted step (paper "
                                 "preset; --no-carry-trust-radius undoes it, as in the QI tokamak start)."),
    "phr.stall_stop": _d("experimental", "Stop an inner solve after K rejected trials at a collapsed radius (0 = off)."),
    "phr.stall_delta_factor": _d("experimental", "Collapsed-radius factor of stall_stop."),
    "phr.full_outer_budget": _d("optimizer", "Run every outer of every mode even when all rows are feasible (paper)."),
    "phr.final_mode_kkt_exit": _d("optimizer", "Final mode exits only when KKT feasibility and stationarity pass."),
    "phr.kkt_stationarity_tol": _d("optimizer", "Stationarity tolerance of the final-mode KKT exit."),
    "phr.kkt_extra_outers": _d("optimizer", "Extra outers allowed by the final-mode KKT exit."),
    "phr.kkt_active_tol": _d("reporting", "Rows with g >= -tol enter the KKT multiplier fit."),
    "phr.kkt_phr_active": _d("reporting", "Rows PHR treats as active (lambda > 0) also enter the KKT fit."),
    "phr.final_coil_stage": _d("coilsolver", "Final coils from the certified cold-start + warm-polish stage."),
    "phr.coil_oracle": _d("coilsolver", "Coil warm-start rule: phr, or anchored (warm only from accepted points, "
                          "per-outer cold audit; paper)."),
    "phr.oracle_d_inherit": _d("coilsolver", "Anchored oracle: boundary distance for an inherited warm solve."),
    "phr.oracle_d_reset": _d("coilsolver", "Anchored oracle: distance for a penalty-reset warm solve (cold beyond)."),
    "phr.oracle_k_max": _d("coilsolver", "Anchored oracle: outers of the inherited warm solve (paper 4)."),
    "phr.oracle_kkt_factor": _d("coilsolver", "Anchored oracle KKT guard factor."),
    "phr.oracle_kkt_floor": _d("coilsolver", "Anchored oracle KKT guard floor."),
    "phr.oracle_audit": _d("coilsolver", "Anchored oracle: per-outer cold audit keeping the lower Lagrangian."),
    "phr.oracle_reset_guard": _d("coilsolver", "Anchored oracle: replace a doubtful reset result by a cold solve."),
    "phr.oracle_d_floor": _d("experimental", "Anchored oracle noise floor distance (0 = off)."),
    "phr.oracle_floor_rule": _d("experimental", "Noise-floor rule: linear or converge."),
    "phr.oracle_floor_k_max": _d("experimental", "Outers of the noise-floor converge rule."),
    "phr.start_step": _d("resume", "Start the continuation at step K (with --resume-from: chain stages)."),
    "phr.resume_keep_target": _d("resume", "Resume keeps the checkpoint's flux target, coil multiplier and penalty "
                                 "(continue an interrupted run)."),
    "phr.skip_coil_on_rejected": _d("perf", "Solve VMEX first and skip coil solves of trials TRF will reject (exact)."),
    "phr.skip_coil_scope": _d("perf", "Which evaluations are sequenced for skipping: head or all."),
    "phr.abort_doomed_coil_solves": _d("perf", "Stop a doomed trial's coil solve before its fallback starts (exact)."),
    "phr.certify_trials": _d("perf", "Certify trial coil solves (flag --no-certify-trials: only accepted coils; "
                             "paper)."),
    "phr.reuse_vmex_state": _d("perf", "Exact Jacobian reuses the evaluation's converged VMEX state (paper)."),
    "phr.overlap_eq_and_coils": _d("perf", "Run the evaluation's VMEX solve in the pool while coils solve (flag "
                                   "--no-overlap)."),
    "phr.fd_ngroups": _d("perf", "Finite-difference groups (PHR's MPI groups)."),
    "phr.eq_cache_size": _d("perf", "Equilibrium results kept in memory (lower it for large QI runs)."),
    "phr.phase_timers": _d("reporting", "Per-evaluation phase timing table phase_timers.jsonl (passive)."),
    "phr.ls_verbose": _d("reporting", "scipy least_squares verbosity."),
    "phr.phase0_reporting": _d("reporting", "provenance.json, eval_x.npz, KKT logging, final checkpoint, coil export "
                               "(flag --no-phase0-reporting turns them off)."),
    # internal
    "phr.v_tol": _d("internal", "Violation tolerance of the PHR updates (1e-4)."),
    "phr.lam_max": _d("internal", "Multiplier cap (1e3)."),
    "phr.flux_phantom_eps": _d("internal", "c0 below which the phantom guard triggers (1e-7)."),
    "phr.reinit_coils_each_alm": _d("internal", "PHR re-initializes the coils each AL outer."),
    "phr.fd_abs_step": _d("internal", "Absolute FD step (1e-5)."),
    "phr.fd_rel_step": _d("internal", "Relative FD step (1e-5)."),
    "phr.diff_method": _d("internal", "FD scheme (forward)."),
    "phr.ls_kwargs": _d("internal", "Extra scipy least_squares keyword arguments (Python API only)."),
    "phr.reuse_eq_cache": _d("internal", "Reuse bitwise-identical equilibrium results."),
    "phr.initial_eval_step": _d("internal", "Resolution step of the seed metrics."),
    "phr.kkt_grad_floor": _d("internal", "Floor of |grad QS| in the KKT stationarity ratio."),
    # ------------------------------------------------------------------ run-only flags
    "run.run_dir": _d("io", "Output folder (required)."),
    "run.self_test": _d("reporting", "Run the PHR update self-test and exit."),
    "run.resume_from": _d("resume", "Checkpoint (al1_checkpoint_*.json) to continue from."),
    "run.init_surface": _d("resume", "Start from the boundary of a checkpoint / JSON, with fresh multipliers."),
    "run.n_workers": _d("perf", "VMEX worker processes (laptop 2-6; GPU node 1 with exact Jacobian)."),
    "run.profile": _d("io", "Problem profile JSON (sets PHRPORT_PROBLEM)."),
    "run.preset": _d("io", "Flag set placed first: production (the published runs' solver settings; 'paper' is a "
                     "deprecated alias) or tiny (laptop smoke run). See docs/profile.md, Presets."),
    "run.help_phr": _d("io", "List the optimizer options and exit."),
    "run.sweep": _d("sweep", "Front sweep from --resume-from instead of the continuation run."),
    "run.sweep_ratio": _d("sweep", "Flux bound ratio between sweep points."),
    "run.sweep_max_points": _d("sweep", "Maximum sweep points."),
    "run.target_flux": _d("sweep", "After the run, ramp the absolute c0 bound down to E in the final mode (0 = off)."),
    "run.sweep_min_target": _d("sweep", "Stop before a point whose bound is below this."),
    "run.sweep_outers": _d("sweep", "Outer budget per sweep point."),
    "run.sweep_max_qs": _d("sweep", "Stop after a point with QS above this (0 = off)."),
    "run.sweep_jf0": _d("sweep", "Reference c0 of the sweep (default: from the seed)."),
    "run.sweep_k0": _d("sweep", "First sweep index."),
    "run.sweep_rho_balance": _d("sweep", "Coil penalty at the start of a point so that J2 = this x QS."),
    "run.sweep_infeasible_stop": _d("sweep", "Stop after this many points that miss their bound."),
    "run.sweep_infeasible_tol": _d("sweep", "A point misses its bound if the scaled coil violation exceeds this."),
    "run.sweep_seed_point": _d("sweep", "Also write the seed as a front point."),
    "run.sweep_polish": _d("sweep", "Point coils from the final coil stage (flag --sweep-no-polish: cold solve)."),
    # ------------------------------------------------------------------ qss-coils / qss-check
    "coilscli.profile": _d("io", "Problem profile (coil limits, deck, finite beta)."),
    "coilscli.run_dir": _d("io", "A qss-run folder: its final boundary and coil settings."),
    "coilscli.boundary": _d("io", "Boundary JSON ({'rc(m,n)': ..}) or a checkpoint."),
    "coilscli.out": _d("io", "Output stem (writes STEM.json, STEM.npz)."),
    "coilscli.override": _d("coilsolver", "JSON merged into the coil configuration."),
    "coilscli.quick": _d("coilsolver", "One certified cold solve instead of the final stage."),
    "coilscli.step": _d("finitebeta", "Continuation step of the equilibrium used for the plasma field."),
    "checkcli.profile": _d("io", "Profile whose input deck is checked."),
    # ------------------------------------------------------------------ qss-seed (docs/seeding.md)
    "seedcli.out": _d("io", "qss-seed: deck to write (use it as equilibrium.input_file)."),
    "seedcli.template": _d("io", "qss-seed: deck whose profiles and settings are kept (default "
                           "inputs/input.LandremanPaul2021_QA)."),
    "seedcli.nfp": _d("problem", "qss-seed: field periods of the new deck (default: the template's or the wout's)."),
    "seedcli.phiedge": _d("problem", "qss-seed: toroidal flux at the edge (default: the template's).", units="Wb"),
    "seedcli.pressure": _d("finitebeta", "qss-seed: generic finite-beta deck, p(s) = P0 (1 - s), current-free.",
                           units="Pa", physics=FB),
    "seedcli.R0": _d("problem", "qss-seed tokamak / rotating-ellipse: major radius.", units="m"),
    "seedcli.aspect": _d("problem", "qss-seed tokamak / rotating-ellipse: aspect ratio (area-equivalent for kappa != 1)."),
    "seedcli.kappa": _d("problem", "qss-seed tokamak: elongation."),
    "seedcli.delta": _d("problem", "qss-seed tokamak: triangularity."),
    "seedcli.brk": _d("problem", "qss-seed tokamak --break: symmetry break qa (1e-3 on (1,1), (2,1)), qi (re-signed "
                      "rotating ellipse + helical axis shift, 2e-3) or none."),
    "seedcli.perturb": _d("problem", "qss-seed tokamak: extra boundary terms m,n,rc,zs (repeatable)."),
    "seedcli.amp": _d("problem", "qss-seed rotating-ellipse: (1,1) ellipse amplitude relative to the minor radius."),
    "seedcli.axis_shift": _d("problem", "qss-seed rotating-ellipse: (0,1) helical axis shift.", units="m"),
    "seedcli.boundary": _d("io", "qss-seed from-boundary: boundary JSON (or a coil export holding one)."),
    "seedcli.wout": _d("io", "qss-seed from-wout: wout file whose last surface becomes the seed."),
    "seedcli.checkpoint": _d("io", "qss-seed from-checkpoint: checkpoint file or run folder."),
    "seedcli.profile": _d("io", "qss-seed from-checkpoint: the run's profile (to rebuild its full boundary)."),
    # ------------------------------------------------------------------ coils: limits
    "coils.n_coils": _d("coilcon", "Modular coils per half field period (stellarator symmetric; total 2 nfp n_coils)."),
    "coils.order": _d("coilcon", "Fourier order of each coil curve (paper 5-8)."),
    "coils.per_coil_length": _d("coilcon", "Length cap of every coil (length_mode per_coil).", units="m"),
    "coils.length_mode": _d("coilcon", "per_coil (each coil <= per_coil_length) or total (sum <= total_length)."),
    "coils.total_length": _d("coilcon", "Cap on the summed base-coil length (length_mode total).", units="m"),
    "coils.max_curvature": _d("coilcon", "Pointwise curvature cap (certified on 4096 samples).", units="1/m"),
    "coils.curvature_margin": _d("coilcon", "Solver-row margin on curvature / MSC (removed at certification)."),
    "coils.lp_curvature_max": _d("coilcon", "Lp-curvature cap per coil, (1/p) int max(kappa - kappa_0, 0)^p dl <= max "
                                 "(0 = off).", units="m^(1-p)"),
    "coils.lp_curvature_p": _d("coilcon", "Exponent p of the Lp curvature."),
    "coils.lp_curvature_threshold": _d("coilcon", "Threshold kappa_0 of the Lp curvature.", units="1/m"),
    "coils.max_msc": _d("coilcon", "Mean-squared curvature cap per coil.", units="1/m^2"),
    "coils.min_cc": _d("coilcon", "Minimum coil-coil distance (all periods / images; exact at certification).",
                       units="m"),
    "coils.min_cs": _d("coilcon", "Minimum coil-surface distance (exact triangles at certification).", units="m"),
    "coils.distance_margin": _d("coilcon", "Solver-row margin on the distance rows."),
    "coils.slot_rows": _d("coilcon", "Toroidal ordering rows: neighbouring coils (and end coils with their images) "
                          "may not cross in the (theta, phi) projection."),
    "coils.ordering_axis": _d("coilcon", "Reference axis of the ordering rows' poloidal angle: frozen_step (default) = "
                              "the boundary's geometric axis (m = 0 modes) at the start of each continuation step, "
                              "fixed within the step (rows depend on the coils only; stored in checkpoints); live = "
                              "recomputed from the current boundary at every evaluation (the previous behaviour)."),
    "coils.slot_gap_deg": _d("coilcon", "Required toroidal gap of the ordering rows (paper 1.5).", units="deg"),
    "coils.slot_kappa": _d("coilcon", "Poloidal-window penalty of the ordering rows.", units="rad"),
    "coils.slot_beta": _d("coilcon", "Soft-min sharpness of the ordering rows.", units="1/rad"),
    "coils.slot_seg": _d("coilcon", "Ordering rows per coil pair (must divide nq)."),
    "coils.link_guard": _d("coilcon", "none, link (reject steps / starts that change the coil linking numbers) or "
                           "link_floor (also keep a minimum distance)."),
    "coils.link_floor_frac": _d("coilcon", "link_floor: minimum distance as a fraction of min_cc."),
    "coils.total_current": _d("coilcon", "Sum of the base-coil currents per half period (fixed); free currents "
                              "redistribute it.", units="A"),
    "coils.free_currents": _d("internal", "Currents are free with a fixed total (PHR); not profile-settable."),
    "coils.max_force": _d("coilcon", "Max force per unit length per turn at the reference scale (0 = off); "
                          "pointwise rows, certified.", units="MN/m"),
    "coils.force_scale": _d("coilcon", "Length factor from the problem to the reference (reactor) scale."),
    "coils.force_current": _d("coilcon", "Total current per half period at the reference scale.", units="A"),
    "coils.force_turns": _d("coilcon", "Turns per coil (0 = 800 / n_coils)."),
    "coils.force_radius": _d("coilcon", "Conductor radius at the reference scale (self-force regularization).",
                             units="m"),
    "coils.force_margin": _d("coilcon", "Solver-row margin on the force rows (removed at certification)."),
    # ------------------------------------------------------------------ coils: solver
    "coils.nfp": _d("internal", "Field periods (from equilibrium.nfp)."),
    "coils.precision": _d("coilsolver", "mixed (FP32 kernels, FP64 accumulation) or fp64 (finite beta)."),
    "coils.start": _d("coilsolver", "Primary start family_label, e.g. offset_3d_L85 (offset curves at 85 % of the "
                      "length cap); families offset_3d, offset_planar, circle."),
    "coils.seed_coils": _d("coilsolver", "Starting coils of the first cold solve: a phrport-coils-v1 JSON / NPZ "
                           "(final_coils.json of a run, any order) or a points JSON (docs/seeding.md)."),
    "coils.warm_max_outer": _d("coilsolver", "AL outers of a warm coil solve (more for 6+ coils)."),
    "coils.viol_tol": _d("coilsolver", "Violation tolerance of the coil solver stop test (certified = v <= this)."),
    "coils.ftol": _d("coilsolver", "Objective stall tolerance of the coil solver."),
    "coils.ok_viol": _d("coilsolver", "A result is ok when its certified violation <= this."),
    "coils.nq": _d("resolution", "Quadrature points per coil."),
    "coils.phr_grid": _d("resolution", "Flux grid nphi x ntheta on the full torus (nphi divisible by 2 nfp)."),
    "coils.dist_grid": _d("resolution", "Coil-surface soft-min grid on the full torus."),
    "coils.verify_grid": _d("internal", "Certification triangulation per field period."),
    "coils.nq_verify": _d("internal", "Certification length quadrature (768)."),
    "coils.mpol_min": _d("internal", "Minimum boundary mode table of the coil surface."),
    "coils.ntor_min": _d("internal", "Minimum boundary mode table of the coil surface."),
    "coils.c0_ref": _d("internal", "Objective normalization."),
    "coils.w_arc": _d("internal", "Arclength gauge regularizer weight."),
    "coils.softmin_alpha": _d("internal", "Soft-min sharpness of the distance rows."),
    "coils.inner": _d("internal", "Inner LM iterations per AL outer."),
    "coils.inner_first": _d("internal", "Inner LM iterations of the first outer."),
    "coils.first_prog": _d("internal", "First-outer progress criterion."),
    "coils.rho0": _d("internal", "Initial AL penalty."),
    "coils.max_outer": _d("internal", "AL outers of a cold solve."),
    "coils.rho_corr": _d("internal", "Penalty of the correction step."),
    "coils.cc_rows": _d("internal", "Coil-coil rows in the partial Jacobian."),
    "coils.fallback_starts": _d("internal", "Fallback starts after the primary."),
    "coils.max_starts": _d("internal", "Cold starts tried before giving up."),
    "coils.certify": _d("internal", "Certify results."),
    "coils.kkt_act_tol": _d("internal", "Active-row tolerance of the multiplier estimate."),
    "coils.lagr_curv": _d("internal", "Lagrangian curvature Hessian in the LM model."),
    "coils.arc_penalty": _d("internal", "XCoil arclength-variation penalty (off)."),
    "coils.rho_warm_reset": _d("internal", "Warm solves restart the penalty (set by the anchored oracle)."),
    # ------------------------------------------------------------------ environment
    "env.PHRPORT_PROBLEM": _d("io", "Problem profile (set by --profile)."),
    "env.PHRPORT_XCOIL": _d("coilsolver", "XCoil backend: unset = the installed xcoil package (default); a path = a local checkout (the folder that contains xcoil/),."),
    "env.PHRPORT_VMEX_DEVICE": _d("perf", "cpu pins VMEX to the CPU (XCoil keeps the GPU)."),
    "env.PHRPORT_FB_CHUNK": _d("perf", "Virtual-casing source chunk (default 1024; 128 for <= 16 GB devices).",
                               physics=FB),
    "env.PHRPORT_INPUT": _d("io", "VMEC deck when the profile gives none."),
    "env.PHRPORT_TRUST_RESET_STEP": _d("optimizer", "1: reset the carried trust radius at a new continuation step "
                                       "(production preset)."),
    "env.PHRPORT_TRUST_FLOOR": _d("optimizer", "Floor on a carried trust radius (production preset 1e-3)."),
    "env.PHRPORT_TRUST_CARRY0": _d("resume", "Carried trust radius for a resumed run."),
    "env.PHRPORT_RESUME_EQ_LABELS": _d("resume", "Row labels of an old checkpoint without eq_labels."),
    "env.PHRPORT_SEED_ANCHOR": _d("resume", "Coil anchor JSON re-converged before the first coil solve."),
    "env.PHRPORT_FRAMES": _d("reporting", "1: write frames.jsonl (every evaluation, for movies)."),
    "env.PHRPORT_PHASE_TIMERS": _d("reporting", "1: same as --phase-timers."),
    "env.PHRPORT_CLEAR_CACHES_EVERY": _d("perf", "Drop JAX caches every N evaluations."),
    "env.PHRPORT_CLEAR_CACHES_RSS_MB": _d("perf", "Drop JAX caches once the process RSS exceeds M MB."),
    "env.PHRPORT_PYCACHE": _d("io", "Bytecode cache folder."),
    "env.PHRPORT_SCRATCH": _d("io", "Scratch folder."),
    "env.PHRPORT_ROOT": _d("io", "Data root (default: the repository)."),
    "env.PHRPORT_DIR": _d("internal", "(not a variable: module constant matched by the scan)"),
}

# flags whose argparse dest differs from the PHRConfig field
DEST_TO_FIELD = {"overlap": "overlap_eq_and_coils"}


# ---------------------------------------------------------------------------------------------------------------
# inventory
# ---------------------------------------------------------------------------------------------------------------
def _sources():
    out = {}
    for fn in sorted(os.listdir(_ROOT)):
        if fn.endswith(".py"):
            out[fn] = open(os.path.join(_ROOT, fn)).read()
    return out


def profile_keys():
    """{section: {key: default source text}} of every _PROB.get(section, key, default) call in phrport."""
    import ast
    keys = {}
    for src in _sources().values():
        for node in ast.walk(ast.parse(src)):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get"
                    and isinstance(node.func.value, (ast.Name, ast.Attribute))
                    and (getattr(node.func.value, "id", None) == "_PROB" or getattr(node.func.value, "attr", None) == "_PROB")
                    and len(node.args) >= 2 and all(isinstance(a, ast.Constant) for a in node.args[:2])):
                d = ast.unparse(node.args[2]) if len(node.args) > 2 else ""
                keys.setdefault(node.args[0].value, {}).setdefault(node.args[1].value, d)
    return keys


def _sig_defaults(fn, skip=()):
    return {k: (None if p.default is inspect.Parameter.empty else p.default)
            for k, p in inspect.signature(fn).parameters.items()
            if p.kind == p.KEYWORD_ONLY or (p.default is not inspect.Parameter.empty)
            if k not in skip}


def inventory():
    """List of option records: dict(key, section, setting, type, default)."""
    from phrport import phr, coils, cli, qi_residual, mercier
    rec = []
    pk = profile_keys()
    parser = phr.build_parser()
    flags = {a.dest: a for a in parser._actions if a.dest != "help"}
    fields = {f.name: f for f in dataclasses.fields(phr.PHRConfig)}
    field_flag = {DEST_TO_FIELD.get(d, d): a for d, a in flags.items()}

    def dflt(f):
        return f.default if f.default is not dataclasses.MISSING else "(factory)"

    rec.append(dict(key="profile.name", setting="profile: name", type="str", default="nfp2_QA"))
    rec.append(dict(key="profile.run", setting="profile: run", type="dict", default="{}"))
    for k, d in sorted(pk.get("equilibrium", {}).items()):
        rec.append(dict(key=f"eq.{k}", setting=f"profile: equilibrium.{k}", type="", default=d))
    phr_prof = set(pk.get("phr", {}))
    for name, f in fields.items():
        sets = []
        if name in phr_prof:
            sets.append(f"profile: phr.{name}")
        a = field_flag.get(name)
        if a is not None:
            sets.extend(a.option_strings)
        rec.append(dict(key=f"phr.{name}", setting=", ".join(sets) or "(not settable)",
                        type=getattr(f.type, "__name__", str(f.type)), default=dflt(f)))
    for k in sorted(phr_prof - set(fields)):
        rec.append(dict(key=f"phr.{k}", setting=f"profile: phr.{k}", type="", default=pk["phr"][k]))
    for d, a in flags.items():
        if DEST_TO_FIELD.get(d, d) in fields:
            continue
        rec.append(dict(key=f"run.{d}", setting=", ".join(a.option_strings),
                        type=(type(a.default).__name__ if a.default is not None else "str"), default=a.default))
    for a in cli.run_parser()._actions:
        if a.dest != "help":
            rec.append(dict(key=f"run.{a.dest}", setting=", ".join(a.option_strings) + " (qss-run)", type="str",
                            default=a.default))
    coil_prof = set(pk.get("coils", {}))
    for f in dataclasses.fields(coils.CoilConfig):
        rec.append(dict(key=f"coils.{f.name}",
                        setting=(f"profile: coils.{f.name}" if f.name in coil_prof else "(not settable)")
                        + ("" if f.name in coil_prof else ""),
                        type=getattr(f.type, "__name__", str(f.type)), default=dflt(f)))
    qi = _sig_defaults(qi_residual.qi_physics, skip=("state", "rt", "surfaces", "mirror_s", "opts"))
    qi.update(_sig_defaults(qi_residual.smooth_constructed_qi_residual,
                            skip=("bmnc_b", "xm_b", "xn_b", "iota_b", "nfp", "bmns_b")))
    qi.update({"nphi": 101, "nalpha": 21, "n_bounce": 31, "mboz": 16, "nboz": 16, "oversample": 2})
    for k, v in sorted(qi.items()):
        rec.append(dict(key=f"qi.{k}", setting=f"profile: equilibrium.qi_options.{k}", type="", default=v))
    for k, v in sorted(_sig_defaults(mercier.mercier_metric).items()):
        rec.append(dict(key=f"merc.{k}", setting=f"profile: equilibrium.mercier_options.{k}", type="", default=v))
    for a in cli.coils_parser()._actions:
        if a.dest != "help":
            rec.append(dict(key=f"coilscli.{a.dest}", setting=", ".join(a.option_strings) + " (qss-coils)",
                            type="str", default=a.default))
    for a in cli.check_parser()._actions:
        if a.dest != "help":
            rec.append(dict(key=f"checkcli.{a.dest}", setting=", ".join(a.option_strings) + " (qss-check)",
                            type="str", default=a.default))
    from phrport import seed as _seed
    seen = set()
    for sp in _seed.build_parser()._subparsers._group_actions[0].choices.values():
        for a in sp._actions:
            if a.dest != "help" and a.dest not in seen:
                seen.add(a.dest)
                rec.append(dict(key=f"seedcli.{a.dest}", setting=", ".join(a.option_strings or [a.dest]) + " (qss-seed)",
                                type="str", default=a.default))
    envs = sorted(set(re.findall(r"PHRPORT_[A-Z0-9_]+", "\n".join(_sources().values()))))
    for e in envs:
        rec.append(dict(key=f"env.{e}", setting=f"environment {e}", type="str", default=""))
    return rec


def undocumented():
    """Inventory keys without a DOC entry (the test asserts this is empty)."""
    return [r["key"] for r in inventory() if r["key"] not in DOC]


def known_profile_keys():
    """{section: set(keys)} a profile may use (for validation)."""
    import dataclasses as _dc
    from phrport import qi_residual, mercier
    pk = profile_keys()
    out = {s: set(v) for s, v in pk.items()}
    q = set(_sig_defaults(qi_residual.qi_physics, skip=("state", "rt", "surfaces", "mirror_s", "opts")))
    q |= set(_sig_defaults(qi_residual.smooth_constructed_qi_residual,
                           skip=("bmnc_b", "xm_b", "xn_b", "iota_b", "nfp", "bmns_b")))
    q |= {"nphi", "nalpha", "n_bounce", "mboz", "nboz", "oversample"}
    out["qi_options"] = q
    out["mercier_options"] = set(_sig_defaults(mercier.mercier_metric))
    return out


def validate_profile(data, run_flags=None):
    """Messages for unknown keys / sections of a profile dict (empty list = clean)."""
    msgs = []
    known = known_profile_keys()
    for sec in data:
        if sec not in ("name", "equilibrium", "phr", "coils", "run") and not sec.startswith("_"):
            msgs.append(f"unknown section {sec!r} (sections: name, equilibrium, phr, coils, run)")
    for sec in ("equilibrium", "phr", "coils"):
        for k in (data.get(sec) or {}):
            if k not in known.get(sec, set()) and not k.startswith("_"):
                msgs.append(f"unknown key {sec}.{k} (ignored by the code)")
    eq = data.get("equilibrium") or {}
    for sub in ("qi_options", "mercier_options"):
        for k in (eq.get(sub) or {}):
            if k not in known[sub] and not k.startswith("_"):
                msgs.append(f"unknown key equilibrium.{sub}.{k}")
    if run_flags is not None:
        for k in (data.get("run") or {}):
            if k not in run_flags and not k.startswith("_"):
                msgs.append(f"unknown run option {k!r} (use the flag name without dashes, e.g. flux_target_frac)")
    return msgs
