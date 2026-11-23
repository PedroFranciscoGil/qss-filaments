# Constraint dictionary

Every objective and constraint the code can impose, with its definition, how to switch it on, typical values, how
strictly it is enforced, how its gradient is computed, what it costs, and the usual pitfalls. The full list of
option keys (including solver and resolution settings) is [reference.md](reference.md); how a profile is written is
[profile.md](profile.md).

## How the problem is posed

    minimize    f(boundary)                                   QS or QI residual (sum of squares)
    subject to  g_i(boundary) <= 0                             equilibrium rows (this page, part A)
                S c0*(boundary) <= S c0_target                 the coil row (part C)
    where       c0*(boundary) = min over coils of c0           the coil subproblem (XCoil), itself
                                subject to h_j(coils) <= 0     subject to the coil limits (part B)

* The outer problem is solved by the PHR augmented Lagrangian: each row gets a multiplier and a penalty, updated
  after every outer iteration; a row is **satisfied at convergence to the PHR tolerance** (v_tol = 1e-4 in its own
  normalized units), not exactly at every iterate. These are the **soft** rows. `final_metrics.json: surface.g_eq`
  lists every row's final value (feasible when <= 0).
* The coil limits are rows of the coil subproblem. XCoil works with smooth (soft-min / soft-max) versions on a
  quadrature, then **certifies** the result against the exact limits (exact segment / triangle distances,
  4096-sample curvature with golden refinement, Gauss linking numbers). These are the **hard** rows: a certified coil
  set has violation v <= `coils.viol_tol` on every one (v <= 1e-5 after `qss-coils`).
* Row values are normalized (relative to their limit or target), so they are O(1) and comparable.

**Gradients.** Equilibrium rows: forward finite differences by default (`--eq-jacobian fd`, any row), or exact
implicit derivatives through the converged VMEX state (`--eq-jacobian exact`, the paper setting; each row's
"exact key" below). The coil row: the envelope theorem (the coil Lagrangian differentiated with respect to the
boundary at fixed optimal coils and multipliers), plus, at finite beta, the virtual-casing field term. Coil limits:
analytic / JAX Jacobians inside XCoil with respect to the coil Fourier coefficients and currents.

**Cost.** "cheap" = negligible next to a VMEX solve; "moderate" = adds a traceable Boozer transform or similar
(tens of percent of an evaluation); "expensive" = can dominate an evaluation or a coil solve.

### Staging constraints (switching rows on later)

Two rows have a built-in "from step" switch: **max-J** (`phr.maxj_from_step`) and the **B_max spread**
(`phr.bmaxsp_from_step`): 0-based continuation step from which the row counts (held at g = -1 before; default 0 =
always). Every other row is either on for the whole run or off. To switch one on later (Mercier, mirror_min, volume,
a tighter coil limit, a higher beta), chain a stage: a new profile with the row, resumed from the previous stage's
final checkpoint with `--start-step K` (the new row starts with multiplier 0; see tutorial 5). Related step options:
`--start-step` (first continuation step of a run), `--carry-flux-from-mode` (with `--carry-flux-multiplier`: keep the
coil row's multiplier only into modes >= this; 1-based mode = step + 1), `maxj_switch_delta` (trust radius from the
max-J switch on).

---

## A. Equilibrium side

### A1. QS residual (objective)
- **Definition**: f = sum of squares of the quasi-symmetry ratio residual of simsopt / VMEX
  `QuasisymmetryRatioResidual` (helicity (M, N)) on `qs_surfaces`, evaluated on the VMEX wout tables.
- **Switch on**: `equilibrium.physics: "qs"` (default), `helicity_m`, `helicity_n` (QA: 1, 0; QH: 1, nfp),
  `qs_surfaces` (default 0, 0.1, ..., 1).
- **Gradient**: exact key `qs` (one row per surface point). **Cost**: cheap (on the wout tables).
- **Pitfalls**: the objective is a sum of squares, so its value scales with the number of surfaces; compare runs with
  the same `qs_surfaces`.

### A2. QI residual (objective)
- **Definition**: the smooth constructed-QI residual (Goodman et al. 2023): on each of `qi_surfaces`, |B| along
  `nalpha` field lines is compared with the quasi-isodynamic field constructed from it (wells squeezed so that the
  bounce distances are alpha-independent); residual = weighted sum of squared differences.
- **Switch on**: `equilibrium.physics: "qi"`, `qi_surfaces`, `qi_options` (`nphi`, `nalpha`, `n_bounce`, `mboz`,
  `nboz`, `soft`, ...).
- **Paper (QI runs)**: surfaces 0.1, 0.3, 0.5, 0.7, 0.9; nphi 101, nalpha 21, n_bounce 31, mboz = nboz = 16,
  soft 0.002, mirror_soft 2e-4.
- **Gradient**: exact key `qs` (traceable Boozer transform of the VMEX state). **Cost**: moderate.
- **Optional terms appended to the residual** (more rows in the sum of squares, not constraints):
  `bmin_weight` + `bmin_surfaces` (well-bottom flatness: alpha-variance of B_min), `bminu_surfaces` (field-line
  B_min uniformity), `djds_surfaces` (shallow-well real-field dJ/ds >= 0) and `jspread_weight` (J spread over alpha).
  Experimental; they change the meaning of the objective value.

### A3. Aspect ratio
- **Definition**: one-sided g = A - A*; two-sided g1 = A - (A* + tol), g2 = (A* - tol) - A.
  A = R_major / a_minor from the wout (VMEC definition).
- **Switch on**: always on unless `phr.aspect_off`; `phr.aspect_target` (default 4.5), `aspect_two_sided`, `aspect_tol`.
- **Paper**: LP QA A* = 6 two-sided, tol 0 (equality); W7-X-like QI A* = 10; NCSX-like A* = 4.4 two-sided.
- **Hard/soft**: soft. **Gradient**: exact key `aspect`. **Cost**: cheap.
- **Pitfalls**: with `aspect_tol: 0` the two rows form an equality: expect a small residual of order v_tol.

### A4. Mean rotational transform
- **Definition**: g1 = iota_mean - (t + tol), g2 = (t - tol) - iota_mean, iota_mean = mean of iota over the
  surfaces (excluding the axis).
- **Switch on**: on by default; `phr.iota_mean_target` (default 0.43), `iota_tol` (default 0); `iota_mean_off`.
- **Paper**: LP QA 0.4193 +- 0.002 (vacuum), 0.4958 +- 0.002 (beta 1 %).
- **Hard/soft**: soft. **Gradient**: exact key `mean_iota`. **Cost**: cheap.

### A5. Axis and edge rotational transform
- **Definition**: bands |iota(0) - t_axis| <= iota_end_tol and |iota(1) - t_edge| <= iota_end_tol (or `iota_edge_tol`
  for the edge only).
- **Switch on**: `phr.iota_axis_target`, `phr.iota_edge_target` (0 = off), `iota_end_tol` (default 0.01),
  `iota_edge_tol`; usually with `iota_mean_off: true`.
- **Paper**: QI 0.86 / 0.95 +- 0.01; NCSX-like 0.40 / 0.65 +- 0.02.
- **Hard/soft**: soft. **Gradient**: exact keys `iota_axis`, `iota_edge`. **Cost**: cheap.
- **Pitfalls**: keep the band away from low-order rationals (iota = 1 for nfp 5 QI) to avoid island chains.

### A6. Elongation
- **Definition**: g = E_max - E*, E_max = maximum over the toroidal cuts of the cross-section elongation
  (the PHR / simsopt MaxElongation metric on a 50 x 50 grid).
- **Switch on**: always on; `phr.elongation_threshold` (default 4.0).
- **Paper**: LP QA 4.602; QI 6.0; NCSX-like 3.5.
- **Hard/soft**: soft. **Gradient**: exact key `elongation`. **Cost**: cheap.

### A7. Magnetic well
- **Definition**: g = W* - W, W = (V'(0) - V'(1)) / V'(0) (simsopt `vacuum_well`; at finite beta the same formula on
  the finite-beta wout).
- **Switch on**: `phr.vacuum_well: true` (or `--vacuum-well`), `well_threshold` (default 0.015).
- **Paper**: LP QA vacuum -0.0304 (a hill allowed), beta 1 % +0.0159, NCSX-like +0.01.
- **Hard/soft**: soft. **Gradient**: exact key `well`. **Cost**: cheap.

### A8. Mirror ratio
- **Definition**: Delta = (B_max - B_min)/(B_max + B_min) on a surface (Goodman et al. Eq. 17-18), from the Boozer
  spectrum. Upper row g = Delta - Delta_max, lower row g = Delta_min - Delta. With `qi_options.mirror_all` Delta is a
  soft max (temperature `mirror_tau`) over every QI surface and the mirror surface, so the cap holds everywhere.
- **Switch on** (QI only): `phr.mirror_max`, `phr.mirror_min` (0 = off); `qi_options.mirror_all`, `mirror_tau`,
  `mirror_soft`; `equilibrium.qi_mirror_s` (default 1/51) when mirror_all is off.
- **Paper**: mirror_max 0.25 with mirror_all, tau 0.002; seed-sensitivity runs mirror_min 0.24-0.29.
- **Hard/soft**: soft. **Gradient**: exact key `mirror`. **Cost**: cheap once the Boozer transform exists.
- **Pitfalls**: the soft max overestimates Delta by <= tau log(n_surfaces): use a target slightly above the true cap.

### A9. Maximum-J
- **Definition**: one row g <= 0 combining (soft max at `maxj_tau`)
  (a) the constructed-field max-J condition: soft max over `maxj_depths` and surfaces of (dJ/ds slope - `maxj_target`);
  (b) if `jreal_surfaces` is set, the real-field J-contour closure: for trapped levels `jreal_fracs`, the J ranges
  of inner (birth) surfaces must be enclosed by outer ones (`jreal_order` max or either; `jreal_measure` range or
  starts); value = soft max of (opening - `jreal_target`).
- **Switch on** (QI only): `phr.maxj: true` plus `qi_options.maxj_target` and/or `qi_options.jreal_*`. Schedule:
  `phr.maxj_from_step` K (row held at g = -1 before step K, so its multiplier is 0; the paper's "max-J late"),
  `maxj_ramp_outers` (lower the target linearly over N outers), `maxj_track_frac` / `maxj_track_floor` (ask for a
  relative decrease per outer), `maxj_switch_delta` (trust radius at the switch).
- **Paper**: constructed-only runs maxj_target -0.06; the final QI runs maxj_target 1.0 (constructed part inactive)
  with the real-field closure on 15 surfaces x 36 lines x 7 levels, order either, measure starts, enclose_inner 8,
  target 0.1, from step 5 (2 in earlier campaigns).
- **Hard/soft**: soft. **Gradient**: exact key `maxj`. **Cost**: expensive with `jreal_*` (the J integrals dominate a
  QI evaluation); constructed-only is moderate.
- **Staging**: `maxj_from_step` K is 0-based (K = 5 means from the sixth continuation step on); before step K the row
  is held at g = -1 (no hinge, no gradient), so its multiplier stays 0 and the number of rows never changes; from step
  K on it is the true row. The step counter is the run's continuation step, so `--start-step` and `--resume-from` keep
  the schedule (a run resumed at step 6 with K = 5 has the row active).
- **Pitfalls**: switching the row on from the first step usually stalls QI progress: use maxj_from_step. The
  closure measure depends on nphi / nalpha: tiny resolutions are not meaningful.

### A10. B_max / B_min spread across field lines
- **Definition**: spread(s) = (max_alpha B_max,alpha - min_alpha B_max,alpha) / B00 (or of B_min with
  `bmaxsp_which: "min"`, both with `"both"`); row = soft max over surfaces s <= `bmaxsp_smax` of (spread - target);
  `bminslope_target` adds d<B_min>/ds / B00 - target terms.
- **Switch on** (QI only): `phr.bmaxsp: true`, `bmaxsp_from_step`; `qi_options.bmaxsp_target`, `bmaxsp_which`,
  `bmaxsp_soft`, `bmaxsp_tau`, `bmaxsp_tau_alpha`, `bmaxsp_smax`, `bminslope_target`.
- **Exploration runs (QH / QI)**: target 0.004, which "min", from step 2-3, bminslope -0.025.
- **Staging**: `bmaxsp_from_step` works exactly like `maxj_from_step` (0-based, row held at g = -1 before).
- **Hard/soft**: soft. **Gradient**: exact key `bmaxsp`. **Cost**: moderate.

### A11. Mercier stability
- **Definition**: merc = soft min (temperature `tau`) of s DMerc(s) over s in [`s_min`, `s_max`] (VMEC Mercier
  criterion on the VMEX state); row g = (margin - merc) / scale.
- **Switch on** (any physics): `equilibrium.mercier_options` {s_min, s_max, tau} (computes the metric) and
  `phr.mercier: true`, `mercier_margin`, `mercier_scale`; `equilibrium.mercier_jac_batch` sets the memory of its
  Jacobian.
- **Paper (NCSX-like, super-high-beta)**: s_min 0.1-0.15, s_max 0.95, tau 0.1; margin 0.05 or 4.0 with scale 10.
- **Hard/soft**: soft. **Gradient**: exact key `merc` (pushed in column batches). **Cost**: expensive (full Nyquist
  real-space grids per Jacobian column).
- **Pitfalls**: DMerc is noisy near the axis and at the edge at low radial resolution: keep s_min >= 0.1,
  s_max <= 0.95 and ns >= 50; check the profile in `final_metrics.json: surface.mercier_soft_min_sDMerc`. Resuming
  a stage that adds the row works (the new multiplier starts at 0).

### A12. Plasma volume and minor radius
- **Definition**: bands |V/V* - 1| <= volume_tol, |a/a* - 1| <= aminor_tol (relative rows); a = sqrt(area / pi) of the
  mean cross-section.
- **Switch on**: `phr.volume_target`, `volume_tol` (default 1e-3), `aminor_target`, `aminor_tol` (default 1e-3);
  `aspect_off` when both fix the shape scale.
- **Paper runs**: aminor_target 0.091, tol 0.01 (runs at a fixed plasma size).
- **Hard/soft**: soft. **Gradient**: exact keys `volume`, `aminor`. **Cost**: cheap.

### A13. QS cap
- **Definition**: g = QS_total / qs_max - 1: an upper bound on the objective, so that a tight coil bound can trade QS
  up to this value.
- **Switch on**: `phr.qs_max` (0 = off). **Paper**: 1e-4 to 5e-4 in the coil-force runs.
- **Hard/soft**: soft. **Gradient**: exact key `qs_total` (2 r^T dr/dx). **Cost**: cheap.

---

## B. Coil side (limits of the coil subproblem)

Coils: `n_coils` modular coils per half field period, Fourier order `order`, replicated by nfp and stellarator
symmetry; currents free with a fixed total `total_current` (per half period). Hard = certified against the exact
limit at the end of every solve; the row value reported is (value - limit)/limit.

| # | constraint | definition (row <= 0) | profile keys (`coils.`) | default | values used in practice | enforced | cost |
|---|---|---|---|---|---|---|---|
| B1 | length | L_c / L_max - 1 per coil, or sum L_c / L_tot - 1 | `per_coil_length` [m], `length_mode` (per_coil / total), `total_length` | 4.0, per_coil | LP QA 4.0; NCSX-like 3.5; QI 1.6 (nfp 5, 6 coils); force runs total 18 | hard (768-point quadrature) | cheap |
| B2 | max curvature | kappa(t) / kappa_max - 1 pointwise (soft max per segment) | `max_curvature` [1/m], `curvature_margin` | 3.0 | NCSX-like 6, QI 13 | hard (4096 samples + golden refinement) | cheap |
| B3 | Lp curvature | (1/p) int max(kappa - kappa_0, 0)^p dl / J_max - 1 per coil (simsopt LpCurveCurvature) | `lp_curvature_max` (0 = off), `lp_curvature_p`, `lp_curvature_threshold` [1/m] | off, p 2, kappa_0 0 | not used in the paper (new) | hard (4096 samples) | moderate on CPU |
| B4 | mean-squared curvature | int kappa^2 dl / L / MSC_max - 1 per coil | `max_msc` [1/m^2] | 4.0 | NCSX-like 12, QI 35 | hard | cheap |
| B5 | coil-coil distance | 1 - d_cc / d_min, soft min over all coil pairs (all periods and images) | `min_cc` [m], `distance_margin` | 0.1 | 0.05 (QI, NCSX-like) | hard (exact segment-segment distance) | moderate |
| B6 | coil-surface distance | 1 - d_cs / d_min, soft min over the boundary grid | `min_cs` [m], `distance_margin`, `dist_grid` | 0.15 | 0.15 LP QA, 0.12 NCSX-like, 0.07 QI | hard (exact distance to the triangulated boundary) | moderate |
| B7 | toroidal ordering (no crossing) | for each neighbouring pair (and end coils with their images): gap - softmin[phi_R - phi_L + kappa (1 - cos dtheta)] <= 0, normalized by the slot width | `slot_rows`, `slot_gap_deg` [deg], `slot_kappa`, `slot_beta`, `slot_seg`, `ordering_axis` | off, 1.0 deg, frozen_step | 1.5 deg (QI, NCSX-like) | **solver rows only**: the hard per-pair gaps are reported (`final_coils.json: slot`, `crossing`) but are not part of the certified violation | expensive on CPU |
| B8 | linking | Gauss linking number of every coil pair = 0 | always; `link_guard` (none / link / link_floor), `link_floor_frac` | none | link guard in the NCSX-like runs | hard (certified; linked results are never ok) | cheap |
| B9 | currents | sum of base-coil currents = `total_current` (equality, by construction) | `total_current` [A] | 1.25e6 | 1.25e6 (nfp 2), 833330 (nfp 3), 1.33e6 (nfp 5) | exact | none |
| B10 | max coil force | F_q / F_lim - (1 - force_margin) at every quadrature point q of every base coil, F = \|I t x (B_mutual + B_self,reg)\| the force per unit length (see below) | `max_force` [MN/m per turn] (0 = off), `force_scale`, `force_current` [A], `force_turns`, `force_radius` [m], `force_margin` | off | 0.413 MN/m per turn, margin 0.015 (LP QA at ARIES scale: x10.13, 72.14 MA) | hard (solver AL rows + certified on the 768-point quadrature) | expensive (C x nq rows) |

Notes and pitfalls:
- **Margins** (`curvature_margin`, `distance_margin`, `force_margin`) tighten only the solver rows; certification
  uses the true limits. Paper runs used distance_margin 0.03 (0.01 earlier).
- **Soft vs certified distances**: the soft-min grid (`dist_grid`) can under-resolve the coil-surface distance on
  folded surfaces; the certificate catches it and the final stage polishes it away.
- **Ordering (B7)** is a projection condition; it prevents crossing but not linking (B8 covers linking). The rows are
  XCoil's (`xcoil/constraints/ordering.py:rows`, XCoil >= 1.3.0): phi is a coil point's cylindrical angle, theta
  its poloidal angle about a **reference axis** R_ax(phi) = sum a_n cos(n nfp phi), Z_ax(phi) = sum b_n sin(n nfp phi)
  (n <= 8). `coils.ordering_axis` chooses that axis:
  * `frozen_step` (default): the geometric axis of the boundary at the **start of each continuation step**, its m = 0
    modes (a_n = rc(0,n), b_n = -zs(0,n), the poloidal mean of R and Z at each phi), fixed for the whole step. The
    rows then depend on the coils only: they do not move with the surface and add no term to the envelope gradient.
    The axis is stored in every checkpoint (`ordering_axis`); a resume inside the same step reuses it,
    `--start-step` and every new step recompute it, a sweep takes its seed checkpoint's. The final coil stage of a run
    and `qss-coils --run-dir` use the frozen axis of the step being certified (from the final checkpoint);
    a standalone `qss-coils --boundary` uses the given boundary's axis. `final_coils.json` records the axis used.
  * `live` (the behaviour before 2026-11-16): the axis of the current boundary at every evaluation (theta-mean of the
    coil-surface distance grid), so the rows also depend on the surface and the envelope gradient carries
    lam^T dg_slot/ds through the moving axis.
  Both give identical rows on the boundary where the axis is frozen (max difference 5e-15 on the PHR seed).
- **Where the coil-side rows are implemented.** Coil constraints live in XCoil; phrport assembles them into its coil
  subproblem. From XCoil: the pointwise length, curvature, MSC, coil-coil and coil-surface rows
  (`xcoil.constraints.pointwise`), the certification geometry (`xcoil.verify.certify`), the coil force
  (`xcoil.constraints.forces`), the toroidal ordering rows (`xcoil.constraints.ordering`) and the Lp curvature
  (`xcoil.constraints.lp_curvature`, XCoil >= 1.4.0; A/B against phrport's former copy on the force-, Lp- and
  ordering-limited QA case: identical c0, certified violation, forces, Lp values, distances and currents). In phrport
  because they belong to its outer problem or its own coil-solver loop: the flux objective and flux bound (PHR's local
  squared flux on PHR's grid, <|B.n|>/<|B|>), the phantom-flux guard, the link guard (a step-acceptance rule of
  phrport's coil solver; XCoil's own solver has the equivalent `link_test`), the frozen ordering axis per continuation
  step and the current parametrization.
- **Force (B10)**: where it is implemented. The physics is XCoil's `xcoil/constraints/forces.py` (`coil_forces`,
  `self_field`, `regularization_circ`, `normalized_limit`; XCoil 1.2.0 and later, where it is also an opt-in constraint
  of XCoil's own solver). phrport builds the rows from it in `phrport/coils.py` (`_Programs`: the `g_force` rows
  appended to the coil constraint rows inside phrport's coil augmented Lagrangian) and the certificate row
  (`_Programs.cert`). Until 2026-11-16 the same code was `phrport/forces.py`; an A/B on the force-limited QA case gave
  bit-identical forces, c0, certified violation and coils.
- **Force (B10)**, as implemented in `xcoil/constraints/forces.py:coil_forces` (simsopt's `coil_force` physics): on base coil c
  with current I_c and unit tangent t, F = |I_c t x (B_mutual + B_self,reg)|, where B_mutual is the Biot-Savart field
  of every other physical coil (all field periods and stellarator images) and B_self,reg the regularized self-field
  of a circular conductor of radius a (delta = a^2 / sqrt(e)). The limit is given per turn at a reference (reactor)
  scale and converted to the normalized problem as F_lim = max_force x turns x S / (I_ref / I)^2 with S =
  `force_scale`, I_ref = `force_current`, turns = `force_turns` (0 = 800 / n_coils) and the conductor radius
  `force_radius` / S (`xcoil.constraints.forces.normalized_limit`). The rows enter the XCoil augmented Lagrangian like every other
  coil limit (multipliers, penalties; JAX Jacobian in the coil DOFs and currents); certification checks the maximum
  over the 768-point quadrature against F_lim (row "force" of the certificate, reported per coil in MN/m at the
  reference scale). On a CPU a coil solve with force rows takes many minutes: run it on a GPU.
- **Start topology**: a start whose coils are linked is rejected with `link_guard`; without it a linked result is
  just not ok and the next start is tried.

## C. The coil row: flux bound

- **Definition**: the coil row of the outer problem is g = S (c0* - c0_target) <= 0 with
  c0 = 0.5 sum (B.n / |B|)^2 |N| / N_points on the flux grid (`coils.phr_grid`), c0* its optimum over coils
  satisfying part B, S = `--flux-metric-scale`.
- **Target forms** (CLI flags or the profile's `"run"` section):
  - `flux_target_frac` F (default 0.5, paper 0.5): at the start of each continuation mode the target is captured as
    F x the achieved c0 (a moving bound that tightens every mode);
  - `flux_phr_target` (absolute, with F <= 0); `flux_target_floor` T: target = max(F c0, T) (stops the F^k
    compounding, paper 2e-5 to 4e-5 in late stages); `final_mode_flux_target` E: absolute in the last mode;
  - `flux_target_bn` b*: the bound stated as <|B.n|>/<|B|> <= b*: the captured target is c0 (b*/b)^2 with b the coil
    solution's `bn_ratio_int` (c0 is quadratic in the normal field), never above the F target;
  - `target_flux` / `--sweep`: drive the bound down after the run / trace the QS vs flux front.
- **Phantom guard**: if a coil set comes within 0.5 `phr.cs_threshold` of the boundary or c0 <= 1e-7, the honest
  mean (B.n/|B|)^2 replaces c0 and the flux gradient is dropped (coils through the plasma would fake a zero flux).
- **Gradient**: envelope theorem (`CoilSolver.envelope_grad`); finite beta adds the virtual-casing field term
  (needs `--eq-jacobian exact`). **Cost**: one coil solve per evaluation (warm when the boundary moved little).
- **Reported metrics**: `squared_flux_Jf` (c0), `avg_Bn_over_B` (grid mean of |B.n|/|B|),
  `avg_Bn_over_B_area_weighted`, `bn_ratio_int` (<|B.n|>/<|B|>, ratio of surface integrals), `max_Bn_over_B`.
- **Pitfalls**: a target the coils cannot reach shows as the Jf row VIOLATED with a growing penalty and QS getting
  worse; loosen F, add a floor, or relax the coil limits.

## D. Not constraints, but checks

- Free boundary: `tools/warm_post.py` (VMEX free boundary with the coils' mgrid) and `tools/desc_free_boundary.py`
  (DESC, the paper's protocol for every free-boundary number).
- Gradient checks: `tools/taylor_envelope.py` (vacuum envelope gradient), `tools/taylor_beta.py` (finite beta).
- Alpha-particle confinement and Boozer post-processing are done outside this package.

## E. Code cross-check

Rows built by `phrport/phr.py:eq_constraint_labels` / `eq_constraint_values` / `exact_constraint_grads`
(A3-A14), the objective in `phrport/equilibrium.py` and `phrport/qi_residual.py` (A1, A2), coil rows in
`phrport/coils.py` (`_Programs`: XCoil pointwise rows + slot, Lp and force rows; certification `cert`).
`tests/test_constraints.py` checks every equilibrium row's label, value, sign and gradient key;
`tests/test_qi_and_physics.py` the QI kernels and the Lp functional; `tests/test_backend.py` certified coil solves.

Not covered by an automated test: the max-force rows (too slow on CPU; checked by hand: certified force-limited QA solve, 2026-11-15), the Mercier values on a real
equilibrium, the `djds` / `jspread` / `bminu` residual terms.
