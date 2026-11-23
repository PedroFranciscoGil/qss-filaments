# The profile file

A **profile** is one JSON file that describes *the design problem*: which equilibrium you start from, what the plasma
must satisfy, what the coils may look like, and (optionally) how long to run. You run it with

```bash
qss-run --profile my_problem.json --preset production --run-dir runs/my_problem
```

Everything the run needs is either in the profile, in a preset, or on that command line. Check a profile before
submitting it:

```bash
qss-check --profile my_problem.json
```

`qss-check` resolves the input deck, and flags every key the code does not know (a misspelled key would otherwise be
silently ignored). `qss-run` prints the same warnings at start.

## Structure

```text
{
  "name":        "...",            a label for the outputs
  "equilibrium": { ... },          the starting equilibrium, the physics objective and the equilibrium resolution
  "phr":         { ... },          the plasma constraints (targets and bounds) and finite-beta settings
  "coils":       { ... },          the coil set and its engineering limits
  "run":         { ... }           optional: run-control options (continuation length, flux bound, ...)
}
```

Every section and every key is optional; a missing key keeps its default (the defaults reproduce PHR's nfp2 QA
problem). The complete list of keys, with defaults and units, is [reference.md](reference.md); the meaning of every
constraint is [constraints.md](constraints.md).

| section | what goes there | most used keys |
|---|---|---|
| `equilibrium` | the seed deck (`input_file`), `nfp`, the objective (`physics` qs or qi, helicity / QI surfaces), the boundary resolution (`surf_mpol`, `surf_ntor`), the VMEC resolution (`vmec_mpol`, `ns_array`, `ftol_array`, `niter_array`), `qi_options`, `mercier_options` | `input_file`, `physics`, `nfp`, `vmec_mpol`, `ns_array` |
| `phr` | targets of the plasma rows (aspect, iota, elongation, well, mirror, max-J, Mercier, volume, ...), finite beta (`finite_beta`, `fb_field_sign`, `fb_current_from_G`) | `aspect_target`, `iota_mean_target`, `elongation_threshold`, `vacuum_well`, `well_threshold` |
| `coils` | coil count and order, length / curvature / MSC / distance limits, ordering, force, currents, coil-solver settings | `n_coils`, `order`, `per_coil_length`, `max_curvature`, `max_msc`, `min_cc`, `min_cs`, `total_current` |
| `run` | any `qss-run` option, written as the flag name without the leading dashes and with `_` for `-` (`"flux_target_frac": 0.5` = `--flux-target-frac 0.5`; `"no_coils": true` = `--no-coils`) | `n_continuation_steps`, `al_outer_iter`, `eq_jacobian`, `flux_target_frac`, `flux_target_floor` |

Paths: `input_file` may be absolute, relative to the profile's own folder, relative to the repository (e.g.
`inputs/input.LandremanPaul2021_QA`), or start with `__ROOT__/`.

Keys starting with `_` are ignored everywhere, so you can annotate a profile: `"_note": "aspect from the 2024 study"`.

## Which value wins

From lowest to highest priority:

1. built-in defaults (PHR nfp2 QA);
2. the profile's `equilibrium` / `phr` / `coils` sections;
3. `--preset production` or `--preset tiny`;
4. the profile's `run` section;
5. options on the command line.

So a profile can carry its own run length (`run`), a preset supplies the solver settings of the paper, and anything
typed on the command line overrides both. The `run` section is read by `qss-run` (and the Slurm template) only; a
direct `python -m phrport.phr` call ignores it. `qss-run` prints the resulting option list at start.

## Presets

`--preset` puts a fixed list of `qss-run` options in front of the others (`phrport/cli.py`: `PRESETS`,
`PRESET_ENV`, applied in `run_main`). It contains solver settings only, no physics targets and no run data. Without a
preset every option keeps its built-in default (PHR's settings: finite-difference Jacobian, flux scale 1, PHR coil
warm-start rule, all trial coil solves certified, 8 VMEX workers), which runs but is slower and less robust.

| option | `production` | `tiny` | built-in default |
|---|---|---|---|
| `--eq-jacobian` | exact | - | fd |
| `--max-nfev` | 30 | 3 | 20 |
| `--flux-metric-scale` | 500 | - | 1 |
| `--flux-target-frac` | 0.5 | - | 0.5 |
| `--coil-oracle` / `--oracle-k-max` | anchored / 4 | - | phr / 2 |
| `--skip-coil-on-rejected`, `--abort-doomed-coil-solves`, `--reuse-vmex-state` | on | - | off |
| `--no-certify-trials` (certify accepted coils only) | on | - | off (every trial certified) |
| `--carry-trust-radius`, `--full-outer-budget` | on | - | off |
| `--n-workers` / `--fd-ngroups` | 1 / 1 | 2 / - | 8 / 8 |
| `--n-continuation-steps` / `--al-outer-iter` | - | 1 / 1 | 4 / 6 |
| `--ls-verbose` | - | 0 | 2 |
| environment `PHRPORT_TRUST_RESET_STEP` / `PHRPORT_TRUST_FLOOR` | 1 / 1e-3 (if not set) | - | unset |

`production` is the setting the published runs used (it contains no data from them); `--preset paper` is a deprecated
alias of it. Every on/off option a preset switches on has an undo form, on the command line or in the profile's
`"run"` section: `--no-carry-trust-radius`, `--no-full-outer-budget`, `--no-skip-coil-on-rejected`,
`--no-abort-doomed-coil-solves`, `--no-reuse-vmex-state`, `--certify-trials` (or `"full_outer_budget": false`, ...).

## Annotated example 1: vacuum QA (`examples/qa_vacuum.json`)

```json
{
 "name": "LP_QA_vacuum",
 "equilibrium": {
  "input_file": "inputs/input.LandremanPaul2021_QA",
  "nfp": 2,
  "helicity_m": 1, "helicity_n": 0,
  "surf_mpol": 8, "surf_ntor": 8,
  "seed_ellipse": 0.0,
  "vmec_mpol": 8,
  "ns_array": [16, 50], "ftol_array": [1e-16, 1e-13], "niter_array": [1200, 5000]
 },
 "phr": {
  "aspect_target": 6.0, "aspect_two_sided": true, "aspect_tol": 0.0,
  "iota_mean_target": 0.4193, "iota_tol": 0.002,
  "elongation_threshold": 4.602,
  "vacuum_well": true, "well_threshold": -0.0304,
  "cs_threshold": 0.15
 },
 "coils": {
  "n_coils": 4, "order": 5,
  "per_coil_length": 4.0, "max_curvature": 3.0, "max_msc": 4.0,
  "min_cc": 0.1, "min_cs": 0.15,
  "total_current": 1.25e6
 },
 "run": {"n_continuation_steps": 5, "al_outer_iter": 5}
}
```

| key | meaning |
|---|---|
| `input_file` | Landreman-Paul precise QA deck: the seed boundary, nfp 2, vacuum (no pressure, no current) |
| `helicity_m/n` | quasi-axisymmetry (M, N) = (1, 0); QH would be (1, nfp) |
| `surf_mpol/ntor` | the boundary has Fourier modes up to m, \|n\| = 8; the continuation frees them step by step |
| `seed_ellipse` | 0 keeps the deck's boundary unchanged (PHR adds 0.07 to a circular seed) |
| `vmec_mpol`, `ns_array`, ... | VMEC resolution 8 x 8, two radial stages (16 then 50 surfaces) with their tolerances and iteration caps |
| `aspect_*` | A = 6 exactly (two-sided band of zero width) |
| `iota_mean_*` | mean rotational transform 0.4193 +- 0.002 |
| `elongation_threshold` | cross sections no more elongated than 4.602 |
| `vacuum_well`, `well_threshold` | magnetic well W >= -0.0304 (a slight hill is allowed) |
| `cs_threshold` | phantom-flux guard distance (coils that touch the plasma do not count as a good fit) |
| `coils` | 4 coils per half period (16 in total), order 5, length <= 4 m, curvature <= 3 /m, MSC <= 4 /m^2, coil-coil >= 0.1 m, coil-plasma >= 0.15 m, 1.25 MA per half period (B ~ 1 T on axis at R = 1 m) |
| `run` | 5 continuation steps (modes 1..5), 5 augmented-Lagrangian outers each |

## Annotated example 2: finite-beta QA (`examples/qa_finite_beta.json`)

Differences from example 1:

```json
"equilibrium": {"input_file": "inputs/input.qa_finite_beta", "...": "..."},
"phr": {"iota_mean_target": 0.42, "iota_tol": 0.005, "well_threshold": 0.0,
        "finite_beta": true, "fb_field_sign": -1.0, "...": "..."},
"coils": {"precision": "fp64", "...": "..."},
"run": {"n_continuation_steps": 5, "al_outer_iter": 5, "eq_jacobian": "exact"}
```

| key | meaning |
|---|---|
| `input_file` | the QA deck of example 1 with a pressure profile p(s) = 7000 (1 - s) Pa and no net current (written with `qss-seed deck --pressure 7000`) |
| `iota_mean_target`, `well_threshold` | the targets of this example: iota 0.42 +- 0.005, a magnetic well W >= 0 |
| `finite_beta` | the coils must also cancel the normal field of the plasma currents: B_plasma by virtual casing on each equilibrium |
| `fb_field_sign` | orientation of the plasma field relative to the coil field; it follows the deck's toroidal field direction (PHIEDGE sign), -1 for decks with this PHIEDGE convention. `CoilSolver.plasma_orientation()` checks it on actual coils |
| `fb_current_from_G` (optional) | scale the coil currents to the current the equilibrium links |
| `precision` | full FP64 coil kernels (the plasma field term is small; mixed precision is not enough) |
| `eq_jacobian: exact` | needed so that the plasma field's contribution to the coil gradient is computed |

## Annotated example 3: QI from a tokamak, with max-J, mirror and coil ordering (`examples/qi_maxj_mirror.json`)

```json
{
 "equilibrium": {
  "input_file": "inputs/input.tokamak_nfp5", "physics": "qi", "nfp": 5,
  "vmec_mpol": 7, "ns_array": [16, 31], "ftol_array": [1e-12, 1e-13], "niter_array": [3000, 10000],
  "qi_surfaces": [0.1, 0.3, 0.5, 0.7, 0.9],
  "qi_options": {"nphi": 101, "nalpha": 21, "n_bounce": 31, "mboz": 16, "nboz": 16, "soft": 0.002,
                 "mirror_soft": 0.0002, "mirror_all": true, "mirror_tau": 0.002,
                 "maxj_target": -0.06, "maxj_depths": [0.1, 0.3, 0.5, 0.7, 0.9]}
 },
 "phr": {"aspect_target": 10.0, "elongation_threshold": 6.0,
         "iota_mean_off": true, "iota_axis_target": 0.85, "iota_edge_target": 0.95, "iota_end_tol": 0.01,
         "mirror_max": 0.25, "vacuum_well": false, "maxj": true, "maxj_from_step": 3, "eq_cache_size": 8},
 "coils": {"n_coils": 6, "order": 5, "per_coil_length": 1.8, "max_curvature": 12.0, "max_msc": 30.0,
           "min_cc": 0.05, "min_cs": 0.08, "total_current": 1.33e6, "phr_grid": [200, 64], "dist_grid": [320, 64],
           "slot_rows": true, "slot_gap_deg": 1.5, "warm_max_outer": 30},
 "run": {"n_continuation_steps": 6, "al_outer_iter": 5, "flux_metric_scale": 100, "flux_target_floor": 1.3e-5,
         "rho_max": 64, "carry_trust_radius": false}
}
```

| key | meaning |
|---|---|
| `input_file` | a circular tokamak, A = 10, nfp 5, with a small rotating ellipse and helical axis shift (`qss-seed tokamak --nfp 5 --aspect 10 --break qi`, see [seeding.md](seeding.md)) |
| `physics: "qi"` | the objective is the constructed quasi-isodynamic residual on `qi_surfaces` |
| `qi_options` resolution keys | field-line grid (101 points x 21 lines), 31 bounce levels, Boozer 16 x 16 |
| `mirror_all`, `mirror_tau`, `mirror_max` | mirror ratio <= 0.25 on every QI surface (soft max at temperature 0.002) |
| `maxj_target`, `maxj_depths`, `maxj`, `maxj_from_step` | the max-J row on the constructed field, switched on at the fourth continuation step (0-based 3) |
| `iota_*` | axis and edge iota bands instead of the mean |
| `eq_cache_size` | keep fewer equilibria in memory (QI states are large) |
| `n_coils` 6, ... | 6 short coils per half period (60 in total) with tight distances |
| `phr_grid`, `dist_grid` | finer flux / distance grids for nfp 5 (nphi divisible by 2 nfp) |
| `slot_rows`, `slot_gap_deg` | neighbouring coils keep a 1.5 degree toroidal gap and cannot cross |
| `run` | the tokamak-start settings: flux scale 100, a floor on the flux bound (a tokamak's coil fit is nearly perfect), penalty cap 64, no carried trust radius |

## Staging constraints

Some rows are better switched on once the shape has developed. Two rows have a built-in switch:

| row | option | default | meaning |
|---|---|---|---|
| max-J | `phr.maxj_from_step` | 0 | 0-based continuation step from which the row counts |
| B_max spread | `phr.bmaxsp_from_step` | 0 | same |

Before that step the row is held at g = -1: no penalty, no gradient, multiplier 0; the number of rows does not change,
so checkpoints and resumes keep working, and `--start-step` / `--resume-from` keep the schedule. For any other row
(Mercier, mirror_min, volume, iota bands, a tighter coil limit, more pressure) use a **stage**: write the stage's
profile with the row, then resume the previous stage's final checkpoint with `--start-step K`; the new row starts with
multiplier 0 ([tutorial 5](tutorials/05_continuation_and_clusters.md)). See also [constraints.md](constraints.md#staging-constraints-switching-rows-on-later).

## Starting your own problem

1. Copy the closest example and point `input_file` at your seed deck (any VMEC input; `qss-check` tells you if it
   is found). `qss-seed` writes decks from a tokamak shape, a rotating ellipse, a wout, a boundary JSON or an old
   run, and `coils.seed_coils` starts the coils from a coil file: see [seeding.md](seeding.md) (with the paper's
   tokamak recipes).
2. Set the physics targets in `phr` (and `qi_options` for QI).
3. Set the coil count and the limits in `coils`. Scale matters: lengths and distances are in the units of the deck
   (the examples have R ~ 1 m).
4. Make a `_tiny` copy (lower `vmec_mpol`, one radial stage, coarser grids) and run it with `--preset tiny` on a
   laptop to catch mistakes in minutes.
5. Submit the full profile with `--preset production` (see the README, "What you submit").
