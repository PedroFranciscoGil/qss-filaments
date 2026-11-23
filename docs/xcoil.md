# The XCoil coil backend

`phrport.coils` solves the coil subproblem with XCoil: on a fixed plasma boundary,

    min_x  c0(x; boundary)  s.t.  the coil limits of docs/constraints.md

by XCoil's device-resident augmented Lagrangian with a Levenberg-Marquardt inner solver (exact Biot-Savart
Jacobians, FP32 kernels with FP64 accumulation by default, `coils.precision: "fp64"` for full double precision).
One compiled program serves every boundary (the surface enters as traced arrays), so a run compiles once per
coil configuration.

## Version and install

XCoil (by Pedro Francisco Gil and Alan A. Kaptanoglu, MIT) is a regular dependency:
`pyproject.toml` pins **`xcoil @ git+https://github.com/PedroFranciscoGil/XCoil@4ccebdeb439f43ea542bb9d4d2e9b9459177313e`**, which `pip install -e .`
installs. XCoil 2.0.0 is the baseline coil solver; phrport also uses its coil force module
`xcoil.constraints.forces`, the toroidal ordering rows `xcoil.constraints.ordering` and the Lp curvature
`xcoil.constraints.lp_curvature` for its max-force, ordering and Lp-curvature rows.

**The paper runs** used the coil solver's 2026-10-01 version. Relative to it, the coil geometry, constraint and
certification functions phrport calls are unchanged; the AL multiplier update (`_lancelot`) gained optional
`rho_fac` / `rho_max` arguments with the old defaults; the coil-coil and coil-surface soft-min rows are evaluated
component-wise (same values); XCoil turns on JAX's persistent compilation cache by default (below). An A/B on five
coil problems gave the same answers to 3-4 digits.

## Selecting the backend

| `PHRPORT_XCOIL` | XCoil used |
|---|---|
| unset (default) | the installed `xcoil` package (the pinned commit) |
| `/path/to/XCoil` | a local checkout: the folder that **contains** the `xcoil` package (the root of an XCoil clone) |

Set it before starting Python. If the selected
tree cannot be imported, `phrport.coils` raises `phrport.backend.XcoilUnavailable` with the reason and how to install
XCoil; mixing two trees in one process is refused. `qss-check` reports the backend in use, and every run records
the XCoil path, tree hash, version and pip's record of the installed commit in its provenance.

There is no simsopt coil backend: phrport never imports simsopt. Exported coils (`final_coils.json`) use simsopt's
`CurveXYZFourier` layout and carry a loader snippet, so simsopt (extra `simsopt`) can read them.

## Devices

XCoil runs on any JAX device. On a machine without a GPU everything runs on the CPU (`JAX_PLATFORMS=cpu` forces
it); the PHR default coil problem (4 coils, order 5) solves and certifies in about 10 s on a laptop CPU. On a GPU host
the usual split is VMEX on the CPU (`PHRPORT_VMEX_DEVICE=cpu`; VMEX is faster there on small GPUs) and XCoil on the
GPU. Pointwise max-force rows (`coils.max_force`) are expensive (C x nq rows with dense Jacobians): use them on a GPU.

Compiled programs go to JAX's persistent compilation cache. XCoil turns it on when it is imported: in
`JAX_COMPILATION_CACHE_DIR` if set, else in `~/.cache/xcoil/jax`; `XCOIL_COMPILE_CACHE=/dir` picks another folder
and `XCOIL_COMPILE_CACHE=off` turns it off. The cache is process-wide, so the VMEX programs are cached too.
Reusing it across restarts removes most of the compile time and makes runs bit-for-bit repeatable on the same
hardware.

## Solves inside the optimization

* **Cold solve** (`CoilSolver.solve(boundary)`): the primary start (`coils.start`, default `offset_3d_L85`: offset
  curves at 85 % of the length cap) and, if it fails, the fallbacks `clear`, `L85_tilt+8`, `L70`, `L85_tilt-8`,
  `L100` (at most `max_starts`). A `seed_coils` JSON starts the first cold solve of a process from given coils.
* **Warm solve** (`solve(boundary, warm=previous)`): continues the AL from the previous coils when the boundary
  moved by less than `--surface-dof-reset-threshold`; `--coil-oracle anchored` (production preset) warm-starts only from
  coils at accepted points, with a per-outer cold audit.
* **Envelope gradient** (`envelope_grad`): d c0*/d boundary from the coil Lagrangian at fixed coils and KKT
  multipliers (re-estimated by NNLS on the near-active rows).
* Speed options (exact, same accepted iterates): `--skip-coil-on-rejected`, `--abort-doomed-coil-solves`,
  `--no-certify-trials` (trial solves not certified, accepted coils certified afterwards).

## Certification: cold start + warm polish

Every result is checked against the TRUE limits in FP64 (`info["cert"]`): length on 768 points, maximum curvature on
4096 samples with golden-section refinement, mean-squared curvature, exact segment-segment coil distances, exact
coil-triangle distances to the triangulated boundary, Gauss linking numbers of every coil pair (all field periods and
stellarator images), and the optional force / Lp-curvature rows. `certified` means violation v <= `viol_tol` and
unlinked; `ok` allows v <= `ok_viol`.

The publication coil stage, `CoilSolver.final_coil_stage(boundary)` (`qss-coils`, `--final-coil-stage`):

1. cold solves from every start of the start family (and optional extra families), all certified;
2. every ok, unlinked, finite result is polished warm (`ftol 1e-10`) until its certified violation is <= 1e-5; when
   the solver rows and the certificate disagree (sampling), the offending solver limits are tightened by the measured
   mismatch and the polish is repeated (at most 4 rounds, margins <= 5 %);
3. the lowest polished Lagrangian wins; `info["final_stage"]` records every candidate and the chosen start, and the
   flux metrics are re-evaluated on unfolded grids with a grid-convergence check.

```bash
qss-coils --profile examples/qa_vacuum.json --run-dir runs/qa --out runs/qa/coils_certified
```

Use `--override '{"per_coil_length": 3.85}'` to re-solve with other limits and `--quick` for one certified cold
solve. Tutorial 4 walks through it.

## Tests

`tests/test_backend.py`: backend selection and error paths; a certified cold solve with the PHR limits; the cold start
+ warm polish stage reaching the 1e-5 certified violation target (CPU, about 40 s together); a tiny 2-coil solve
(slow marker). They skip if XCoil cannot be imported.
