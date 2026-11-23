# Tutorial 2: finite-beta quasi-axisymmetry

Goal: the QA problem of tutorial 1 at finite plasma pressure (a current-free pressure profile
p(s) = 7000 (1 - s) Pa on the QA deck, beta of about 1 %), with the plasma's own field in the coil problem.

Profile: [`examples/qa_finite_beta.json`](../../examples/qa_finite_beta.json) (laptop variant
[`examples/qa_finite_beta_tiny.json`](../../examples/qa_finite_beta_tiny.json)). The deck
`inputs/input.qa_finite_beta` was written with `qss-seed deck --pressure 7000` (any pressure and current profile in
a VMEC deck works). The keys that make it a finite-beta problem:

```json
"phr": {"finite_beta": true, "fb_field_sign": -1.0, "well_threshold": 0.0, "iota_mean_target": 0.42, ...},
"coils": {"precision": "fp64", ...}
```

* `finite_beta`: the coil residual is (B_coil + B_plasma).n / |B_coil + B_plasma| with B_plasma from virtual casing
  on the VMEX state (`phrport/plasma_field.py`, `phrport/finite_beta.py`). The plasma-field context is refreshed at
  every Jacobian point; the field term of the envelope gradient (d c0*/d boundary through B_plasma) is added to the
  coil row.
* `fb_field_sign`: orientation of the plasma field relative to the coils (it follows the deck's toroidal-field direction, the PHIEDGE sign);
  `CoilSolver.plasma_orientation` checks it.
* `fb_current_from_G` (optional): scale the coil currents to the current linked by the equilibrium (lambda_G), used
  for the NCSX- and W7-X-like designs.
* the virtual-casing work is jitted (`--fb-jit vjp`, default; `off` = eager path, `check` = compare both); it falls
  back to the eager path for the rest of a mode if compiling fails (e.g. GPU memory).

## Laptop run (about 3 minutes on a CPU)

```bash
qss-run --profile examples/qa_finite_beta_tiny.json --preset tiny --run-dir runs/t2
```

The field term needs the state tangents of the exact equilibrium Jacobian: the profile's `"run"` section sets
`"eq_jacobian": "exact"` (with the finite-difference Jacobian the log says `explicit coil gradient only`). Look for

```
[finite-beta] plasma field in the coil problem (virtual casing, 16^2 grid, 4 digits values / 3 digits field term; jit vjp)
[finite-beta] field term of the envelope gradient: |g_field|=...
```

## Production run

```bash
qss-run --profile examples/qa_finite_beta.json --preset production --run-dir runs/qa_beta
```

On GPUs with <= 16 GB set `PHRPORT_FB_CHUNK=128`. `--fb-reuse-tol 2e-3 --fb-reuse-every 4` reuses the plasma field
between nearby Jacobian points (1.5x faster, slightly different iterates).

## Checks

* gradient: `PHRPORT_PROBLEM=examples/qa_finite_beta.json python tools/taylor_beta.py --ndir 3` (Taylor test of the
  full envelope gradient including the field term);
* free boundary: `tools/desc_free_boundary.py --beta --wout runs/qa_beta/wout_optimized.nc --coils runs/qa_beta/coils_certified.json ...`
  (DESC, optional extra `desc`).
