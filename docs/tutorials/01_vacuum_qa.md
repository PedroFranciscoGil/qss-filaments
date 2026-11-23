# Tutorial 1: vacuum quasi-axisymmetry with coils

Goal: optimize a two-field-period QA boundary (aspect 6, iota 0.42, magnetic well) together with four modular coils
per half period, starting from the Landreman-Paul QA deck.

Profile: [`examples/qa_vacuum.json`](../../examples/qa_vacuum.json) (laptop variant
[`examples/qa_vacuum_tiny.json`](../../examples/qa_vacuum_tiny.json): boundary and VMEC resolution 4, one radial
stage, coarser coil flux grid).

## 1. Check the install

```bash
qss-check --profile examples/qa_vacuum_tiny.json
```

## 2. Laptop run (about 2 minutes on a CPU)

```bash
qss-run --profile examples/qa_vacuum_tiny.json --preset tiny --run-dir runs/t1
```

`--preset tiny` is one continuation step, one AL outer and three function evaluations. What happens:

1. the seed boundary is solved with VMEX (`wout_initial.nc`, `initial_metrics.json`);
2. the coils are solved cold on the seed (XCoil, certified): their squared flux c0 sets the flux bound of the mode,
   `--flux-target-frac 0.5` x c0;
3. the least-squares subproblem moves the boundary to reduce the QS residual + the PHR penalties of the violated rows
   (aspect band, iota band, elongation, well, coil flux bound). Every function evaluation re-solves the coils (warm
   when the boundary moved little) and adds the envelope gradient of c0* to the coil row;
4. multipliers and penalties are updated, a checkpoint is written, and the final coils are solved and exported.

The log ends with

```
Final: QS=5.49e-04 aspect=5.9994 iota=0.4194 c0_local=8.80e-05
```

(the numbers vary slightly with the platform). Inspect the result:

```bash
python -c "import json; d=json.load(open('runs/t1/final_metrics.json')); print(d['surface']['g_eq']); print({k: d['coils'][k] for k in ('squared_flux_Jf', 'avg_Bn_over_B', 'certified', 'linked')})"
```

`g_eq` lists every constraint row (feasible when <= 0); `avg_Bn_over_B` is the coils' mean |B.n|/|B| on the
boundary. `final_coils.json` contains a simsopt loader snippet.

## 3. Production run

```bash
qss-run --profile examples/qa_vacuum.json --preset production --run-dir runs/qa_vac
```

`--preset production` uses the exact equilibrium Jacobian, a flux metric scale of 500, the anchored coil oracle and the
exact speed-ups (see [reference.md](../reference.md)). The profile's `"run"` section asks for five continuation
steps (boundary modes |m|, |n| <= 1..5) of five outers each ([profile.md](../profile.md)).
On a GPU node this takes a few hours; resume and cluster use are in [tutorial 5](05_continuation_and_clusters.md).

## 4. Certify and post-process

```bash
qss-coils --profile examples/qa_vacuum.json --run-dir runs/qa_vac --out runs/qa_vac/coils_certified
```

```bash
PHRPORT_PROBLEM=examples/qa_vacuum.json python tools/warm_post.py runs/qa_vac/coils_certified.json runs/qa_vac/post
```

`tools/warm_post.py` writes the coil metrics on fine grids, the mgrid of the coils, a VMEX free-boundary check,
Poincare sections and VTK files;
`tools/desc_free_boundary.py` is the DESC free-boundary verification used for the paper's free-boundary numbers.

## Variations

* two-stage baseline: `qss-run ... --no-coils`, then `qss-coils` on its final boundary;
* a tighter coil fit: `--flux-target-frac 0.3`, or an absolute bound on <|B.n|>/<|B|> with `--flux-target-bn 2e-3`;
* other coil limits: edit the profile's `"coils"` section (see [constraints.md](../constraints.md)).
