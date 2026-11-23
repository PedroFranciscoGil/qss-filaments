# Tutorial 6: from a tokamak to a stellarator, and seeding the coils

Goal: start the QA problem of tutorial 1 from a circular tokamak instead of a stellarator deck, make your own seed
deck, and start the coils from an earlier design. Background: [seeding.md](../seeding.md).

## 1. The ready-made tokamak seed (about 3 minutes on a laptop CPU)

```bash
qss-run --profile examples/qa_from_tokamak_tiny.json --preset tiny --run-dir runs/t6
```

The profile points at `inputs/input.tokamak_nfp2` (written by `qss-seed tokamak --aspect 6 --break qa`): a circular torus with R0 = 1, a = 1/6, plus 1e-3 on the
(m, n) = (1, 1) and (2, 1) modes so that the optimizer can leave the axisymmetric saddle. The log's first metrics
show the tokamak: `iota` is about 0. Two continuation steps later the boundary has non-axisymmetric shaping and
iota has moved toward the 0.4193 target; the coils are 5 per half period. With `examples/qa_from_tokamak.json`
and `--preset production` (5 steps x 5 outers on a GPU) this is the full QA tokamak start.

## 2. Your own tokamak seed

```bash
qss-seed tokamak --aspect 6 --kappa 1.3 --delta 0.3 --break qa --out inputs/input.my_tokamak
```

writes a D-shaped seed (area-equivalent aspect 6, elongation 1.3, triangularity 0.3) with the QA symmetry break,
keeping the vacuum profiles and PHIEDGE of `inputs/input.LandremanPaul2021_QA`. Use it by copying
`examples/qa_from_tokamak_tiny.json` and changing `equilibrium.input_file`, then check it:

```bash
qss-check --profile my_tokamak_tiny.json
```

For a QI target use `--nfp 5 --aspect 10 --break qi` (the re-signed rotating ellipse and helical axis shift of the
QI tokamak start, as in `inputs/input.tokamak_nfp5`) and follow the three-stage recipe in [seeding.md](../seeding.md#13-recipe-qi-from-a-tokamak):
the flux bound floor (`--flux-target-floor`) is what keeps the near-perfect coil fit of a tokamak from freezing the
shaping.

Other seeds go the same way: `qss-seed rotating-ellipse`, `qss-seed from-wout wout.nc`,
`qss-seed from-boundary boundary.json`, `qss-seed from-checkpoint runs/old --profile old.json`.

## 3. Starting the coils from an earlier design

The coils of run `runs/t6` can seed the first coil solve of another run with the same nfp and coil count. Add to
a profile's `"coils"` section

```json
"seed_coils": "runs/t6/final_coils.json"
```

or test it on one boundary:

```bash
qss-coils --profile examples/qa_from_tokamak_tiny.json --run-dir runs/t6 --out runs/t6/coils_reseeded --quick --override '{"seed_coils": "runs/t6/final_coils.json"}'
```

The log line `[seed_coils] first cold solve starts from ...` confirms it, and the result's start label is `seed`.
Coils from simsopt (or MAKEGRID files loaded with simsopt) are given as a points JSON, see
[seeding.md](../seeding.md#22-starting-from-your-own-coils).
