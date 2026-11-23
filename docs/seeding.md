# Seeding: the starting boundary and the starting coils

A run needs two starting points: a **plasma boundary** (the seed equilibrium) and, inside it, **starting coils**
for the first coil solve. This page lists every way to provide each, and the recipes that work for starting from a
tokamak.

## 1. The seed boundary

The seed is always a VMEC input deck, given by the profile key `equilibrium.input_file`. Everything else (a
tokamak shape, a wout, a boundary file, an old run) is turned into a deck with `qss-seed`, or loaded with a run
flag:

| your seed | how | notes |
|---|---|---|
| a VMEC / VMEX input deck | `"equilibrium": {"input_file": "inputs/input.my_seed"}` | the deck also gives nfp, pressure and current profiles, PHIEDGE |
| a tokamak (circular, elliptic, D-shaped) | `qss-seed tokamak --aspect 6 --break qa --out inputs/input.my_tok` | sections 1.1-1.3 |
| a rotating ellipse | `qss-seed rotating-ellipse --aspect 6 --amp 0.05 --out inputs/input.my_re` | the classic stellarator seed |
| a wout file (VMEC / VMEX / DESC export) | `qss-seed from-wout wout_x.nc --out inputs/input.from_wout` | takes the last surface and nfp from the wout |
| a boundary JSON `{"rc(m,n)": .., "zs(m,n)": ..}` | `qss-seed from-boundary boundary.json --out inputs/input.from_json` | also reads the `boundary` stored in coil exports (`final_coils.json`) |
| a previous run (checkpoint or run folder), restarted from mode 1 | `qss-seed from-checkpoint runs/old --profile old_profile.json --out inputs/input.from_run` | a new problem from an old design |
| a previous run, same problem, continued | `qss-run ... --resume-from runs/old/al1_checkpoint_latest.json --resume-keep-target` | keeps multipliers, penalties and flux target (tutorial 5) |
| a previous run, fresh multipliers | `qss-run ... --init-surface runs/old` | keeps the boundary and the continuation step, resets multipliers |
| the default | no `input_file`: `inputs/input.nfp2_QA` | PHR's start: a circular tokamak, see `seed_ellipse` below |

`qss-seed` keeps everything except the boundary from a template deck (`--template`, default
`inputs/input.LandremanPaul2021_QA`: vacuum, nfp 2, PHIEDGE 0.0839); set `--nfp` and `--phiedge` for other devices.
`qss-seed --help` and `qss-seed <kind> --help` list the options. Check the result with
`qss-check --profile my.json`; a seed whose VMEX solve fails stops the run at "initial equilibrium failed"
(see [troubleshooting.md](troubleshooting.md)).

Boundary names follow simsopt: `rc(m,n)`, `zs(m,n)`, stellarator symmetric, n in units of nfp
(VMEC decks write `RBC(n,m)`). `equilibrium.surf_mpol` / `surf_ntor` decide which of them become optimization
variables; a seed with higher modes keeps them fixed.

`equilibrium.seed_ellipse` (default 0.07): if the deck is exactly axisymmetric, this amplitude is added to
`rc(2,1)` and `zs(2,1)` (a rotating ellipse). That is PHR's own start: `input.nfp2_QA` is a circular torus.
Set it to 0 to keep a deck unchanged (the examples do).

### 1.1 Starting from a tokamak

An axisymmetric boundary is a valid VMEC equilibrium, so any tokamak can seed a stellarator optimization. Three
things are different from a stellarator seed:

1. **Break the symmetry.** At an exactly axisymmetric vacuum boundary the rotational transform is zero and the
   gradient of every non-axisymmetric quantity vanishes (a saddle point), so the optimizer cannot leave it. Add a
   small non-axisymmetric term. `qss-seed tokamak --break` provides the two that worked:
   * **QA**: 1e-3 on (m, n) = (1, 1) and (2, 1) (`--break qa`; `inputs/input.tokamak_nfp2` is this seed at A = 6);
   * **QI**: a 2e-3 rotating ellipse tall at phi = 0, (1, 1) rc = -2e-3, zs = +2e-3, and a 2e-3 outward helical axis
     shift, (0, 1) rc = +2e-3, zs = -2e-3 (`--break qi`; `inputs/input.tokamak_nfp5` is this seed at A = 10, nfp 5). The signs matter: with
     the opposite orientation QI runs can settle on a branch with a much worse QI.
   Without a break, the default `seed_ellipse` adds a 0.07 (2, 1) ellipse (the PHR start).
2. **Choose nfp up front.** A tokamak has no field periods, but the deck's `NFP` fixes the symmetry of everything
   that follows (boundary modes, coils): use the target's nfp (`--nfp`). The aspect ratio is usually the target's
   (`--aspect`); `--kappa` / `--delta` give an elliptic or D-shaped start.
3. **Expect zero iota and a near-zero coil error at the start.** A vacuum tokamak has iota ~ 0, so the iota rows
   start far from feasible; the continuation (mode 1 frees the helical shaping that makes iota) and the PHR
   penalties take care of it, no special setting is needed. The coils fit an axisymmetric boundary almost
   perfectly, so the default flux bound (0.5 x the fit achieved at each mode start) becomes tiny and strangles the
   shaping: **set a floor** on it (`--flux-target-floor`, below).

Finite beta: a current-free tokamak holds no pressure (iota ~ 0), so start in vacuum and raise the pressure stage
by stage once the shape has iota (one stage per continuation mode works; a smaller VMEC time step, `DELT 0.5`
in the deck, helps at high beta). Constraints that need a shaped field (max-J, mirror floor, Mercier) are better switched
on later: `phr.maxj_from_step`, `phr.bmaxsp_from_step`, or a later stage with `--start-step`.

### 1.2 Recipe: QA from a tokamak

Seed `inputs/input.tokamak_nfp2` (circular torus, R0 = 1, a = 1/6, the QA break), the LP QA targets of example 1
(aspect 6, iota 0.4193, elongation 4.602, well >= -0.0304), 5 coils per half period with the PHR coil limits, 5
continuation steps x 5 outers with the production preset (flux metric scale 500):

```bash
qss-run --profile examples/qa_from_tokamak.json --preset production --run-dir runs/qa_tok
```

### 1.3 Recipe: QI from a tokamak

Seed `inputs/input.tokamak_nfp5` (circular torus, R0 = 1, a = 0.1, nfp 5, the QI break) with the QI targets of
`examples/qi_maxj_mirror.json` (aspect 10, iota 0.85 / 0.95 at axis / edge, mirror <= 0.25). The settings that made
the QI tokamak starts work: flux metric scale 100, a PHR penalty cap of 64, no carried trust radius (the carried
radius collapsed in these starts), and a flux bound floor of 1.3e-5 in modes 1-4, then a looser floor (4.5e-5) and
more outers in the last modes. As three stages:

```bash
qss-run --profile examples/qi_maxj_mirror.json --preset production --run-dir runs/qi_tok_s1 --n-continuation-steps 4
```

```bash
qss-run --profile examples/qi_maxj_mirror.json --preset production --run-dir runs/qi_tok_s2 --resume-from runs/qi_tok_s1/al1_checkpoint_final.json --start-step 4 --n-continuation-steps 5 --al-outer-iter 8 --flux-target-floor 4.5e-5
```

```bash
qss-run --profile examples/qi_maxj_mirror.json --preset production --run-dir runs/qi_tok_s3 --resume-from runs/qi_tok_s2/al1_checkpoint_final.json --start-step 5 --n-continuation-steps 6 --al-outer-iter 8 --flux-target-floor 4.5e-5
```

(The profile's `"run"` section holds the first stage's settings; the command-line options override them.) A little
shaping of the seed (`--kappa 1.3 --delta 0.3`) can help; freeing toroidal modes early (`equilibrium.ntor_extra`) did
not help QI starts.

## 2. The starting coils

### 2.1 Default starts

The first coil solve of a run (and every cold solve) builds its starting coils from the current boundary, using
XCoil's start families. `coils.start` is `family_label`:

| family | coils |
|---|---|
| `offset_3d` (default) | curves on a surface offset outward from the plasma, one per toroidal slot |
| `offset_planar` | planar coils on the offset surface |
| `circle` | circles around the magnetic axis |

| label | size of the start |
|---|---|
| `L85` (default), `L70`, `L100` | offset chosen so the total coil length is 85 / 70 / 100 % of the length cap |
| `clear` | the smallest offset that clears 1.1 x `min_cs` from the plasma |
| `L85_tilt+8`, `L85_tilt-8` | L85 tilted by +-8 degrees |

The base coils sit at the toroidal angles of `n_coils` evenly spaced slots per half period, are fitted at Fourier
order `coils.order`, and start with equal currents `total_current / n_coils` (the currents are free afterwards, with
a fixed total). If the primary start fails, the fallbacks `clear`, `L85_tilt+8`, `L70`, `L85_tilt-8`, `L100` are
tried (at most 3 cold solves). With `coils.link_guard: "link"` a start whose coils are linked is rejected and the same
label is tried in the other families. The publication stage (`qss-coils`, `--final-coil-stage`) solves every start
and keeps the best certified one.

### 2.2 Starting from your own coils

`coils.seed_coils` gives a coil file the **first cold solve of the process** starts from (then the defaults take
over as fallbacks). Accepted files:

* a **phrport-coils-v1** file from this package: `final_coils.json` / `coils_certified.json` / `.npz` of a run or of
  `qss-coils`. Any Fourier order (it is refitted at `coils.order`); nfp and `n_coils` must match the problem.
* a **points JSON** `{"points": [[[x, y, z], ...], ...], "currents_A": [...]}`: one list of points per base coil
  (the coils of one half period that generate the full set by symmetry), and their currents. From simsopt:

```python
import json
# base_curves / base_currents: the simsopt curves and currents of one half period
json.dump({"points": [c.gamma().tolist() for c in base_curves],
           "currents_A": [I.get_value() for I in base_currents]}, open("my_coils.json", "w"))
```

MAKEGRID coil files and FOCUS files are not read directly: load them with simsopt
(`simsopt.field.load_coils_from_makegrid_file`), keep the base coils of one half period, and write the points
JSON above. Currents are rescaled to `coils.total_current`.

In a profile:

```json
"coils": {"n_coils": 5, "seed_coils": "runs/qa_tok/final_coils.json"}
```

or for a single coil solve on a fixed boundary:

```bash
qss-coils --profile examples/qa_vacuum_tiny.json --run-dir runs/first --out runs/first/coils_from_seed --quick --override '{"seed_coils": "runs/first/final_coils.json"}'
```

### 2.3 Warm starts during a run, and resumes

* Inside a run, the coils of the previous evaluation are reused (warm start) when the boundary moved by less than
  `--surface-dof-reset-threshold`; with `--coil-oracle anchored` (production preset) warm starts come only from the coils
  at accepted points, with a cold audit every outer. The seed coils matter for the first solve only.
* At a new continuation mode the coils continue warm; the flux bound is re-captured.
* A resumed run (`--resume-from`) starts with a cold coil solve on the checkpoint boundary.
  `PHRPORT_SEED_ANCHOR` (a JSON with `solver_x`, `currents`, `c0`, `boundary`) restores a previous anchor instead.

### 2.4 Pitfalls

* **Seed coils too close to the new plasma**: if the boundary differs from the one the coils were made for, they
  may violate `min_cs` or even cut through it. The solver pushes them out or fails over to the default starts;
  the phantom-flux guard (`phr.cs_threshold`) keeps coils that cut the plasma from faking a good fit.
* **Linked seed**: a seed whose coils are linked is never accepted (certification); with `link_guard` it is
  rejected at once.
* **Ordering rows at a crossing start** (`slot_rows`): coils that start crossed must first untangle; the ordering
  rows have a large violation and the first solve is slow. Start from uncrossed coils or from the default starts.
* **Different nfp or coil count**: the file is refused with a clear error. Change `n_coils` or regenerate the seed.
* **Tokamak seeds**: the default starts work (they are built from the boundary); the near-zero coil error of an
  axisymmetric boundary is why the flux bound needs a floor (section 1.1).
