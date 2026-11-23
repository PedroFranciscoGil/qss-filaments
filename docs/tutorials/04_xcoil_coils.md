# Tutorial 4: XCoil coil solves and certification

Goal: solve, certify and export coils for a fixed boundary, change the coil limits, and call the coil solver from
Python. Background: [xcoil.md](../xcoil.md); limits: [constraints.md](../constraints.md#b-coil-side-limits-of-the-coil-subproblem).

## 1. One certified cold solve (about 10 s on a CPU)

On the final boundary of tutorial 1 (or any run folder):

```bash
qss-coils --profile examples/qa_vacuum_tiny.json --run-dir runs/t1 --out runs/t1/coils_quick --quick
```

The summary line reports c0 (squared flux), `certified`, `linked`, the certified violation `v` and
`bn_ratio_int` = <|B.n|>/<|B|>. `runs/t1/coils_quick.json` is a phrport-coils-v1 file (simsopt
`CurveXYZFourier` DOFs per base coil, currents, nfp / stellarator symmetry and a simsopt loader snippet).

## 2. Publication stage: cold start + warm polish (about 2 minutes on a laptop CPU)

```bash
qss-coils --profile examples/qa_vacuum_tiny.json --run-dir runs/t1 --out runs/t1/coils_certified
```

Every start is solved cold; every ok, unlinked one is polished warm until the CERTIFIED violation of the true
limits (exact distances, 4096-sample curvature, linking) is <= 1e-5, and the lowest polished Lagrangian wins. The
summary reports `chosen_start`, `n_polished`, `n_met` and `v_target_met`.

## 3. Other limits

```bash
qss-coils --profile examples/qa_vacuum_tiny.json --run-dir runs/t1 --out runs/t1/coils_short --quick --override '{"per_coil_length": 3.5, "max_curvature": 4.0}'
```

Any `CoilConfig` field can be overridden, e.g. `{"lp_curvature_max": 0.5, "lp_curvature_threshold": 1.0}` (Lp
curvature rows), `{"slot_rows": true, "slot_gap_deg": 1.5}` (toroidal ordering), `{"link_guard": "link"}`,
`{"n_coils": 5, "order": 8}`, `{"precision": "fp64"}`. The same keys go into a profile's `"coils"` section to use
them inside the optimization. Ordering, Lp-curvature and force rows have dense JAX Jacobians: on a laptop CPU they
make a coil solve minutes instead of seconds; on a GPU the cost is small.

## 4. From Python

```python
import os
os.environ["PHRPORT_PROBLEM"] = "examples/qa_vacuum_tiny.json"     # before importing phrport
os.environ.setdefault("JAX_ENABLE_X64", "1")
from phrport import equilibrium as E
from phrport.coils import CoilConfig, CoilSolver

b = E.seed_boundary()                       # {'rc(m,n)': ..., 'zs(m,n)': ...}
cs = CoilSolver(CoilConfig(n_coils=4, order=5, per_coil_length=4.0))
cold = cs.solve(b)                          # cold start (primary + fallbacks), certified
print(cold.ok, cold.certified, cold.c0_local, cold.info["v"], cold.info["bn_ratio_int"])

b2 = dict(b); b2["rc(1,0)"] *= 1.01         # a nearby boundary
warm = cs.solve(b2, warm=cold)              # warm start from the previous coils
grad = cs.envelope_grad(b2, warm, ["rc(1,0)", "zs(1,0)"])   # d c0*/d boundary (envelope theorem)

final = cs.final_coil_stage(b2)             # cold starts + certified warm polish
print(final.info["final_stage"]["chosen_start"], final.info["v"])
```

## 5. Backend and devices

`qss-check` shows the XCoil tree in use. `PHRPORT_XCOIL=/path/to/folder` selects a local XCoil checkout
([xcoil.md](../xcoil.md#selecting-the-backend)); the
default is the installed `xcoil` package at the commit pinned in `pyproject.toml`. On a GPU host, `PHRPORT_VMEX_DEVICE=cpu` keeps
VMEX on the CPU and gives XCoil the GPU.

## 6. Export for other codes

* simsopt: the `loader` field of the JSON;
* finite-build VTK (CoilPy): `python tools/coildump.py ...` then `python tools/coilfb.py ...`;
* DESC free-boundary check: `python tools/desc_free_boundary.py --wout W --coils runs/t1/coils_certified.json --tag t1 --out runs/t1/desc_fb`
  (optional extra `desc`).
