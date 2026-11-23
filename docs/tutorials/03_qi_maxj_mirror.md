# Tutorial 3: quasi-isodynamic with max-J, mirror ratio and coil ordering

Goal: a five-field-period QI configuration (aspect 10) started from a tokamak, with the constructed-QI residual as
objective, a mirror-ratio cap on every QI surface, axis and edge iota bands, a max-J row switched on part-way, and six
coils per half period that keep their toroidal order.

Profile: [`examples/qi_maxj_mirror.json`](../../examples/qi_maxj_mirror.json) (laptop variant
[`examples/qi_maxj_mirror_tiny.json`](../../examples/qi_maxj_mirror_tiny.json): three QI surfaces, coarser field-line
grids and Boozer resolution, two continuation steps). The annotated version is in
[profile.md](../profile.md#annotated-example-3-qi-from-a-tokamak-with-max-j-mirror-and-coil-ordering-examplesqi_maxj_mirrorjson).

## The QI keys

* `physics: "qi"` replaces the QS residual by the smooth constructed-QI residual (Goodman et al. 2023) on the
  `qi_surfaces`, computed from a traceable Boozer transform of the VMEX state (exact Jacobian available);
* `mirror_max` caps (B_max - B_min)/(B_max + B_min); with `mirror_all` every QI surface is capped through a soft max;
* `maxj` adds the max-J row on the constructed field (`maxj_target`, `maxj_depths`); the real-field J-contour closure
  (`qi_options.jreal_*`) can be folded into the same row ([constraints.md](../constraints.md#a9-maximum-j));
  `maxj_from_step: 3` switches it on at the fourth continuation step;
* `slot_rows` keeps neighbouring coils (and the end coils with their stellarator images) toroidally ordered with a
  1.5 degree gap, so coils cannot cross; linking numbers are always certified;
* the `"run"` section holds the tokamak-start settings (see [seeding.md](../seeding.md#13-recipe-qi-from-a-tokamak)).

Other QI rows: `mirror_min`, `bmaxsp` (B_max spread across field lines), `mercier` (see
[constraints.md](../constraints.md)).

## Laptop run (a few minutes on a CPU)

```bash
qss-run --profile examples/qi_maxj_mirror_tiny.json --preset tiny --run-dir runs/t3
```

```bash
python -c "import json; d=json.load(open('runs/t3/final_metrics.json')); print(d['surface']['g_eq']); print(d['coils'].get('slot'))"
```

`g_eq` now contains `mirror`, `maxj`, `iota_axis_hi/lo` and `iota_edge_hi/lo`; the coil report has the ordering
gaps per coil pair (`min_gap_deg`, negative = crossing).

## Production run

```bash
qss-run --profile examples/qi_maxj_mirror.json --preset production --run-dir runs/qi
```

The real-field J closure (15 surfaces x 36 field lines x 7 pitches) is the expensive part; run it on a GPU node.
