# Tutorial 5: continuation, resume and running on a cluster

## Continuation inside one run

`--n-continuation-steps N` frees the boundary modes |m|, |n| <= k at step k = 1..N (with `equilibrium.ntor_extra`
the toroidal range can run ahead). Each step runs `--al-outer-iter` AL outers; at the start of a step the coil flux
bound is re-captured (`--flux-target-frac` x the current c0, or the absolute targets). Every outer writes
`al1_checkpoint_stepSS_iterII.json` and `al1_checkpoint_latest.json`; the end of the run writes
`al1_checkpoint_final.json`.

## Resume an interrupted run (laptop example)

Run tutorial 1 with two steps and stop it (Ctrl-C) during step 2, or let it finish:

```bash
qss-run --profile examples/qa_vacuum_tiny.json --preset tiny --n-continuation-steps 2 --run-dir runs/t5
```

Continue from the newest checkpoint into a new folder, keeping its flux target, multipliers and penalties:

```bash
qss-run --profile examples/qa_vacuum_tiny.json --preset tiny --n-continuation-steps 2 --run-dir runs/t5_r2 --resume-from runs/t5/al1_checkpoint_latest.json --resume-keep-target
```

## Chain stages (change the problem between stages)

A stage can resume from the previous stage's final checkpoint with a different profile and start at a given
continuation step, e.g. switch on a Mercier row from step 4, or raise beta:

```bash
qss-run --profile examples/qa_vacuum_tiny.json --preset tiny --n-continuation-steps 3 --run-dir runs/t5_stage2 --resume-from runs/t5/al1_checkpoint_final.json --start-step 2
```

Checkpoints store the constraint-row labels, so multipliers are re-aligned when rows are added or removed (new rows
start at lambda = 0). For old checkpoints without labels, `PHRPORT_RESUME_EQ_LABELS=aspect,iota_mean_hi,...` gives
their row order. For a beta ramp, the deck of the next stage is a new profile with the new pressure profile; the
equilibrium of the resumed boundary is recomputed at the new beta.

## Front sweeps

From a finished run, `--sweep --resume-from RUN` traces the trade-off between QS and the coil flux bound (geometric
bound ratio `--sweep-ratio`, `--sweep-max-points`); `tools/front_collect.py` gathers the points of several sweeps into
one front. `--target-flux E` drives a single run to an absolute flux bound.

## Slurm

`tools/slurm/qss_run.slurm` is a generic one-GPU job: install the repository in a venv at `./.venv` (or set
`VENV`), adjust the `#SBATCH` lines (partition, account, time), then

```bash
PROFILE=examples/qi_maxj_mirror.json RUN=qi1 sbatch tools/slurm/qss_run.slurm
```

The job runs `qss-run --preset production` into `runs/$RUN`; if the folder already has a checkpoint it continues from
the newest one in a new segment `runs/${RUN}_r2`, ... with `--resume-keep-target`, so resubmitting the same command
after a time limit simply continues. After the run finishes it writes the certified coils
(`coils_certified.json`, cold starts + warm polish). Stage chaining: `RESUME_FROM=runs/qi1/al1_checkpoint_final.json START_STEP=5`.
`EXTRA="--flux-target-floor 4e-5"` passes other options.

Practical settings on GPU nodes (all set or suggested in the template):

* `XLA_PYTHON_CLIENT_PREALLOCATE=false` (VMEX worker processes and XCoil share the device);
* `JAX_COMPILATION_CACHE_DIR` on the project file system: restarts reuse the compiled programs, and runs are
  bitwise repeatable with the same cache;
* `PHRPORT_VMEX_DEVICE=cpu` on small GPUs; on H100/H200-class GPUs VMEX can stay on the GPU;
* `PHRPORT_FB_CHUNK=128` for finite beta on GPUs with <= 16 GB;
* `PHRPORT_CLEAR_CACHES_RSS_MB=90000` for very long runs whose JAX caches grow;
* two runs per GPU: `tools/slurm/pack2_mps.sbatch 'qss-run ...' 'qss-run ...'` (private CUDA MPS daemon, 1.65x
  throughput, results identical to solo runs).
