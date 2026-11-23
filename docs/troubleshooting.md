# Troubleshooting and FAQ

**Before anything: `qss-check --profile my.json`.** It checks the install, the devices, the XCoil backend, the
input deck and every profile key (misspelled keys are ignored by the code, `qss-check` tells you).

### The run stops with "initial equilibrium failed"
VMEX did not converge on the seed boundary of the deck. Check the deck alone (`inputs/...`): raise `niter_array`,
use a two-stage `ns_array` such as [25, 50] with `ftol_array` [1e-14, 1e-13], or lower `vmec_mpol`. Very low aspect
ratio or strongly shaped seeds may need a smaller time step (`DELT` in the deck). For a resumed stage whose deck has
no converged seed, the run continues from the checkpoint boundary with a warning (that is expected).

### A tokamak seed does not move (iota stays ~0, QS / QI stalls in mode 1-2)
The seed is exactly axisymmetric or the flux bound strangles it: add a symmetry break (`qss-seed tokamak --break
qa|qi`) and a flux bound floor (`--flux-target-floor`, 1.3e-5 in the paper's QI start); for QI also try
`--rho-max 64 --no-carry-trust-radius`. See [seeding.md](seeding.md#11-starting-from-a-tokamak).

### Many "VMEX failed" trials, the inner solve makes no progress
The trust region asks for boundaries VMEX cannot solve. Use `--preset production` (it carries the trust radius and
resets it per step: `PHRPORT_TRUST_RESET_STEP=1`, `PHRPORT_TRUST_FLOOR=1e-3`), lower `--max-nfev` per inner solve
so multipliers update more often, or start the next continuation step from the last good checkpoint.

### "coil solve not certified" / `final_coils.json: certified: false`
In the loop, every trial coil solve is certified by default; with `--no-certify-trials` (the production preset) only
accepted coils are. For the final design always run `qss-coils` (cold starts + warm polish to a
certified violation <= 1e-5); its summary lists `n_met` and `v_target_met`. If no start meets the target:
- the limits may be infeasible for this boundary: compare `final_metrics.json: coils` (`individual_lengths`,
  `max_curvature`, `individual_msc`, `min_cc_distance`, `min_cs_distance`) with the profile's limits;
- try more starts (`--override '{"max_starts": 6}'`) or another start family (`"start": "offset_planar_L85"`);
- linked results (`linked: true`) are never accepted: use `"link_guard": "link"`.

### The flux bound row `Jf` stays VIOLATED and its penalty grows
The coils cannot reach the requested fit with these limits. Options: a larger `flux_target_frac` (0.7-0.9),
a floor (`--flux-target-floor`), more coils or looser limits, or accept the trade-off with a QS cap
(`phr.qs_max`). Check `final_metrics.json: coils.min_cs_distance` against `cs_threshold`: coils that hug the plasma
trigger the phantom-flux guard.

### Mercier row behaves erratically
DMerc is noisy close to the axis and the edge at low radial resolution. Use `mercier_options` s_min 0.1-0.15,
s_max <= 0.95 and `ns_array` ending at >= 50 surfaces.

### Out of memory
- GPU, finite beta: `PHRPORT_FB_CHUNK=128`, or `--fb-jit off`; the jitted virtual casing falls back to the eager
  path by itself if compiling fails.
- GPU in general: `XLA_PYTHON_CLIENT_PREALLOCATE=false` (set by the Slurm template), `PHRPORT_VMEX_DEVICE=cpu`.
- QI runs: lower `phr.eq_cache_size` (4 in the paper profile) and `equilibrium.mercier_jac_batch`.
- Long runs whose memory grows: `PHRPORT_CLEAR_CACHES_RSS_MB=90000`.
- Laptop: fewer VMEX workers (`--n-workers 2`).

### The job hit its time limit
Resubmit the same `sbatch` command: the template continues from the newest `al1_checkpoint_latest.json` in a new
folder `runs/<RUN>_r2` with `--resume-keep-target`. By hand, see tutorial 5. Setting `JAX_COMPILATION_CACHE_DIR`
on the project file system (the template does) avoids recompiling.

### Resuming after adding a constraint row
Checkpoints store their row labels; new rows start with multiplier 0. For a checkpoint written by an older version
without labels, set `PHRPORT_RESUME_EQ_LABELS` to its rows (comma separated, in order).

### Results differ slightly between machines
Fresh XLA compiles and CPU vs GPU change rounding; the trajectory of a nonlinear optimization amplifies it. Runs are
bitwise repeatable on the same hardware with the same compile cache.

### Slow coil solves on a laptop
Ordering (`slot_rows`), Lp curvature and force rows have dense Jacobians: on a CPU a coil solve goes from seconds to
minutes. Keep them for GPU runs, or test with the plain limits first.

### XCoil cannot be imported
`qss-check` prints the reason. Usually XCoil is not installed: `pip install -e .` installs it from GitHub
(see [xcoil.md](xcoil.md#version-and-install)); a failed `git clone` of `PedroFranciscoGil/XCoil` during
`pip install` means exactly that. Otherwise `PHRPORT_XCOIL` must be
unset or the folder that contains the `xcoil` package.
