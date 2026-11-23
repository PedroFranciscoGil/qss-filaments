#!/bin/bash
# Generic Slurm job for one quasi-single-stage optimization (one GPU), with automatic resume and the certified
# coil stage at the end. Copy it next to your runs, adjust the #SBATCH lines to your cluster, then:
#
#   PROFILE=examples/qa_vacuum.json RUN=qa_vac sbatch tools/slurm/qss_run.slurm
#
# Variables (environment, all optional except PROFILE and RUN):
#   PROFILE     problem profile (JSON)                      RUN        run name (folder runs/$RUN)
#   VENV        python environment (default: ./.venv)        PRESET     qss-run preset (default: production)
#   NSTEPS      continuation steps                           NOUT       AL outers per step
#               (both optional: default = the profile's "run" section, else 4 and 6)
#   EXTRA       extra qss-run options, e.g. "--flux-target-floor 4e-5"
#   RESUME_FROM checkpoint to start from (a previous stage's al1_checkpoint_final.json); START_STEP its step
#   MAX_TRIES   restarts after a crash / time limit (default 4)  CERTIFY 0 skips the coil stage (default 1)
#
# Resume logic: if runs/$RUN already holds a checkpoint, a new segment runs/${RUN}_r2, _r3, ... continues from the
# newest al1_checkpoint_latest.json with --resume-keep-target (same flux target, multipliers and penalties). The job
# can therefore simply be resubmitted after a time limit. The finished run has al1_checkpoint_final.json.
#
#SBATCH --job-name=qss
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=runs/%x_%j.out

set -u
HERE=${SLURM_SUBMIT_DIR:-$PWD}; cd "$HERE"
: "${PROFILE:?set PROFILE=path/to/profile.json}"; : "${RUN:?set RUN=name}"
VENV=${VENV:-$HERE/.venv}; PY=$VENV/bin/python; QSS_RUN=$VENV/bin/qss-run; QSS_COILS=$VENV/bin/qss-coils
PRESET=${PRESET:-production}; MAX_TRIES=${MAX_TRIES:-4}; CERTIFY=${CERTIFY:-1}

# environment: x64, VMEX on the CPU when the GPU is small (set PHRPORT_VMEX_DEVICE=cpu), XCoil on the GPU,
# no JAX memory preallocation, caches inside the run folder (compiles are reused across restarts)
export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false MALLOC_ARENA_MAX=2
export PHRPORT_TRUST_RESET_STEP=${PHRPORT_TRUST_RESET_STEP:-1} PHRPORT_TRUST_FLOOR=${PHRPORT_TRUST_FLOOR:-1e-3}
export XDG_CACHE_HOME=$HERE/.cache JAX_COMPILATION_CACHE_DIR=$HERE/.cache/jax PHRPORT_PYCACHE=$HERE/.cache/py
mkdir -p "$HERE/runs" "$HERE/.cache"
command -v nvidia-smi >/dev/null && nvidia-smi -L; date

newest() {   # newest segment of $1 that holds a checkpoint: runs/$1, runs/$1_r2, ...
  local d L=; for d in "$HERE/runs/$1" $(ls -d "$HERE/runs/$1"_r[0-9]* 2>/dev/null | sort -V); do
    [ -f "$d/al1_checkpoint_latest.json" ] && L=$d; done; echo "$L"; }

TRIES=0
while :; do
  LAST=$(newest "$RUN")
  if [ -n "$LAST" ] && [ -f "$LAST/al1_checkpoint_final.json" ]; then echo "run finished in $LAST"; break; fi
  TRIES=$((TRIES + 1)); [ $TRIES -gt "$MAX_TRIES" ] && { echo "giving up after $((TRIES - 1)) attempts"; exit 1; }
  if [ -z "$LAST" ]; then SEG=$RUN
    START=""; [ -n "${RESUME_FROM:-}" ] && START="--resume-from $RESUME_FROM ${START_STEP:+--start-step $START_STEP}"
  else n=2; while [ -e "$HERE/runs/${RUN}_r$n" ]; do n=$((n + 1)); done; SEG=${RUN}_r$n
    START="--resume-from $LAST/al1_checkpoint_latest.json --resume-keep-target"; fi
  echo "attempt $TRIES: segment $SEG $START"; date
  export PHRPORT_SCRATCH=$HERE/.cache/scratch/$SEG
  "$QSS_RUN" --profile "$PROFILE" --preset "$PRESET" --run-dir "$HERE/runs/$SEG" \
      ${NSTEPS:+--n-continuation-steps $NSTEPS} ${NOUT:+--al-outer-iter $NOUT} ${EXTRA:-} $START > "$HERE/runs/$SEG.log" 2>&1
  rc=$?; echo "=== exit $rc" >> "$HERE/runs/$SEG.log"; echo "attempt $TRIES: exit $rc"; date
done

# certified coil stage: every start, cold solve, warm polish to certified violation <= 1e-5, lowest Lagrangian
if [ "$CERTIFY" = 1 ] && [ ! -f "$LAST/coils_certified.json" ]; then
  "$QSS_COILS" --profile "$PROFILE" --run-dir "$LAST" --out "$LAST/coils_certified" > "$LAST/coils_certified.log" 2>&1 \
    && echo "certified coils: $LAST/coils_certified.json" || echo "coil stage failed (see $LAST/coils_certified.log)"
fi
date
