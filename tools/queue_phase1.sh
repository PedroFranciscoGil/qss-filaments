#!/bin/bash
# Phase-1 runs on the host (roadmap): A1 polish with the B1 KKT exit, then baseline vs speed-bundle timing.
cd "$(dirname "$0")/.." && . tools/host_env.sh
E="--flux-metric-scale 200 --eq-jacobian exact --n-continuation-steps 4 --al-outer-iter 6 --max-nfev 20 --n-workers 6"
run() { name=$1; shift; echo "== $name start $(date)"; "$PHRPORT_PY" -u -m phrport.phr --run-dir runs_phrport/$name "$@" > runs_phrport/$name.log 2>&1; echo "== $name end $(date) rc=$?"; }
run p1_a1_frac0.3seed_kkt --resume-from runs_phrport/s200_exact_frac0.3/al1_checkpoint_step03_iter01.json --flux-target-frac 0 --flux-phr-target 1.753e-4 --final-mode-kkt-exit $E
run p1_a1_frac0.2seed_kkt --resume-from runs_phrport/s200_exact_frac0.2/al1_checkpoint_step03_iter01.json --flux-target-frac 0 --flux-phr-target 1.512e-4 --final-mode-kkt-exit $E
run p1_exact_baseline $E
run p1_exact_speedbundle $E --abort-doomed-coil-solves --no-certify-trials --reuse-vmex-state
