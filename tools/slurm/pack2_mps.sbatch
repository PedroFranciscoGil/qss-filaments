#!/bin/bash
# Pack two independent runs (e.g. two phrport optimisations) on ONE GPU under a private CUDA MPS daemon.
#
#   cd <repo or run folder> && mkdir -p runs
#   sbatch [--job-name=..] [--time=..] tools/slurm/pack2_mps.sbatch '<command 1>' '<command 2>'
#
# Each command runs with bash -c from the submit directory ($SLURM_SUBMIT_DIR, or PACK_DIR), so $SLURM_JOB_ID
# etc. expand inside the job.  Both runs share the GPU through MPS (XLA_PYTHON_CLIENT_MEM_FRACTION=${PACK_MEMFRAC:-0.45}
# each, XLA_PYTHON_CLIENT_PREALLOCATE=false).  Do not start another MPS daemon inside the commands.
#
# Measured on an HPC cluster (H200, 8 CPUs, 2026-11-04): two runs packed under MPS = 1.66x throughput (two qh4 probes
# unpinned: 1294 s each vs 1072 s solo; full qh4 + finite-beta pair: 89 min vs 145-149 min solo, 1.63-1.67x), each run
# 12-19 % slower per VMEX iteration.  With the default PACK_PIN=0 the packed runs are bit-for-bit the solo 8-CPU runs
# (159/159 f-evals compared).  Two processes on one GPU without MPS were slower (0.87x).
#
# Options (environment):
#   PACK_PIN=0 (default)  no taskset and no thread variables: each run sees all the job's CPUs, as a solo run does,
#                         and is bitwise identical to a solo run on the same CPU count.
#   PACK_PIN=1            pin each run to half of the CPUs (taskset; child VMEX workers inherit it) with
#                         OMP/OPENBLAS/MKL_NUM_THREADS = that half; same speed, but the half-width BLAS/XLA
#                         rounding differs from full-width baselines (not bitwise).
#   PACK_THREADS_REPORT='<cmd>'  run <cmd> once per run's CPU set before starting (e.g. a python thread report).
#   PACK_MEMFRAC          per-run XLA memory fraction (default 0.45).
#   PACK_DIR / PACK_OUT   working directory (default $SLURM_SUBMIT_DIR) / log folder (default <PACK_DIR>/runs).
# Output: <PACK_OUT>/pack_<job>_{1,2}.log (each ends with a 'PACK_RUN exit <rc> seconds <s>' line),
#         pack_<job>_smi.csv (GPU every 60 s), pack_<job>_rss.txt (RSS per process tree + GPU memory per pid).
# Example:
#   sbatch tools/slurm/pack2_mps.sbatch \
#     'python -m phrport.phr --run-dir runs/A --phase-timers ...' \
#     'python -m phrport.phr --run-dir runs/B --fb-reuse-tol 2e-3 --fb-reuse-every 4 ...'
# If a node is known to hang jobs before import, add --exclude=<node>.
#SBATCH --job-name=pack2
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2:30:00
#SBATCH --output=runs/%x_%j.out
S=${PACK_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}; cd $S
O=${PACK_OUT:-$S/runs}; mkdir -p $O
[ $# -eq 2 ] || { echo "need two commands"; exit 1; }
J=$SLURM_JOB_ID; T=${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/pack_$J
export CUDA_MPS_PIPE_DIRECTORY=$T/mps_pipe CUDA_MPS_LOG_DIRECTORY=$T/mps_log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
nvidia-smi -L; hostname; date
nvidia-cuda-mps-control -d && echo "MPS daemon started ($CUDA_MPS_PIPE_DIRECTORY)" || { echo "MPS start failed"; exit 1; }
SMI=$O/pack_${J}_smi.csv RSS=$O/pack_${J}_rss.txt
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 60 > $SMI &
SMIP=$!
cleanup() { kill $SMIP $SAMP 2>/dev/null; echo quit | nvidia-cuda-mps-control; }
trap cleanup EXIT
# split the allowed CPUs in two halves
read C1 C2 < <(python3 -c "import os;c=sorted(os.sched_getaffinity(0));h=len(c)//2;print(','.join(map(str,c[:h])),','.join(map(str,c[h:])))")
NT=$(( $(echo $C1 | tr ',' '\n' | wc -l) ))
echo "cpus run1: $C1   run2: $C2   threads $NT"
PIN=${PACK_PIN:-0}
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=${PACK_MEMFRAC:-0.45}
if [ "$PIN" = 1 ]; then export OMP_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT MKL_NUM_THREADS=$NT
else C1=$(python3 -c "import os;print(','.join(map(str,sorted(os.sched_getaffinity(0)))))"); C2=$C1; echo "no pinning: both runs on cpus $C1"; fi
if [ -n "$PACK_THREADS_REPORT" ]; then for C in $C1 $C2; do echo "--- thread report (cpus $C):"; taskset -c $C $PACK_THREADS_REPORT; done; fi
CMDS=("$1" "$2"); PIDS=(); T0=$(date +%s)
for k in 0 1; do C=$C1; [ $k = 1 ] && C=$C2
  echo "=== $(date +%T) run $((k+1)) [cpus $C]: ${CMDS[$k]}"
  taskset -c $C bash -c "${CMDS[$k]}"'; echo "PACK_RUN exit $? seconds $SECONDS"' > $O/pack_${J}_$((k+1)).log 2>&1 &
  PIDS+=($!)
done
tree_rss() {   # total RSS (MB) of a process and all its descendants (run_speed.slurm uses setsid, so walk ppid)
  ps -eo pid=,ppid=,rss= | awk -v r=$1 '{p[$1]=$2; m[$1]=$3} END{s=0; for(i in p){j=i; while(j>1){ if(j==r){s+=m[i];break}; j=p[j]}} printf "%.0f", s/1024}'
}
( M1=0; M2=0; while :; do a=$(tree_rss ${PIDS[0]}); b=$(tree_rss ${PIDS[1]})
    [ $a -gt $M1 ] && M1=$a; [ $b -gt $M2 ] && M2=$b
    echo "$(date +%T) rss1 ${a} MB rss2 ${b} MB  peak1 ${M1} peak2 ${M2}" >> $RSS
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader >> $RSS 2>/dev/null; sleep 60; done ) &
SAMP=$!
for k in 0 1; do wait ${PIDS[$k]}; echo "=== $(date +%T) run $((k+1)) exit $?  (per-run time: PACK_RUN line in its log)"; done
echo "job wall $(( $(date +%s) - T0 )) s"; tail -1 $RSS; date; echo PACK2_DONE
