#!/bin/bash
# Start workers for one suite, split over two GPUs.
# Usage: bash launch.sh <ecg|derm> <suite> <workers_gpu0> <workers_gpu1> <cache_dir> <out_dir> <log_dir>
# Each worker holds the cohort on its GPU (about 2 GB); choose worker counts to fit free GPU memory.
# Rerunning the same command resumes: finished jobs are skipped. With WAIT=1 the script returns when all workers have finished.
set -e
TASK=${1:?task: ecg or derm}; SUITE=${2:?suite}; W0=${3:?workers on gpu 0}; W1=${4:?workers on gpu 1}
CACHE=${5:?cache dir}; OUT=${6:?output dir}; LOGS=${7:?log dir}
case "$TASK" in
  ecg)  SCRIPT=run.py ;;
  derm) SCRIPT=run_derm.py ;;
  *) echo "task must be ecg or derm" >&2; exit 1 ;;
esac
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT" "$LOGS"
N=$((W0+W1)); k=0
for g in 0 1; do
  W=$W0; [ $g -eq 1 ] && W=$W1
  for i in $(seq 1 $W); do
    nohup python3 "$HERE/$SCRIPT" worker --suite "$SUITE" --gpu $g --shard $k --nshards $N \
        --cache-dir "$CACHE" --out-dir "$OUT" > "$LOGS/${SUITE}_w${k}.txt" 2>&1 &
    k=$((k+1))
  done
done
[ -n "$WAIT" ] && wait
echo "started $N workers ($W0 on gpu0, $W1 on gpu1) for suite $SUITE; logs in $LOGS"
