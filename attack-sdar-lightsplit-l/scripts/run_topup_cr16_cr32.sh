#!/usr/bin/env bash
# Top-up seeds for the F-vs-L comparison at CR16 and CR32, where the SDAR victim collapsed on one
# side in the first pass (F cr32 0 healthy, L cr16 0 healthy; F cr16 & L cr32 only 1 healthy).
# Adds seeds 3,4,5 for BOTH LightSplit-F (fixed_random) and LightSplit-L (fixed_random_ll) at CR{16,32}.
# Both run from THIS copy (its fixed_random is byte-identical to production; fixed_random_ll is here).
# USL / L7 / cifar10 / wcc=0. 12 runs, 4 free GPUs. Health-gate + average afterward (gate_sweep.py).
set -uo pipefail
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=3
PY=python
OUT=./reproduce/logs

# job = mode:cr:seed  (mode f=LightSplit-F, l=LightSplit-L)
JOBS=()
for mode in f l; do for cr in 16 32; do for s in 3 4 5; do JOBS+=("$mode:$cr:$s"); done; done; done

run_one() {
  local gpu="$1" mode="$2" cr="$3" seed="$4"
  local args prefix
  if [ "$mode" = "f" ]; then
    args="--bottleneck_mode fixed_random"; prefix="fr_usl_l7"
  else
    args="--bottleneck_mode fixed_random_ll --mlp_hidden 512"; prefix="frll_usl_l7"
  fi
  local log="$OUT/${prefix}_cr${cr}_run${seed}.log"
  echo "[$(date +%H:%M:%S)] GPU$gpu START $mode cr$cr seed$seed -> $(basename $log)"
  CUDA_VISIBLE_DEVICES=$gpu $PY run_sdar.py usl resnet cifar10 7 "$seed" \
    $args --compression_ratio "$cr" --wcc_lambda 0 --print_to_stdout > "$log" 2>&1
  echo "[$(date +%H:%M:%S)] GPU$gpu DONE  $mode cr$cr seed$seed"
}

for g in 0 1 2 3; do
  (
    i=$g
    while [ $i -lt ${#JOBS[@]} ]; do
      IFS=: read -r mode cr seed <<< "${JOBS[$i]}"
      run_one "$g" "$mode" "$cr" "$seed"
      i=$((i+4))
    done
  ) &
done
wait
echo "[$(date +%H:%M:%S)] all top-up runs complete"
