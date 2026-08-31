# Shared settings for the FORA reproduction scripts. Source, do not run.
#
#   PYTHON     python with torch, torchvision, piq, tensorboard          default: python
#   OUT        output root                                              default: <attack-fora>/results/reproduce
#   DATA_ROOT  torchvision download directory for CIFAR-10             default: $OUT/<table>/data (downloaded there)
#   GPUS       GPU ids to spread the runs over, one run per GPU         default: 0
#   ITERS      attack iterations                                        default: 40000 (the paper)
REPRO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${CODE_DIR:-$(cd "$REPRO_DIR/.." && pwd)}"
PY="${PYTHON:-python}"
OUT="${OUT:-$CODE_DIR/results/reproduce}"
GPUS="${GPUS:-0}"
ITERS="${ITERS:-40000}"
COMMON=(--layer_id 2 --lr 1e-4 --dlr 3e-4 --a 0.8 --dataset cifar --batch_size 64 --gid 0)
# run_one <gpu> <tag> <main.py defense args...>
# Runs main.py inside $RUN_DIR (it writes log/, model/ and recon_pics/ relative to its cwd) with
# --save_path cifar_level2_<tag>.pth, logging to $RUN_DIR/logs/cifar_level2_<tag>_full.log; a run
# whose log already ends with main.py's "Training Complete" line is skipped.
run_one() { local gpu="$1" tag="cifar_level2_$2"; shift 2; local log="$RUN_DIR/logs/${tag}_full.log"
  grep -qs "Training Complete" "$log" && { echo "[skip] $tag"; return; }
  echo "[$(date +%H:%M:%S)] gpu$gpu $tag"
  ( cd "$RUN_DIR" && CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$CODE_DIR/main.py" "${COMMON[@]}" --iteration "$ITERS" \
      --save_path "$tag.pth" "$@" > "$log" 2>&1 ) || echo "[fail] $tag (see $log)"; }
# run_specs: SPECS holds "cell|seeds|args" lines; every (cell, seed) becomes one run tagged <cell>_s<seed>
# with --seed <seed>; runs are assigned round-robin to the GPUs in $GPUS, one lane per GPU
run_specs() { local -a gpus jobs; read -r -a gpus <<< "$GPUS"; local ng=${#gpus[@]} g i spec cell seeds args
  for spec in "${SPECS[@]}"; do IFS='|' read -r cell seeds args <<< "$spec"
    for sd in $seeds; do jobs+=("${cell}_s${sd}|--seed $sd $args"); done; done
  for g in "${!gpus[@]}"; do ( i=0; for job in "${jobs[@]}"; do
      if [ $((i % ng)) -eq "$g" ]; then run_one "${gpus[$g]}" "${job%%|*}" ${job#*|}; fi; i=$((i+1)); done ) & done
  wait; }
