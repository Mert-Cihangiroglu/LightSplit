# Shared settings for the utility-llm reproduction scripts. Source, do not run.
#
#   PYTHON          python with torch, transformers, datasets        default: python
#   HF_HOME         Hugging Face cache (GPT-2 weights and WikiText-103 are downloaded on first use)
#   CUDA_VISIBLE_DEVICES  GPU to use                                  default: 0
#   OUT             output root                                       default: <utility-llm>/results/reproduce
#   MODELS          subset of gpt2 gpt2-medium gpt2-large gpt2-xl (space separated), default all; runs are
#                   independent, so models can be split across GPUs by invoking the script once per GPU
#   SMOKE=1         gpt2 only, a few hundred tokens; pipeline test only
REPRO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${CODE_DIR:-$(cd "$REPRO_DIR/.." && pwd)}"
PY="${PYTHON:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" TOKENIZERS_PARALLELISM=false
export SLLM_TOK_CACHE="${SLLM_TOK_CACHE:-$CODE_DIR/.tok_cache}"
OUT="${OUT:-$CODE_DIR/results/reproduce}"
MODELS="${MODELS:-gpt2 gpt2-medium gpt2-large gpt2-xl}"
MTB="${MTB:-80000}"; MVB="${MVB:-1500}"      # training / validation blocks of 256 tokens (~20.5M training tokens)
if [ "${SMOKE:-0}" = 1 ]; then MODELS=gpt2; MTB=16; MVB=4; fi
DATASET=(--dataset-repo Salesforce/wikitext --dataset-config wikitext-103-raw-v1 --dataset-name wikitext103)
cut_for()   { case "$1" in gpt2) echo 1;; gpt2-medium) echo 2;; gpt2-large) echo 4;; gpt2-xl) echo 5;; esac; }   # round(0.1 * n_layer)
batch_for() { case "$1" in gpt2-xl) echo 4;; *) echo 8;; esac; }
# run <model> <mode> <cr>  -> $OUT/<figure>/<model>_<mode>_cr<cr>.csv ; skips finished runs
run() { local model="$1" mode="$2" cr="$3" tag="$1_$2_cr$3"; mkdir -p "$OUT/$FIG"
  if [ -s "$OUT/$FIG/$tag.csv" ] && [ "$(grep -vc '^model,' "$OUT/$FIG/$tag.csv")" -ge 1 ]; then echo "[skip] $tag"; return; fi
  echo "[$(date +%H:%M:%S)] $FIG/$tag"
  "$PY" "$CODE_DIR/train_split.py" --model-name "$model" "${DATASET[@]}" --mode "$mode" --compression-ratio "$cr" \
      --cut "$(cut_for "$model")" --epochs 1 --batch-size "$(batch_for "$model")" \
      --max-train-blocks "$MTB" --max-val-blocks "$MVB" --seed 0 --summary-csv "$OUT/$FIG/$tag.csv"; }
