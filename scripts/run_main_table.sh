#!/bin/bash
# Main results (Table 1): accuracy across all model-benchmark pairs at budget N=4.
# Four baselines + UAB, averaged over three seeds.
set -e
cd "$(dirname "$0")/.."

MODELS=("qwen2.5-1.5b" "qwen2.5-7b" "llama3.2-3b" "gemma3-12b" "gptoss" "gemma3-27b")
DATA=("deepscaler" "gpqa" "hh_rlhf" "formal_logic" "math500")
BUDGET_MODES=("vote" "random" "length" "llm_judge" "uav_base")
SEEDS=(42 44 46)
N=4
TAU=0.2

get_data_size () {
  case "$1" in
    formal_logic) echo 0   ;;
    math500)      echo 500 ;;
    *)            echo 300 ;;
  esac
}

for data in "${DATA[@]}"; do
  SIZE=$(get_data_size "$data")
  for model in "${MODELS[@]}"; do
    for mode in "${BUDGET_MODES[@]}"; do
      for seed in "${SEEDS[@]}"; do
        python src/main.py \
          --model        "$model" \
          --data         "$data" \
          --data_size    "$SIZE" \
          --budget_mode  "$mode" \
          --num_agents   "$N" \
          --tau          "$TAU" \
          --score_fn     exp \
          --seed         "$seed"
      done
    done
  done
done
