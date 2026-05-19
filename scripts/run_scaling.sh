#!/bin/bash
# Budget scaling and cost-at-fixed-accuracy experiments.
#   Block 1 — scaling figure:        N in {2, 4, 8}
#   Block 2 — cost-at-accuracy curve: N in {1, 2, 4, 6, 8, 12, 16}
# UAB vs. the Uniform (vote) baseline.
set -e
cd "$(dirname "$0")/.."

MODELS=("qwen2.5-1.5b" "qwen2.5-7b" "llama3.2-3b")
DATA=("deepscaler" "hh_rlhf" "formal_logic" "math500")
SEEDS=(42 44 46)

get_data_size () {
  case "$1" in
    formal_logic) echo 0   ;;
    math500)      echo 500 ;;
    *)            echo 300 ;;
  esac
}

TAU=0.2

run () {  # run <model> <data> <N>
  local model=$1 data=$2 N=$3
  local SIZE TAU
  SIZE=$(get_data_size "$data")
  for seed in "${SEEDS[@]}"; do
    python src/main.py --model "$model" --data "$data" --data_size "$SIZE" \
      --budget_mode uav_base --num_agents "$N" --tau "$TAU" --score_fn exp --seed "$seed"
    python src/main.py --model "$model" --data "$data" --data_size "$SIZE" \
      --budget_mode vote --num_agents "$N" --seed "$seed"
  done
}

echo "=== Block 1: budget scaling N in {2,4,8} ==="
for model in "${MODELS[@]}"; do
  for data in "${DATA[@]}"; do
    for N in 2 4 8; do
      run "$model" "$data" "$N"
    done
  done
done

echo "=== Block 2: cost-at-accuracy N in {1,2,4,6,8,12,16} (Qwen2.5-1.5B, Llama3.2-3B) ==="
for model in "qwen2.5-1.5b" "llama3.2-3b"; do
  for data in "math500" "formal_logic"; do
    for N in 1 6 12 16; do   # 2, 4 and 8 already covered by Block 1
      run "$model" "$data" "$N"
    done
  done
done
