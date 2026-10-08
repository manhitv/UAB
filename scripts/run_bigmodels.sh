#!/bin/bash
set -e
source "$(dirname "$0")/common.sh"

N=4
MODES=("vote" "random" "length" "llm_judge" "asc" "uav_base")

for data in "${DATASETS[@]}"; do
  for model in "${BIG_MODELS[@]}"; do
    for mode in "${MODES[@]}"; do
      for seed in "${SEEDS[@]}"; do
        run_one "$model" "$data" "$mode" "$N" "$seed"
      done
    done
  done
done
