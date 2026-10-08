#!/bin/bash
set -e
source "$(dirname "$0")/common.sh"

BUDGETS=(4 8 12 16)
MODES=("vote" "asc" "uav_base")

for model in "${MAIN_MODELS[@]}"; do
  for data in "${DATASETS[@]}"; do
    for N in "${BUDGETS[@]}"; do
      for mode in "${MODES[@]}"; do
        for seed in "${SEEDS[@]}"; do
          run_one "$model" "$data" "$mode" "$N" "$seed"
        done
      done
    done
  done
done
