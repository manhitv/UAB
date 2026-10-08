#!/bin/bash
set -e
source "$(dirname "$0")/common.sh"

ABLATION_MODEL="qwen2.5-1.5b"

for data in "${DATASETS[@]}"; do
  for tau in 0.1 0.2 0.5 1 2; do
    for seed in "${SEEDS[@]}"; do
      run_one "$ABLATION_MODEL" "$data" uav_base 4 "$seed" --tau "$tau"
    done
  done
done

for data in "${DATASETS[@]}"; do
  for N in 4 8 12 16; do
    for K in 2 3 4 5 6 7 8; do
      [ "$K" -le "$N" ] || continue
      for seed in "${SEEDS[@]}"; do
        run_one "$ABLATION_MODEL" "$data" uav_base "$N" "$seed" --phase1_samples "$K"
      done
    done
  done
done

for model in "${MAIN_MODELS[@]}"; do
  for data in "${DATASETS[@]}"; do
    for metric in anll nll token_var min_token_nll; do
      for seed in "${SEEDS[@]}"; do
        run_one "$model" "$data" uav_base 4 "$seed" --uncertainty_mode "$metric" --phase1_samples 1
      done
    done
    for seed in "${SEEDS[@]}"; do
      run_one "$model" "$data" uav_base 4 "$seed" --no_phase1_vote
    done
  done
done
