#!/bin/bash
# Ablation studies:
#   Block 1 — temperature tau
#   Block 2 — uncertainty / difficulty metric
#   Block 3 — hard-threshold exit
#   Block 4 — easy-threshold exit
#   Block 5 — Phase-1 vote contribution
#   Block 6 — verbalized confidence (VCS) vs. ANLL
set -e
cd "$(dirname "$0")/.."

MODELS=("qwen2.5-1.5b" "llama3.2-3b")
SEEDS=(42 44 46)

get_data_size () {
  case "$1" in
    formal_logic) echo 0   ;;
    math500)      echo 500 ;;
    *)            echo 300 ;;
  esac
}

TAU=0.2

echo "=== Block 1: temperature tau in {0.1,0.2,0.5,1,2} ==="
for model in "${MODELS[@]}"; do
  for data in "formal_logic" "math500" "hh_rlhf" "deepscaler"; do
    SIZE=$(get_data_size "$data")
    for tau in 0.1 0.2 0.5 1 2; do
      for seed in "${SEEDS[@]}"; do
        python src/main.py --model "$model" --data "$data" --data_size "$SIZE" \
          --budget_mode uav_base --num_agents 4 --tau "$tau" --score_fn exp --seed "$seed"
      done
    done
  done
done

echo "=== Block 2: uncertainty / difficulty metric ==="
for model in "${MODELS[@]}"; do
  for data in "formal_logic" "math500"; do
    SIZE=$(get_data_size "$data")
    for metric in anll nll token_var min_token_nll; do
      for seed in "${SEEDS[@]}"; do
        python src/main.py --model "$model" --data "$data" --data_size "$SIZE" \
          --budget_mode uav_base --num_agents 4 --tau "$TAU" --score_fn exp \
          --diff_metric "$metric" --seed "$seed"
      done
    done
    # closed-form allocation as an additional comparison point
    for seed in "${SEEDS[@]}"; do
      python src/main.py --model "$model" --data "$data" --data_size "$SIZE" \
        --budget_mode closed_form --num_agents 4 --tau "$TAU" --score_fn exp --seed "$seed"
    done
  done
done

echo "=== Block 3: hard-threshold exit (theta_hard in {0.3,0.5,0.7}) ==="
for model in "${MODELS[@]}"; do
  for data in "formal_logic" "math500" "hh_rlhf" "deepscaler"; do
    SIZE=$(get_data_size "$data")
    for hard_mode in redistribute skip; do
      for theta in 0.3 0.5 0.7; do
        for seed in "${SEEDS[@]}"; do
          python src/main.py --model "$model" --data "$data" --data_size "$SIZE" \
            --budget_mode uav_base --num_agents 4 --tau "$TAU" --score_fn exp \
            --hard_mode "$hard_mode" --theta_hard "$theta" --n_max 8 --seed "$seed"
        done
      done
    done
  done
done

echo "=== Block 4: easy-threshold exit (theta_easy in {0.3,0.5,0.7}) ==="
for model in "${MODELS[@]}"; do
  for data in "formal_logic" "math500" "hh_rlhf" "deepscaler"; do
    SIZE=$(get_data_size "$data")
    for easy_mode in redistribute skip; do
      for theta in 0.3 0.5 0.7; do
        for seed in "${SEEDS[@]}"; do
          python src/main.py --model "$model" --data "$data" --data_size "$SIZE" \
            --budget_mode uav_base --num_agents 4 --tau "$TAU" --score_fn exp \
            --easy_mode "$easy_mode" --theta_easy "$theta" --seed "$seed"
        done
      done
    done
  done
done

echo "=== Block 5: Phase-1 vote contribution ablation ==="
for model in "${MODELS[@]}"; do
  for data in "formal_logic" "math500"; do
    SIZE=$(get_data_size "$data")
    for seed in "${SEEDS[@]}"; do
      python src/main.py --model "$model" --data "$data" --data_size "$SIZE" \
        --budget_mode uav_base --num_agents 4 --tau "$TAU" --score_fn exp \
        --no_phase1_vote --seed "$seed"
    done
  done
done

echo "=== Block 6: verbalized confidence (VCS) ==="
for model in "${MODELS[@]}"; do
  for data in "formal_logic" "math500"; do
    SIZE=$(get_data_size "$data")
    for seed in "${SEEDS[@]}"; do
      python src/main.py --model "$model" --data "$data" --data_size "$SIZE" \
        --budget_mode uav_base --num_agents 4 --tau "$TAU" --score_fn exp \
        --uncertainty_mode vcs --seed "$seed"
    done
  done
done

echo "=== All ablations complete ==="
