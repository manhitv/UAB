#!/bin/bash
# Black-box & big-model setting.
#   Block 1 — large open-weight models (GPT-OSS-20B, Gemma3-27B), local backend.
#   Block 2 — black-box Cohere API (Command-A), single seed to limit API cost.
# Random/Length are omitted here (weaker than LLM-Judge/Uniform) to save compute.
set -e
cd "$(dirname "$0")/.."

DATA=("deepscaler" "gpqa" "hh_rlhf" "formal_logic" "math500")

get_data_size () {
  case "$1" in
    formal_logic) echo 0   ;;
    math500)      echo 500 ;;
    *)            echo 300 ;;
  esac
}

echo "=== Block 1: large open-weight models (local backend) ==="
for model in "gptoss" "gemma3-27b"; do
  for data in "${DATA[@]}"; do
    SIZE=$(get_data_size "$data")
    for mode in "vote" "llm_judge" "uav_base"; do
      for seed in 42 44 46; do
        python src/main.py --model "$model" --data "$data" --data_size "$SIZE" \
          --budget_mode "$mode" --num_agents 4 --tau 0.2 --score_fn exp --seed "$seed"
      done
    done
  done
done

echo "=== Block 2: black-box Cohere API ==="
# Requires:  export COHERE_API_KEY="your_key"   (or pass --cohere_api_key)
if [[ -z "$COHERE_API_KEY" ]]; then
  echo "WARNING: COHERE_API_KEY is not set — Block 2 will fail. Export it first."
fi
for data in "${DATA[@]}"; do
  SIZE=$(get_data_size "$data")
  for mode in "vote" "llm_judge" "uav"; do
    python src/main.py --backend cohere --model command-a-03-2025 \
      --data "$data" --data_size "$SIZE" \
      --budget_mode "$mode" --num_agents 4 --tau 0.2 --seed 42
  done
done
