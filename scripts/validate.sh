#!/bin/bash
# Quick end-to-end validation of the UAB pipeline with Qwen2.5-1.5B.
# Uses --debug (small slice) so it finishes in a few minutes.
set -e
cd "$(dirname "$0")/.."

echo "=== [1/4] Uniform baseline (vote) ==="
python src/main.py --model qwen2.5-1.5b --data formal_logic --data_size 0 \
  --budget_mode vote --num_agents 4 --seed 42 --debug

echo "=== [2/4] UAB on Formal Logic ==="
python src/main.py --model qwen2.5-1.5b --data formal_logic --data_size 0 \
  --budget_mode uav_base --num_agents 4 --tau 0.2 --seed 42 --debug

echo "=== [3/4] UAB on MATH-500 ==="
python src/main.py --model qwen2.5-1.5b --data math500 --data_size 500 \
  --budget_mode uav_base --num_agents 4 --tau 0.2 --seed 42 --debug

echo "=== [4/4] Threshold-exit analysis tables ==="
python src/main.py --analyze --tau 0.2

echo "=== Validation complete — see out/ for results ==="
