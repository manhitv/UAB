#!/bin/bash
#SBATCH --job-name=u.m0
#SBATCH --qos=batch-short
#SBATCH --output=log_output/log_%A_%a.out
#SBATCH --error=log_error/log_%A_%a.err
#SBATCH --nodes=1 # Number of nodes required
#SBATCH --gpus=1 # Number of GPUs required
#SBATCH --gpus-per-node=1 # Number of GPU per node
#SBATCH --ntasks-per-node=1 # Number of tasks per node
#SBATCH --cpus-per-task=1 # Number of CPUs per task
#SBATCH --mem=40G 
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu-large
#SBATCH --sockets-per-node=1 # Number of sockets per node
#SBATCH --cores-per-socket=8 # Number of cores per socket

module load Anaconda3
source activate
conda activate uab

set -e
# cd "$(dirname "$0")/.."

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

python src/main.py --model gptoss --data formal_logic --data_size 0 \
  --budget_mode uav_base --num_agents 4 --tau 0.5 --seed 42 --debug

echo "=== Validation complete — see out/ for results ==="