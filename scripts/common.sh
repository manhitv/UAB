cd "$(dirname "${BASH_SOURCE[0]}")/.."

SEEDS=(42 44 46)
MAIN_MODELS=("qwen2.5-1.5b" "qwen2.5-7b" "llama3.2-3b")
BIG_MODELS=("gptoss" "gemma3-27b")
DATASETS=("math500" "formal_logic" "deepscaler" "gsm8k" "prog_expr")
ASC_THRESH=${ASC_THRESH:-0.95}

get_data_size () {
  case "$1" in
    formal_logic) echo 0   ;;
    math500)      echo 500 ;;
    prog_expr)    echo 100 ;;
    *)            echo 300 ;;
  esac
}

run_one () {
  local model=$1 data=$2 mode=$3 N=$4 seed=$5
  shift 5
  python src/main.py --model "$model" --data "$data" --data_size "$(get_data_size "$data")" \
    --budget_mode "$mode" --num_agents "$N" --asc_thresh "$ASC_THRESH" --seed "$seed" "$@"
}
