#!/bin/bash
set -e
source "$(dirname "$0")/common.sh"

for mode in vote asc uav_base; do
  run_one qwen2.5-1.5b formal_logic "$mode" 4 42 --debug
done
run_one qwen2.5-1.5b math500 uav_base 4 42 --debug
