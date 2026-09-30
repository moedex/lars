#!/bin/zsh
# Full attention-only LoRA on Qwen3-30B-A3B, its suite with per-row dumps, and the routing
# simulation with the adapted 4B as primary and the adapted 30B as fallback.
# Needs data/train/*.jsonl, evals/data/*.jsonl, and the committed 4B row dumps; see HANDOFF.md.
# Each step logs to logs/<step>.log; a failing step stops the queue.
set -e
cd "$(dirname "$0")/.."
UV=${UV:-uv}
export LARS_MLX_CACHE_GB=${LARS_MLX_CACHE_GB:-16}  # the 4 GB serving default can slow training
mkdir -p logs
M30=mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit
RECORDS=(data/train/open-jev.train.jsonl data/train/jev-bench.train.jsonl data/train/tasksource-jev.train.jsonl)
step() { echo "=== $(date '+%H:%M:%S') $1"; }

step "lora-30b: attention-only LoRA, full corpus, one epoch"
$UV run python -m lars.train.lora --model $M30 --keys attn --records $RECORDS --limit 20000 \
  --out checkpoints/lora-30b > logs/lora-30b.log 2>&1
if grep -q '"improved": false' checkpoints/lora-30b/adapter_config.json; then
  step "STOP: no checkpoint beat the untrained 30B; the saved adapter is the identity"; exit 1
fi
step "suite-30b-lora: adapter, per-config calibration, per-row dumps"
$UV run python evals/run_suite.py --backend mlx --model $M30 --adapter checkpoints/lora-30b --tag lora-rows \
  --dump-rows > logs/suite-30b-lora.log 2>&1
step "cascade: 4B+LoRA primary, 30B+LoRA fallback"
$UV run python evals/cascade.py evals/results/rows/lora-4b-fused-bf16-rows \
  evals/results/rows/qwen3-30b-a3b-instruct-2507-4bit-lora-rows \
  --suite evals/results/lora-4b-fused-bf16-rows.json > logs/cascade-30b-lora.log 2>&1
step "ALL DONE"
