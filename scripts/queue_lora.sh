#!/bin/zsh
# Serial GPU queue for LoRA plus head on the 4B, then the 30B-A3B Tier A run it was ordered
# after. Written 2026-09-22 while the GPU was lent out; start it once the GPU is free.
# Each step logs to logs/<step>.log; a failing step stops the queue.
set -e
cd "$(dirname "$0")/.."
UV=${UV:-~/Library/Python/3.9/bin/uv}
mkdir -p logs
M4=mlx-community/Qwen3-4B-Instruct-2507-4bit
M30=mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit
RECORDS=(data/train/open-jev.train.jsonl data/train/jev-bench.train.jsonl data/train/tasksource-jev.train.jsonl)
step() { echo "=== $(date '+%H:%M:%S') $1"; }

step "smoke-30b (strict restore check, bf16-scaled split check)"
$UV run python scripts/smoke_mlx_backend.py $M30 > logs/smoke-30b.log 2>&1

step "lora-probe: 400 records, speed and peak memory before the long run"
$UV run python -m lars.train.lora --model $M4 --records $RECORDS --limit 400 --eval-every 50 \
  --out checkpoints/lora-probe > logs/lora-probe.log 2>&1
step "lora-4b: full corpus, one epoch"
$UV run python -m lars.train.lora --model $M4 --records $RECORDS --limit 20000 \
  --out checkpoints/lora-4b > logs/lora-4b.log 2>&1
step "suite-4b-lora: adapter alone, per-config calibration"
$UV run python evals/run_suite.py --backend mlx --model $M4 --adapter checkpoints/lora-4b --tag lora \
  > logs/suite-4b-lora.log 2>&1
step "extract-4b-lora: features from the adapted model"
$UV run python -m lars.train.extract --model $M4 --adapter checkpoints/lora-4b --records $RECORDS \
  --test-records evals/data/*.test.jsonl --test-limit 1100 --limit 20000 --holdout-fraction 0.2 --max-options 160 \
  --out data/features-lora > logs/extract-4b-lora.log 2>&1
step "train head on the adapted features"
$UV run python -m lars.train.residual --train data/features-lora/train.npz --heldout data/features-lora/heldout.npz \
  --epochs 10 --out checkpoints/pointer_head_lora.npz > logs/train-4b-lora-head.log 2>&1
step "suite-4b-lora-head"
$UV run python evals/run_suite.py --backend mlx --model $M4 --adapter checkpoints/lora-4b \
  --head checkpoints/pointer_head_lora.npz --projection data/features-lora/projection.npy --tag lora-head \
  > logs/suite-4b-lora-head.log 2>&1

step "suite-30b"
$UV run python evals/run_suite.py --backend mlx --model $M30 > logs/suite-30b.log 2>&1
step "ALL DONE"
