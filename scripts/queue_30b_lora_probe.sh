#!/bin/zsh
# Two 400-record LoRA probes on Qwen3-30B-A3B before committing to a full run: attention only
# (a few million parameters) and attention plus every expert. Each log reports trainable
# parameters, steps per minute, peak memory, and held-out metrics every 50 steps.
set -e
cd "$(dirname "$0")/.."
UV=${UV:-~/Library/Python/3.9/bin/uv}
mkdir -p logs
M30=mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit
RECORDS=(data/train/open-jev.train.jsonl data/train/jev-bench.train.jsonl data/train/tasksource-jev.train.jsonl)
step() { echo "=== $(date '+%H:%M:%S') $1"; }

for keys in attn attn+experts; do
  tag=${keys/+/-}
  step "lora-30b-probe-$tag"
  $UV run python -m lars.train.lora --model $M30 --keys $keys --records $RECORDS --limit 400 --eval-every 50 \
    --out checkpoints/lora-30b-probe-$tag > logs/lora-30b-probe-$tag.log 2>&1 || echo "probe $tag failed; see its log"
done
step "ALL DONE"
