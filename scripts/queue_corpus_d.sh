#!/bin/zsh
# Attention-only LoRA on Qwen3-30B-A3B over corpus D (corpus C with each licensed, trained
# jev-bench source filled to 1,000 rows), seed 0, choosing the checkpoint by macro accuracy on
# jev-bench validation rows, then its suite with per-row dumps. About 33 GB peak; roughly
# 3 h training + 45 min suite. Build the corpus first:
#   uv run --extra evals python scripts/build_corpus_d.py
# Compare with lora-30b-c-s0 (same seed, corpus C). Every evaluated step is kept in
# checkpoints/lora-30b-d-s0.steps/, and the history logs held-out Brier too, so the effect of
# the new selection rule can be read apart from the new data.
set -e
cd "$(dirname "$0")/.."
UV=${UV:-uv}
export MOELARS_MLX_CACHE_GB=${MOELARS_MLX_CACHE_GB:-16}
mkdir -p logs
M30=mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit
RECORDS=(data/train-d/open-jev.train.jsonl data/train-d/jev-bench.train.jsonl data/train-d/tasksource-jev.train.jsonl)
HOLDOUT=jev-bench/civil_comments,jev-bench/fever_evidence,jev-bench/helpsteer2_helpfulness,tasksource-jev/babi_nli/three-arg-relations
step() { echo "=== $(date '+%H:%M:%S') $1"; }

SEED=${SEED:-0}
NAME=lora-30b-d-s$SEED
step "$NAME: attention-only LoRA, corpus D, seed $SEED, one epoch, selection on validation macro"
$UV run python -m moelars.train.lora --model $M30 --keys attn --records $RECORDS --limit 40000 --seed $SEED \
  --holdout-sources $HOLDOUT --select-records data/train/jev-bench.validation.jsonl \
  --out checkpoints/$NAME > logs/$NAME.log 2>&1
if grep -q '"improved": false' checkpoints/$NAME/adapter_config.json; then
  step "STOP: no checkpoint beat the untrained 30B; the saved adapter is the identity"; exit 1
fi
step "suite-$NAME: adapter, per-config calibration, per-row dumps"
$UV run python evals/run_suite.py --backend mlx --model $M30 --adapter checkpoints/$NAME --tag $NAME-rows \
  --dump-rows > logs/suite-$NAME.log 2>&1
step "ALL DONE"
