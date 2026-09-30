#!/bin/zsh
# Attention-only LoRA on Qwen3-30B-A3B over the commercially licensed corpus (data/train-c,
# DESIGN.md section 8.1), seeds 0 and 1, each followed by its suite with per-row dumps.
# About 33 GB peak and 2 x (2.3 h training + 1.6 h suite). Build the corpus first:
#   uv run --extra evals python scripts/build_corpus_c.py
# Each step logs to logs/<step>.log; a failing step stops the queue.
set -e
cd "$(dirname "$0")/.."
UV=${UV:-uv}
export LARS_MLX_CACHE_GB=${LARS_MLX_CACHE_GB:-16}  # the 4 GB serving default can slow training
mkdir -p logs
M30=mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit
RECORDS=(data/train-c/open-jev.train.jsonl data/train-c/jev-bench.train.jsonl data/train-c/tasksource-jev.train.jsonl)
# The old corpus's seeded held-out sources that corpus C still has, named so that neither the
# seed nor the changed source list moves which configs the adapter trains on.
HOLDOUT=jev-bench/civil_comments,jev-bench/fever_evidence,jev-bench/helpsteer2_helpfulness,tasksource-jev/babi_nli/three-arg-relations
step() { echo "=== $(date '+%H:%M:%S') $1"; }

for SEED in 0 1; do
  NAME=lora-30b-c-s$SEED
  step "$NAME: attention-only LoRA, corpus C, seed $SEED, one epoch"
  $UV run python -m lars.train.lora --model $M30 --keys attn --records $RECORDS --limit 20000 --seed $SEED --holdout-sources $HOLDOUT \
    --out checkpoints/$NAME > logs/$NAME.log 2>&1
  if grep -q '"improved": false' checkpoints/$NAME/adapter_config.json; then
    step "STOP: no checkpoint beat the untrained 30B; the saved adapter is the identity"; exit 1
  fi
  step "suite-$NAME: adapter, per-config calibration, per-row dumps"
  $UV run python evals/run_suite.py --backend mlx --model $M30 --adapter checkpoints/$NAME --tag $NAME-rows \
    --dump-rows > logs/suite-$NAME.log 2>&1
done
step "ALL DONE"
