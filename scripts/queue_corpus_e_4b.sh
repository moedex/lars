#!/bin/zsh
# LoRA (every linear layer, as the earlier 4B runs) on Qwen3-4B over corpus E (corpus D diluted with Open-Jev rows back to
# 40% jev-bench), seed 0, then its suite with per-row dumps. About 10 GB peak; roughly 6 h
# training + 45 min suite. Build the corpus first:
#   uv run --extra evals python scripts/build_corpus_e.py
# Selection weighs seen and unseen jev-bench validation sources equally (the held-out sources are
# the unseen ones). The license-dropped sources stay out of selection as well as training.
set -e
cd "$(dirname "$0")/.."
UV=${UV:-uv}
export LARS_MLX_CACHE_GB=${LARS_MLX_CACHE_GB:-16}
mkdir -p logs
MODEL=mlx-community/Qwen3-4B-Instruct-2507-4bit
RECORDS=(data/train-e/open-jev.train.jsonl data/train-e/jev-bench.train.jsonl data/train-e/tasksource-jev.train.jsonl)
HOLDOUT=jev-bench/civil_comments,jev-bench/fever_evidence,jev-bench/helpsteer2_helpfulness,tasksource-jev/babi_nli/three-arg-relations
LICENSE_DROPPED=jev-bench/sst5,jev-bench/yelp5,jev-bench/stsb
step() { echo "=== $(date '+%H:%M:%S') $1"; }

SEED=${SEED:-0}
NAME=lora-4b-e-s$SEED
step "$NAME: full LoRA on the 4B, corpus E, seed $SEED, one epoch, selection on seen + unseen macro"
$UV run python -m lars.train.lora --model $MODEL --records $RECORDS --limit 50000 --seed $SEED \
  --holdout-sources $HOLDOUT --select-records data/train/jev-bench.validation.jsonl --select-skip $LICENSE_DROPPED \
  --out checkpoints/$NAME > logs/$NAME.log 2>&1
if grep -q '"improved": false' checkpoints/$NAME/adapter_config.json; then
  step "STOP: no checkpoint beat the untrained 30B; the saved adapter is the identity"; exit 1
fi
step "suite-$NAME: adapter, per-config calibration, per-row dumps"
$UV run python evals/run_suite.py --backend mlx --model $MODEL --adapter checkpoints/$NAME --tag $NAME-rows \
  --dump-rows > logs/suite-$NAME.log 2>&1
step "ALL DONE"
