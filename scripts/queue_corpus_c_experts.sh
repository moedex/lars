#!/bin/zsh
# Attention-plus-experts LoRA on Qwen3-30B-A3B over corpus C, seed 0, then its suite with
# per-row dumps. 93 GB peak on the old corpus: refuses to start with under 100 GB reclaimable.
set -e
cd "$(dirname "$0")/.."
UV=${UV:-uv}
export LARS_MLX_CACHE_GB=${LARS_MLX_CACHE_GB:-16}
mkdir -p logs
M30=mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit
RECORDS=(data/train-c/open-jev.train.jsonl data/train-c/jev-bench.train.jsonl data/train-c/tasksource-jev.train.jsonl)
# The old corpus's seeded held-out sources that corpus C still has, named so that neither the
# seed nor the changed source list moves which configs the adapter trains on.
HOLDOUT=jev-bench/civil_comments,jev-bench/fever_evidence,jev-bench/helpsteer2_helpfulness,tasksource-jev/babi_nli/three-arg-relations
NAME=lora-30b-c-experts
step() { echo "=== $(date '+%H:%M:%S') $1"; }

page=$(sysctl -n hw.pagesize)
free_gb=$(vm_stat | awk -v p=$page '/Pages (free|inactive|speculative)/ {gsub("\\.","",$NF); s+=$NF} END {printf "%d", s*p/1e9}')
if (( free_gb < ${MIN_FREE_GB:-100} )); then
  step "STOP: about $free_gb GB reclaimable, under ${MIN_FREE_GB:-100} GB; free memory first"; exit 1
fi

step "$NAME: attention plus every expert LoRA, corpus C, seed 0, one epoch ($free_gb GB free)"
$UV run python -m lars.train.lora --model $M30 --keys attn+experts --records $RECORDS --limit 20000 --holdout-sources $HOLDOUT \
  --out checkpoints/$NAME > logs/$NAME.log 2>&1
if grep -q '"improved": false' checkpoints/$NAME/adapter_config.json; then
  step "STOP: no checkpoint beat the untrained 30B; the saved adapter is the identity"; exit 1
fi
step "suite-$NAME: adapter, per-config calibration, per-row dumps"
$UV run python evals/run_suite.py --backend mlx --model $M30 --adapter checkpoints/$NAME --tag $NAME-rows \
  --dump-rows > logs/suite-$NAME.log 2>&1
step "ALL DONE"
