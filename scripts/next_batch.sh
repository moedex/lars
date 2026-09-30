#!/bin/zsh
# Serial GPU job queue for the next batch, exactly as queued and then cancelled on 2026-09-22.
# Each step logs to logs/<step>.log; a failing step stops the queue. Expect four to five hours on an M5 Max.
set -e
cd "$(dirname "$0")/.."
UV=${UV:-~/Library/Python/3.9/bin/uv}
mkdir -p logs
M4=mlx-community/Qwen3-4B-Instruct-2507-4bit
M9=mlx-community/Qwen3.5-9B-MLX-4bit
RECORDS=(data/train/open-jev.train.jsonl data/train/jev-bench.train.jsonl data/train/tasksource-jev.train.jsonl)
step() { echo "=== $(date '+%H:%M:%S') $1"; }

step "extract-4b-full (~40 min)"
$UV run python -m lars.train.extract --model $M4 --records $RECORDS \
  --test-records evals/data/*.test.jsonl --test-limit 1100 --limit 20000 --holdout-fraction 0.2 --max-options 160 \
  --out data/features-full > logs/extract-4b-full.log 2>&1
step "train-4b-v2 (seconds)"
$UV run python -m lars.train.residual --train data/features-full/train.npz --heldout data/features-full/heldout.npz \
  --epochs 10 --out checkpoints/pointer_head_v2.npz > logs/train-4b-v2.log 2>&1
$UV run python -m lars.train.report --shard data/features-full/test.npz --head checkpoints/pointer_head_v2.npz \
  --sources data/features-full/sources.json --out logs/report-4b-v2.json > logs/report-4b-v2.log 2>&1
step "suite-4b-head-v2 (~30 min)"
$UV run python evals/run_suite.py --backend mlx --model $M4 --head checkpoints/pointer_head_v2.npz \
  --projection data/features-full/projection.npy --tag head-v2 > logs/suite-4b-head-v2.log 2>&1

step "smoke-9b: prefix cache and feature path on the hybrid-attention model"
$UV run python scripts/smoke_mlx_backend.py $M9 > logs/smoke-9b.log 2>&1
step "suite-9b (~60 min)"
$UV run python evals/run_suite.py --backend mlx --model $M9 > logs/suite-9b.log 2>&1
step "extract-9b (~80 min)"
$UV run python -m lars.train.extract --model $M9 --records $RECORDS \
  --test-records evals/data/*.test.jsonl --test-limit 1100 --limit 20000 --holdout-fraction 0.2 --max-options 160 \
  --out data/features-9b > logs/extract-9b.log 2>&1
step "train-9b"
$UV run python -m lars.train.residual --train data/features-9b/train.npz --heldout data/features-9b/heldout.npz \
  --epochs 10 --out checkpoints/pointer_head_9b.npz > logs/train-9b.log 2>&1
$UV run python -m lars.train.report --shard data/features-9b/test.npz --head checkpoints/pointer_head_9b.npz \
  --sources data/features-9b/sources.json --out logs/report-9b.json > logs/report-9b.log 2>&1
step "suite-9b-head (~60 min)"
$UV run python evals/run_suite.py --backend mlx --model $M9 --head checkpoints/pointer_head_9b.npz \
  --projection data/features-9b/projection.npy --tag head > logs/suite-9b-head.log 2>&1

step "molar triage walkthrough numbers"
EX=examples/molar_triage
for spec in "qwen3-4b|$M4|checkpoints/pointer_head_v2.npz|data/features-full/projection.npy" \
            "qwen3.5-9b|$M9|checkpoints/pointer_head_9b.npz|data/features-9b/projection.npy"; do
  IFS='|' read -r slug model head proj <<< "$spec"
  $UV run lars eval --backend mlx --model $model --data $EX/molar_triage.test.jsonl > logs/molar-$slug-raw.json
  $UV run lars calibrate --backend mlx --model $model --data $EX/molar_triage.calibration.jsonl \
    --out calibration/molar_triage.$slug.json > logs/molar-$slug-cal.log 2>&1
  $UV run lars eval --backend mlx --model $model --calibration calibration/molar_triage.$slug.json \
    --data $EX/molar_triage.test.jsonl > logs/molar-$slug-calibrated.json
  $UV run lars calibrate --backend mlx --model $model --head $head --projection $proj \
    --data $EX/molar_triage.calibration.jsonl --out calibration/molar_triage.$slug-head.json > logs/molar-$slug-head-cal.log 2>&1
  $UV run lars eval --backend mlx --model $model --head $head --projection $proj \
    --calibration calibration/molar_triage.$slug-head.json --data $EX/molar_triage.test.jsonl > logs/molar-$slug-head.json
done

step "load cost: load time, peak memory, warm 3-question request latency"
$UV run python scripts/load_cost.py $M4 $M9 > logs/load-cost.log 2>&1
step "ALL DONE"
