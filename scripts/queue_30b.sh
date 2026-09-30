#!/bin/zsh
# Serial GPU queue that replaced the tail of next_batch.sh on 2026-09-22: the 9B extraction,
# head, and head suite were cut (9B Tier A was not beating 4B + head v2 at 2-3x the cost), and
# a mixture-of-experts backbone, Qwen3-30B-A3B (about 3B active parameters), took their place.
# Each step logs to logs/<step>.log; a failing step stops the queue.
set -e
cd "$(dirname "$0")/.."
UV=${UV:-~/Library/Python/3.9/bin/uv}
mkdir -p logs
M4=mlx-community/Qwen3-4B-Instruct-2507-4bit
M9=mlx-community/Qwen3.5-9B-MLX-4bit
M30=mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit
step() { echo "=== $(date '+%H:%M:%S') $1"; }

step "smoke-30b: prefix cache and feature path on the MoE model"
$UV run python scripts/smoke_mlx_backend.py $M30 > logs/smoke-30b.log 2>&1
step "suite-30b"
$UV run python evals/run_suite.py --backend mlx --model $M30 > logs/suite-30b.log 2>&1

step "molar triage walkthrough numbers"
EX=examples/molar_triage
for spec in "qwen3-4b|$M4|checkpoints/pointer_head_v2.npz|data/features-full/projection.npy" \
            "qwen3.5-9b|$M9||" \
            "qwen3-30b-a3b|$M30||"; do
  IFS='|' read -r slug model head proj <<< "$spec"
  $UV run lars eval --backend mlx --model $model --data $EX/molar_triage.test.jsonl > logs/molar-$slug-raw.json
  $UV run lars calibrate --backend mlx --model $model --data $EX/molar_triage.calibration.jsonl \
    --out calibration/molar_triage.$slug.json > logs/molar-$slug-cal.log 2>&1
  $UV run lars eval --backend mlx --model $model --calibration calibration/molar_triage.$slug.json \
    --data $EX/molar_triage.test.jsonl > logs/molar-$slug-calibrated.json
  [[ -z $head ]] && continue
  $UV run lars calibrate --backend mlx --model $model --head $head --projection $proj \
    --data $EX/molar_triage.calibration.jsonl --out calibration/molar_triage.$slug-head.json > logs/molar-$slug-head-cal.log 2>&1
  $UV run lars eval --backend mlx --model $model --head $head --projection $proj \
    --calibration calibration/molar_triage.$slug-head.json --data $EX/molar_triage.test.jsonl > logs/molar-$slug-head.json
done

step "load cost: load time, peak memory, warm 3-question request latency"
$UV run python scripts/load_cost.py $M4 $M9 $M30 > logs/load-cost.log 2>&1
step "ALL DONE"
