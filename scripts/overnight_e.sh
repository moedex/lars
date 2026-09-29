#!/bin/zsh
# Unattended follow-up to the corpus-E seed-1 run, needing no model and no network: run it detached
# (its own session, so it outlives the terminal or agent that started it) under caffeinate:
#
#   python3 -c 'import os,sys; os.setsid(); os.execvp("caffeinate", ["caffeinate","-i","-s","zsh",sys.argv[1]])' \
#       scripts/overnight_e.sh > logs/overnight.log 2>&1 &
#
# Steps, each waiting for the previous to finish rather than for a clock, each with a timeout:
#   1. wait for the seed-1 queue (training + suite) to end;
#   2. seed comparison, seed average and seen/unseen split, by paired bootstrap on the row dumps;
#   3. the two seeds served live as an ensemble (must match the simulated average), and its latency;
#   4. the 4B tier on corpus E (scripts/queue_corpus_e_4b.sh), then its bootstrap against the 30B.
# A failing step is logged and the next one still runs. Everything lands in logs/ and
# logs/overnight-summary.md; nothing is committed or pushed.
set -u
cd "$(dirname "$0")/.."
UV=${UV:-uv}
M30=mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit
P=qwen3-30b-a3b-instruct-2507-4bit
R=evals/results/rows/$P-lora-30b
S=logs/overnight-summary.md
step() { echo "=== $(date '+%F %T') $1"; }
wait_for() {  # wait_for <seconds> <description> <shell test>
  local deadline=$(( $(date +%s) + $1 ))
  until eval "$3"; do
    if (( $(date +%s) > deadline )); then step "TIMEOUT waiting for $2"; return 1; fi
    sleep 60
  done
}
busy='pgrep -f "moelars.train.lora|evals/run_suite.py|evals/ensemble_check.py" >/dev/null'

step "waiting for the corpus-E seed-1 queue"
wait_for 43200 "seed 1" 'grep -q "ALL DONE\|STOP" logs/queue-corpus-e-s1.log' || exit 1
wait_for 3600 "the GPU to be free" "! $busy"

step "seed comparison and seed average (bootstrap)"
{
  echo "# Overnight results, corpus E ($(date '+%F %T'))"
  echo; echo "## Seed 1 suite"; sed -n '/Macro/,$p' evals/results/$P-lora-30b-e-s1-rows.md
  echo; echo "## Paired bootstrap, test rows"; echo
  $UV run python evals/bootstrap.py $R-c-s0-rows $R-e-s0-rows $R-e-s1-rows $R-e-s0-rows+$R-e-s1-rows
  echo; echo "## Seed 1 against seed 0"; echo
  $UV run python evals/bootstrap.py $R-e-s0-rows $R-e-s1-rows $R-e-s0-rows+$R-e-s1-rows
  echo; echo "## Seen and unseen configs, against corpus C seed 0"; echo
  $UV run python - <<'EOF'
import sys, numpy as np
sys.path.insert(0, "evals")
from bootstrap import outcomes
R = "evals/results/rows/qwen3-30b-a3b-instruct-2507-4bit-lora-30b-"
base = outcomes(R + "c-s0-rows")
unseen = ["sst5", "chaosnli", "yelp5", "stsb", "fever_evidence", "helpsteer2_helpfulness", "civil_comments"]
seen = [c for c in base if c not in unseen]
rng = np.random.default_rng(0)
for name in ("e-s0-rows", "e-s1-rows", "e-s0-rows+" + R + "e-s1-rows"):
    other = outcomes(R + name)
    parts = []
    for label, cfgs in (("seen", seen), ("unseen", unseen)):
        point = np.mean([other[c][0].mean() - base[c][0].mean() for c in cfgs])
        draws = []
        for _ in range(10000):
            m = []
            for c in cfgs:
                i = rng.integers(0, len(base[c][0]), len(base[c][0]))
                m.append(other[c][0][i].mean() - base[c][0][i].mean())
            draws.append(np.mean(m))
        lo, hi = np.percentile(draws, [2.5, 97.5])
        parts.append(f"{label} {point * 100:+.1f} ({lo * 100:+.1f} to {hi * 100:+.1f})")
    print(f"- {name.replace(R, '')}: " + "; ".join(parts))
EOF
} >> $S 2>> logs/overnight.err

step "seed average served live"
$UV run python evals/ensemble_check.py --model $M30 --adapter checkpoints/lora-30b-e-s0 --slug $P-lora-30b-e-s0-rows \
  --adapter checkpoints/lora-30b-e-s1 --slug $P-lora-30b-e-s1-rows --out evals/results/ensemble-check-e-s0-s1.json \
  > logs/ensemble-check-e-s0-s1.log 2>&1
{ echo; echo "## Seed average served live"; tail -1 logs/ensemble-check-e-s0-s1.log; } >> $S

step "latency, one and two adapters"
$UV run python scripts/load_cost.py $M30@checkpoints/lora-30b-e-s0 \
  $M30@checkpoints/lora-30b-e-s0,checkpoints/lora-30b-e-s1 > logs/load-cost-e.log 2>&1
{ echo; echo "## Load cost"; echo; grep "|" logs/load-cost-e.log; } >> $S

step "4B tier on corpus E"
wait_for 3600 "the GPU to be free" "! $busy"
zsh scripts/queue_corpus_e_4b.sh > logs/queue-corpus-e-4b.log 2>&1
P4=qwen3-4b-instruct-2507-4bit
{
  echo; echo "## 4B tier"; cat logs/queue-corpus-e-4b.log
  sed -n '/Macro/,$p' evals/results/$P4-lora-4b-e-s0-rows.md 2>/dev/null
  echo; $UV run python evals/bootstrap.py $R-e-s0-rows evals/results/rows/$P4-lora-4b-e-s0-rows 2>&1 | tail -3
} >> $S
step "ALL DONE"
