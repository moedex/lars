# Evals

Every change to rendering, backends, or calibration should move a number here, not a vibe.

## Data sources

Use only these. The reasoning is in `DESIGN.md` under "What we will not do".

| Source | Use | License |
|---|---|---|
| `Praveenrajus/jev-bench` | Primary benchmark. 22 configs, human vote distributions on 4 of them. Rows are already in wire format. | mixed, see its manifest |
| `ZefanCai/Open-Jev` | Training and OOD evaluation. Synthetic and controlled. | CC0 |
| `tasksource/tasksource-jev` | Broad choice coverage recast from 575 public tasks. Filter by `source` to avoid benchmark contamination. | per upstream |

Do not use any corpus whose labels were produced by Jev, including the `yuri_v3` stream of `SargeDev/jev-distill-corpus-v3`.

## JSONL format

One object per line:

```json
{"state": "...", "question": {"type": "choice", "instructions": "...", "criteria": {"a": "...", "b": "..."}}, "label": "a"}
```

`state` and `question` may also be JSON-encoded strings. `soft_label` is an optional map from option key to human probability.

## Commands

```bash
uv run python evals/fetch_jevbench.py --config banking77 --split test --out evals/data/banking77.test.jsonl
uv run lars eval --backend mock --data evals/data/banking77.test.jsonl
uv run lars calibrate --backend mlx --model <path> --data evals/data/banking77.validation.jsonl --out calibration/banking77.json
uv run lars eval --backend mlx --model <path> --calibration calibration/banking77.json --data evals/data/banking77.test.jsonl
```

Reported: accuracy, ECE, Brier, and coverage at a 5% error budget with the threshold that achieves it.

A tiny sample set lives in `evals/samples/` so the harness runs without downloads.
