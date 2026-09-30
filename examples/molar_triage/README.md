# Molar Triage

A dental practice's inbox, sorted by LARS. Sixty fictional patient messages, each
answered three ways:

| key | primitive | question |
|---|---|---|
| `in_pain` | noul | The sender says they are in pain right now. |
| `department` | choice | hygiene, restorative, orthodontics, surgery, billing, front_desk |
| `urgency` | score | routine, soon, this week, today, emergency |

All messages were written for this example. There are no real patients, practices, or
invoice numbers in it. The labels are one person's judgment, which is exactly the
situation you are in when you calibrate LARS on your own data: a few dozen rows you
labeled yourself, split in two.

Files:

- `build.py` holds the messages and labels and writes the two JSONL files.
- `molar_triage.calibration.jsonl`: 30 messages, 90 rows. Fit on these.
- `molar_triage.test.jsonl`: the other 30 messages, 90 rows. Report on these.
- `request.json`: one message as a full System One request with all three questions,
  evidence spans switched on, and a 15-point abstain margin.

## Walkthrough

Ask the model first, with nothing fitted:

```bash
uv run lars eval --backend mlx --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
    --data examples/molar_triage/molar_triage.test.jsonl
```

Fit a calibrator on the other half, then ask again:

```bash
uv run lars calibrate --backend mlx --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
    --data examples/molar_triage/molar_triage.calibration.jsonl --out calibration/molar_triage.json
uv run lars eval --backend mlx --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
    --calibration calibration/molar_triage.json --data examples/molar_triage/molar_triage.test.jsonl
```

What changes: `noul` rows get a Platt fit on the raw yes-minus-no logit, which can move
the decision boundary as well as the confidence. `choice` and `score` rows get a
temperature, which changes confidence only. Accuracy on those two can only move if you
also serve a Tier B head.

Serve with the calibrator and send the sample request:

```bash
uv run lars serve --backend mlx --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
    --calibration calibration/molar_triage.json
curl -s localhost:8600/v1/systemone -H 'content-type: application/json' \
    -d @examples/molar_triage/request.json | python -m json.tool
```

## Numbers

Filled in from a real run; see the table at the end of `evals/RESULTS.md` for the
same models on jev-bench.

<!-- MOLAR_NUMBERS -->
