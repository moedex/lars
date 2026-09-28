# moe-LARS

**Moe Limited but Accurate Response System.** A local-models-only typed-decision engine.

State plus typed questions go in. Calibrated probability distributions come out, in one
forward pass per question, with no text generation. The answer space is limited to the
options you declare, which is what makes calibration and zero structural errors possible.

moe-LARS is wire-compatible with the System One HTTP API. Point any existing System One
client at it by changing the base URL.

```
POST /v1/systemone
GET  /v1/models
```

Three primitives from the System One shape, one moe-LARS addition:

| primitive | question | answer |
|---|---|---|
| `noul` | is this true? | `noul`: P(yes) |
| `choice` | which one of these? | `choice`, `probabilities`, `confidence` |
| `score` | which ordered level? | `score` (may be fractional), `legend`, `probabilities`, `confidence` |
| `multi` | which of these apply? | `probabilities` per option, `selected` |

moe-LARS extensions, all opt-in under a `moelars` request key: permutation-averaged choice
answers with an `order_sensitivity` metric, declared constraints between nouls,
abstention below a probability margin, evidence spans by leave-one-out ablation, and evidence fusion: named numeric features per
noul (`moelars.features`), combined with the model's logit by a logistic fitted in `calibrate`.

## Quickstart

```bash
uv sync --extra dev
uv run moelars serve --backend mock            # no model needed, deterministic demo answers
```

```bash
curl -s localhost:8600/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Help! My payouts have been failing for 3 days.",
  "questions": {
    "department": {"type": "choice", "instructions": "Which team should handle this?",
                   "criteria": {"billing": "Payments, payouts", "technical": "Bugs, outages"}},
    "is_urgent":  {"type": "noul", "instructions": "The message conveys urgency"}
  }
}'
```

With a real model on Apple Silicon:

```bash
uv sync --extra dev --extra mlx
uv run moelars serve --backend mlx --model mlx-community/Qwen3.5-4B-Instruct-4bit
```

The MLX backend caps MLX's buffer cache at 4 GB (`MOELARS_MLX_CACHE_GB` changes it). Uncapped, varied prompt lengths grew it to about 100 GB within a few hundred requests.

Or any GGUF model anywhere:

```bash
uv sync --extra dev --extra llamacpp
uv run moelars serve --backend llamacpp --model ./models/qwen3.5-4b-instruct-q4_k_m.gguf --template chatml
```

In Docker (Linux has no Metal, so the container serves the mock backend or GGUF models on
CPU through llama.cpp; use the MLX backend natively on Apple Silicon):

```bash
docker build -t moelars .                                        # mock backend
docker run --rm -p 8600:8600 moelars
docker build -t moelars:llamacpp --build-arg EXTRAS=llamacpp .   # compiles llama.cpp for CPU
docker run --rm -p 8600:8600 -v "$PWD/models:/models:ro" moelars:llamacpp \
  --backend llamacpp --model /models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf --template chatml
```

Arguments after the image name go to `moelars serve`; set `MOELARS_API_KEY` with `-e` to
require a bearer token.

Use it from the official System One SDKs by swapping the base URL:

```python
from typesafe_sdk import Choice, Noul, TypeSafeClient

client = TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8600")
r = client.system_one(
    state="I was charged twice.",
    questions={"refund": Noul(instructions="Is the customer asking for money back?"),
               "team": Choice(instructions="Which team?", criteria={"billing": "Charges", "technical": "Bugs"})},
)
print(r.nouls["refund"].noul, r.choices["team"].choice)
```

## Run it at login, for coding agents (MCP)

On macOS, `moelars service` runs the server as a launchd agent that starts at login. Coding
agents reach it over MCP at `/mcp`. The model loads on the first request and unloads after
15 minutes without one. Details are in `SERVICE.md`.

```bash
uv tool install '.[mlx,mcp]'                     # a fixed install for the service to run from
moelars service install --backend mlx --model mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit \
  --adapter checkpoints/lora-30b-c-s1 --calibration calibration/served/lora-30b-c-s1.json
moelars service status                           # also: start, stop, restart, logs, uninstall
claude mcp add --transport http --scope user moelars http://127.0.0.1:8600/mcp \
  --header "Authorization: Bearer $(cat ~/.config/moelars/api-key)"
```

The tools are `moelars_check` (the probability that a claim holds for a text),
`moelars_classify` (one option out of several), `moelars_decide` (a full System One request)
and `moelars_status`. Clients that only speak stdio run `moelars mcp-bridge`, which holds
no model and forwards to the service. The service binds localhost, refuses non-local
`Origin` headers, and requires the key that `install` writes to `~/.config/moelars/api-key`.
For your own server, `moelars serve --mcp --idle-unload 15m` does the same without launchd.

## Calibrate on your own data

Raw model logits are not calibrated. Fit temperatures on a labeled set, then serve with them:

```bash
uv run moelars calibrate --backend mlx --model <model> --data my_labels.jsonl --out calibration/mine.json
uv run moelars serve     --backend mlx --model <model> --calibration calibration/mine.json
uv run moelars eval      --backend mlx --model <model> --calibration calibration/mine.json --data my_test.jsonl
```

`eval` reports accuracy, expected calibration error, Brier score, and the coverage you
can automate at a 5% error budget. The JSONL format is in `evals/README.md`. Nouls get
a Platt fit on the raw yes-minus-no logit, which can move a biased model's decision
boundary; choice and score get a temperature. Rows may carry `"features": {"name": value}` for nouls; `calibrate` then also fits an evidence fusion, used
whenever a request supplies the same features in `moelars.features`. `examples/molar_triage/` walks through
this end to end on a small hand-labeled dental inbox.

## Train a decision head (Tier B)

The backbone stays frozen. A small zero-initialized pointer head learns a residual on
the model's own label logits from cached features, in seconds, and serves through the
same engine:

```bash
uv run python -m moelars.train.build   --out data/train
uv run python -m moelars.train.extract --model <model> --records data/train/*.train.jsonl --out data/features
uv run python -m moelars.train.residual --train data/features/train.npz --heldout data/features/heldout.npz
uv run moelars serve --backend mlx --model <model> --head checkpoints/pointer_head.npz --projection data/features/projection.npy
```

Details and the data policy are in `src/moelars/train/README.md`.

## Layout

```
src/moelars/
  schema.py        wire models, request validation, moe-LARS extensions
  render.py        prompt rows: fenced state prefix + per-question suffix
  labels.py        single-token option labels verified per tokenizer
  primitives.py    softmax, confidence formulas, expected score, order sensitivity
  calibration.py   temperature and Platt fitting, ECE, Brier, coverage-at-error
  engine.py        rows -> backend -> answers; permutations, constraints, abstain, evidence
  backends/        mock, mlx, llamacpp
  server.py        FastAPI app, System One error and header conventions
  evalset.py       labeled JSONL loading, eval and calibrate
  cli.py           moelars serve | eval | calibrate
tests/             unit tests plus a conformance test that runs the official SDK in-process
evals/             benchmark data conventions and a jev-bench fetcher
```

## Results

jev-bench, 22 configs, 200 test rows each, per-config calibration fitted on validation.
Jev's numbers are quoted from jev-bench's published jev-1.13.0 run on full splits.

| configuration | macro acc | Brier | ECE |
|---|---|---|---|
| Jev (published) | 0.733 | 0.349 | 0.113 |
| Qwen3-4B zero-shot | 0.662 | 0.404 | 0.088 |
| Qwen3-4B + LoRA | 0.731 | 0.317 | 0.074 |
| Qwen3-30B-A3B zero-shot | 0.680 | 0.372 | 0.079 |
| **Qwen3-30B-A3B + attention LoRA** | **0.752** | 0.295 | 0.060 |
| Qwen3-30B-A3B + attention-plus-experts LoRA | 0.751 | 0.286 | 0.072 |
| both 30B adapters averaged (simulated from row dumps) | 0.765 | 0.282 | |

- A paired bootstrap over test rows puts the attention LoRA 1.9 points above Jev (95% CI
  0.7 to 3.1) and averaging both adapters 1.3 above it alone (0.7 to 1.9). Neither covers
  seed variance; a second seed is in `CLOSEOUT.md`.
- civil_comments (calibrated 0.930, the majority baseline) is worth about 1.8 macro points;
  without it the attention LoRA is 0.743 against Jev's 0.733.
- LoRA regresses stsb (0.425 to 0.350 on the 30B) and mmlu (0.780 to 0.750).

Full tables in `evals/RESULTS.md`, newest section last.

## Status

Pre-alpha, closing out to v0.1.0 (`CLOSEOUT.md`). The mock backend and the HTTP contract
are tested. The MLX backend has been run on Apple Silicon with Qwen3-4B, Qwen3.5-9B and
Qwen3-30B-A3B, zero-shot, with Tier B heads, and with LoRA adapters. The llama.cpp
backend has run in the Docker image on CPU (Qwen3-4B Q4_K_M: boolq 0.84, sst5 0.48 on 100
rows each, about 2.5 s per row), but has not been benchmarked on the full suite.
See `DESIGN.md` for the architecture, the reasoning, and the roadmap.

## License

MIT.
