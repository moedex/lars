# Codebase review

Reviewed 2026-09-23 at `39e4ce1` (`main`). This is a read-only review of the repository's Python source, tests, shell scripts, example builder, package and CI configuration, and project documentation. Four review subagents covered the API, decision engine, backends and training, and evaluation tooling; the lead checked integration paths and reproduced selected findings. Locally present untracked `scripts/queue_cascade.sh` and evaluation outputs were considered. Bulk calibration/result JSON, row dumps, model checkpoints, downloaded datasets, caches, and binary artifacts were inspected only where relevant to a code path; their contents were not individually audited. No external dependency vulnerability scan or real-model inference was performed.

No Critical or High findings were verified. Findings below describe current behavior and the conditions that trigger it.

## Medium

### M1. A request can expand into thousands of model passes

**Status (2026-09-24): fixed.** `Engine` refuses a request over `max_rows` (512) planned rows or `max_input_tokens` (32,768) before any model pass (422 `invalid_request`); the server refuses bodies over 1 MB by `Content-Length` (413). `lars serve --max-rows/--max-input-tokens/--max-body-bytes`. A chunked body without `Content-Length` is still only bounded by the token budget.

**Evidence:** `src/moelar/schema.py:127-130` has no upper bound on state length, question count, or aggregate option text. `src/moelar/render.py:171-210` multiplies base rows by options and evidence ablations. A schema-valid request with one 255-option `multi` question, 24 state units, and `moelar.explain=true` plans 6,375 rows.

**Impact and trigger:** A client allowed to call `/v1/systemone` can tie up the model for a long time or exhaust memory with one request. The default server binds to localhost, but the CLI permits other bind addresses (`src/moelar/cli.py:80`). This is a code-path finding; it was not load-tested on a real model. **Suggested fix:** Enforce request-body, prompt-token, question, and planned-row budgets before inference, with explicit limits for evidence and multi-option expansion.

### M2. Synchronous inference blocks the HTTP event loop

**Status (2026-09-24): fixed.** `/v1/systemone` runs `Engine.evaluate` on a worker thread behind a one-slot limiter, so inference is serialized over the shared model and `/healthz` answers while a request is scored (tested).

**Evidence:** The `async` route at `src/moelar/server.py:73-76` calls synchronous `Engine.evaluate`; `src/moelar/engine.py:93-99,120-136` calls synchronous backend inference. In an ASGI reproduction with a 350 ms sleeping engine, a concurrent `/healthz` request completed only after the inference request (364 ms).

**Impact and trigger:** Any slow model request delays unrelated requests handled by that worker, including health checks. **Suggested fix:** Offload inference to a bounded worker or use a synchronous route, and control concurrency around shared model state.

### M3. The requested model is silently ignored

**Status (2026-09-25): fixed.** `Engine.evaluate` serves `lars-latest`, `jev-latest` (the official SDKs' default, so a base-URL swap still works) and the loaded model's exact ID, and refuses any other name with 422 `invalid_request` before any model pass (tested over HTTP).

**Evidence:** `src/moelar/schema.py:128` accepts any model string. `src/moelar/engine.py:93-99` does not read it and always returns the loaded engine's model ID. A POST specifying `model: "totally-unknown"` returned 200 with the mock model ID.

**Impact and trigger:** A caller can believe it used a requested model while receiving decisions from another, invalidating routing or evaluation assumptions. **Suggested fix:** Reject IDs other than the loaded model and documented alias, or implement actual model selection.

### M4. A sentinel inside caller state changes the prompt boundary

**Status (2026-09-25): fixed.** `compose_prompt` splits at the sentinel's last occurrence; only template text follows the inserted one. Prompts for ordinary state are byte-identical (tested for every template).

**Evidence:** `src/moelar/render.py:29,222-224` inserts a fixed sentinel and splits at its *first* occurrence. Supplying the literal sentinel in the state caused `compose_prompt` to place `QUESTION: x` before the closing state fence and leave the inserted sentinel after it.

**Impact and trigger:** State text containing `\u0000MOELAR_QUESTION\u0000` corrupts the prompt structure and can move the question into the data region. **Suggested fix:** Split at a known insertion offset or use a marker guaranteed absent from all caller content.

### M5. Overlapping constraints can leave answers violating a declared constraint

**Status (2026-09-25): fixed.** Constraints that share no question behave as before. When they overlap, answers become the least-squares nearest probabilities satisfying all of them (Dykstra's projections over each constraint and the unit box). Exclusive members round down, and a complement's second answer is 1 minus its rounded first, so rounding cannot break either. The review's case gives 0.5, 0.5, 0.0.

**Evidence:** `src/moelar/engine.py:252-267` applies constraints once in request order. Applying `exclusive(a,b,c)` to three 0.9 nouls and then `complement(a,b)` produced `a=0.5, b=0.5, c=0.3333`; the exclusive sum is 1.3333. Independent rounding can also make a normalized exclusive group sum to 1.0002.

**Impact and trigger:** A valid request with overlapping constraints can return probabilities inconsistent with its declared rules. **Suggested fix:** Resolve constraints jointly or iterate to a validated fixed point, and round only after enforcing the final invariants.

### M6. Evidence ablation drops every state unit after the first 24

**Status (2026-09-25): fixed.** Only the first 24 units are ablated, but every ablated state keeps all the other units, including those past 24 (tested with 30).

**Evidence:** `src/moelar/render.py:161-163` truncates units to 24; `:205-210` builds each ablated state only from that truncated list. The base decision still sees the complete state. With a 30-unit state, every ablation omits units 25–30 as well as its named span.

**Impact and trigger:** For long states with `moelar.explain=true`, reported evidence effects can be attributed to the wrong span. **Suggested fix:** Keep the unablated tail in every comparison, or explicitly decline evidence when the state exceeds the supported span count.

### M7. Pointer-head features can come from user text instead of option lines

**Status (2026-09-25): fixed.** `render` records where each option line ends (`Row.option_ends`), and the engine and feature extraction pass those offsets to `label_logits_with_features`; parsing is only a fallback for callers that don't pass them. Instructions and criteria are not escaped, so prompts stay byte-identical to what the adapters and heads were trained on (checked over 23,033 eval rows, where the recorded offsets also equal the old parse).

**Evidence:** `src/moelar/spans.py:7,14-20` takes the first lines matching `A) ` or `B) ` anywhere in a suffix. `src/moelar/render.py:114-118,127-137` inserts caller instructions and descriptions without escaping newlines. Instructions beginning with `A) ...\nB) ...` are therefore selected before the generated choices; `src/moelar/backends/mlx.py:211-214` uses those offsets for option features.

**Impact and trigger:** With a pointer head, such instructions or multiline criteria make it score unrelated token positions and can change decisions. **Suggested fix:** Carry the generated option offsets from rendering, or restrict parsing to the generated options block and escape embedded line breaks.

### M8. Multi-question examples crash the evaluation and calibration commands

**Status (2026-09-25): fixed.** `eval` and `calibrate` refuse a `multi` example with a `ValueError` naming the fix (one noul per option) instead of an `IndexError`. Scoring per-option labels is post-release.

**Evidence:** `src/moelar/render.py:184-189` labels multi rows `base:<key>`, while `src/moelar/engine.py:104-106` keeps only rows whose variant equals `base` and then indexes the empty result. `src/moelar/evalset.py:90-96` uses this path for both commands. Calling `raw_logits` for a valid two-option `MultiQuestion` raised `IndexError: list index out of range`.

**Impact and trigger:** Any `multi` example in an eval or calibration JSONL stops the command, though the request API supports that question type. **Suggested fix:** Define and collect per-option multi labels for these workflows, or reject multi examples with a clear validation error before evaluation.

### M9. Noul soft labels are discarded during evaluation and fitting

**Status (2026-09-25): fixed.** Nouls use the normalized soft label as the Brier and Platt target (accuracy still scores the hard label). Platt smoothing now applies to hard 0/1 targets only, so soft targets survive fitting. No committed eval set has noul soft labels, so no published number moves.

**Evidence:** `src/moelar/evalset.py:78-83` can construct a target from `soft_label`, but `:108-116,148-150` substitutes a hard yes/no vector for nouls in both metrics and Platt fitting. A row labeled yes with `soft_label={"yes":0.6,"no":0.4}` is treated as `[1,0]`.

**Impact and trigger:** Noul datasets with human probability labels produce Brier scores and fitted calibrators against a different target than the supplied one. **Suggested fix:** Use the normalized soft target for nouls as for choice/score, or reject and document unsupported soft labels for this kind.

### M10. Coverage can report a threshold that exceeds its error budget

**Status (2026-09-25): fixed.** `coverage_at_error` only considers cut points at the end of a run of equal confidences, so a reported threshold always accepts exactly the set whose error was checked.

**Evidence:** `src/moelar/calibration.py:156-162` evaluates sorted samples one by one without grouping equal confidence values. For confidences `[0.9, 0.9]`, correctness `[1, 0]`, and a zero error budget, `coverage_at_error` returns `(0.5, 0.9)`; accepting all decisions at threshold 0.9 yields 50% error.

**Impact and trigger:** Tied confidence values can make reported coverage and threshold disagree, overstating safe automation coverage. **Suggested fix:** Evaluate complete equal-confidence groups and return only a threshold whose actual selected set meets the budget.

### M11. LoRA can publish an adapter worse than its baseline

**Status (2026-09-23): fixed.** The untrained adapter (identity) is saved with `"improved": false` when no checkpoint beats it; the last state goes to `<out>.last/`.

**Evidence:** `src/moelar/train/lora.py:250-268` saves an improved checkpoint only when held-out Brier is strictly below the unadapted baseline. If none improves, `:292-296` saves the *last trained* adapter anyway. The training guide at `src/moelar/train/README.md:69-73` says it saves the best held-out checkpoint.

**Impact and trigger:** A degrading run still leaves an adapter that later evaluation or serving can load as the selected result. This was verified by code path; MLX training was not run here. **Suggested fix:** Save or restore the initial baseline state, and distinguish an intentionally retained failed candidate from a selected adapter.

### M12. Re-extraction can leave stale held-out or test shards

**Status (2026-09-25): fixed.** Extraction deletes `train`/`heldout`/`test` shards and `manifest.json` in `--out` before writing, and `Shard` refuses a shard whose `manifest.json` beside it does not list it with the same number of presentations. The three committed feature directories pass.

**Evidence:** `src/moelar/train/extract.py:94-110` skips empty subsets without removing existing `heldout.npz` or `test.npz` in `--out`; `src/moelar/train/features.py:101-102` also writes nothing for an empty presentation stream. The documented next step reads `heldout.npz` by path (`src/moelar/train/README.md:53-55`).

**Impact and trigger:** Reusing an output directory after changing model, split, or test inputs can let later training or reports silently consume shards from a previous run. **Suggested fix:** Use a fresh output directory or remove obsolete shards and verify shard provenance against the current manifest before consuming them.

### M13. Suite caching can mix old metrics with new run metadata and skip row dumps

**Status (2026-09-25): fixed.** Each config's entry carries a fingerprint of model, backend, template, rows, adapter/head/projection files (size and mtime) and the config's data. Entries that don't match the current run are dropped, and with `--dump-rows` a config without its row dump is recomputed (tested end to end on the mock backend). Committed results have no fingerprint, so a rerun under an old slug recomputes them.

**Evidence:** `evals/run_suite.py:127-137` keys the output by model basename plus optional tag, loads existing results, and overwrites top-level run metadata. `:152-155` skips each cached config before the `--dump-rows` writer at `:179-189`.

**Impact and trigger:** Repeating a run with the same slug after changing `--rows`, backend, adapter, head, template, or data can label old per-config metrics as the new setup. Running once without `--dump-rows` and then rerunning with it exits successfully without creating those rows. **Suggested fix:** Fingerprint effective inputs per config and invalidate mismatches; explicitly generate or require recomputation for missing row dumps.

### M14. Cascade results do not verify that paired rows are the same examples

**Status (2026-09-24): fixed.** Row dumps carry `id`, a content hash of state, question, and label (`Example.id`); `evals/cascade.py` refuses rows whose IDs differ and warns on dumps without IDs. The committed dumps were backfilled, every row's target checked against `evals/data`.

**Evidence:** `src/moelar/evalset.py:124-125` dumps probabilities, target, and hit but no example ID. `evals/cascade.py:33-36` accepts pairs if lengths and target vectors match. Reordered or replaced examples with the same target, common in binary tasks, therefore pass the check.

**Impact and trigger:** Combining dumps made from different row orders or dataset revisions can silently calculate invalid cascade accuracy and escalation. The currently saved shared dumps have matching lengths, keys, and targets; actual misalignment was not established. **Suggested fix:** Store a stable example ID or content hash in each dump and require it to match before combining rows.

## Low

### L1. Authentication failures omit the SDK request ID header

**Status (2026-09-24): fixed.** 401 and 413 responses carry both request ID headers.

**Evidence:** `src/moelar/server.py:49-51` returns a 401 with only `x-moelar-request-id`; normal responses add `x-typesafe-request-id` at `:52-55`, and the module documents that the SDK reads it. A reproduced 401 omitted the SDK-facing header.

**Impact and trigger:** SDK callers cannot correlate authentication failures by the request ID they normally receive. **Suggested fix:** Add both headers on the early 401 path.

## Verification and limits

`ruff check .` passed. Forty-one tests passed when MLX-dependent modules and one MLX-dependent test were excluded. The full suite aborted during `mlx.core` import; a narrower run confirmed `ImportError: [metal::load_device] No Metal device available` in this environment. Reproductions used the mock backend or isolated functions and did not modify project files. GPU inference, training, and dependency vulnerability status remain unverified.
