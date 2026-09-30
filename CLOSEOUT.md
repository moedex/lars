# Closeout to v0.1.0, planned 2026-09-25

Goal: a tagged GitHub release, `v0.1.0` on `moedex/lars`, that serves the most accurate
configuration that answers the `load_cost.py` request in under 2 s warm on this machine
(128 GB M5 Max). The adapters go on the HF Hub under `moedex`. Check boxes as work lands; the
newest state wins over `HANDOFF.md`.

## Decisions (2026-09-25)

| question | decision |
|---|---|
| what "ship" means | tagged GitHub release, pinned default model + adapter(s) + calibrator, release notes |
| latency | accuracy first; the default must answer the `load_cost.py` request in under 2 s warm |
| two adapters (0.765) | confirm with a second seed; if the gain holds and fits the budget, it becomes the default |
| score-task regressions | document them as a known limitation; no fix in this release |
| where the weights go | HF Hub, `moedex` org, with model cards |
| training-data licenses | audit before publishing weights; this blocks the weights, not the code |
| code review | fix every open Medium (M3 to M10, M12, M13) before the tag |

Where things stand on 2026-09-25: a paired bootstrap over the committed test row dumps (10k
resamples, rows resampled within each config) puts averaging the two adapters at +1.3 over
attention-only alone, 95% CI [+0.7, +1.9], and attention-only at +1.9 over Jev's 0.733, CI
[+0.7, +3.1]. That covers test-row noise only, not seed variance, and treats Jev's number as exact.

## Order

```
Phase 0  license audit  ──┐        latency baseline (quiet machine, ~1 h)
                           ▼
Phase 1  GPU: seed-1 run (+ retrains if the audit drops a source)    ║  code: review fixes
                           ▼                                          ║
Phase 2  gate: does the two-adapter gain hold?  ──►  build two-adapter serving, or skip
                           ▼
Phase 3  as-served numbers, preset, weights on the Hub, docs, tag
```

The audit comes first because dropping a source means retraining, and the second-seed run
should be on the corpus we actually ship. The review fixes use no GPU and run alongside Phase 1.

---

## 1. Latency decides the architecture

- [x] Quiet machine (nothing else on the GPU). Run:
  ```
  M30=mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit
  uv run python scripts/load_cost.py $M30 $M30@checkpoints/lora-30b $M30@checkpoints/lora-30b-experts
  ```
  Record load time, peak memory, and warm ms per request in `evals/RESULTS.md`.
  **Done 2026-09-25:** 265 ms zero-shot, 299 ms attention, 384 ms experts; two adapters about 0.7 s.
- [ ] Two-adapter estimate for now: the sum of the two adapter rows. Measure it for real once §2
  builds it.
- [x] ~~Only if the 30B + LoRA takes over 2 s:~~ not needed (0.3 s). fuse the 4B here (`mlx_lm.fuse`, local snapshot
  path) and time `lora-4b-fused-bf16`, which brings 4B-to-30B routing (0.757 at 46%
  escalated) back into play.

**Rule:** the default is the most accurate configuration under 2 s. In order of preference:
the two 30B adapters averaged (if §2's gate passes), then the 30B + attention-only adapter,
then 4B routed to the 30B.

## 2. Two adapters on one 30B

### 2a. Confirm the gain is real (GPU, on `data/train-c`; see §5)

Every adapter is retrained on the option-C corpus, so the gate uses the new runs: attention
seed 0, attention seed 1, and experts seed 0 (`scripts/queue_corpus_c.sh`, then
`scripts/queue_corpus_c_experts.sh` once the machine has about 100 GB free).

- [x] Second seed of the attention-only adapter on corpus C (`checkpoints/lora-30b-c-s1`):
  0.745 macro, against 0.748 for seed 0.
- [x] Suites for attn s0, attn s1 and experts, with row dumps.
- [x] Paired bootstrap, committed as `evals/bootstrap.py` (test rows, 10,000 resamples):
  - attn s1 against attn s0: -0.3 pts (-1.1 to +0.4). The seeds are indistinguishable.
  - **attn s1 + experts** against attn s1: +0.4 pts (-0.4 to +1.1). **Gate failed.**
  - attn s0 + experts against attn s0: -0.2 pts (-0.9 to +0.5).
  - **attn s0 + attn s1** (the control) against attn s0: +0.5 pts (-0.0 to +1.0), and it
    beats both mixed pairs.

**Gate:** attn s1 + experts beats attn s1 alone with a bootstrap 95% CI above zero, and the
choice of averaging over switching is made on validation. If the control does as well as the
mixed pair, ship whichever pair is cheaper to serve (two attention-only adapters are smaller).
If the gate fails, the default is one adapter (s0 or s1, whichever has the better validation
macro) and averaging goes in the release notes as future work.

**Outcome (2026-09-25): the gate failed; the default is attention seed 1.** The corpus-C
experts adapter is 0.710 alone (the old-corpus one was 0.751), mostly on
helpsteer2_verbosity (0.205 against 0.670), chaosnli (0.520 against 0.650),
measuring_hate_speech and strategyqa_closed. A re-run of attn s0 on those configs with
the current suite code reproduced its committed numbers exactly, so the harness is not the
cause. The likely cause is checkpoint selection. Held-out Brier chose step 500, half an
epoch (0.310, the best held-out score of any run, against 0.330 at step 1000). The
old-corpus experts run kept its full-epoch checkpoint. Only the selected weights are saved,
so this is untested. Seeds 0 and 1 answer the same 3,082 validation rows right. Seed 1 has
the higher validation macro (0.7494 against 0.7493, from configs with fewer validation rows)
and the better validation Brier (0.3093 against 0.3114), so it is the default. Seed 0 is
0.003 higher on test, which is not used to choose.

### 2b. Serving, only if the gate passes (it did not: the multi-adapter code stays, with no `30b-duo` preset)

- [x] `MLXBackend` takes several adapters (`lars.backends.adapters.AdapterSet`; swapping matches each adapter loaded alone, tested on a tiny Qwen3): load the base once, build LoRA layers for the
  union of the adapters' keys (experts ⊇ attention), keep each adapter's weights in memory,
  and swap them with `model.load_weights(..., strict=False)` between passes. Expert keys stay
  zero while the attention-only adapter is loaded.
- [x] `EnsembleEngine` averages the adapters' calibrated probabilities per question (the blend in
  `evals/cascade.py`), then applies constraints once on the average.
- [x] CLI: `--adapter` and `--calibration` can be repeated. Tests: two identity adapters must match one; the
  average must equal the per-pass mean.
- [ ] Re-run §1's `load_cost.py` with both adapters and check it is under 2 s.
- [ ] Suite with both adapters live (not simulated) must reproduce the cascade number within
  ±0.5 (`evals/ensemble_check.py`).

## 3. Score-task regressions: document them

- [x] A table in `evals/RESULTS.md` (2026-09-25, as served) of every config where the shipped
  default (attention seed 1, pooled calibrator) scores below the zero-shot 30B: six, none by
  more than 2 points. stsb no longer regresses (0.455 against 0.425), since corpus C does not
  train on it.
- [x] Headline stated honestly: 0.743 as served against Jev's 0.733; 0.735 against 0.733
  without civil_comments, a tie. Below Jev on 14 of 22 configs. The clear win is calibration
  (Brier 0.317 against 0.349). Seeds tie on macro but move single configs by up to 9.5
  points (ledgar, stsb). Our numbers use 200 test rows per config; Jev's use full splits.
- [ ] Ordinal loss and more score sources go on the post-release list (below).

## 4. Code-review Mediums (no GPU; alongside Phase 1)

Each fix gets a test and a status line in `CODEBASE-REVIEW.md`, same format as M1.

- [x] **M3** Reject a request `model` that isn't the loaded model's ID or its documented
  alias (`lars` or the preset name); 400 `invalid_request`.
- [x] **M4** Split the prompt at the offset where the sentinel was inserted, not at the first
  match; test with the sentinel inside state.
- [x] **M5** Apply constraints until nothing changes (with an iteration cap) and check the
  invariants; round only at the end; test the exclusive+complement case from the review.
- [x] **M6** Keep units after 24 in every ablated state (or decline evidence past 24 units
  with a warning); test with a 30-unit state.
- [x] **M7** Pass option offsets from `render` to the pointer head instead of re-parsing
  `A) `. (No escaping: prompts stay byte-identical, so no suite re-run is needed for M4 or M7.)
- [x] **M8** Reject `multi` examples in eval/calibrate with a clear error (per-option labels
  are post-release).
- [x] **M9** Use the normalized soft target for nouls in metrics and Platt fitting.
- [x] **M10** `coverage_at_error` evaluates groups of equal confidence together; test
  `[0.9, 0.9]` / `[1, 0]`.
- [x] **M12** Extraction deletes stale `heldout.npz` / `test.npz` and writes a manifest
  fingerprint; consumers check it.
- [x] **M13** Fingerprint suite inputs per config (rows, backend, adapter, head, template, data
  hash); a mismatch or a missing row dump means recomputing.
- [ ] After the M4, M5 and M7 fixes, re-run the suite for the shipped default and confirm the
  headline hasn't moved by more than noise (render changes affect every prompt).

## 5. Data license audit (Phase 0, blocks the weights)

- [x] Per-source table in `DESIGN.md` §8.1 (2026-09-25). ANLI is CC BY-NC 4.0 (3,300 rows);
  yelp5's terms forbid redistributing derived work; sst5 and stsb could not be verified;
  LEDGAR is CC BY 4.0 (not share-alike). Everything else is permissive, with attribution.
- [x] **Decision (2026-09-25): option C.** Drop yelp5, sst5, stsb and ANLI (4,200 rows) and
  backfill 3,300 SNLI choice rows (CC BY-SA 4.0), giving `data/train-c` with 13,385 rows
  (`scripts/build_corpus_c.py`, tasksource-jev pinned at `18332e5`). Adapters get Apache
  2.0 with attributions.
- [ ] Retrain on `data/train-c`: attention seed 0 and seed 1, then experts. These replace the
  §2a runs; the old adapters stay research-only and unpublished.
- [ ] Qwen3-30B-A3B base is Apache 2.0; the model card credits it and every §8.1 source.

## 6. Release (Phase 3)

- [x] **Calibrator as served.** Fitted on 4,120 pooled validation rows (`calibration/served/lora-30b-c-s1.json`): 0.743 macro, Brier 0.317, ECE 0.090, against 0.745 / 0.310 / 0.073 per-config. Per-config calibrators can't be used for arbitrary requests.
  Fit one pooled calibrator over all configs' validation rows for the shipped default and add
  `--calibration FILE` to `evals/run_suite.py`. Report the pooled number as the headline and
  per-config as secondary. (Raw and per-config calibrated accuracy differ only on
  civil_comments, so expect about 0.75.)
- [x] `lars serve --preset 30b` (no `30b-duo`: §2 failed) resolves the model, the
  adapter from the Hub (`huggingface_hub.snapshot_download`) and the pooled calibrator.
  `--backend mock` stays the default with no preset. Add a preset smoke test that uses the
  mock backend in CI.
- [ ] HF Hub: `moedex/lars-qwen3-30b-a3b-lora-attn` (attention seed 1, `checkpoints/lora-30b-c-s1`), with
  `adapter_config.json`, safetensors, the calibrator, and a model card: base model, recipe and
  seed, training sources and licenses from §5, the jev-bench table, known limitations, and the
  data policy (no Jev-labeled data).
- [ ] Docs: README quickstart using the preset, the headline table, the civil_comments caveat,
  and the Docker images (mock, and llama.cpp on CPU: first hardware pass 2026-09-25 found
  and fixed zeroed logits; Qwen3-4B Q4_K_M boolq 0.84, sst5 0.48 on 100 rows). Update `HANDOFF.md` and `evals/RESULTS.md`.
- [ ] Bump `version` to 0.1.0 in `pyproject.toml` and `src/lars/__init__.py`, and add a
  `CHANGELOG.md` entry.
- [ ] Final checks on a clean checkout: `uv sync`, `ruff`, `pytest` (MLX tests here), CI green
  on `main`, and a quiet `load_cost.py` run with the preset under 2 s.
- [ ] Tag `v0.1.0`, GitHub release with notes (headline, latency, known limitations, links to
  the Hub).

## Not in this release

- The two-adapter ensemble. Retrain the corpus-C experts adapter keeping the full-epoch
  checkpoint, or save every evaluated checkpoint, and re-run the §2a gate. Select on
  validation macro over the suite rather than Brier on four held-out sources; that choice
  picked the worst suite adapter of the run.
- Ordinal-aware loss and more score sources (the stsb and mmlu regressions).
- Molar Triage numbers (`examples/molar_triage/README.md` placeholder).
- Per-option labels for `multi` in eval/calibrate (M8's full fix).
- Gating the pointer head on K, shortlist-then-rerank for high K, complement-consistency loss.
- A hardware pass for the llama.cpp backend.
- Merged adapters as GGUF for CPU deployment, for machines without Apple Silicon or a GPU.
  Fuse each LoRA into a de-quantized base (`mlx_lm.fuse --de-quantize`), convert with
  llama.cpp's `convert_hf_to_gguf.py`, quantize to Q4_K_M, and publish next to the MLX
  adapters. A two-adapter ensemble cannot merge into one file (it averages outputs, not
  weights), so it ships as two GGUFs at twice the CPU cost. The bar to beat on CPU: the 4B
  GGUF took about 2.5 s per row in Docker, and Laya answers in about 0.17 s at 0.559 macro
  (`evals/RESULTS.md`, 2026-09-25). The 30B-A3B has only 3B active parameters, so a Q4 GGUF
  (about 18 GB of RAM) may be practical; measure it before promising it.
