# Handoff, 2026-09-24 midday

Everything below is committed and pushed to `main` on `git@github.com:moedex/lars.git`.
Nothing of ours is running.

## State in one paragraph

LARS (Limited but Accurate Response System; `lars` in code) serves the System
One wire shape from a local model with no text generation. On jev-bench (22 configs, 200
test rows each, per-config calibration on every side), **Qwen3-30B-A3B with an
attention-only LoRA is 0.752 macro accuracy against Jev's published 0.733**, Brier 0.295
against 0.349, and 0.743 against 0.733 without civil_comments. Qwen3-4B with its LoRA is
0.731. Routing the 4B to the adapted 30B reaches 0.757 at 46% escalated, inside the noise
of the 30B alone. A second 30B adapter over attention plus every expert is also 0.752
(0.751), better on high-K choice and worse on ordinal scores; **averaging the two 30B
adapters is 0.765**, the best so far, chosen on validation but near the noise.
Full tables: `evals/RESULTS.md`, newest section last.

## What happened since (2026-09-23 night to 2026-09-24)

- Full attention-plus-experts LoRA on the 30B (422M parameters): 169 minutes, 92.9 GB
  peak, held-out Brier 0.458 at step 1000, level with attention only. Suite 0.751, Brier
  0.286; choice 0.783 against 0.769, score 0.523 against 0.555. Adapter in
  `checkpoints/lora-30b-experts` on this machine; history, config, suite, calibration and
  row dumps committed.
- Codebase review M1 (request budgets: `--max-rows`, `--max-input-tokens`,
  `--max-body-bytes`), M2 (inference off the event loop, one at a time), M14 (example IDs
  in row dumps; `evals/cascade.py` refuses misaligned rows) and L1 (request IDs on 401)
  fixed; old dumps backfilled with IDs, routing tables unchanged. 75 tests.
- `scripts/load_cost.py` takes `MODEL@ADAPTER` to time an unfused adapter.

## What happened 2026-09-23 (afternoon)

- Codebase review M11 fixed: if no checkpoint beats the untrained model, `lars.train.lora`
  saves the untrained adapter (identity) with `"improved": false`, keeps the last state in
  `<out>.last/`, and clears a stale adapter in `--out` first. `scripts/queue_30b_lora.sh`
  stops before the suite in that case. One new test (68 total).
- Full attention-only LoRA on the 30B-A3B: 1095 steps, 135 minutes on this machine, 33 GB
  peak, held-out Brier 0.742 to 0.454 (step 1000 selected, `"improved": true`).
- Suite with the adapter (unfused) and row dumps, and routing against the 4B dumps:
  committed under `evals/results/` and `calibration/`.
- Probes here: attention plus every expert (422M parameters) now fits, 4.0 steps/min at
  93 GB peak, and its 31-step held-out Brier (0.396) beats attention only (0.437) on 55
  rows.

## Next

1. **Latency decides the architecture.** Accuracy no longer separates the 30B + LoRA alone
   from the routed pair. Run `scripts/load_cost.py` on this machine (quiet) for the 4B
   fused bf16 (copy it from the old machine, or fuse it again here), the 30B,
   and the 30B as `MODEL@checkpoints/lora-30b` (and `@checkpoints/lora-30b-experts`),
   then decide what `lars serve` defaults to.
2. **Two adapters on one 30B.** Averaging the attention-only and expert adapters is 0.765
   (+1.3). Before building it, confirm it is not noise (a second seed of either adapter,
   or more test rows), then measure the cost with `load_cost.py` for both adapters. The
   cheap version: one base model, two adapters hot-swapped or batched per request.
3. **stsb and mmlu under LoRA** regress on both models (30B: stsb 0.425 to 0.350, mmlu
   0.780 to 0.750; the expert adapter is worse still on score tasks, stsb 0.275):
   ordinal-aware loss for score tasks, more score sources in `lars.train.build`, or a
   gentler adapter.
4. Codebase review findings M3 to M10, M12 and M13 (M1, M2, M11, M14 and L1 are fixed;
   see the status lines in `CODEBASE-REVIEW.md`).
5. Molar Triage numbers (`scripts/queue_30b.sh` has the loop; the `<!-- MOLAR_NUMBERS -->`
   placeholder in `examples/molar_triage/README.md`).

## Things to know

- Two machines. This one (128 GB M5 Max): checkout at `~/Code/personal/moelars`, `uv` on
  PATH, 30B weights in the HF cache, `checkpoints/lora-30b` and `checkpoints/lora-30b-experts`
  (the selected adapters) and the two probe adapters. The worktree
  `~/Code/personal/moelars-experts-run` ran the expert queue; its checkpoint is moved
  here, so it can be removed with `git worktree remove --force`.
- The old 64 GB one: `uv` at `~/Library/Python/3.9/bin/uv`, checkout at
  `~/Code/moeLAR`, and `checkpoints/lora-4b-fused-bf16`. Older queue scripts default `UV`
  to the old path; run them here with `UV=uv`.
- `data/` and `evals/data/` came from the old machine as a zip (same 14285-record corpus
  and split); do not rebuild them.
- The user shares GPUs with other work. Treat in-queue ms/row as rough; quote latency
  only from a quiet-machine `load_cost.py` run. Accuracy comes first.
- HF cache weights sit behind symlinks into xet blob folders; `du` on a model directory
  shows megabytes. Use `ls -laL snapshots/*/`. If the environment is offline,
  `mlx_lm.fuse` needs the local snapshot path as `--model`.
- LoRA's `scale` multiplies every update. Defaults are lr 2e-5, rank 8, scale 20.
- chaosnli borrows mnli's calibrator (no validation split).
- civil_comments' calibrated 0.930 is the majority baseline and is worth about 1.8 macro
  points wherever LARS leads; the headline flatters us by about that much.
- The zero-shot 30B scores 0.110 on helpsteer2_verbosity (all mass on one end of the
  scale); calibration cannot fix an argmax.
- Data policy is unchanged: no Jev-labeled data, no calls to the hosted API, Jev numbers
  quoted only from jev-bench's published run. See `DESIGN.md` section 8.

## Open ideas, not started

- Gate the pointer head on K: on LoRA it helps high-K routing and hurts elsewhere. Read
  off test, so check on held-out data first.
- Shortlist-then-rerank for high-K configs (banking77, clinc150, massive, ledgar).
- Complement-consistency loss needs negated noul pairs generated in `lars.train.build`.
- llama.cpp backend still needs a hardware pass.
