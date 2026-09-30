# Tier B: residual decision head

## Recipe

The backbone stays frozen. For each row, the engine already computes label logits
`z` (one per option) by reading the next-token distribution after `Answer:`. Tier B
adds a small head that produces a correction `r` and a learned scale, and serves

    z' = s * z + r,     r = head(h, E_labels)

where `h` is the backbone's last hidden state at the answer position and `E_labels`
are the output-embedding rows of the label tokens. In practice the head is a pointer:
it scores each option's line-end hidden state against the answer-position hidden
state. Both are projected with a fixed seeded matrix and RMS-normalized, identically
at training and serving time, because raw final-layer hidden states have norms in the
hundreds and an unnormalized head diverges on the first step. The head's query
projection is zero-initialized so training starts exactly at the Tier A model. This is the
"residual on the model's own scorer" design that jev-bench's maintainers found keeps
the in-distribution gain without regressing on sources the head never saw.

Losses, summed:

1. cross-entropy against the target distribution (soft labels when available)
2. Brier against the same target, which is what calibration is scored on
3. permutation-KL: each choice row is also presented with shuffled option order, and
   the two realigned distributions are pulled together
4. complement consistency: for noul pairs generated as negations, P(yes) of one plus
   P(yes) of the other is pulled toward 1

Sources are split by whole source, never by row, so the held-out number measures
generalization to question forms the head never saw.

## Data

    uv run python -m lars.train.build --out data/train

pulls Open-Jev (CC0), tasksource-jev, and the jev-bench train and validation splits,
converts them to `Record` JSONL, and writes a manifest with per-source counts.

No data labeled by Jev is used. See `DESIGN.md` section 8.

## Commands

```bash
# 1. corpus (CPU, downloads)
uv run python -m lars.train.build --out data/train --max-per-source 4000 --jev-bench-per-config 300

# 2. features from the frozen backbone (GPU). Whole sources are held out.
uv run python -m lars.train.extract --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
    --records data/train/open-jev.train.jsonl data/train/jev-bench.train.jsonl data/train/tasksource-jev.train.jsonl \
    --test-records evals/data/*.test.jsonl --limit 3000 --holdout-fraction 0.2 --out data/features

# 3. train the head on cached features (seconds)
uv run python -m lars.train.residual --train data/features/train.npz --heldout data/features/heldout.npz \
    --out checkpoints/pointer_head.npz

# 4. serve with it
uv run lars serve --backend mlx --model <model> --head checkpoints/pointer_head.npz \
    --projection data/features/projection.npy
```

### LoRA plus head

The half of Tier B that `DESIGN.md` specifies beyond the frozen-feature head: adapt the
backbone with LoRA through the same label readout, then train a head on the adapted
model's features. `lars.train.lora` uses the same sampling, `--limit`, and seed as
`extract`, so both hold out the same sources.

```bash
# 1. LoRA through the label readout (GPU, the long step). Saves the best held-out checkpoint;
#    if none beats the untrained model, saves the identity with "improved": false in adapter_config.json.
uv run python -m lars.train.lora --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
    --records data/train/open-jev.train.jsonl data/train/jev-bench.train.jsonl data/train/tasksource-jev.train.jsonl \
    --limit 20000 --out checkpoints/lora-4b

# 2. features from the adapted model, then a head on them, as above
uv run python -m lars.train.extract --model mlx-community/Qwen3-4B-Instruct-2507-4bit --adapter checkpoints/lora-4b \
    --records ... --test-records evals/data/*.test.jsonl --test-limit 1100 --limit 20000 --max-options 160 \
    --out data/features-lora
uv run python -m lars.train.residual --train data/features-lora/train.npz --heldout data/features-lora/heldout.npz \
    --epochs 10 --out checkpoints/pointer_head_lora.npz

# 3. fair suite; --adapter works on serve, eval, and calibrate too
uv run python evals/run_suite.py --backend mlx --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
    --adapter checkpoints/lora-4b --tag lora
```

Loss: cross-entropy plus Brier on the K label logits at the answer position, with a fresh
option order for every choice record each epoch. Only the answer position is projected to
the vocabulary. LoRA's `scale` multiplies every update, so keep the learning rate low
(default 2e-5 at scale 20); on a toy model 1e-2 oscillated where 3e-3 converged.

## Status

First real run done on Qwen3-4B features, see `evals/RESULTS.md`: +2.6 macro accuracy,
ECE down a third, Brier down a fifth on the jev-bench test shard with no per-config
calibration; held-out sources neutral on accuracy and better on Brier. Checkpoint
selection is by held-out Brier. `python -m lars.train.report` gives the per-source
table. The fair comparison (head plus per-config calibration against Tier A plus the
same) is +1.8 macro accuracy, all on sources the head trained on; held-out sources are
neutral. `evals/run_suite.py --head ... --tag head` reproduces it. Next: train on the
full corpus (`--limit 20000`) and on a 9B backbone; see `HANDOFF.md`.

Known gaps: the head is per-letter-position-blind by design (it scores option text
through the option-line hidden state, not the letter), so it cannot fix letter bias in
the backbone's own `z`; the learned scale can only shrink it. Complement-consistency
loss is described above but not yet implemented, since no source in the corpus
supplies negated noul pairs; generate them in `build` first.
