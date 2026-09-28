"""LoRA on the backbone, trained through the same label readout the engine serves.

The loss is on the K label logits at the answer position, cross-entropy plus Brier against
the record's target, not on generated text, so training optimizes exactly what serving
reads. Choice records get a fresh option order every epoch, so the adapter cannot learn
letter positions. Whole sources are held out (the same split as the pointer head) and
scored at every evaluation. Which checkpoint becomes the adapter:

- With `--select-records` (the benchmark's validation rows, e.g.
  `data/train/jev-bench.validation.jsonl`), the one with the best macro accuracy over the
  selection sources, the metric the suite reports. Held-out Brier on a few sources proved
  a poor proxy: on the corpus-C experts run it chose a half-epoch checkpoint that was the
  worst adapter of the run on the suite.
- Otherwise, the one with the best held-out Brier.

Either way the untrained model is the first candidate. If no checkpoint beats it, the saved
adapter is the untrained one (the backbone exactly), marked `"improved": false`, and the
last trained state goes to `<out>.last/` for inspection only. Every evaluated checkpoint is
also kept in `<out>.steps/step-<n>/`, so a different choice needs no retraining.

Adapters are saved in mlx-lm's format (`adapters.safetensors` plus `adapter_config.json`),
so `mlx_lm.load(model, adapter_path=...)` and `--adapter` on every moe-LARS entry point load
them. A pointer head for the adapted model is then trained on features extracted with
`python -m moelars.train.extract --adapter ...`.

    uv run python -m moelars.train.lora --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
        --records data/train/open-jev.train.jsonl data/train/jev-bench.train.jsonl \
        data/train/tasksource-jev.train.jsonl --limit 20000 --out checkpoints/lora-4b
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from moelars.labels import assign_labels
from moelars.render import compose_prompt, render_content
from moelars.train.data import Record, read_records, split_by_group
from moelars.train.features import _render


@dataclass
class Example:
    """One tokenized presentation: the full prompt, its label token ids, and the target in presented order."""

    ids: list[int]
    label_ids: list[int]
    target: np.ndarray
    source: str


def encode(backend: Any, record: Record, labels: list[str], order: list[int]) -> Example:
    """Render `record` exactly as the engine would, with its options presented in `order`."""
    body, _, _ = _render(record, labels, order)
    prefix, suffix = compose_prompt(backend.template(), render_content(record.state), body)
    k = len(record.options)
    label_ids = [backend._label_id(label) for label in labels[:k]]
    if any(i is None for i in label_ids):
        raise ValueError(f"labels not single-token for this tokenizer: {labels[:k]}")
    target = np.asarray([record.target[i] for i in order], np.float32)
    return Example(backend._encode(prefix + suffix), label_ids, target, record.source)


def presentations(backend: Any, records: list[Record], labels: list[str], rng: random.Random | None,
                  max_tokens: int) -> tuple[list[Example], int]:
    """Encode every record; shuffle choice options when `rng` is given. Returns (examples, skipped as too long)."""
    examples, skipped = [], 0
    for record in records:
        order = list(range(len(record.options)))
        if rng is not None and record.kind == "choice":
            rng.shuffle(order)
        example = encode(backend, record, labels, order)
        if len(example.ids) > max_tokens:
            skipped += 1
            continue
        examples.append(example)
    return examples, skipped


def batches(examples: list[Example], batch_tokens: int, rng: random.Random | None) -> list[list[Example]]:
    """Group by length so padding stays small; each batch holds at most `batch_tokens` padded tokens."""
    order = sorted(range(len(examples)), key=lambda i: len(examples[i].ids))
    groups: list[list[Example]] = []
    current: list[Example] = []
    for i in order:
        longest = max([len(e.ids) for e in current] + [len(examples[i].ids)])
        if current and longest * (len(current) + 1) > batch_tokens:
            groups.append(current)
            current = []
        current.append(examples[i])
    if current:
        groups.append(current)
    if rng is not None:
        rng.shuffle(groups)
    return groups


def collate(mx: Any, batch: list[Example]):
    """Right-pad token ids and label ids. Causal attention means padding after a row's last token never reaches it."""
    t_max = max(len(e.ids) for e in batch)
    k_max = max(len(e.label_ids) for e in batch)
    ids = np.zeros((len(batch), t_max), np.int32)
    label_ids = np.zeros((len(batch), k_max), np.int32)
    kmask = np.zeros((len(batch), k_max), bool)
    target = np.zeros((len(batch), k_max), np.float32)
    lengths = np.zeros(len(batch), np.int32)
    for row, e in enumerate(batch):
        ids[row, : len(e.ids)] = e.ids
        k = len(e.label_ids)
        label_ids[row, :k] = e.label_ids
        kmask[row, :k] = True
        target[row, :k] = e.target
        lengths[row] = len(e.ids)
    return mx.array(ids), mx.array(lengths), mx.array(label_ids), mx.array(kmask), mx.array(target)


def text_stack(model: Any) -> Any:
    """Vision-language checkpoints (Qwen3.5) keep the text model under `language_model`."""
    return getattr(model, "language_model", model)


def readout(mx: Any, model: Any, ids, lengths, label_ids, kmask):
    """Label logits at each row's last prompt token, (B, K), padded options at -1e9.

    Runs the transformer body over the whole batch but projects only the answer position to
    the vocabulary, which is the only position the engine reads.
    """
    text = text_stack(model)
    hidden = text.model(ids)  # (B, T, H), final norm applied
    last = hidden[mx.arange(ids.shape[0]), lengths - 1]  # (B, H)
    if text.args.tie_word_embeddings:
        vocab = text.model.embed_tokens.as_linear(last)
    else:
        vocab = text.lm_head(last)
    z = mx.take_along_axis(vocab, label_ids, axis=1).astype(mx.float32)
    return mx.where(kmask, z, mx.array(-1e9, mx.float32))


def _log_softmax(mx: Any, logits):
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def loss_fn(mx: Any, model: Any, ids, lengths, label_ids, kmask, target, brier_weight: float):
    logp = _log_softmax(mx, readout(mx, model, ids, lengths, label_ids, kmask))
    ce = -(target * mx.where(kmask, logp, 0.0)).sum(-1).mean()
    p = mx.exp(logp)
    brier = (((p - target) ** 2) * kmask).sum(-1).mean()
    return ce + brier_weight * brier


def macro_by_source(mx: Any, model: Any, examples: list[Example], batch_tokens: int) -> dict[str, Any]:
    """Accuracy and Brier per source, and their unweighted means: the suite's macro average."""
    by_source: dict[str, list[Example]] = {}
    for example in examples:
        by_source.setdefault(example.source, []).append(example)
    per = {source: evaluate(mx, model, group, batch_tokens) for source, group in sorted(by_source.items())}
    return {"n": len(examples), "sources": len(per),
            "macro_acc": float(np.mean([r["acc"] for r in per.values()])),
            "macro_brier": float(np.mean([r["brier"] for r in per.values()])),
            "per_source": {source: round(r["acc"], 4) for source, r in per.items()}}


def evaluate(mx: Any, model: Any, examples: list[Example], batch_tokens: int) -> dict[str, float]:
    from moelars.calibration import ece

    hits, conf, brier = [], [], 0.0
    for batch in batches(examples, batch_tokens, rng=None):
        ids, lengths, label_ids, kmask, target = collate(mx, batch)
        p = mx.exp(_log_softmax(mx, readout(mx, model, ids, lengths, label_ids, kmask)))
        mx.eval(p)
        p_np, t_np, m_np = np.asarray(p), np.asarray(target), np.asarray(kmask)
        hits.extend((p_np.argmax(-1) == t_np.argmax(-1)).tolist())
        conf.extend(p_np.max(-1).tolist())
        brier += float((((p_np - t_np) ** 2) * m_np).sum())
    n = len(hits)
    if not n:
        return {"n": 0}
    return {"n": n, "acc": float(np.mean(hits)), "ece": float(ece(np.asarray(conf), np.asarray(hits, float))),
            "brier": brier / n}


# Module paths inside a decoder block, for `--keys`. "all" leaves the choice to mlx-lm, which
# adapts every linear layer in the block; on a mixture-of-experts model that includes the
# router and every expert (hundreds of millions of parameters on Qwen3-30B-A3B at rank 8).
KEY_PRESETS: dict[str, list[str] | None] = {
    "all": None,
    "attn": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"],
    "attn+experts": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                     "mlp.switch_mlp.gate_proj", "mlp.switch_mlp.up_proj", "mlp.switch_mlp.down_proj"],
}


def stop_router_index_gradients() -> None:
    """Make mlx-lm's Qwen3-MoE block trainable. It routes with argpartition indices that carry
    no stop_gradient, so any backward pass through it fails with "Cannot calculate VJP with
    respect to indices"; mlx-lm's other MoE models stop that gradient. The forward pass is
    unchanged. Idempotent, and a no-op when the module is missing."""
    try:
        from mlx_lm.models import qwen3_moe
    except ImportError:
        return
    block = qwen3_moe.Qwen3MoeSparseMoeBlock
    if getattr(block, "_moelars_routes_without_index_gradients", False):
        return

    def __call__(self, x):
        import mlx.core as mx

        gates = mx.softmax(self.gate(x), axis=-1, precise=True)
        k = self.top_k
        inds = mx.stop_gradient(mx.argpartition(gates, kth=-k, axis=-1)[..., -k:])
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / mx.sum(scores, axis=-1, keepdims=True)
        y = self.switch_mlp(x, inds)
        return (y * scores[..., None]).sum(axis=-2)

    block.__call__ = __call__
    block._moelars_routes_without_index_gradients = True


def add_lora(model: Any, num_layers: int, rank: int, scale: float, dropout: float,
             keys: list[str] | None = None) -> dict[str, Any]:
    """Freeze the backbone and add LoRA to the last `num_layers` blocks (-1 for all). Returns the adapter config.

    `keys` restricts LoRA to those module paths within each block; None adapts every linear layer."""
    from mlx_lm.tuner.utils import linear_to_lora_layers

    total = len(model.layers)
    num_layers = total if num_layers < 0 else min(num_layers, total)
    params: dict[str, Any] = {"rank": rank, "scale": scale, "dropout": dropout}
    if keys is not None:
        params["keys"] = list(keys)  # mlx-lm's load_adapters reads them back from the config
    model.freeze()
    linear_to_lora_layers(model, num_layers, params)
    return {"fine_tune_type": "lora", "num_layers": num_layers, "lora_parameters": params}


def save_adapter(model: Any, out: Path, config: dict[str, Any]) -> None:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    out.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out / "adapters.safetensors"), dict(tree_flatten(model.trainable_parameters())))
    (out / "adapter_config.json").write_text(json.dumps(config, indent=2))


def train(
    backend: Any,
    train_records: list[Record],
    heldout_records: list[Record],
    out: str | Path,
    epochs: int = 1,
    lr: float = 2e-5,
    rank: int = 8,
    scale: float = 20.0,
    dropout: float = 0.0,
    num_layers: int = -1,
    brier_weight: float = 1.0,
    batch_tokens: int = 4096,
    max_tokens: int = 2048,
    eval_every: int = 500,
    grad_checkpoint: bool = True,
    seed: int = 0,
    meta: dict[str, Any] | None = None,
    keys: list[str] | None = None,
    select_records: list[Record] | None = None,
) -> list[dict[str, Any]]:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten

    mx.random.seed(seed)
    rng = random.Random(seed)
    out = Path(out)
    model = backend.model
    # Labels come in a fixed canonical order, so the first K are the ones the engine uses for
    # a K-option question; only as many as the widest record are needed.
    everything = [*train_records, *heldout_records, *(select_records or [])]
    labels = assign_labels(max(len(r.options) for r in everything), backend.is_single_token)
    stop_router_index_gradients()
    config = add_lora(model, num_layers, rank, scale, dropout, keys)
    config["moelars"] = {**(meta or {}), "lr": lr, "epochs": epochs, "brier_weight": brier_weight,
                        "batch_tokens": batch_tokens, "max_tokens": max_tokens, "seed": seed}
    if grad_checkpoint:
        # Recompute each block's activations in the backward pass instead of keeping them;
        # patches the block class, so only do it in a training process.
        from mlx_lm.tuner.trainer import grad_checkpoint as checkpoint_blocks

        checkpoint_blocks(model.layers[0])
    trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    # LoRA's B starts at zero, so this is the backbone exactly: the adapter to fall back to.
    untrained = [(k, mx.array(v)) for k, v in tree_flatten(model.trainable_parameters())]
    stale = out / "adapters.safetensors"
    if stale.exists():
        stale.unlink()  # an adapter from an earlier run must not pass for this run's selection
    print(f"trainable parameters: {trainable:,} across {config['num_layers']} blocks", flush=True)

    heldout, skipped_heldout = presentations(backend, heldout_records, labels, rng=None, max_tokens=max_tokens)
    select, _ = presentations(backend, select_records or [], labels, rng=None, max_tokens=max_tokens)
    if select:
        print(f"selecting on macro accuracy over {len(select)} rows from "
              f"{len({e.source for e in select})} sources", flush=True)
    first_epoch, skipped = presentations(backend, train_records, labels, rng=rng, max_tokens=max_tokens)
    steps_per_epoch = len(batches(first_epoch, batch_tokens, rng=None))
    total_steps = max(1, steps_per_epoch * epochs)
    warmup = min(100, total_steps // 10)
    schedule = optim.join_schedules(
        [optim.linear_schedule(lr / 100, lr, max(warmup, 1)), optim.cosine_decay(lr, max(total_steps - warmup, 1))],
        [max(warmup, 1)],
    )
    optimizer = optim.AdamW(learning_rate=schedule, weight_decay=0.0)
    print(f"train {len(first_epoch)} presentations ({skipped} over {max_tokens} tokens), heldout {len(heldout)} "
          f"({skipped_heldout} skipped), {steps_per_epoch} steps per epoch", flush=True)

    def step_loss(model, ids, lengths, label_ids, kmask, target):
        return loss_fn(mx, model, ids, lengths, label_ids, kmask, target, brier_weight)

    grad_fn = nn.value_and_grad(model, step_loss)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    baseline = evaluate(mx, model, heldout, batch_tokens) if heldout else None
    baseline_select = macro_by_source(mx, model, select, batch_tokens) if select else None
    print(json.dumps({"step": 0, "heldout": baseline, "select": baseline_select}), flush=True)

    def score(entry: dict[str, Any] | None, select_entry: dict[str, Any] | None) -> float | None:
        """Higher is better: macro accuracy on the selection rows, else minus held-out Brier."""
        if select:
            return select_entry["macro_acc"] if select_entry else None
        return -entry["brier"] if entry and entry.get("n") else None

    best = score(baseline, baseline_select)
    step, running, seen = 0, 0.0, 0

    def checkpoint(epoch: int) -> None:
        nonlocal best
        entry: dict[str, Any] = {"step": step, "epoch": epoch, "loss": running / max(seen, 1),
                                 "minutes": round((time.perf_counter() - started) / 60, 1),
                                 "lr": float(optimizer.learning_rate),
                                 "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2)}
        if heldout:
            entry["heldout"] = evaluate(mx, model, heldout, batch_tokens)
        if select:
            entry["select"] = macro_by_source(mx, model, select, batch_tokens)
        current = score(entry.get("heldout"), entry.get("select"))
        improved = current is None or best is None or current > best
        config["moelars"]["improved"] = True
        save_adapter(model, out.parent / f"{out.name}.steps" / f"step-{step}", config)
        if improved:
            best = current
            save_adapter(model, out, config)
            entry["selected"] = True
        history.append(entry)
        (out.parent / f"{out.name}.history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps(entry), flush=True)

    for epoch in range(epochs):
        examples = first_epoch if epoch == 0 else presentations(
            backend, train_records, labels, rng=rng, max_tokens=max_tokens)[0]
        model.train()
        for batch in batches(examples, batch_tokens, rng=rng):
            loss, grads = grad_fn(model, *collate(mx, batch))
            grads, _ = optim.clip_grad_norm(grads, max_norm=1.0)
            optimizer.update(model, grads)
            mx.eval(model.trainable_parameters(), optimizer.state, loss)
            running, seen, step = running + float(loss), seen + 1, step + 1
            if eval_every and step % eval_every == 0:
                model.eval()
                checkpoint(epoch + 1)
                running, seen = 0.0, 0
                model.train()
        model.eval()
        if not eval_every or step % eval_every:
            checkpoint(epoch + 1)
    if (out / "adapters.safetensors").exists():
        # Leave the model at the selected checkpoint, as the head trainer does.
        model.load_weights(str(out / "adapters.safetensors"), strict=False)
    else:
        print("no checkpoint beat the untrained model; saving the untrained adapter (identity), "
              f"last trained state in {out.name}.last/", flush=True)
        config["moelars"]["improved"] = False
        save_adapter(model, out.parent / f"{out.name}.last", config)
        model.load_weights(untrained, strict=False)
        save_adapter(model, out, config)
    return history


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--records", nargs="+", required=True)
    parser.add_argument("--limit", type=int, default=20000, help="records sampled across --records")
    parser.add_argument("--max-options", type=int, default=160, help="skip records with more options")
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--holdout-sources", default=None,
                        help="comma list of sources to hold out, instead of a seeded fraction of them")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5, help="peak; LoRA's scale multiplies every update")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--scale", type=float, default=20.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num-layers", type=int, default=-1, help="blocks to adapt, counted from the top; -1 for all")
    parser.add_argument("--brier-weight", type=float, default=1.0)
    parser.add_argument("--batch-tokens", type=int, default=4096, help="padded tokens per step")
    parser.add_argument("--max-tokens", type=int, default=2048, help="skip prompts longer than this")
    parser.add_argument("--eval-every", type=int, default=500, help="steps between held-out evaluations")
    parser.add_argument("--no-grad-checkpoint", action="store_true", help="keep activations; faster, more memory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keys", choices=sorted(KEY_PRESETS), default="all",
                        help="which layers in each block get LoRA; 'attn' or 'attn+experts' for MoE models")
    parser.add_argument("--select-records", nargs="+", default=None,
                        help="choose the checkpoint by macro accuracy over these records' sources "
                             "(e.g. data/train/jev-bench.validation.jsonl) instead of held-out Brier")
    parser.add_argument("--select-per-source", type=int, default=100, help="selection rows kept per source")
    parser.add_argument("--out", default="checkpoints/lora")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    records: list[Record] = []
    for path in args.records:
        records.extend(read_records(path))
    records = [r for r in records if len(r.options) <= args.max_options]
    # Same sampling and split as `moelars.train.extract` with the same seed, so the adapter
    # and a head trained after it hold out the same sources.
    rng.shuffle(records)
    records = records[: args.limit]
    held = [x.strip() for x in args.holdout_sources.split(",") if x.strip()] if args.holdout_sources else None
    train_records, heldout_records = split_by_group(records, args.holdout_fraction, seed=args.seed,
                                                    sources_to_hold=held)
    print(f"records: {len(records)} -> train {len(train_records)} / heldout {len(heldout_records)} "
          f"(held-out sources: {sorted({r.source for r in heldout_records})[:8]}...)", flush=True)

    select_records = None
    if args.select_records:
        # A selection row whose state is also a training state would reward memorizing it.
        seen_states = {json.dumps(r.state, sort_keys=True, default=str) for r in records}
        per_source: dict[str, list[Record]] = {}
        for path in args.select_records:
            for r in read_records(path):
                if len(r.options) <= args.max_options and \
                        json.dumps(r.state, sort_keys=True, default=str) not in seen_states:
                    per_source.setdefault(r.source, []).append(r)
        select_records = [r for group in per_source.values() for r in group[: args.select_per_source]]

    from moelars.backends.mlx import MLXBackend

    backend = MLXBackend(args.model)
    train(backend, train_records, heldout_records, args.out, epochs=args.epochs, lr=args.lr, rank=args.rank,
          scale=args.scale, dropout=args.dropout, num_layers=args.num_layers, brier_weight=args.brier_weight,
          batch_tokens=args.batch_tokens, max_tokens=args.max_tokens, eval_every=args.eval_every,
          grad_checkpoint=not args.no_grad_checkpoint, seed=args.seed, keys=KEY_PRESETS[args.keys],
          select_records=select_records,
          meta={"model": args.model, "keys": args.keys, "records": args.records, "limit": args.limit,
                "holdout_fraction": args.holdout_fraction,
                "heldout_sources": sorted({r.source for r in heldout_records}),
                "selection": {"records": args.select_records, "per_source": args.select_per_source}
                if args.select_records else None})
    return 0


if __name__ == "__main__":
    sys.exit(main())
