"""Pointer residual head and its training loop, in MLX.

    logits_i = exp(log_s) * z_i + (W_q h_ans) . (W_k h_opt_i) / sqrt(rank) + b_kind

W_q is zero-initialized so training starts exactly at the backbone's own answer.
Loss = cross-entropy + Brier on the presented order, plus permutation-KL between the
canonical presentation and each shuffled one after realignment.

    uv run python -m lars.train.residual --train data/features/train.npz --heldout data/features/heldout.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

KINDS = {"noul": 0, "choice": 1, "score": 2}


def _mlx():
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    return mx, nn, optim


def _check_manifest(path: Path, presentations: int) -> None:
    """Refuse a shard that the `manifest.json` beside it does not describe (a stale leftover)."""
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.exists():
        return
    entry = json.loads(manifest_path.read_text()).get(path.stem)
    if not isinstance(entry, dict) or entry.get("presentations") != presentations:
        raise ValueError(f"{path} is not the {path.stem!r} shard in {manifest_path} "
                         "(left over from an earlier extraction?); re-extract into a fresh directory")


class Shard:
    def __init__(self, path: str | Path):
        from lars.heads import rms_normalize

        data = np.load(path)
        _check_manifest(Path(path), len(data["z"]))
        self.h_ans = rms_normalize(data["h_ans"])
        self.h_opt = rms_normalize(data["h_opt"])
        self.z = data["z"]
        self.target = data["target"]
        self.mask = data["mask"]
        self.perm = data["perm"]
        self.ids = data["ids"]
        self.kinds = np.asarray([KINDS[k] for k in data["kinds"]], np.int32)
        self.n = len(self.ids)
        # canonical (perm == identity) row index per record id, for the permutation loss
        self.base_index: dict[str, int] = {}
        for i in range(self.n):
            k = int(self.mask[i].sum())
            if list(self.perm[i, :k]) == list(range(k)):
                self.base_index[str(self.ids[i])] = i
        self.pairs = np.asarray(
            [(self.base_index[str(rid)], i) for i, rid in enumerate(self.ids)
             if str(rid) in self.base_index and self.base_index[str(rid)] != i],
            np.int64,
        ).reshape(-1, 2)


def build_head(proj_dim: int, rank: int):
    mx, nn, _ = _mlx()

    class PointerHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q = nn.Linear(proj_dim, rank, bias=False)
            self.k = nn.Linear(proj_dim, rank, bias=False)
            self.q.weight = mx.zeros_like(self.q.weight)  # start exactly at the backbone
            self.log_s = mx.zeros((3,))  # per-kind scale on the backbone logits
            self.bias = mx.zeros((3,))  # per-kind bias on the yes label (noul) / unused otherwise

        def __call__(self, h_ans, h_opt, z, mask, kind):
            q = self.q(h_ans)  # (B, r)
            k = self.k(h_opt)  # (B, K, r)
            residual = (k * q[:, None, :]).sum(-1) / (rank**0.5)
            scale = mx.exp(self.log_s)[kind][:, None]
            logits = scale * z + residual
            yes_bias = mx.where(kind == 0, self.bias[0], 0.0)[:, None]
            first = mx.concatenate([yes_bias, mx.zeros_like(logits[:, 1:])], axis=1)
            logits = logits + first
            return mx.where(mask, logits, mx.array(-1e9))

    return PointerHead()


def _batch(mx, shard: Shard, idx: np.ndarray):
    return (
        mx.array(shard.h_ans[idx]),
        mx.array(shard.h_opt[idx]),
        mx.array(shard.z[idx]),
        mx.array(shard.mask[idx]),
        mx.array(shard.kinds[idx]),
        mx.array(shard.target[idx]),
    )


def _log_softmax(mx, logits):
    m = logits.max(axis=-1, keepdims=True)
    return logits - m - mx.log(mx.exp(logits - m).sum(axis=-1, keepdims=True))


def loss_fn(mx, head, batch, brier_weight: float):
    h_ans, h_opt, z, mask, kind, target = batch
    logp = _log_softmax(mx, head(h_ans, h_opt, z, mask, kind))
    ce = -(target * logp).sum(-1).mean()
    p = mx.exp(logp)
    brier = (((p - target) ** 2) * mask).sum(-1).mean()
    return ce + brier_weight * brier


def perm_loss_fn(mx, head, base_batch, shuf_batch, perm_shuf):
    """KL between the canonical distribution and the shuffled one realigned to canonical order."""
    b_h, b_o, b_z, b_m, b_k, _ = base_batch
    s_h, s_o, s_z, s_m, s_k, _ = shuf_batch
    p_base = mx.exp(_log_softmax(mx, head(b_h, b_o, b_z, b_m, b_k)))
    logp_shuf = _log_softmax(mx, head(s_h, s_o, s_z, s_m, s_k))
    # Padded positions hold log-probabilities near -1e9; zero them before the scatter so
    # they never land on a real option. Realign: shuffled position j holds canonical
    # option perm[j].
    logp_shuf = mx.where(s_m, logp_shuf, mx.zeros_like(logp_shuf))
    rows = mx.arange(perm_shuf.shape[0])[:, None]
    safe_perm = mx.maximum(perm_shuf, 0)
    aligned = mx.zeros_like(logp_shuf)
    aligned = aligned.at[rows, safe_perm].add(logp_shuf)
    kl = (p_base * (mx.log(p_base + 1e-9) - aligned) * b_m).sum(-1).mean()
    return kl


def evaluate(mx, head, shard: Shard, batch_size: int = 256) -> dict[str, float]:
    correct, conf, brier, n = 0.0, [], 0.0, 0
    hits = []
    for start in range(0, shard.n, batch_size):
        idx = np.arange(start, min(start + batch_size, shard.n))
        h_ans, h_opt, z, mask, kind, target = _batch(mx, shard, idx)
        p = mx.exp(_log_softmax(mx, head(h_ans, h_opt, z, mask, kind)))
        p_np = np.asarray(p)
        t_np = shard.target[idx]
        pred = p_np.argmax(-1)
        gold = t_np.argmax(-1)
        hit = pred == gold
        correct += hit.sum()
        hits.extend(hit.tolist())
        conf.extend(p_np.max(-1).tolist())
        brier += (((p_np - t_np) ** 2) * shard.mask[idx]).sum(-1).sum()
        n += len(idx)
    from lars.calibration import ece

    return {
        "n": int(n),
        "acc": float(correct / n),
        "ece": float(ece(np.asarray(conf), np.asarray(hits, float))),
        "brier": float(brier / n),
    }


def baseline(mx, shard: Shard) -> dict[str, float]:
    """Metrics for the backbone alone (scale 1, no residual)."""
    head = build_head(shard.h_ans.shape[1], rank=1)
    return evaluate(mx, head, shard)


def train(
    train_shard: Shard,
    heldout: Shard | None,
    rank: int = 64,
    epochs: int = 3,
    batch_size: int = 64,
    lr: float = 1e-3,
    brier_weight: float = 1.0,
    perm_weight: float = 0.5,
    seed: int = 0,
    out: str | Path | None = None,
):
    mx, nn, optim = _mlx()
    mx.random.seed(seed)
    rng = np.random.default_rng(seed)
    head = build_head(train_shard.h_ans.shape[1], rank)
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=1e-4)

    def step_loss(head, batch, base_batch, shuf_batch, perm_shuf):
        loss = loss_fn(mx, head, batch, brier_weight)
        if base_batch is not None:
            loss = loss + perm_weight * perm_loss_fn(mx, head, base_batch, shuf_batch, perm_shuf)
        return loss

    grad_fn = nn.value_and_grad(head, step_loss)
    history = []
    from mlx.utils import tree_flatten, tree_unflatten

    def snapshot():
        return tree_unflatten([(k, mx.array(v)) for k, v in tree_flatten(head.parameters())])

    base_train = baseline(mx, train_shard)
    print("baseline train:", base_train, flush=True)
    base_heldout = baseline(mx, heldout) if heldout is not None else None
    if base_heldout is not None:
        print("baseline heldout:", base_heldout, flush=True)
    # The untrained head is the backbone exactly, so it is the checkpoint to beat: an epoch is
    # kept only if it improves on no head at all.
    best_score = -(base_heldout or base_train)["brier"]
    best_params = snapshot()
    for epoch in range(epochs):
        order = rng.permutation(train_shard.n)
        started, total, steps = time.perf_counter(), 0.0, 0
        for start in range(0, train_shard.n, batch_size):
            idx = order[start : start + batch_size]
            batch = _batch(mx, train_shard, idx)
            base_batch = shuf_batch = perm_shuf = None
            if train_shard.pairs.size and perm_weight > 0:
                pair_idx = train_shard.pairs[rng.integers(0, len(train_shard.pairs), size=min(batch_size, 32))]
                base_batch = _batch(mx, train_shard, pair_idx[:, 0])
                shuf_batch = _batch(mx, train_shard, pair_idx[:, 1])
                perm_shuf = mx.array(train_shard.perm[pair_idx[:, 1]])
            loss, grads = grad_fn(head, batch, base_batch, shuf_batch, perm_shuf)
            grads, _ = optim.clip_grad_norm(grads, max_norm=1.0)
            optimizer.update(head, grads)
            mx.eval(head.parameters(), optimizer.state)
            total += float(loss)
            steps += 1
        elapsed = round(time.perf_counter() - started, 1)
        metrics: dict = {"epoch": epoch + 1, "loss": total / max(steps, 1), "seconds": elapsed}
        metrics["train"] = evaluate(mx, head, train_shard)
        if heldout is not None:
            metrics["heldout"] = evaluate(mx, head, heldout)
        # Select by held-out Brier when held-out sources exist; in-distribution gains are
        # cheap, generalization is what we are buying.
        score = -metrics["heldout"]["brier"] if heldout is not None else -metrics["train"]["brier"]
        if score > best_score:
            best_score = score
            best_params = snapshot()
            metrics["selected"] = True
        history.append(metrics)
        print(json.dumps(metrics), flush=True)
    if not any(entry.get("selected") for entry in history):
        print("no epoch beat the backbone alone; saving the untrained head (identity)", flush=True)
    head.update(best_params)
    if out:
        save_head(head, out, history)
    return head, history


def save_head(head, out: str | Path, history: list | None = None) -> None:
    """Save as npz with flat keys (q.weight, k.weight, log_s, bias) for the numpy serving scorer."""
    from mlx.utils import tree_flatten

    mx, _, _ = _mlx()
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = {name: np.asarray(value.astype(mx.float32)) for name, value in tree_flatten(head.parameters())}
    np.savez(path.with_suffix(".npz"), **flat)
    if history is not None:
        path.with_suffix(".history.json").write_text(json.dumps(history, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--heldout", default=None)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--brier-weight", type=float, default=1.0)
    parser.add_argument("--perm-weight", type=float, default=0.5)
    parser.add_argument("--out", default="checkpoints/pointer_head.npz")
    args = parser.parse_args()
    train_shard = Shard(args.train)
    heldout = Shard(args.heldout) if args.heldout else None
    train(train_shard, heldout, args.rank, args.epochs, args.batch_size, args.lr, args.brier_weight, args.perm_weight,
          out=args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
