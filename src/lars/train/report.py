"""Per-source comparison of the backbone alone versus a trained head on a feature shard.

    uv run python -m lars.train.report --shard data/features/test.npz \
        --head checkpoints/pointer_head.npz --sources data/features/sources.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from lars.calibration import ece
from lars.train.residual import Shard, _batch, _log_softmax, _mlx, build_head


def _load_head(mx, path: Path, proj_dim: int):
    data = np.load(path)
    head = build_head(proj_dim, rank=int(data["q.weight"].shape[0]))
    head.q.weight = mx.array(data["q.weight"])
    head.k.weight = mx.array(data["k.weight"])
    head.log_s = mx.array(data["log_s"])
    head.bias = mx.array(data["bias"])
    return head


def _probs(mx, head, shard: Shard, batch_size: int = 256) -> np.ndarray:
    out = np.zeros_like(shard.z)
    for start in range(0, shard.n, batch_size):
        idx = np.arange(start, min(start + batch_size, shard.n))
        h_ans, h_opt, z, mask, kind, _ = _batch(mx, shard, idx)
        out[idx] = np.asarray(mx.exp(_log_softmax(mx, head(h_ans, h_opt, z, mask, kind))))
    return out


def _metrics(p: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    pred, gold = p.argmax(-1), target.argmax(-1)
    hit = (pred == gold).astype(float)
    return {
        "n": int(len(hit)),
        "acc": float(hit.mean()),
        "ece": float(ece(p.max(-1), hit)),
        "brier": float((((p - target) ** 2) * mask).sum(-1).mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--out", default=None, help="write the per-source table as JSON")
    args = parser.parse_args()

    mx, _, _ = _mlx()
    shard = Shard(args.shard)
    sources = json.loads(Path(args.sources).read_text())
    heldout_sources = set(sources.pop("__heldout__", []))
    identity = build_head(shard.h_ans.shape[1], rank=1)
    head = _load_head(mx, Path(args.head), shard.h_ans.shape[1])
    p_base, p_head = _probs(mx, identity, shard), _probs(mx, head, shard)

    by_source: dict[str, list[int]] = defaultdict(list)
    for i, rid in enumerate(shard.ids):
        by_source[sources.get(str(rid), "unknown")].append(i)

    rows = []
    for source, idx in sorted(by_source.items()):
        idx_arr = np.asarray(idx)
        base = _metrics(p_base[idx_arr], shard.target[idx_arr], shard.mask[idx_arr])
        with_head = _metrics(p_head[idx_arr], shard.target[idx_arr], shard.mask[idx_arr])
        rows.append({"source": source, "held_out": source in heldout_sources, "n": base["n"],
                     "base": base, "head": with_head})

    print(f"{'source':38s} {'n':>5s} {'acc base':>9s} {'acc head':>9s} {'ece base':>9s} {'ece head':>9s} "
          f"{'brier b':>8s} {'brier h':>8s}  held-out")
    for r in rows:
        print(f"{r['source'][:38]:38s} {r['n']:5d} {r['base']['acc']:9.3f} {r['head']['acc']:9.3f} "
              f"{r['base']['ece']:9.3f} {r['head']['ece']:9.3f} {r['base']['brier']:8.3f} {r['head']['brier']:8.3f}"
              f"  {'yes' if r['held_out'] else ''}")
    overall = {"base": _metrics(p_base, shard.target, shard.mask), "head": _metrics(p_head, shard.target, shard.mask)}

    def macro(key: str, which: str) -> float:
        return float(np.mean([r[which][key] for r in rows]))

    print()
    for which in ("base", "head"):
        m = overall[which]
        print(f"micro  {which} acc={m['acc']:.3f} ece={m['ece']:.3f} brier={m['brier']:.3f}")
    for which in ("base", "head"):
        print(f"macro  {which} acc={macro('acc', which):.3f} ece={macro('ece', which):.3f} "
              f"brier={macro('brier', which):.3f}")
    if args.out:
        Path(args.out).write_text(json.dumps({"rows": rows, "micro": overall}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
