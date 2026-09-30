"""Extract pointer-head feature shards with the MLX backend.

    uv run python -m lars.train.extract \
        --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
        --records data/train/open-jev.train.jsonl data/train/jev-bench.train.jsonl \
        --limit 3000 --holdout-fraction 0.2 --out data/features

Writes `train.npz`, `heldout.npz` (whole sources held out), and, with --test-records,
`test.npz`. Also writes `projection.npy` so serving can reproduce the features.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

from lars.backends.mlx import MLXBackend
from lars.evalset import read_examples
from lars.train.data import Record, from_jev_bench, read_records, split_by_group
from lars.train.features import extract, projection, write_shard


def _records_from_eval_jsonl(paths: list[str]) -> list[Record]:
    """Turn evals/data/*.jsonl (jev-bench wire rows) into records for a test shard."""
    records: list[Record] = []
    for path in paths:
        config = Path(path).name.split(".")[0]
        rows = [
            {"state": e.state, "question": e.question, "label": e.label, "soft_label": e.soft_label}
            for e in read_examples(path)
        ]
        records.extend(from_jev_bench(rows, config))
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default=None, help="LoRA adapter directory from lars.train.lora")
    parser.add_argument("--records", nargs="+", required=True)
    parser.add_argument("--test-records", nargs="*", default=[])
    parser.add_argument("--limit", type=int, default=3000, help="records sampled across --records")
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--shuffles", type=int, default=1)
    parser.add_argument("--proj-dim", type=int, default=512)
    parser.add_argument("--max-options", type=int, default=64, help="skip records with more options")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/features")
    parser.add_argument("--sources-only", action="store_true",
                        help="only write sources.json (record id -> source), no model")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    records: list[Record] = []
    for path in args.records:
        records.extend(read_records(path))
    records = [r for r in records if len(r.options) <= args.max_options]
    rng.shuffle(records)
    records = records[: args.limit]
    train, heldout = split_by_group(records, args.holdout_fraction, seed=args.seed)
    print(f"records: {len(records)} -> train {len(train)} / heldout {len(heldout)} "
          f"(held-out sources: {sorted({r.source for r in heldout})[:8]}...)", flush=True)

    test: list[Record] = []
    if args.test_records:
        test = _records_from_eval_jsonl(args.test_records)
        test = [r for r in test if len(r.options) <= args.max_options]
        if args.test_limit:
            rng.shuffle(test)
            test = test[: args.test_limit]
    sources = {r.id: r.source for r in [*train, *heldout, *test]}
    sources["__heldout__"] = sorted({r.source for r in heldout})
    (out / "sources.json").write_text(json.dumps(sources))
    if args.sources_only:
        print(f"wrote {out / 'sources.json'} with {len(sources) - 1} ids")
        return 0

    # Shards and the manifest from an earlier run in this directory would otherwise survive
    # when this run writes fewer subsets; consumers check a shard against the manifest.
    for stale in ("train.npz", "heldout.npz", "test.npz", "manifest.json"):
        for path in (out / stale, out / f"{stale}.json"):
            path.unlink(missing_ok=True)

    backend = MLXBackend(args.model, adapter=args.adapter)
    proj = projection(backend.hidden_size, args.proj_dim, seed=args.seed)
    np.save(out / "projection.npy", proj)

    manifest = {"model": args.model, "adapter": args.adapter, "proj_dim": args.proj_dim, "shuffles": args.shuffles,
                "seed": args.seed}
    for name, subset, shuffles in [("train", train, args.shuffles), ("heldout", heldout, 0)]:
        if not subset:
            continue
        started = time.perf_counter()
        n = write_shard(extract(backend, subset, proj, shuffles=shuffles, seed=args.seed), out / f"{name}.npz")
        elapsed = time.perf_counter() - started
        manifest[name] = {"records": len(subset), "presentations": n, "seconds": round(elapsed, 1),
                          "sources": sorted({r.source for r in subset})}
        print(f"{name}: {n} presentations from {len(subset)} records in {elapsed:.0f}s", flush=True)

    if test:
        started = time.perf_counter()
        n = write_shard(extract(backend, test, proj, shuffles=0, seed=args.seed), out / "test.npz")
        manifest["test"] = {"records": len(test), "presentations": n,
                            "seconds": round(time.perf_counter() - started, 1),
                            "sources": sorted({r.source for r in test})}
        print(f"test: {n} presentations from {len(test)} records", flush=True)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
