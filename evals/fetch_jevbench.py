"""Convert a jev-bench config into LARS eval JSONL. Requires `pip install lars-engine[evals]`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="e.g. banking77, boolq, sst5, chaosnli")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    from datasets import load_dataset

    rows = load_dataset("Praveenrajus/jev-bench", args.config, split=args.split)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out.open("w") as handle:
        for row in rows:
            record = {"state": row["state"], "question": row["question"], "label": row["label"]}
            if row.get("soft_label"):
                record["soft_label"] = row["soft_label"]
            handle.write(json.dumps(record) + "\n")
            written += 1
            if args.limit and written >= args.limit:
                break
    print(f"wrote {written} rows to {out}")


if __name__ == "__main__":
    main()
