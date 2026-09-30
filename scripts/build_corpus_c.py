"""Derive the commercially licensed training corpus (option C, DESIGN.md section 8) from data/train.

    uv run --extra evals python scripts/build_corpus_c.py            # writes data/train-c/

Starts from the committed corpus in data/train, never a rebuild (upstream datasets have
changed since it was built), and:

- drops jev-bench/yelp5 (Yelp Dataset Agreement: no redistribution of derived works),
  jev-bench/sst5 and jev-bench/stsb (licenses not verifiable), and tasksource-jev/anli/*
  (CC BY-NC 4.0);
- adds as many SNLI choice rows (CC BY-SA 4.0, human labels) as ANLI rows were dropped,
  sampled with a fixed seed from tasksource/tasksource-jev at a pinned revision, keeping
  none whose state matches an eval state.

Writes the three training streams plus manifest.json recording what changed.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from lars.train.data import from_tasksource_jev, read_records, write_records

STREAMS = ("open-jev", "jev-bench", "tasksource-jev")
DROP_EXACT = {"jev-bench/yelp5", "jev-bench/sst5", "jev-bench/stsb"}
DROP_PREFIX = ("tasksource-jev/anli/",)
TASKSOURCE_REVISION = "18332e529f10cde1e808966235d3932171d56e69"
BACKFILL_SOURCE = "snli"


def dropped(source: str) -> bool:
    return source in DROP_EXACT or source.startswith(DROP_PREFIX)


def eval_states(data_dir: Path) -> set[str]:
    states = set()
    for path in data_dir.glob("*.jsonl"):
        for line in path.open():
            state = json.loads(line)["state"]
            states.add(state if isinstance(state, str) else json.dumps(state))
    return states


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/train")
    parser.add_argument("--out", default="data/train-c")
    parser.add_argument("--eval-data", default="evals/data")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest: dict = {"from": str(src), "dropped": {}, "added": {}}
    kept: dict[str, list] = {}
    for stream in STREAMS:
        records = list(read_records(src / f"{stream}.train.jsonl"))
        kept[stream] = [r for r in records if not dropped(r.source)]
        for source, n in Counter(r.source for r in records if dropped(r.source)).items():
            manifest["dropped"][source] = n
    n_backfill = sum(n for s, n in manifest["dropped"].items() if s.startswith(DROP_PREFIX))

    from datasets import load_dataset

    rows = load_dataset("tasksource/tasksource-jev", split="train", revision=TASKSOURCE_REVISION)
    rows = rows.filter(lambda r: r["source"] == BACKFILL_SOURCE)
    seen = eval_states(Path(args.eval_data))
    candidates = [r for r in from_tasksource_jev(rows) if r.state not in seen]
    random.Random(args.seed).shuffle(candidates)
    if len(candidates) < n_backfill:
        raise SystemExit(f"only {len(candidates)} {BACKFILL_SOURCE} rows for {n_backfill} to backfill")
    kept["tasksource-jev"].extend(candidates[:n_backfill])
    manifest["added"][f"tasksource-jev/{BACKFILL_SOURCE}"] = {
        "rows": n_backfill, "revision": TASKSOURCE_REVISION, "seed": args.seed, "license": "CC BY-SA 4.0",
    }

    for stream in STREAMS:
        count = write_records(kept[stream], out / f"{stream}.train.jsonl")
        manifest[f"{stream}.train"] = {"rows": count, "sources": dict(Counter(r.source for r in kept[stream]))}
        print(f"{stream}.train: {count}", flush=True)
    manifest["total_rows"] = sum(len(v) for v in kept.values())
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"total {manifest['total_rows']}; dropped {sum(manifest['dropped'].values())}, added {n_backfill}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
