"""Build the Tier B training corpus from the clean public sources.

    uv run python -m lars.train.build --out data/train --max-per-source 20000

Writes one JSONL per source plus `manifest.json`. Requires `pip install lars-engine[evals]`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from lars.train.data import from_jev_bench, from_open_jev, from_tasksource_jev, write_records

JEV_BENCH_CONFIGS = [
    "banking77", "boolq", "sst5", "clinc150", "massive", "ledgar", "go_emotions", "mmlu", "arc_challenge",
    "mnli", "yelp5", "helpsteer2_helpfulness", "helpsteer2_verbosity", "stsb", "measuring_hate_speech",
    "fever_evidence", "paws", "civil_comments", "sms_spam", "strategyqa_closed", "strategyqa_grounded",
]


def _take(iterator, limit: int):
    for i, item in enumerate(iterator):
        if limit and i >= limit:
            return
        yield item


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/train")
    parser.add_argument("--max-per-source", type=int, default=20000)
    parser.add_argument("--jev-bench-per-config", type=int, default=1000)
    parser.add_argument("--skip", default="", help="comma list of: open-jev, tasksource, jev-bench")
    args = parser.parse_args()
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    from datasets import load_dataset

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}

    if "open-jev" not in skip:
        ds = load_dataset("ZefanCai/Open-Jev", "release-v2-redistributable")
        for split in ["train", "calibration", "validation", "test", "ood"]:
            records = list(_take(from_open_jev(ds[split], source=f"open-jev/{split}"), args.max_per_source))
            count = write_records(records, out / f"open-jev.{split}.jsonl")
            manifest[f"open-jev.{split}"] = {"rows": count, "kinds": dict(Counter(r.kind for r in records))}
            print(f"open-jev {split}: {count}", flush=True)

    if "tasksource" not in skip:
        ds = load_dataset("tasksource/tasksource-jev")
        for split in ["train", "validation", "test"]:
            records = list(_take(from_tasksource_jev(ds[split]), args.max_per_source))
            count = write_records(records, out / f"tasksource-jev.{split}.jsonl")
            manifest[f"tasksource-jev.{split}"] = {
                "rows": count, "sources": len({r.source for r in records}),
            }
            print(f"tasksource-jev {split}: {count}", flush=True)

    if "jev-bench" not in skip:
        for split in ["train", "validation"]:
            all_records = []
            for config in JEV_BENCH_CONFIGS:
                try:
                    rows = load_dataset("Praveenrajus/jev-bench", config, split=split)
                except Exception as error:  # noqa: BLE001 - some configs lack a split
                    print(f"jev-bench {config} {split}: skipped ({error})", flush=True)
                    continue
                all_records.extend(_take(from_jev_bench(rows, config), args.jev_bench_per_config))
            count = write_records(all_records, out / f"jev-bench.{split}.jsonl")
            manifest[f"jev-bench.{split}"] = {"rows": count, "kinds": dict(Counter(r.kind for r in all_records))}
            print(f"jev-bench {split}: {count}", flush=True)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote manifest to {out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
