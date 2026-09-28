"""Corpus E: corpus D diluted with general Open-Jev rows back to corpus C's share of jev-bench rows.

    uv run --extra evals python scripts/build_corpus_e.py            # writes data/train-e/

Corpus D (jev-bench sources filled to 1,000 rows) raised the suite by 1.1 points but lost 2.1 on
the configs it does not train on, with jev-bench-format rows at 73% of the corpus against 40% in
corpus C. This keeps corpus D's jev-bench rows and adds Open-Jev train rows (the same
redistributable release corpus C already uses, so no new licenses) until jev-bench is
`--jev-share` of the total. New rows are taken in dataset order after the ones already in the
corpus, skipping any whose state is in an eval split, or whose state and question together are
already in the corpus (Open-Jev asks several questions about one state; each is a separate row).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from moelars.train.data import from_open_jev, read_records, write_records

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_corpus_c import STREAMS, eval_states  # noqa: E402
from build_corpus_d import state_key  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/train-d")
    parser.add_argument("--out", default="data/train-e")
    parser.add_argument("--jev-share", type=float, default=0.40)
    parser.add_argument("--eval-data", default="evals/data")
    args = parser.parse_args()
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    streams = {stream: list(read_records(src / f"{stream}.train.jsonl")) for stream in STREAMS}
    jev = len(streams["jev-bench"])
    total = sum(len(v) for v in streams.values())
    need = max(0, round(jev / args.jev_share) - total)
    have = {(state_key(r.state), r.question) for records in streams.values() for r in records}
    evals: set[str] = set()
    for s in eval_states(Path(args.eval_data)):
        evals.add(state_key(s))
        if s[:1] in "{[\"":
            evals.add(state_key(json.loads(s)))
    have_ids = {r.id for r in streams["open-jev"]}

    from datasets import load_dataset

    rows = load_dataset("ZefanCai/Open-Jev", "release-v2-redistributable", split="train")
    added = 0
    for record in from_open_jev(rows, source="open-jev/train"):
        if added >= need:
            break
        key = (state_key(record.state), record.question)
        if record.id in have_ids or key in have or key[0] in evals:
            continue
        have.add(key)
        streams["open-jev"].append(record)
        added += 1
    if added < need:
        raise SystemExit(f"only {added} new Open-Jev rows for {need}")

    manifest = {"from": str(src), "jev_share": args.jev_share, "added": {"open-jev/train": added}}
    for stream in STREAMS:
        count = write_records(streams[stream], out / f"{stream}.train.jsonl")
        manifest[f"{stream}.train"] = {"rows": count, "kinds": dict(Counter(r.kind for r in streams[stream]))}
    manifest["total_rows"] = sum(len(v) for v in streams.values())
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"total {manifest['total_rows']}, jev-bench {jev} ({jev / manifest['total_rows']:.0%}), "
          f"added {added} Open-Jev rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
