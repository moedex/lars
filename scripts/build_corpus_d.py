"""Corpus D: corpus C with more rows from each licensed, trained jev-bench source.

    uv run --extra evals python scripts/build_corpus_d.py            # writes data/train-d/

Corpus C caps every jev-bench source at 300 rows. This fills each one up to `--per-source`
from its train split at a pinned revision, in dataset order, skipping any row whose state is
already in the corpus, in an eval split (`evals/data`, test and validation), or in the
selection rows (`data/train/jev-bench.validation.jsonl`). Sources corpus C drops for their
licenses (DESIGN.md 8.1) stay out, and so do the held-out sources, which measure
generalization and must not be trained on. open-jev and tasksource-jev are copied unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from lars.train.data import from_jev_bench, read_records, write_records

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_corpus_c import DROP_EXACT, STREAMS, eval_states  # noqa: E402

JEV_BENCH_REVISION = "18f88da81c28c2bec55edc31f63f2afdfba109ea"
# The held-out sources of every corpus-C run (scripts/queue_corpus_c.sh).
HELD_OUT = {"jev-bench/civil_comments", "jev-bench/fever_evidence", "jev-bench/helpsteer2_helpfulness"}


def state_key(state: object) -> str:
    return state if isinstance(state, str) else json.dumps(state, sort_keys=True, default=str)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/train-c")
    parser.add_argument("--out", default="data/train-d")
    parser.add_argument("--per-source", type=int, default=1000)
    parser.add_argument("--eval-data", default="evals/data")
    parser.add_argument("--selection", default="data/train/jev-bench.validation.jsonl")
    args = parser.parse_args()
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    streams = {stream: list(read_records(src / f"{stream}.train.jsonl")) for stream in STREAMS}
    seen = {state_key(r.state) for records in streams.values() for r in records}
    seen |= {state_key(s) for s in eval_states(Path(args.eval_data))}
    # eval_states keeps a string state's JSON form; decode it too so both spellings match.
    seen |= {state_key(json.loads(s)) for s in eval_states(Path(args.eval_data)) if s[:1] in "{[\""}
    seen |= {state_key(r.state) for r in read_records(args.selection)}

    from datasets import load_dataset

    have = Counter(r.source for r in streams["jev-bench"])
    manifest: dict = {"from": str(src), "revision": JEV_BENCH_REVISION, "per_source": args.per_source, "added": {}}
    for source in sorted(have):
        if source in DROP_EXACT or source in HELD_OUT:
            continue
        config = source.split("/", 1)[1]
        rows = load_dataset("Praveenrajus/jev-bench", config, split="train", revision=JEV_BENCH_REVISION)
        added = 0
        for record in from_jev_bench(rows, config):
            if have[source] + added >= args.per_source:
                break
            key = state_key(record.state)
            if key in seen:
                continue
            seen.add(key)
            streams["jev-bench"].append(record)
            added += 1
        manifest["added"][source] = added
        print(f"{source}: {have[source]} + {added}", flush=True)

    for stream in STREAMS:
        count = write_records(streams[stream], out / f"{stream}.train.jsonl")
        manifest[f"{stream}.train"] = {"rows": count, "sources": dict(Counter(r.source for r in streams[stream]))}
    manifest["total_rows"] = sum(len(v) for v in streams.values())
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"total {manifest['total_rows']}, added {sum(manifest['added'].values())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
