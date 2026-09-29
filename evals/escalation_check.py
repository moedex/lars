"""What escalating low-confidence answers to a hosted model buys, measured on the suite's rows.

    uv run --extra escalate python evals/escalation_check.py evals/results/rows/<slug> --max-below 0.9

Local answers are the calibrated probabilities a suite run dumped (`run_suite.py --dump-rows`),
so no local model is loaded. Every test and validation row whose top probability is below
`--max-below` is put to the hosted model once (`moelars.escalate.Escalator.pick`, the serving
path), and its picks are cached in `--cache`, so a re-run or a lower threshold costs nothing.
Then, for each threshold up to `--max-below`, rows below it are blended as the server blends
them (`weight * one_hot(pick) + (1 - weight) * local`) and scored the way `evals/cascade.py`
scores its blend: right when the answer's top option is the target's top option. The threshold
is chosen on validation and reported on test, with the share of rows that left the machine.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np

from moelars.escalate import DEFAULT_MODEL, DEFAULT_WEIGHT, Escalator, resolve_api_key
from moelars.evalset import read_examples
from moelars.schema import SystemOneRequest

HERE = Path(__file__).resolve().parent
THRESHOLDS = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


def load(rows_dir: Path, data_dir: Path) -> list[dict]:
    """Every dumped row joined to its example, with the option keys the hosted model sees."""
    items = []
    for dump in sorted(rows_dir.glob("*.json")):
        cfg = dump.stem
        rows = json.loads(dump.read_text())
        for split in ("test", "validation"):
            path = data_dir / f"{cfg}.{split}.jsonl"
            if not rows.get(split) or not path.exists():
                continue
            by_id = {e.id: e for e in read_examples(path)}
            for row in rows[split]:
                example = by_id.get(row["id"])
                if example is None:
                    raise ValueError(f"[{cfg}/{split}] row {row['id']} has no example in {path}")
                items.append({"cfg": cfg, "split": split, "id": row["id"], "keys": row["keys"],
                              "p": np.asarray(row["p"]), "target": np.asarray(row["target"]), "example": example})
    return items


def blended_hit(item: dict, pick: str | None, weight: float) -> tuple[bool, float]:
    p = item["p"]
    if pick is not None:
        p = weight * (np.asarray(item["keys"]) == pick) + (1 - weight) * p
    target = item["target"]
    return bool(target[int(np.argmax(p))] == target.max()), float(np.sum((p - target) ** 2))


async def fetch(items: list[dict], escalator: Escalator, cache: dict, concurrency: int) -> None:
    gate = asyncio.Semaphore(concurrency)
    todo = [i for i in items if i["id"] not in cache]

    async def one(item: dict) -> None:
        async with gate:
            e = item["example"]
            question = SystemOneRequest(state=e.state, questions={"q": e.question}).questions["q"]
            try:
                cache[item["id"]] = await escalator.pick(e.state, question)
            except Exception as error:  # noqa: BLE001 - counted, and the row keeps its local answer
                cache[item["id"]] = None
                print(f"[{item['cfg']}] hosted call failed: {type(error).__name__}", file=sys.stderr)

    await asyncio.gather(*(one(i) for i in todo))


def macro(items: list[dict], cache: dict, below: float, weight: float) -> dict:
    per: dict[str, list[tuple[bool, float, bool]]] = {}
    for item in items:
        escalate = item["p"].max() < below
        pick = cache.get(item["id"]) if escalate else None
        hit, brier = blended_hit(item, pick, weight)
        per.setdefault(item["cfg"], []).append((hit, brier, escalate))
    return {"acc": float(np.mean([np.mean([h for h, _, _ in v]) for v in per.values()])),
            "brier": float(np.mean([np.mean([b for _, b, _ in v]) for v in per.values()])),
            "escalated": float(np.mean([np.mean([e for _, _, e in v]) for v in per.values()])),
            "per_config": {c: float(np.mean([h for h, _, _ in v])) for c, v in per.items()}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("rows", type=Path, help="row-dump directory from run_suite.py --dump-rows")
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--weight", type=float, default=DEFAULT_WEIGHT)
    parser.add_argument("--max-below", type=float, default=0.9, help="rows below this are sent (once, cached)")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    key = resolve_api_key()
    if not key:
        raise SystemExit("no API key (ANTHROPIC_API_KEY, MOELARS_ANTHROPIC_KEY_FILE, ~/.config/moelars/anthropic-key)")
    cache_path = args.cache or HERE / "results" / "escalation" / f"picks-{args.model}-{args.rows.name}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    items = load(args.rows, args.data_dir)
    due = [i for i in items if i["p"].max() < args.max_below]
    print(f"{len(items)} rows; {len(due)} below {args.max_below}; {sum(i['id'] not in cache for i in due)} to send",
          flush=True)
    escalator = Escalator(key, model=args.model)
    try:
        asyncio.run(fetch(due, escalator, cache, args.concurrency))
    finally:
        cache_path.write_text(json.dumps(cache))
    failed = sum(cache.get(i["id"]) is None for i in due)

    thresholds = [t for t in THRESHOLDS if t <= args.max_below]
    split = {s: [i for i in items if i["split"] == s] for s in ("validation", "test")}
    results = {t: {s: macro(split[s], cache, t, args.weight) for s in split} for t in [0.0, *thresholds]}
    hosted_on_due = [blended_hit(i, cache.get(i["id"]), 1.0)[0] for i in due if i["split"] == "test"]
    local_on_due = [blended_hit(i, None, 1.0)[0] for i in due if i["split"] == "test"]
    chosen = max(thresholds, key=lambda t: (round(results[t]["validation"]["acc"], 4), -t))

    print(f"\nhosted model {args.model}, weight {args.weight}; {failed} failed calls\n")
    print("| escalate below | val acc | test acc | test Brier | test rows escalated |")
    print("|---|---|---|---|---|")
    for t, r in results.items():
        name = "never (local only)" if t == 0.0 else f"{t}" + (" (chosen on validation)" if t == chosen else "")
        print(f"| {name} | {r['validation']['acc']:.3f} | {r['test']['acc']:.3f} | {r['test']['brier']:.3f} | "
              f"{r['test']['escalated']:.1%} |")
    print(f"\non the {len(hosted_on_due)} test rows below {args.max_below}: local right {np.mean(local_on_due):.3f}, "
          f"hosted pick right {np.mean(hosted_on_due):.3f}")
    base, best = results[0.0]["test"]["per_config"], results[chosen]["test"]["per_config"]
    moved = sorted(((best[c] - base[c], c) for c in base), reverse=True)
    print("largest per-config changes at the chosen threshold:",
          ", ".join(f"{c} {d:+.3f}" for d, c in moved[:4] + moved[-3:] if abs(d) >= 0.005))
    if args.out:
        args.out.write_text(json.dumps({"model": args.model, "weight": args.weight, "chosen": chosen,
                                        "failed": failed, "results": {str(t): r for t, r in results.items()}},
                                       indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
