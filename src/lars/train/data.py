"""Unified training records and converters from the clean public sources.

One record is one typed decision:

    {"id", "source", "kind", "state", "question", "options", "target"}

- `kind`: noul | choice | score
- `options`: for noul always ["yes", "no"]; for choice a list of option keys, with an
  optional parallel `descriptions` list; for score the ordered level descriptions
- `target`: probability distribution aligned with `options` (sums to 1)

Sources (none labeled by Jev; per-source licenses in DESIGN.md section 8.1, where some are
non-commercial or unverified; `scripts/build_corpus_c.py` drops those for published weights):

- ZefanCai/Open-Jev (CC0). Options come as "key: description" strings for choice, as
  ["no", "yes"] for noul with target [P(no), P(yes)], and as level texts for score.
- tasksource/tasksource-jev (per upstream). Choice only, one-hot targets.
- Praveenrajus/jev-bench train and validation splits (mixed licenses, see its manifest).
  Rows are already in wire format; soft labels are used as targets when present.

The converters are deterministic and never look at held-out fields.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

NOUL_OPTIONS = ["yes", "no"]


@dataclass
class Record:
    id: str
    source: str
    kind: str
    state: Any
    question: str
    options: list[str]
    target: list[float]
    descriptions: list[str | None] = field(default_factory=list)

    def validate(self) -> None:
        if self.kind not in {"noul", "choice", "score"}:
            raise ValueError(f"{self.id}: bad kind {self.kind}")
        if len(self.options) != len(self.target):
            raise ValueError(f"{self.id}: options/target length mismatch")
        if self.kind == "noul" and self.options != NOUL_OPTIONS:
            raise ValueError(f"{self.id}: noul options must be {NOUL_OPTIONS}")
        total = sum(self.target)
        if not 0.99 <= total <= 1.01:
            raise ValueError(f"{self.id}: target sums to {total:.3f}")
        if self.descriptions and len(self.descriptions) != len(self.options):
            raise ValueError(f"{self.id}: descriptions length mismatch")

    def to_question(self) -> dict[str, Any]:
        """The wire-format question this record asks, for rendering with the engine."""
        if self.kind == "noul":
            return {"type": "noul", "instructions": self.question}
        if self.kind == "choice":
            criteria = {
                key: (self.descriptions[i] if self.descriptions else None) for i, key in enumerate(self.options)
            }
            return {"type": "choice", "instructions": self.question, "criteria": criteria}
        return {"type": "score", "instructions": self.question, "criteria": list(self.options)}


def _stable_id(*parts: str) -> str:
    return hashlib.sha1("\u0001".join(parts).encode("utf-8")).hexdigest()[:16]


def _normalize(target: list[float]) -> list[float]:
    total = float(sum(target))
    if total <= 0:
        raise ValueError("empty target")
    return [float(t) / total for t in target]


# --------------------------------------------------------------------------- Open-Jev


def _split_key_description(option: str) -> tuple[str, str | None]:
    key, sep, description = option.partition(": ")
    return (key.strip(), description.strip() or None) if sep else (option.strip(), None)


def from_open_jev(rows: Iterable[dict[str, Any]], source: str = "open-jev") -> Iterator[Record]:
    for row in rows:
        kind = row["kind"]
        state = json.loads(row["state_json"]) if "state_json" in row else row["state"]
        options = list(row["options"])
        target = [float(t) for t in row["target"]]
        if kind == "noul":
            # Open-Jev orders noul options ["no", "yes"]; records use ["yes", "no"].
            index = {o.strip().lower(): i for i, o in enumerate(options)}
            target = [target[index["yes"]], target[index["no"]]]
            record = Record(row["id"], source, "noul", state, row["question"], list(NOUL_OPTIONS), _normalize(target))
        elif kind == "choice":
            keys, descriptions = zip(*(_split_key_description(o) for o in options), strict=True)
            record = Record(
                row["id"], source, "choice", state, row["question"], list(keys), _normalize(target), list(descriptions)
            )
        elif kind == "score":
            record = Record(row["id"], source, "score", state, row["question"], options, _normalize(target))
        else:
            continue  # multilabel and other kinds are not modeled yet
        record.validate()
        yield record


# --------------------------------------------------------------------------- tasksource-jev


def from_tasksource_jev(rows: Iterable[dict[str, Any]], source: str = "tasksource-jev") -> Iterator[Record]:
    for row in rows:
        if row.get("kind") != "choice":
            continue
        options = [str(o) for o in row["options"]]
        if len(options) < 2 or len(options) > 255:
            continue
        record = Record(row["id"], f"{source}/{row.get('source', '')}", "choice", row["state"], row["question"],
                        options, _normalize([float(t) for t in row["target"]]))
        record.validate()
        yield record


# --------------------------------------------------------------------------- jev-bench


def from_jev_bench(rows: Iterable[dict[str, Any]], config: str) -> Iterator[Record]:
    for i, row in enumerate(rows):
        question = row["question"] if isinstance(row["question"], dict) else json.loads(row["question"])
        state = row["state"]
        if isinstance(state, str):
            try:
                state = json.loads(state)
            except json.JSONDecodeError:
                pass
        soft = row.get("soft_label")
        if isinstance(soft, str) and soft:
            soft = json.loads(soft)
        kind = question["type"]
        label = str(row["label"])
        rid = _stable_id("jev-bench", config, str(i), label)
        if kind == "noul":
            yes = label in {"1", "true", "yes"}
            target = [1.0 if yes else 0.0, 0.0 if yes else 1.0]
            if isinstance(soft, dict) and "yes" in soft:
                target = _normalize([float(soft.get("yes", 0.0)), float(soft.get("no", 0.0))])
            record = Record(
                rid, f"jev-bench/{config}", "noul", state, question["instructions"], list(NOUL_OPTIONS), target
            )
        elif kind == "choice":
            criteria = question["criteria"]
            keys = list(criteria)
            if isinstance(soft, dict):
                target = _normalize([float(soft.get(k, 0.0)) for k in keys])
            else:
                target = [1.0 if k == label else 0.0 for k in keys]
            descriptions = [criteria[k] if isinstance(criteria[k], str) else None for k in keys]
            record = Record(
                rid, f"jev-bench/{config}", "choice", state, question["instructions"], keys, target, descriptions
            )
        else:
            levels = [str(c) for c in question["criteria"]]
            if isinstance(soft, dict):
                target = _normalize([float(soft.get(str(i), 0.0)) for i in range(len(levels))])
            else:
                target = [1.0 if str(i) == label else 0.0 for i in range(len(levels))]
            record = Record(rid, f"jev-bench/{config}", "score", state, question["instructions"], levels, target)
        record.validate()
        yield record


# --------------------------------------------------------------------------- io


def write_records(records: Iterable[Record], path: str | Path) -> int:
    count = 0
    with Path(path).open("w") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_records(path: str | Path) -> Iterator[Record]:
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield Record(**json.loads(line))


def split_by_group(
    records: list[Record], holdout_fraction: float, seed: int = 0, sources_to_hold: list[str] | None = None
) -> tuple[list[Record], list[Record]]:
    """Hold out whole sources, never individual rows, so generalization is measured honestly.

    By default a seeded shuffle picks `holdout_fraction` of the sources, so the choice moves
    with the seed and with the source list. `sources_to_hold` names them instead, which keeps
    the held-out set fixed across seeds and corpus revisions.
    """
    if sources_to_hold is not None:
        missing = sorted(set(sources_to_hold) - {r.source for r in records})
        if missing:
            raise ValueError(f"held-out sources not in the records: {missing}")
        held = set(sources_to_hold)
        return [r for r in records if r.source not in held], [r for r in records if r.source in held]
    sources = sorted({r.source for r in records})
    rng = random.Random(seed)
    rng.shuffle(sources)
    held = set(sources[: max(1, int(len(sources) * holdout_fraction))]) if len(sources) > 1 else set()
    return [r for r in records if r.source not in held], [r for r in records if r.source in held]
