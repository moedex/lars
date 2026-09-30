"""Labeled evaluation sets.

One JSON object per line with `state`, `question`, and `label`, optionally `soft_label`.
`state` and `question` may be JSON values or JSON-encoded strings, which is the
jev-bench convention. `label` is the option key for choice, the level index as a
string for score, and "0" or "1" for noul.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lars.calibration import Calibrator, brier, coverage_at_error, ece, fit_fusion, fit_platt, fit_temperature
from lars.engine import Engine
from lars.primitives import softmax
from lars.schema import SystemOneRequest


@dataclass
class Example:
    state: Any
    question: dict[str, Any]
    label: str
    soft_label: dict[str, float] | None = None
    # Named numeric evidence for a noul, fused with the model when a calibration is fitted with it.
    features: dict[str, float] | None = None

    @property
    def id(self) -> str:
        """A stable content hash, so per-row dumps from two runs can be checked for alignment."""
        blob = json.dumps([self.state, self.question, self.label], sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _maybe_json(value: Any) -> Any:
    """Decode a JSON-encoded string (object, array, or quoted string) when that is what it is."""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped[:1] in '{["':
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    return value


def read_examples(path: str | Path) -> Iterator[Example]:
    with Path(path).open() as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            question = _maybe_json(raw["question"])
            if not isinstance(question, dict):
                raise ValueError("question must be a JSON object")
            soft = _maybe_json(raw.get("soft_label"))
            features = raw.get("features")
            yield Example(
                state=_maybe_json(raw["state"]),
                question=question,
                label=str(raw["label"]),
                soft_label=soft if isinstance(soft, dict) else None,
                features={k: float(v) for k, v in features.items()} if isinstance(features, dict) else None,
            )


@dataclass
class EvalResult:
    count: int
    accuracy: float
    ece: float
    brier: float
    coverage_at_5pct: float
    threshold_at_5pct: float
    per_kind: dict[str, dict[str, float]]
    ms_per_example: float = 0.0
    temperatures: dict[str, float] | None = None
    platt: dict[str, tuple[float, float] | None] | None = None


def _target_vector(example: Example, keys: tuple[str, ...]) -> np.ndarray:
    if example.soft_label:
        vec = np.asarray([float(example.soft_label.get(k, 0.0)) for k in keys])
        total = vec.sum()
        return vec / total if total > 0 else vec
    return np.asarray([1.0 if k == example.label else 0.0 for k in keys])


def _noul_keys_label(label: str) -> str:
    return "yes" if label in {"1", "true", "yes"} else "no"


def _noul_target(example: Example) -> np.ndarray:
    """[P(yes), P(no)]: the normalized soft label when there is one, else the hard label."""
    if example.soft_label:
        yes = sum(float(v) for k, v in example.soft_label.items() if _noul_keys_label(str(k).lower()) == "yes")
        total = sum(float(v) for v in example.soft_label.values())
        if total > 0:
            return np.asarray([yes / total, 1.0 - yes / total])
    return np.asarray([1.0, 0.0]) if _noul_keys_label(example.label) == "yes" else np.asarray([0.0, 1.0])


def collect(engine: Engine, examples: list[Example]) -> list[tuple[Example, str, tuple[str, ...], np.ndarray]]:
    rows = []
    for example in examples:
        request = SystemOneRequest(state=example.state, questions={"q": example.question})
        kind = request.questions["q"].type
        if kind == "multi":
            raise ValueError("eval and calibrate do not support multi questions yet (one label cannot score "
                             "per-option answers); split them into one noul per option")
        keys, logits = engine.raw_logits(example.state, "q", request.questions["q"])
        rows.append((example, kind, keys, logits))
    return rows


@dataclass
class Collected:
    """Raw logits for a list of examples and the time the model took to produce them.

    Raw logits do not depend on the calibrator, so one collection serves the raw score, the
    calibration fit, and the calibrated score and row dump of the same examples.
    """

    rows: list[tuple[Example, str, tuple[str, ...], np.ndarray]]
    ms_per_example: float


def collect_timed(engine: Engine, examples: list[Example]) -> Collected:
    started = time.perf_counter()
    rows = collect(engine, examples)
    return Collected(rows, (time.perf_counter() - started) * 1000.0 / max(len(rows), 1))


def evaluate(
    engine: Engine, examples: list[Example], rows: list[dict] | None = None, collected: Collected | None = None
) -> EvalResult:
    """Score examples; when `rows` is given, append each example's probabilities and hit to it in input order.

    `collected` reuses raw logits from `collect_timed` on the same examples instead of running the model again.
    """
    if collected is None:
        collected = collect_timed(engine, examples)
    confidences, correct, probs, targets = [], [], [], []
    per_kind: dict[str, list[bool]] = {}
    for example, kind, keys, logits in collected.rows:
        if kind in {"noul", "multi"}:
            p_yes = engine.noul_prob(float(logits[0] - logits[1]), kind, features=example.features)
            p = np.asarray([p_yes, 1.0 - p_yes])
        else:
            p = softmax(logits, engine.calibrator.temperature_for(kind))
        label = _noul_keys_label(example.label) if kind in {"noul", "multi"} else example.label
        target = _noul_target(example) if kind in {"noul", "multi"} else _target_vector(example, keys)
        predicted = keys[int(p.argmax())]
        hit = predicted == label
        confidences.append(float(p.max()))
        correct.append(hit)
        probs.append(p)
        targets.append(target)
        per_kind.setdefault(kind, []).append(hit)
        if rows is not None:
            rows.append({"id": example.id, "keys": list(keys), "p": p.tolist(), "target": target.tolist(),
                         "hit": bool(hit)})
    conf = np.asarray(confidences)
    hits = np.asarray(correct, dtype=float)
    cov, thr = coverage_at_error(conf, hits, 0.05)
    return EvalResult(
        count=len(collected.rows),
        accuracy=float(hits.mean()) if len(hits) else 0.0,
        ece=ece(conf, hits),
        brier=brier(probs, targets),
        coverage_at_5pct=cov,
        threshold_at_5pct=thr,
        per_kind={k: {"count": len(v), "accuracy": float(np.mean(v))} for k, v in per_kind.items()},
        ms_per_example=collected.ms_per_example,
        temperatures={k: engine.calibrator.temperature_for(k) for k in sorted({c[1] for c in collected.rows})},
        platt={k: engine.calibrator.platt_for(k) for k in sorted({c[1] for c in collected.rows})},
    )


def calibrate(
    engine: Engine, examples: list[Example], source: str | None = None, collected: Collected | None = None
) -> Calibrator:
    """Fit per-kind temperatures and Platt scalings; `collected` reuses raw logits as in `evaluate`."""
    collected_rows = (collected or collect_timed(engine, examples)).rows
    calibrator = Calibrator(fitted_on=source)
    by_kind: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {}
    for example, kind, keys, logits in collected_rows:
        target = _noul_target(example) if kind in {"noul", "multi"} else _target_vector(example, keys)
        logits_list, targets_list = by_kind.setdefault(kind, ([], []))
        logits_list.append(np.asarray(logits))
        targets_list.append(target)
    for kind, (logits_list, targets_list) in by_kind.items():
        if kind in {"noul", "multi"}:
            # Platt on the raw yes-minus-no logit subsumes temperature (a = 1/T) and adds a bias.
            z = np.asarray([float(row[0] - row[1]) for row in logits_list])
            y = np.asarray([float(t[0]) for t in targets_list])
            calibrator.platt[kind] = fit_platt(z, y)
            calibrator.temperatures[kind] = 1.0
            evidence = [example.features for example, k, _, _ in collected_rows if k == kind]
            if evidence and all(evidence):
                calibrator.fusion[kind] = fit_fusion(z, evidence, y)  # type: ignore[arg-type]
        else:
            calibrator.temperatures[kind] = fit_temperature(logits_list, targets_list)
    return calibrator
