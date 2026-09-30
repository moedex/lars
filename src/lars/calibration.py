"""Calibration.

Raw label logits from a language model are not calibrated. LARS fits, on the
user's own labeled data:

- a temperature per primitive kind (noul, choice, score, multi), which reshapes
  distributions without changing the argmax
- an optional Platt scaler (a, b) for noul-style probabilities, which can also move a
  decision across 0.5 when the raw model is biased toward one side

Metrics: expected calibration error (ECE), Brier score, and coverage at an error budget.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from lars.primitives import softmax

KINDS = ("noul", "choice", "score", "multi")


@dataclass
class Calibrator:
    temperatures: dict[str, float] = field(default_factory=lambda: dict.fromkeys(KINDS, 1.0))
    platt: dict[str, tuple[float, float]] = field(default_factory=dict)
    # Evidence fusion per kind: sigmoid(weights . [logit, *features] + bias), fitted by `fit_fusion`.
    fusion: dict[str, dict[str, Any]] = field(default_factory=dict)
    fitted_on: str | None = None
    # Per-decision escalation, {"below": p, "weight": w}: set only where a measurement showed the
    # hosted model helps this decision (`lars.escalate`). None keeps the decision local.
    escalate: dict[str, float] | None = None

    def temperature_for(self, kind: str) -> float:
        return float(self.temperatures.get(kind, 1.0))

    def platt_for(self, kind: str) -> tuple[float, float] | None:
        value = self.platt.get(kind)
        return (float(value[0]), float(value[1])) if value else None

    def fusion_for(self, kind: str, features: dict[str, float] | None) -> dict[str, Any] | None:
        """The fitted fusion for this kind when every feature it was fitted on is supplied."""
        spec = self.fusion.get(kind)
        if not spec or not features or any(name not in features for name in spec["features"]):
            return None
        return spec

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> Calibrator:
        raw = json.loads(Path(path).read_text())
        platt = {k: (float(v[0]), float(v[1])) for k, v in raw.get("platt", {}).items()}
        return cls(
            temperatures=raw.get("temperatures", {}),
            platt=platt,
            fusion=raw.get("fusion", {}),
            fitted_on=raw.get("fitted_on"),
            escalate=raw.get("escalate"),
        )


# --------------------------------------------------------------------------- fitting


def fit_temperature(logit_rows: list[np.ndarray], targets: list[np.ndarray]) -> float:
    """Grid-search the temperature minimizing cross-entropy against (possibly soft) targets."""
    if not logit_rows:
        return 1.0
    grid = np.geomspace(0.05, 20.0, 400)
    best_t, best_nll = 1.0, float("inf")
    for t in grid:
        nll = 0.0
        for logits, target in zip(logit_rows, targets, strict=True):
            p = np.clip(softmax(logits, t), 1e-9, 1.0)
            nll -= float((np.asarray(target) * np.log(p)).sum())
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return best_t


def _logistic_loss(z: np.ndarray, y: np.ndarray) -> float:
    # log(1 + exp(z)) - y * z, computed stably
    return float((np.logaddexp(0.0, z) - y * z).sum())


def _platt_targets(labels: np.ndarray) -> np.ndarray:
    """Platt's smoothed targets for hard 0/1 labels; soft labels (strictly between) pass through."""
    y = np.asarray(labels, dtype=np.float64).copy()
    hard = (y == 0.0) | (y == 1.0)
    n_pos, n_neg = float((y[hard] == 1.0).sum()), float((y[hard] == 0.0).sum())
    if n_pos and n_neg:
        y[hard & (y == 1.0)] = (n_pos + 1) / (n_pos + 2)
        y[hard & (y == 0.0)] = 1 / (n_neg + 2)
    return y


def fit_platt(scores: np.ndarray, labels: np.ndarray, l2: float = 1e-2, iterations: int = 100) -> tuple[float, float]:
    """Fit sigmoid(a * s + b) to binary labels.

    Scores are standardized before fitting so the L2 penalty and the step sizes are
    scale-free, and each Newton step is backtracked until the penalized loss decreases.
    Raw language-model logit differences can be tens of units wide and nearly separable,
    which makes an undamped Newton fit overshoot into a flipped or exploding solution.
    Platt's original label smoothing keeps the separable case finite.
    """
    s_raw = np.asarray(scores, dtype=np.float64)
    y_raw = np.asarray(labels, dtype=np.float64)
    if s_raw.size == 0:
        return 1.0, 0.0
    scale = float(s_raw.std()) or 1.0
    s = s_raw / scale
    y = _platt_targets(y_raw)

    def penalized(a: float, b: float) -> float:
        return _logistic_loss(a * s + b, y) + 0.5 * l2 * a * a

    a, b = 1.0, 0.0
    current = penalized(a, b)
    for _ in range(iterations):
        z = a * s + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        w = p * (1 - p) + 1e-12
        grad_a = float(((p - y) * s).sum() + l2 * a)
        grad_b = float((p - y).sum())
        h_aa = float((w * s * s).sum() + l2)
        h_ab = float((w * s).sum())
        h_bb = float(w.sum())
        det = h_aa * h_bb - h_ab * h_ab
        if abs(det) < 1e-12:
            break
        da = (h_bb * grad_a - h_ab * grad_b) / det
        db = (h_aa * grad_b - h_ab * grad_a) / det
        step = 1.0
        improved = False
        for _ in range(30):
            candidate = penalized(a - step * da, b - step * db)
            if candidate < current:
                a, b, current, improved = a - step * da, b - step * db, candidate, True
                break
            step *= 0.5
        if not improved or (abs(step * da) < 1e-9 and abs(step * db) < 1e-9):
            break
    return float(a / scale), float(b)


def fit_logistic(
    x: np.ndarray, labels: np.ndarray, l2: float = 1e-2, iterations: int = 100
) -> tuple[np.ndarray, float]:
    """Fit sigmoid(x @ w + b) to binary labels with a damped, L2-penalized Newton method.

    Columns are standardized before fitting, as in `fit_platt`, and the returned weights
    are mapped back to the raw scale. Platt's label smoothing keeps separable data finite.
    """
    x_raw = np.asarray(x, dtype=np.float64)
    y_raw = np.asarray(labels, dtype=np.float64)
    if x_raw.ndim != 2 or x_raw.shape[0] == 0:
        return np.zeros(x_raw.shape[1] if x_raw.ndim == 2 else 0), 0.0
    mean = x_raw.mean(axis=0)
    scale = x_raw.std(axis=0)
    scale[scale == 0] = 1.0
    design = np.hstack([(x_raw - mean) / scale, np.ones((x_raw.shape[0], 1))])
    y = _platt_targets(y_raw)
    penalty = np.full(design.shape[1], l2)
    penalty[-1] = 0.0  # the intercept is not penalized

    def loss(theta: np.ndarray) -> float:
        return _logistic_loss(design @ theta, y) + 0.5 * float((penalty * theta * theta).sum())

    theta = np.zeros(design.shape[1])
    current = loss(theta)
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-np.clip(design @ theta, -500, 500)))
        grad = design.T @ (p - y) + penalty * theta
        hessian = (design * (p * (1 - p) + 1e-12)[:, None]).T @ design + np.diag(penalty + 1e-9)
        delta = np.linalg.solve(hessian, grad)
        step, improved = 1.0, False
        for _ in range(30):
            candidate = loss(theta - step * delta)
            if candidate < current:
                theta, current, improved = theta - step * delta, candidate, True
                break
            step *= 0.5
        if not improved or float(np.abs(step * delta).max()) < 1e-9:
            break
    weights = theta[:-1] / scale
    bias = float(theta[-1] - (weights * mean).sum())
    return weights, bias


def fit_fusion(
    logits: np.ndarray, features: list[dict[str, float]], labels: np.ndarray, l2: float = 1e-2
) -> dict[str, Any]:
    """Logistic fusion of the model's yes-minus-no logit with caller-supplied numeric evidence."""
    names = sorted(set.intersection(*(set(f) for f in features))) if features else []
    x = np.column_stack([np.asarray(logits, dtype=np.float64), *[[float(f[n]) for f in features] for n in names]])
    weights, bias = fit_logistic(x, labels, l2=l2)
    return {"features": names, "weights": [float(w) for w in weights], "bias": bias}


def fused_probability(spec: dict[str, Any], logit: float, features: dict[str, float]) -> float:
    evidence = zip(spec["weights"][1:], spec["features"], strict=True)
    z = spec["weights"][0] * logit + sum(w * float(features[n]) for w, n in evidence)
    return float(1.0 / (1.0 + np.exp(-np.clip(z + spec["bias"], -500, 500))))


# --------------------------------------------------------------------------- metrics


def ece(confidences: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    conf = np.asarray(confidences, dtype=np.float64)
    hit = np.asarray(correct, dtype=np.float64)
    if conf.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if mask.any():
            total += mask.mean() * abs(hit[mask].mean() - conf[mask].mean())
    return float(total)


def brier(probs: list[np.ndarray], targets: list[np.ndarray]) -> float:
    if not probs:
        return 0.0
    return float(np.mean([((np.asarray(p) - np.asarray(t)) ** 2).sum() for p, t in zip(probs, targets, strict=True)]))


def coverage_at_error(confidences: np.ndarray, correct: np.ndarray, max_error: float = 0.05) -> tuple[float, float]:
    """Largest fraction of decisions acceptable above one threshold with empirical error <= max_error.

    Returns (coverage, threshold). Coverage is 0 when no threshold meets the budget.
    """
    conf = np.asarray(confidences, dtype=np.float64)
    hit = np.asarray(correct, dtype=np.float64)
    if conf.size == 0:
        return 0.0, 1.0
    # A threshold accepts every decision at or above it, so tied confidences enter together:
    # only the last index of each run of equal values is a threshold that can be reported.
    order = np.argsort(-conf, kind="stable")
    ranked, errors = conf[order], np.cumsum(1.0 - hit[order])
    best_cov, best_thr = 0.0, 1.0
    for k in range(1, conf.size + 1):
        if k < conf.size and ranked[k] == ranked[k - 1]:
            continue
        if errors[k - 1] / k <= max_error:
            best_cov, best_thr = k / conf.size, float(ranked[k - 1])
    return best_cov, best_thr
