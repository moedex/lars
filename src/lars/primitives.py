"""Turning label logits into typed answers.

Everything here is plain numpy over small vectors. Formulas:

- probabilities: softmax over the allowed labels only, at a calibrated temperature
- choice confidence: (N * max_p - 1) / (N - 1), the published System One formula
- score: expected level index, sum(i * p_i); may fall between levels
- score confidence: 1 - 2 * E|i - score| / (K - 1), one minus the normalized spread
  around the expected level. System One's score confidence formula is unpublished, so
  this is LARS's own definition. It is 1.0 for a one-hot distribution and rewards
  mass on adjacent levels more than mass on distant ones.
- order sensitivity: mean total variation distance of each ordering's distribution
  from their mean
"""

from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / max(temperature, 1e-6)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0))))


def logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return float(np.log(p / (1 - p)))


def choice_confidence(probs: np.ndarray) -> float:
    n = len(probs)
    if n < 2:
        return 1.0
    return float(np.clip((n * probs.max() - 1.0) / (n - 1.0), 0.0, 1.0))


def expected_score(probs: np.ndarray) -> float:
    levels = np.arange(len(probs), dtype=np.float64)
    return float((levels * probs).sum())


def score_confidence(probs: np.ndarray) -> float:
    k = len(probs)
    if k < 2:
        return 1.0
    levels = np.arange(k, dtype=np.float64)
    center = expected_score(probs)
    spread = float((probs * np.abs(levels - center)).sum())
    return float(np.clip(1.0 - 2.0 * spread / (k - 1.0), 0.0, 1.0))


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum())


def order_sensitivity(distributions: list[np.ndarray]) -> float:
    if len(distributions) < 2:
        return 0.0
    stack = np.stack(distributions)
    mean = stack.mean(axis=0)
    return float(np.mean([total_variation(d, mean) for d in stack]))


def top_margin(probs: np.ndarray) -> float:
    if len(probs) < 2:
        return 1.0
    ordered = np.sort(probs)[::-1]
    return float(ordered[0] - ordered[1])


def round_probs(keys: tuple[str, ...], probs: np.ndarray, digits: int = 4) -> dict[str, float]:
    return {k: round(float(p), digits) for k, p in zip(keys, probs, strict=True)}
