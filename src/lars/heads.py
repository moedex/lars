"""Serving-side pointer head: numpy only, loads what `lars.train.residual` saves.

    logits_i = exp(log_s[kind]) * z_i + (h_ans W_q) . (h_opt_i W_k) / sqrt(rank) + [i == 0] * bias[kind == noul]

Features arrive already projected with the same fixed matrix used at training time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

KIND_INDEX = {"noul": 0, "choice": 1, "score": 2, "multi": 0}


def rms_normalize(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Scale each vector (last axis) to unit RMS. Backbone hidden states have norms in the
    hundreds; the head sees unit-scale inputs at both training and serving time."""
    x = np.asarray(x, dtype=np.float32)
    rms = np.sqrt((x * x).mean(axis=-1, keepdims=True)) + eps
    return x / rms


class PointerHeadScorer:
    def __init__(self, q: np.ndarray, k: np.ndarray, log_s: np.ndarray, bias: np.ndarray, projection: np.ndarray):
        self.q = q.astype(np.float32)  # (rank, proj_dim), torch/mlx Linear layout
        self.k = k.astype(np.float32)
        self.log_s = log_s.astype(np.float32)
        self.bias = bias.astype(np.float32)
        self.projection = projection.astype(np.float32)  # (hidden, proj_dim)
        self.rank = self.q.shape[0]

    @classmethod
    def load(cls, head_path: str | Path, projection_path: str | Path) -> PointerHeadScorer:
        data = np.load(head_path)
        return cls(data["q.weight"], data["k.weight"], data["log_s"], data["bias"], np.load(projection_path))

    def project(self, hidden: np.ndarray) -> np.ndarray:
        """Fixed projection then RMS normalization, matching `lars.train.residual.Shard`."""
        return rms_normalize(hidden.astype(np.float32) @ self.projection)

    def adjust(self, z: np.ndarray, h_ans: np.ndarray, h_opt: np.ndarray, kind: str) -> np.ndarray:
        """Return adjusted logits for one row. h_ans (proj_dim,), h_opt (K, proj_dim), z (K,)."""
        q = h_ans @ self.q.T  # (rank,)
        k = h_opt @ self.k.T  # (K, rank)
        residual = (k @ q) / np.sqrt(self.rank)
        index = KIND_INDEX[kind]
        logits = np.exp(self.log_s[index]) * z.astype(np.float32) + residual
        if kind in {"noul", "multi"}:
            logits = logits.copy()
            logits[0] += self.bias[0]
        return logits.astype(np.float64)
