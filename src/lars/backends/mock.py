"""Deterministic mock backend for tests and the demo server.

Logits are a hash of (state, option) plus a keyword bonus when an option's key or
description appears in the state, so demos behave sensibly without a model. The
answers mean nothing beyond that.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

from lars.backends.base import Backend
from lars.render import TemplateFn, template_plain

_OPTION_LINE = re.compile(r"^([A-Z]{1,2})\) (.+)$", re.MULTILINE)
_STATE_BLOCK = re.compile(r"<STATE [0-9a-f]+>\n(.*)\n</STATE [0-9a-f]+>", re.DOTALL)


def _hash_unit(*parts: str) -> float:
    digest = hashlib.sha256("\u0001".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


class MockBackend(Backend):
    name = "mock"
    model_name = "lars-mock"

    def __init__(self, keyword_bonus: float = 3.0, noise: float = 2.0) -> None:
        self.keyword_bonus = keyword_bonus
        self.noise = noise

    def template(self) -> TemplateFn:
        return template_plain

    def is_single_token(self, label: str) -> bool:
        return len(label) <= 2

    def count_tokens(self, text: str) -> int:
        return max(1, len(text.split()))

    def label_logits(self, prefix: str, suffixes: list[str], labels: list[tuple[str, ...]]) -> list[np.ndarray]:
        match = _STATE_BLOCK.search(prefix)
        state_text = (match.group(1) if match else prefix).lower()
        state_words = set(re.findall(r"[a-z0-9]+", state_text))
        results: list[np.ndarray] = []
        for suffix, row_labels in zip(suffixes, labels, strict=True):
            described = {m.group(1): m.group(2) for m in _OPTION_LINE.finditer(suffix)}
            question = suffix.split("\n", 1)[0].lower()
            question_words = {w for w in re.findall(r"[a-z0-9]+", question) if len(w) > 3}
            logits = []
            for label in row_labels:
                text = described.get(label, label).lower()
                option_words = {w for w in re.findall(r"[a-z0-9]+", text) if len(w) > 2 and w not in {"yes", "no"}}
                value = (_hash_unit(state_text, text) - 0.5) * 2 * self.noise
                if option_words & state_words:
                    value += self.keyword_bonus
                if text.startswith("yes") and question_words & state_words:
                    value += self.keyword_bonus
                logits.append(value)
            results.append(np.asarray(logits, dtype=np.float64))
        return results
