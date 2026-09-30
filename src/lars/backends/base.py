"""Backend protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from lars.render import TemplateFn


class Backend(ABC):
    """A model that can prefill a prefix once and score many suffixes against label tokens.

    Implementations own tokenization, chat templating, KV-cache reuse, and batching.
    The engine never sees token ids.
    """

    name: str = "backend"
    model_name: str = "unknown"

    @abstractmethod
    def template(self) -> TemplateFn:
        """The chat template used to compose prompts for this model."""

    @abstractmethod
    def is_single_token(self, label: str) -> bool:
        """True if `" " + label` encodes to exactly one token (the form read after `Answer:`)."""

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Token count for usage reporting."""

    @abstractmethod
    def label_logits(self, prefix: str, suffixes: list[str], labels: list[tuple[str, ...]]) -> list[np.ndarray]:
        """For each suffix, the next-token logits restricted to its labels, in label order."""
