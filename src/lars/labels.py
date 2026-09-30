"""Answer labels.

Options are presented to the model as short letter labels and read back as a single
next-token logit each. A label is usable only if the backend tokenizer encodes it as
one token, so the pool is verified per model at load time.
"""

from __future__ import annotations

import string
from collections.abc import Callable, Iterator


def label_candidates() -> Iterator[str]:
    """Yield A..Z then AA..ZZ, 702 candidates, enough for the 255-option maximum with margin."""
    letters = string.ascii_uppercase
    yield from letters
    for first in letters:
        for second in letters:
            yield first + second


def assign_labels(count: int, is_single_token: Callable[[str], bool]) -> list[str]:
    """Return `count` verified single-token labels in canonical order."""
    labels: list[str] = []
    for candidate in label_candidates():
        if is_single_token(candidate):
            labels.append(candidate)
            if len(labels) == count:
                return labels
    raise ValueError(f"tokenizer offers only {len(labels)} single-token labels, need {count}")
