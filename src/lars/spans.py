"""Locating option lines inside a rendered prompt, in token space."""

from __future__ import annotations

import re

OPTION_LINE = re.compile(r"^([A-Z]{1,2})\) ", re.MULTILINE)


def option_end_char_offsets(prefix: str, suffix: str, count: int) -> list[int]:
    """Exclusive character offset, in prefix + suffix, of the end of each option line."""
    ends: list[int] = []
    base = len(prefix)
    for match in OPTION_LINE.finditer(suffix):
        line_end = suffix.find("\n", match.start())
        ends.append(base + (line_end if line_end != -1 else len(suffix)))
        if len(ends) == count:
            break
    if len(ends) != count:
        raise ValueError(f"found {len(ends)} option lines, expected {count}")
    return ends


def char_offsets_to_token_indexes(offsets: list[tuple[int, int]], char_ends: list[int]) -> list[int]:
    """For each exclusive char offset, the index of the last token that starts before it."""
    starts = [s for s, _ in offsets]
    result = []
    for end in char_ends:
        index = 0
        for i, s in enumerate(starts):
            if s < end:
                index = i
            else:
                break
        result.append(index)
    return result
