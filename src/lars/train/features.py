"""Extract pointer-head features from the frozen backbone.

For each record and each presentation (canonical order plus optional shuffles) we
render the exact prompt the engine would send, run one forward pass, and keep:

- `h_ans`: hidden state at the last prompt token (the position whose next token is read)
- `h_opt`: hidden state at the end of each option line, one per option
- `z`: the backbone's own label logits, `label_rows @ h_ans`
- `perm`: presented index -> canonical option index, so shuffled presentations can be
  realigned for the permutation loss

Hidden states are projected with a fixed seeded Gaussian matrix to `proj_dim` before
caching. The same projection is stored in the head checkpoint, so serving matches.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lars.backends.mlx import MLXBackend
from lars.labels import assign_labels
from lars.render import compose_prompt, render_choice, render_content, render_noul, render_score
from lars.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion
from lars.train.data import Record


@dataclass
class Presentation:
    record_id: str
    kind: str
    perm: list[int]
    h_ans: np.ndarray
    h_opt: np.ndarray
    z: np.ndarray
    target: np.ndarray


def projection(hidden_size: int, proj_dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((hidden_size, proj_dim)) / np.sqrt(proj_dim)).astype(np.float32)


def _render(record: Record, labels: list[str], order: list[int]) -> tuple[str, tuple[str, ...], tuple[int, ...]]:
    question = record.to_question()
    if record.kind == "noul":
        return render_noul(NoulQuestion(**question), labels)
    if record.kind == "choice":
        return render_choice(ChoiceQuestion(**question), labels, order)
    return render_score(ScoreQuestion(**question), labels)


def extract(
    backend: MLXBackend,
    records: Iterable[Record],
    proj: np.ndarray,
    shuffles: int = 1,
    seed: int = 0,
) -> Iterable[Presentation]:
    labels = assign_labels(255, backend.is_single_token)
    template = backend.template()
    rng = random.Random(seed)
    for record in records:
        k = len(record.options)
        orders = [list(range(k))]
        if record.kind == "choice":
            for _ in range(shuffles):
                order = list(range(k))
                rng.shuffle(order)
                orders.append(order)
        state_text = render_content(record.state)
        suffixes: list[str] = []
        option_ends: list[tuple[int, ...]] = []
        prefix = ""
        for order in orders:
            body, _keys, ends = _render(record, labels, order)
            prefix, suffix = compose_prompt(template, state_text, body)
            suffixes.append(suffix)
            option_ends.append(ends)
        # Same code path as serving, with the prefix prefilled once for all presentations.
        features = backend.label_logits_with_features(prefix, suffixes, [tuple(labels[:k])] * len(orders), option_ends)
        for order, (z, h_ans, h_opt) in zip(orders, features, strict=True):
            target = np.asarray([record.target[i] for i in order], dtype=np.float32)
            yield Presentation(
                record_id=record.id,
                kind=record.kind,
                perm=order,
                h_ans=(h_ans @ proj).astype(np.float16),
                h_opt=(h_opt @ proj).astype(np.float16),
                z=np.asarray(z, dtype=np.float32),
                target=target,
            )


def write_shard(presentations: Iterable[Presentation], path: str | Path, max_k: int = 255) -> int:
    """Pad to a shard-wide K and save as npz. Returns the number of presentations."""
    items = list(presentations)
    if not items:
        return 0
    k_max = min(max_k, max(len(p.perm) for p in items))
    n, p = len(items), items[0].h_ans.shape[0]
    h_ans = np.zeros((n, p), np.float16)
    h_opt = np.zeros((n, k_max, p), np.float16)
    z = np.zeros((n, k_max), np.float32)
    target = np.zeros((n, k_max), np.float32)
    mask = np.zeros((n, k_max), bool)
    perm = np.full((n, k_max), -1, np.int32)
    ids, kinds = [], []
    for i, item in enumerate(items):
        k = len(item.perm)
        h_ans[i] = item.h_ans
        h_opt[i, :k] = item.h_opt
        z[i, :k] = item.z
        target[i, :k] = item.target
        mask[i, :k] = True
        perm[i, :k] = item.perm
        ids.append(item.record_id)
        kinds.append(item.kind)
    np.savez_compressed(
        path, h_ans=h_ans, h_opt=h_opt, z=z, target=target, mask=mask, perm=perm,
        ids=np.asarray(ids), kinds=np.asarray(kinds),
    )
    Path(str(path) + ".json").write_text(json.dumps({"n": n, "k_max": k_max, "proj_dim": p}))
    return n
