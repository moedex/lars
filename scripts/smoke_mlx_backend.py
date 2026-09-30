"""Prefix-cache and feature-path consistency check for an MLX model. Exits non-zero on mismatch.

Run this before benchmarking a new architecture. Checks, strictest first:

- restoring the prefix snapshot gives the same logits as prefilling the prefix and running
  the suffix on that same cache (identical kernels and shapes, so this isolates the
  snapshot and restore logic, which is where cache-type bugs show up);
- repeated rows from one snapshot are identical (a row must not leak into the next);
- the cached split agrees with one full prefill, and the feature path's dequantized label
  rows reproduce the forward pass. These two differ by bf16 rounding that grows with the
  logits' magnitude, so the bound is two bf16 steps at the largest logit, at least 0.15.

    uv run python scripts/smoke_mlx_backend.py mlx-community/Qwen3.5-9B-MLX-4bit
"""

import sys

import numpy as np

from lars.backends.mlx import MLXBackend
from lars.render import compose_prompt, render_choice, render_content
from lars.schema import ChoiceQuestion


def _bf16_step(magnitude: float) -> float:
    """Spacing of bfloat16 values (8 significant bits) around `magnitude`."""
    return 2.0 ** (np.floor(np.log2(max(magnitude, 1e-6))) - 7)


def _direct(backend: MLXBackend, prefix: str, suffix: str, labels: tuple[str, ...]) -> np.ndarray:
    """Prefill the prefix, then run the suffix on the same cache: no snapshot, no restore."""
    prefix_ids = backend._encode(prefix)
    suffix_ids = backend._encode(prefix + suffix)[len(prefix_ids) :]
    cache = backend._make_cache(backend.model)
    backend._forward(prefix_ids, cache)
    logits = backend._forward(suffix_ids, cache)
    return logits[[backend._label_id(label) for label in labels]].astype(np.float64)


def main() -> int:
    backend = MLXBackend(sys.argv[1])
    print("hidden_size", backend.hidden_size)
    question = ChoiceQuestion(
        type="choice",
        instructions="Which team should handle this?",
        criteria={"billing": "Payments, payouts", "technical": "Bugs, outages", "sales": "Pricing, demos"},
    )
    labels = ("A", "B", "C")
    body, _, _ = render_choice(question, list(labels), [0, 1, 2])
    state = render_content("Help! My payouts have been failing for 3 days.")
    prefix, suffix = compose_prompt(backend.template(), state, body)
    print("prompt tail:", repr((prefix + suffix)[-80:]))
    z_cached = backend.label_logits(prefix, [suffix, suffix], [labels] * 2)
    z_full = backend.label_logits("", [prefix + suffix], [labels])
    z_feat, h_ans, h_opt = backend.label_logits_with_features(prefix, [suffix], [labels])[0]
    print("cached", np.round(z_cached[0], 3), "full", np.round(z_full[0], 3), "features", np.round(z_feat, 3))
    print("h_ans norm", round(float(np.linalg.norm(h_ans)), 1), "h_opt", h_opt.shape)
    z_direct = _direct(backend, prefix, suffix, labels)
    print("direct", np.round(z_direct, 3))

    restore_vs_direct = float(np.abs(z_cached[0] - z_direct).max())
    repeat = float(np.abs(z_cached[0] - z_cached[1]).max())
    cached_vs_full = float(np.abs(z_cached[0] - z_full[0]).max())
    cached_vs_feat = float(np.abs(z_cached[0] - z_feat).max())
    rounding = max(0.15, 2 * _bf16_step(float(np.abs(z_full[0]).max())))
    print("max diff restore-vs-direct", restore_vs_direct, "repeat", repeat,
          "cached-vs-full", cached_vs_full, "cached-vs-features", cached_vs_feat, "rounding bound", rounding)
    ok = (restore_vs_direct < 1e-3 and repeat < 1e-3 and cached_vs_full <= rounding
          and cached_vs_feat <= rounding and int(np.argmax(z_cached[0])) == 0)
    print("SMOKE", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
