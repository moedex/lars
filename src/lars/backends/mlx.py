"""MLX backend for Apple Silicon.

Prefills the shared prefix once into a KV cache, snapshots it, and for each suffix
restores the snapshot and runs the suffix to read the last-position logits.

Requires `pip install lars-engine[mlx]`. Untested paths are guarded: if snapshot or restore
fails on a given cache type, the backend falls back to a full prefill per row.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from lars.backends.base import Backend
from lars.render import TEMPLATES, TemplateFn
from lars.spans import char_offsets_to_token_indexes, option_end_char_offsets


class MLXBackend(Backend):
    name = "mlx"

    def __init__(self, model_path: str, template: str | None = None, adapter: str | list[str] | None = None) -> None:
        """`adapter` is one LoRA adapter directory, or several to serve as an ensemble (`use_adapter`)."""
        try:
            import mlx.core as mx
            from mlx_lm import load
            from mlx_lm.models.cache import make_prompt_cache
        except ImportError as error:  # pragma: no cover - depends on platform
            raise ImportError("install with `pip install lars-engine[mlx]` on Apple Silicon") from error

        self._mx = mx
        self._make_cache = make_prompt_cache
        # MLX keeps freed buffers for reuse, up to the memory limit by default. Every request has a different
        # prompt length, so those buffers rarely match and the cache grows until the OS kills the server
        # (about 100 GB after roughly 100 requests on a 128 GB machine, with 2.2 GB actually in use).
        mx.set_cache_limit(int(float(os.environ.get("LARS_MLX_CACHE_GB", "4")) * 2**30))
        adapters = [adapter] if isinstance(adapter, str) else list(adapter or [])
        self.adapters = None
        if len(adapters) > 1:
            from lars.backends.adapters import AdapterSet

            self.model, self.tokenizer = load(model_path)
            self.adapters = AdapterSet(self.model, adapters)  # same module paths as mlx-lm load_adapters
        else:
            # A LoRA adapter from `lars.train.lora` loads through mlx-lm's own adapter path.
            self.model, self.tokenizer = load(model_path, adapter_path=adapters[0] if adapters else None)
        # Vision-language checkpoints (Qwen3.5) wrap the text stack in `language_model`;
        # everything that touches the transformer body or the output projection goes there.
        self._text = getattr(self.model, "language_model", self.model)
        self.model_name = "+".join([model_path, *(Path(a).name for a in adapters)])
        self._template: TemplateFn
        if template and template in TEMPLATES:
            self._template = TEMPLATES[template]
        elif getattr(self.tokenizer, "chat_template", None):
            self._template = self._hf_template
        else:
            self._template = TEMPLATES["chatml"]
        self._label_cache: dict[str, int | None] = {}

    def use_adapter(self, index: int) -> None:
        """Make adapter `index` of an ensemble the active one for the next passes."""
        if self.adapters is None:
            if index != 0:
                raise ValueError("this backend has one adapter")
            return
        self.adapters.use(index)

    # ----------------------------------------------------------------- template + tokens

    def _hf_template(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        # Hybrid-thinking templates (Qwen3, Qwen3.5) open a <think> block by default; the
        # label readout needs the plain answer position, so thinking is switched off. Templates
        # without the switch ignore the keyword.
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    def template(self) -> TemplateFn:
        return self._template

    def _encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def _label_id(self, label: str) -> int | None:
        if label not in self._label_cache:
            ids = self._encode(" " + label)
            self._label_cache[label] = ids[0] if len(ids) == 1 else None
        return self._label_cache[label]

    def is_single_token(self, label: str) -> bool:
        return self._label_id(label) is not None

    def count_tokens(self, text: str) -> int:
        return len(self._encode(text))

    # ----------------------------------------------------------------- training features

    @property
    def hidden_size(self) -> int:
        return int(self._text.args.hidden_size)

    def hidden_states(self, text: str) -> np.ndarray:
        """Final-layer hidden states for every token of `text`, shape (T, hidden_size), float32."""
        mx = self._mx
        ids = self._encode(text)
        hidden = self._text.model(mx.array(ids)[None])[0]
        mx.eval(hidden)
        return np.asarray(hidden.astype(mx.float32))

    def label_rows(self, labels: list[str]) -> np.ndarray:
        """Output-projection rows for the label tokens, shape (K, hidden_size). z = rows @ h.

        Quantized models pack the projection weight; the selected rows are dequantized
        with the layer's scales and biases so the result matches the full forward pass.
        """
        mx = self._mx
        ids = [self._label_id(label) for label in labels]
        if any(i is None for i in ids):
            raise ValueError(f"labels not single-token for this tokenizer: {labels}")
        layer = self._text.model.embed_tokens if self._text.args.tie_word_embeddings else self._text.lm_head
        index = mx.array(ids)
        if hasattr(layer, "scales"):
            rows = mx.dequantize(
                layer.weight[index], layer.scales[index], layer.biases[index], layer.group_size, layer.bits
            )
        else:
            rows = layer.weight[index]
        mx.eval(rows)
        return np.asarray(rows.astype(mx.float32))

    def token_offsets(self, text: str) -> list[tuple[int, int]]:
        """Character span of each token, aligned with `hidden_states(text)`."""
        base = getattr(self.tokenizer, "_tokenizer", self.tokenizer)
        try:
            encoded = base(text, add_special_tokens=False, return_offsets_mapping=True)
            return [tuple(span) for span in encoded["offset_mapping"]]
        except Exception:  # pragma: no cover - slow tokenizer fallback
            spans: list[tuple[int, int]] = []
            count = 0
            for end in range(1, len(text) + 1):
                n = len(self._encode(text[:end]))
                if n > count:
                    spans.extend((end - 1, end) for _ in range(n - count))
                    count = n
            return spans

    # ----------------------------------------------------------------- inference

    def _forward(self, ids: list[int], cache: Any) -> np.ndarray:
        mx = self._mx
        logits = self.model(mx.array(ids, dtype=mx.int32)[None], cache=cache)
        last = logits[0, -1, :]
        mx.eval(last)
        return np.asarray(last.astype(mx.float32))

    @staticmethod
    def _detach(state: Any) -> Any:
        # ArraysCache (the Gated DeltaNet layers in Qwen3.5) hands out and adopts its internal
        # list, then writes through `cache[i] = ...`; copying the list keeps the snapshot
        # untouched by the rows that restore it. KV tuples rebind on append and need no copy.
        return list(state) if isinstance(state, list) else state

    def _snapshot(self, cache: Any) -> list[Any] | None:
        try:
            return [(self._detach(c.state), getattr(c, "meta_state", None)) for c in cache]
        except Exception:  # pragma: no cover - cache type without state API
            return None

    def _restore(self, snapshot: list[Any]) -> Any:
        cache = self._make_cache(self.model)
        for c, (state, meta) in zip(cache, snapshot, strict=True):
            c.state = self._detach(state)
            if meta is not None and hasattr(c, "meta_state"):
                c.meta_state = meta
        return cache

    def label_logits(self, prefix: str, suffixes: list[str], labels: list[tuple[str, ...]]) -> list[np.ndarray]:
        return [z for z, _, _ in self._score_rows(prefix, suffixes, labels, want_hidden=False)]

    def label_logits_with_features(
        self,
        prefix: str,
        suffixes: list[str],
        labels: list[tuple[str, ...]],
        option_ends: list[tuple[int, ...]] | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Per suffix: (label logits, hidden at the answer position, hidden at each option line end).

        `option_ends` gives, per suffix, the character offset in it where each option line
        ends, as `render` records them (`Row.option_ends`). Without it the suffix is parsed
        for option lines, which caller text shaped like `A) ...` can fool.
        Hidden states are unprojected (hidden_size). The pointer head projects them.
        """
        return self._score_rows(prefix, suffixes, labels, want_hidden=True, option_ends=option_ends)

    def _hidden_forward(self, ids: list[int], cache: Any):
        """Run the transformer body only; returns (T, hidden) for these positions."""
        mx = self._mx
        hidden = self._text.model(mx.array(ids, dtype=mx.int32)[None], cache=cache)[0]
        mx.eval(hidden)
        return hidden

    def _score_rows(
        self,
        prefix: str,
        suffixes: list[str],
        labels: list[tuple[str, ...]],
        want_hidden: bool,
        option_ends: list[tuple[int, ...]] | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        mx = self._mx
        prefix_ids = self._encode(prefix)
        snapshot = None
        if prefix_ids:
            prefix_cache = self._make_cache(self.model)
            self._forward(prefix_ids, prefix_cache)
            snapshot = self._snapshot(prefix_cache)

        results = []
        ends_per_row = option_ends if option_ends is not None else [None] * len(suffixes)
        for suffix, row_labels, row_ends in zip(suffixes, labels, ends_per_row, strict=True):
            ids = [self._label_id(label) for label in row_labels]
            if any(i is None for i in ids):
                raise ValueError(f"labels not single-token for this tokenizer: {row_labels}")
            full_ids = self._encode(prefix + suffix)
            shares_prefix = snapshot is not None and full_ids[: len(prefix_ids)] == prefix_ids
            if shares_prefix:
                cache, run_ids, offset = self._restore(snapshot), full_ids[len(prefix_ids) :], len(prefix_ids)
            else:
                cache, run_ids, offset = self._make_cache(self.model), full_ids, 0

            if not want_hidden:
                vocab_logits = self._forward(run_ids, cache)
                results.append((vocab_logits[np.asarray(ids)].astype(np.float64), None, None))
                continue

            hidden = self._hidden_forward(run_ids, cache)  # (T_run, hidden)
            rows = self.label_rows(list(row_labels))  # (K, hidden)
            h_ans = np.asarray(hidden[-1].astype(mx.float32))
            z = (rows @ h_ans).astype(np.float64)
            offsets = self.token_offsets(prefix + suffix)
            if row_ends:
                if len(row_ends) != len(row_labels):
                    raise ValueError(f"{len(row_ends)} option offsets for {len(row_labels)} labels")
                ends = [len(prefix) + e for e in row_ends]
            else:
                ends = option_end_char_offsets(prefix, suffix, len(row_labels))
            token_idx = [max(i - offset, 0) for i in char_offsets_to_token_indexes(offsets, ends)]
            h_opt = np.asarray(hidden[mx.array(token_idx)].astype(mx.float32))
            results.append((z, h_ans, h_opt))
        return results
