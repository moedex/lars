"""llama.cpp backend via llama-cpp-python, for GGUF models on any platform.

Prefills the prefix, saves the llama state, and for each suffix restores it before
evaluating the suffix tokens and reading the last logits row.

Requires `pip install lars-engine[llamacpp]`.
"""

from __future__ import annotations

import numpy as np

from lars.backends.base import Backend
from lars.render import TEMPLATES, TemplateFn


class LlamaCppBackend(Backend):
    name = "llamacpp"

    def __init__(self, model_path: str, template: str = "chatml", n_ctx: int = 8192, n_gpu_layers: int = -1) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as error:  # pragma: no cover
            raise ImportError("install with `pip install lars-engine[llamacpp]`") from error
        if template not in TEMPLATES:
            raise ValueError(f"unknown template {template!r}; choose from {sorted(TEMPLATES)}")
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, logits_all=False, verbose=False)
        self.model_name = model_path
        self._template = TEMPLATES[template]
        self._label_cache: dict[str, int | None] = {}

    def template(self) -> TemplateFn:
        return self._template

    def _encode(self, text: str, bos: bool = False) -> list[int]:
        return list(self.llm.tokenize(text.encode("utf-8"), add_bos=bos, special=True))

    def _label_id(self, label: str) -> int | None:
        if label not in self._label_cache:
            ids = self.llm.tokenize((" " + label).encode("utf-8"), add_bos=False, special=False)
            self._label_cache[label] = ids[0] if len(ids) == 1 else None
        return self._label_cache[label]

    def is_single_token(self, label: str) -> bool:
        return self._label_id(label) is not None

    def count_tokens(self, text: str) -> int:
        return len(self._encode(text))

    def _last_logits(self) -> np.ndarray:
        """Next-token logits after the last evaluated token.

        Read from the context: with `logits_all=False`, llama-cpp-python (0.3.35) leaves its
        `scores` array at zero, which made every label tie.
        """
        import llama_cpp

        pointer = llama_cpp.llama_get_logits_ith(self.llm.ctx, -1)
        return np.ctypeslib.as_array(pointer, shape=(self.llm.n_vocab(),)).astype(np.float64)

    def label_logits(self, prefix: str, suffixes: list[str], labels: list[tuple[str, ...]]) -> list[np.ndarray]:
        llm = self.llm
        prefix_ids = self._encode(prefix, bos=True)
        llm.reset()
        llm.eval(prefix_ids)
        state = llm.save_state()

        results: list[np.ndarray] = []
        for suffix, row_labels in zip(suffixes, labels, strict=True):
            full_ids = self._encode(prefix + suffix, bos=True)
            if full_ids[: len(prefix_ids)] == prefix_ids:
                llm.load_state(state)
                llm.eval(full_ids[len(prefix_ids) :])
            else:
                llm.reset()
                llm.eval(full_ids)
            vocab_logits = self._last_logits()
            ids = [self._label_id(label) for label in row_labels]
            if any(i is None for i in ids):
                raise ValueError(f"labels not single-token for this tokenizer: {row_labels}")
            results.append(vocab_logits[np.asarray(ids)])
        return results
