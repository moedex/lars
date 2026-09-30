"""MLX backend plumbing that does not need a real checkpoint. Skipped where MLX is unavailable."""

from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from lars.backends.mlx import MLXBackend  # noqa: E402


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text.strip()] if len(text.strip()) == 1 else [1, 2]


def _backend(model) -> MLXBackend:
    backend = object.__new__(MLXBackend)
    backend._mx = mx
    backend.model = model
    backend._text = getattr(model, "language_model", model)
    backend.tokenizer = _Tokenizer()
    backend._label_cache = {}
    return backend


def test_vision_language_wrapper_resolves_text_stack():
    """Qwen3.5-style checkpoints keep the text model under `language_model`."""
    weight = mx.arange(128 * 8, dtype=mx.float32).reshape(128, 8)
    text = SimpleNamespace(args=SimpleNamespace(hidden_size=8, tie_word_embeddings=False),
                           model=SimpleNamespace(embed_tokens=None), lm_head=SimpleNamespace(weight=weight))
    backend = _backend(SimpleNamespace(language_model=text))
    assert backend.hidden_size == 8
    rows = backend.label_rows(["A", "B"])
    assert rows.shape == (2, 8)
    assert np.allclose(rows, np.asarray(weight)[[ord("A"), ord("B")]])


def test_plain_model_with_tied_embeddings():
    weight = mx.arange(128 * 4, dtype=mx.float32).reshape(128, 4)
    model = SimpleNamespace(args=SimpleNamespace(hidden_size=4, tie_word_embeddings=True),
                            model=SimpleNamespace(embed_tokens=SimpleNamespace(weight=weight)))
    backend = _backend(model)
    assert backend.hidden_size == 4
    assert np.allclose(backend.label_rows(["Z"]), np.asarray(weight)[[ord("Z")]])


def test_label_rows_rejects_multi_token_labels():
    backend = _backend(SimpleNamespace(args=SimpleNamespace(hidden_size=4, tie_word_embeddings=True),
                                       model=SimpleNamespace(embed_tokens=SimpleNamespace(weight=mx.zeros((128, 4))))))
    with pytest.raises(ValueError):
        backend.label_rows(["AB"])


def test_restored_snapshot_survives_writes_from_earlier_rows():
    """Gated DeltaNet layers write through `cache[i] = ...`; one row must not leak into the next."""
    from mlx_lm.models.cache import ArraysCache, KVCache

    backend = _backend(SimpleNamespace())
    backend._make_cache = lambda model: [ArraysCache(size=2), KVCache()]
    prefix = backend._make_cache(None)
    prefix[0][0], prefix[0][1] = mx.zeros((1, 3)), mx.ones((1, 3))
    prefix[1].update_and_fetch(mx.zeros((1, 1, 2, 4)), mx.zeros((1, 1, 2, 4)))
    snapshot = backend._snapshot(prefix)

    first = backend._restore(snapshot)
    first[0][1] = mx.full((1, 3), 7.0)
    first[1].update_and_fetch(mx.ones((1, 1, 1, 4)), mx.ones((1, 1, 1, 4)))
    second = backend._restore(snapshot)
    assert np.allclose(np.asarray(second[0][1]), 1.0)
    assert second[1].offset == 2


def test_empty_prefix_skips_the_prefill():
    """`mx.array([])` is float; an empty prefix must not reach the embedding gather."""
    calls = []

    def model(inputs, cache=None):
        assert inputs.dtype == mx.int32
        calls.append(inputs.shape[1])
        return mx.broadcast_to(mx.arange(128, dtype=mx.float32), (1, inputs.shape[1], 128))

    class Tokenizer(_Tokenizer):
        def encode(self, text, add_special_tokens=False):
            return super().encode(text) if text else []

    backend = _backend(model)
    backend.tokenizer = Tokenizer()
    backend._make_cache = lambda model: []
    (z,) = backend.label_logits("", ["xy"], [("A", "B")])
    assert calls == [2]
    assert np.allclose(z, [ord("A"), ord("B")])


def test_backend_caps_the_mlx_buffer_cache(monkeypatch):
    # Regression: an uncapped buffer cache grew to about 100 GB over a run of varied prompts.
    limits = []
    monkeypatch.setattr(mx, "set_cache_limit", lambda value: limits.append(value) or 0)
    monkeypatch.setenv("LARS_MLX_CACHE_GB", "2")
    import lars.backends.mlx as backend_module

    monkeypatch.setattr("mlx_lm.load", lambda *args, **kwargs: (object(), type("T", (), {"chat_template": None})()))
    backend_module.MLXBackend("unused")
    assert limits == [2 * 2**30]
