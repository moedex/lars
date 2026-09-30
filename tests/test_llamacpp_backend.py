"""llama.cpp backend plumbing with a stand-in llama_cpp module (the real one is built in the Docker image)."""

import ctypes
import sys
import types

import numpy as np

from lars.backends.llamacpp import LlamaCppBackend

VOCAB = 8


class _FakeLlama:
    """Keeps logits in the context, like llama.cpp, and leaves `scores` at zero, like llama-cpp-python 0.3.35."""

    def __init__(self):
        self.ctx = object()
        self.n_tokens = 0
        self.scores = np.zeros((64, VOCAB), dtype=np.float32)
        self.logits = (ctypes.c_float * VOCAB)()

    def n_vocab(self):
        return VOCAB

    def tokenize(self, text, add_bos=False, special=False):
        ids = [1] if add_bos else []
        return ids + [2 + (b % (VOCAB - 2)) for b in text]

    def reset(self):
        self.n_tokens = 0

    def eval(self, ids):
        self.n_tokens += len(ids)
        for i in range(VOCAB):
            self.logits[i] = float((sum(ids) * (i + 1)) % 7)

    def save_state(self):
        return self.n_tokens

    def load_state(self, state):
        self.n_tokens = state


def test_label_logits_come_from_the_context_not_the_zeroed_scores(monkeypatch):
    fake = _FakeLlama()
    module = types.SimpleNamespace(llama_get_logits_ith=lambda ctx, i: fake.logits)
    monkeypatch.setitem(sys.modules, "llama_cpp", module)
    backend = object.__new__(LlamaCppBackend)
    backend.llm, backend._label_cache = fake, {}
    backend._label_id = lambda label: {"A": 3, "B": 5}[label]
    (z,) = backend.label_logits("prefix ", ["suffix"], [("A", "B")])
    assert z.shape == (2,) and z[0] != z[1]
    np.testing.assert_array_equal(z, [fake.logits[3], fake.logits[5]])
