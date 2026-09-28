"""LoRA training through the label readout, on a tiny random Qwen3. Skipped where MLX is unavailable."""

import random

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx.utils import tree_flatten  # noqa: E402
from mlx_lm.models import qwen3  # noqa: E402
from mlx_lm.tuner.utils import load_adapters  # noqa: E402

from moelars.render import TEMPLATES  # noqa: E402
from moelars.train import lora  # noqa: E402
from moelars.train.data import Record  # noqa: E402

VOCAB = 97


def _tiny(tie: bool = True, seed: int = 0):
    mx.random.seed(seed)
    args = qwen3.ModelArgs(model_type="qwen3", hidden_size=32, num_hidden_layers=2, intermediate_size=64,
                           num_attention_heads=4, rms_norm_eps=1e-6, vocab_size=VOCAB, num_key_value_heads=2,
                           max_position_embeddings=512, rope_theta=10000.0, head_dim=8, tie_word_embeddings=tie)
    model = qwen3.Model(args)
    mx.eval(model.parameters())
    return model


class _Backend:
    """Character-level stand-in for MLXBackend: one token per character, single-letter labels."""

    def __init__(self, model):
        self.model = model

    def template(self):
        return TEMPLATES["chatml"]

    def is_single_token(self, label):
        return len(label) == 1

    def _encode(self, text):
        return [ord(c) % (VOCAB - 1) + 1 for c in text]

    def _label_id(self, label):
        return ord(label) % (VOCAB - 1) + 1 if len(label) == 1 else None


def _find(ids, needle):
    return next(i for i in range(len(ids)) if ids[i : i + len(needle)] == needle)


def _records():
    choice = Record("c1", "src/a", "choice", "The printer is jammed again.", "Which team?",
                    ["billing", "hardware", "sales"], [0.0, 1.0, 0.0])
    noul = Record("n1", "src/b", "noul", "Refund me now!", "Is the sender angry?", ["yes", "no"], [1.0, 0.0])
    return [choice, noul]


@pytest.mark.parametrize("tie", [True, False])
def test_readout_matches_each_row_run_alone(tie):
    """Right padding must not reach a row's last token, and only the answer position is projected."""
    model = _tiny(tie)
    rows = [[5, 9, 13, 21, 34], [7, 3, 2]]
    labels = [[11, 12, 13], [14, 15, 0]]
    examples = [lora.Example(r, [x for x in lab if x], np.ones(len([x for x in lab if x])) / 3, "s")
                for r, lab in zip(rows, labels, strict=True)]
    z = np.asarray(lora.readout(mx, model, *lora.collate(mx, examples)[:4]))
    for i, example in enumerate(examples):
        alone = np.asarray(model(mx.array(example.ids)[None])[0, -1])[example.label_ids]
        assert np.allclose(z[i, : len(example.label_ids)], alone, atol=1e-4)
    assert z[1, 2] < -1e8  # padded option


def test_batches_respect_the_token_budget_and_keep_every_example():
    examples = [lora.Example(list(range(n)), [1, 2], np.array([1.0, 0.0]), "s") for n in (3, 50, 7, 49, 20, 1)]
    groups = lora.batches(examples, batch_tokens=100, rng=random.Random(0))
    assert sorted(len(e.ids) for g in groups for e in g) == [1, 3, 7, 20, 49, 50]
    for group in groups:
        assert max(len(e.ids) for e in group) * len(group) <= 100 or len(group) == 1


def test_encode_shuffles_choice_targets_with_the_presented_order():
    backend = _Backend(_tiny())
    choice, noul = _records()
    labels = ["A", "B", "C"]
    example = lora.encode(backend, choice, labels, [2, 0, 1])
    assert example.target.tolist() == [0.0, 0.0, 1.0]  # "hardware" is presented third
    assert example.label_ids == [backend._label_id(x) for x in labels]
    at = {key: _find(example.ids, backend._encode(key)) for key in ("sales", "billing", "hardware")}
    assert at["sales"] < at["billing"] < at["hardware"]
    assert lora.encode(backend, noul, labels, [0, 1]).target.tolist() == [1.0, 0.0]


def test_lora_fits_a_readout_that_depends_on_content():
    """LoRA alone (frozen backbone) must be able to move the label readout toward a content-dependent target."""
    import mlx.nn as nn
    import mlx.optimizers as optim

    model = _tiny()
    lora.add_lora(model, num_layers=-1, rank=8, scale=8.0, dropout=0.0)
    examples = [lora.Example([first, 40, 41, 42, 43], [11, 12], np.array(target, np.float32), "s")
                for first, target in [(1, [1.0, 0.0]), (2, [0.0, 1.0]), (3, [1.0, 0.0]), (4, [0.0, 1.0])]]
    batch = lora.collate(mx, examples)
    optimizer = optim.Adam(learning_rate=3e-3)  # 1e-2 oscillates: LoRA's scale multiplies every update
    step = nn.value_and_grad(model, lambda m, *b: lora.loss_fn(mx, m, *b, 1.0))
    first = None
    for _ in range(150):
        loss, grads = step(model, *batch)
        optimizer.update(model, grads)
        mx.eval(model.trainable_parameters(), optimizer.state)
        first = float(loss) if first is None else first
    assert float(loss) < 0.05 < first


def test_training_loop_keeps_the_backbone_frozen_and_saves_the_best_adapter(tmp_path):
    backend = _Backend(_tiny())
    frozen_before = {k: np.asarray(v) for k, v in tree_flatten(backend.model.parameters())}
    history = lora.train(backend, _records() * 8, _records(), tmp_path / "adapter", epochs=3, lr=3e-3, rank=4,
                         scale=2.0, batch_tokens=4096, max_tokens=2048, eval_every=0, grad_checkpoint=False)
    assert len(history) == 3 and all(entry["loss"] > 0 for entry in history)
    # LoRA wraps each adapted base layer, so `q_proj.weight` now lives at `q_proj.linear.weight`.
    trained = {k.replace(".linear.", "."): v for k, v in tree_flatten(backend.model.parameters())}
    for name, before in frozen_before.items():
        assert np.array_equal(before, np.asarray(trained[name])), f"base weight {name} moved"
    assert any("lora_a" in k for k in trained)
    assert (tmp_path / "adapter" / "adapters.safetensors").exists()
    assert (tmp_path / "adapter.history.json").exists()


def test_saved_adapter_loads_with_mlx_lm(tmp_path):
    """The adapter directory must be loadable by mlx_lm's own loader, which is what --adapter uses,
    and the trained model is left at the checkpoint that was saved."""
    backend = _Backend(_tiny(seed=1))
    lora.train(backend, _records() * 4, _records(), tmp_path / "adapter", epochs=2, lr=3e-3, rank=4, scale=2.0,
               eval_every=0, grad_checkpoint=False)
    examples, _ = lora.presentations(backend, _records(), ["A", "B", "C"], rng=None, max_tokens=2048)
    batch = lora.collate(mx, examples)[:4]
    trained = np.asarray(lora.readout(mx, backend.model, *batch))

    fresh = load_adapters(_tiny(seed=1), str(tmp_path / "adapter"))
    assert np.allclose(np.asarray(lora.readout(mx, fresh, *batch)), trained, atol=1e-4)


def test_keys_restrict_lora_to_attention_and_reload_the_same_way(tmp_path):
    backend = _Backend(_tiny(seed=2))
    lora.train(backend, _records() * 2, _records(), tmp_path / "adapter", epochs=1, lr=3e-3, rank=4, scale=2.0,
               eval_every=0, grad_checkpoint=False, keys=lora.KEY_PRESETS["attn"])
    adapted = {k.split(".lora_")[0] for k, _ in tree_flatten(backend.model.trainable_parameters())}
    assert adapted and all(".self_attn." in k for k in adapted), adapted
    assert len(adapted) == 2 * 4  # two blocks, four attention projections

    examples, _ = lora.presentations(backend, _records(), ["A", "B", "C"], rng=None, max_tokens=2048)
    batch = lora.collate(mx, examples)[:4]
    fresh = load_adapters(_tiny(seed=2), str(tmp_path / "adapter"))
    assert np.allclose(np.asarray(lora.readout(mx, fresh, *batch)),
                       np.asarray(lora.readout(mx, backend.model, *batch)), atol=1e-4)


def _tiny_moe(seed: int = 0):
    from mlx_lm.models import qwen3_moe

    mx.random.seed(seed)
    args = qwen3_moe.ModelArgs(model_type="qwen3_moe", hidden_size=32, num_hidden_layers=2, intermediate_size=64,
                               num_attention_heads=4, num_experts=8, num_experts_per_tok=2, decoder_sparse_step=1,
                               mlp_only_layers=[], moe_intermediate_size=16, rms_norm_eps=1e-6, vocab_size=VOCAB,
                               num_key_value_heads=2, head_dim=8, rope_theta=10000.0, tie_word_embeddings=False,
                               max_position_embeddings=512, norm_topk_prob=True)
    model = qwen3_moe.Model(args)
    mx.eval(model.parameters())
    return model


@pytest.mark.parametrize("keys", ["attn", "attn+experts"])
def test_lora_trains_through_a_mixture_of_experts_block(tmp_path, keys):
    """Qwen3-MoE routes by argpartition; without a stop_gradient on the indices the first backward pass fails."""
    model = _tiny_moe()
    examples, _ = lora.presentations(_Backend(model), _records(), ["A", "B", "C"], rng=None, max_tokens=2048)
    batch = lora.collate(mx, examples)[:4]
    before = np.asarray(lora.readout(mx, model, *batch))
    lora.stop_router_index_gradients()
    assert np.allclose(np.asarray(lora.readout(mx, model, *batch)), before, atol=1e-5)  # same forward pass

    backend = _Backend(model)
    history = lora.train(backend, _records() * 4, _records(), tmp_path / "adapter", epochs=2, lr=3e-3, rank=4,
                         scale=2.0, eval_every=0, grad_checkpoint=True, keys=lora.KEY_PRESETS[keys])
    assert len(history) == 2 and all(np.isfinite(entry["loss"]) for entry in history)
    adapted = {k.split(".lora_")[0] for k, _ in tree_flatten(model.trainable_parameters())}
    assert any("switch_mlp" in k for k in adapted) == (keys == "attn+experts")
    assert not any(k.endswith("mlp.gate") for k in adapted)  # the router is never adapted


def test_a_run_that_never_beats_the_untrained_model_saves_the_identity(tmp_path):
    """With a learning rate that only hurts, the saved adapter must be the untrained one, not the last state."""
    import json

    backend = _Backend(_tiny(seed=3))
    examples, _ = lora.presentations(backend, _records(), ["A", "B", "C"], rng=None, max_tokens=2048)
    batch = lora.collate(mx, examples)[:4]
    before = np.asarray(lora.readout(mx, backend.model, *batch))
    out = tmp_path / "adapter"
    out.mkdir()
    (out / "adapters.safetensors").write_bytes(b"stale")  # must not pass for this run's selection
    # Train on the opposite of the held-out targets, so held-out Brier only gets worse.
    flipped = [Record(r.id, r.source, r.kind, r.state, r.question, r.options, r.target[::-1]) for r in _records()]
    history = lora.train(backend, flipped * 8, _records(), out, epochs=2, lr=3e-2, rank=4, scale=2.0,
                         eval_every=0, grad_checkpoint=False)
    assert not any(entry.get("selected") for entry in history)
    assert json.loads((out / "adapter_config.json").read_text())["moelars"]["improved"] is False
    assert (tmp_path / "adapter.last" / "adapters.safetensors").exists()
    assert np.allclose(np.asarray(lora.readout(mx, backend.model, *batch)), before, atol=1e-5)
    fresh = load_adapters(_tiny(seed=3), str(out))
    assert np.allclose(np.asarray(lora.readout(mx, fresh, *batch)), before, atol=1e-5)


def test_selection_rows_decide_the_checkpoint_and_every_step_is_kept(tmp_path):
    """With --select-records, the checkpoint follows macro accuracy on those rows even when held-out
    Brier gets worse, and every evaluated checkpoint is saved under <out>.steps/."""
    import json

    backend = _Backend(_tiny(seed=3))
    flipped = [Record(r.id, r.source, r.kind, r.state, r.question, r.options, r.target[::-1]) for r in _records()]
    out = tmp_path / "adapter"
    # Training on the flipped targets worsens held-out Brier on the originals but fits the flipped
    # selection rows, so selection by held-out Brier would keep the identity and this must not.
    history = lora.train(backend, flipped * 8, _records(), out, epochs=3, lr=3e-3, rank=4, scale=2.0,
                         eval_every=0, grad_checkpoint=False, select_records=flipped)
    assert history[-1]["select"]["macro_acc"] > 0.0
    assert any(entry.get("selected") for entry in history)
    assert all("select" in entry and entry["select"]["sources"] == 2 for entry in history)
    assert json.loads((out / "adapter_config.json").read_text())["moelars"]["improved"] is True
    assert sorted(p.name for p in (tmp_path / "adapter.steps").iterdir()) == [f"step-{e['step']}" for e in history]
