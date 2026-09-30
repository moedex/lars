"""Swapping LoRA adapters on one model matches loading each alone. Skipped where MLX is unavailable."""

import json

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx.utils import tree_flatten  # noqa: E402
from mlx_lm.models import qwen3  # noqa: E402
from mlx_lm.tuner.utils import linear_to_lora_layers, load_adapters  # noqa: E402

from lars.backends.adapters import AdapterSet  # noqa: E402

ATTN = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]
MLP = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
IDS = mx.array([[5, 17, 42, 8, 91, 3]])


def _tiny():
    mx.random.seed(0)
    args = qwen3.ModelArgs(model_type="qwen3", hidden_size=32, num_hidden_layers=2, intermediate_size=64,
                           num_attention_heads=4, rms_norm_eps=1e-6, vocab_size=97, num_key_value_heads=2,
                           max_position_embeddings=512, rope_theta=10000.0, head_dim=8, tie_word_embeddings=True)
    model = qwen3.Model(args)
    mx.eval(model.parameters())
    return model


def _adapter(path, keys, seed, rank=4, scale=2.0):
    model = _tiny()
    params = {"rank": rank, "scale": scale, "dropout": 0.0, "keys": keys}
    model.freeze()
    linear_to_lora_layers(model, 2, params)
    mx.random.seed(seed)
    weights = {k: mx.random.normal(v.shape) * 0.1 for k, v in tree_flatten(model.trainable_parameters())}
    path.mkdir()
    mx.save_safetensors(str(path / "adapters.safetensors"), weights)
    config = {"fine_tune_type": "lora", "num_layers": 2, "lora_parameters": params}
    (path / "adapter_config.json").write_text(json.dumps(config))
    return path


def _logits(model):
    out = model(IDS)
    mx.eval(out)
    return np.asarray(out)


def test_swapping_matches_each_adapter_alone(tmp_path):
    attn = _adapter(tmp_path / "attn", ATTN, seed=1)
    both = _adapter(tmp_path / "both", ATTN + MLP, seed=2)
    alone = [_logits(load_adapters(_tiny(), str(d))) for d in (attn, both)]
    base = _logits(_tiny())
    assert not np.allclose(alone[0], base) and not np.allclose(alone[0], alone[1])

    model = _tiny()
    adapters = AdapterSet(model, [str(attn), str(both)])
    for index in (0, 1, 0, 1):
        adapters.use(index)
        np.testing.assert_allclose(_logits(model), alone[index], rtol=1e-5, atol=1e-5)


def test_adapters_must_share_rank_and_scale(tmp_path):
    a = _adapter(tmp_path / "a", ATTN, seed=1, rank=4)
    b = _adapter(tmp_path / "b", ATTN, seed=2, rank=2)
    with pytest.raises(ValueError, match="rank or scale"):
        AdapterSet(_tiny(), [str(a), str(b)])
