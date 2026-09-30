"""Head math on tiny synthetic shards. Skipped where MLX is unavailable."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from lars.train.residual import Shard, baseline, build_head, evaluate, train  # noqa: E402


def _synthetic_shard(tmp_path, n=240, k=4, p=16, seed=0):
    """Backbone logits z are weak; the hidden features carry the answer via a fixed direction."""
    rng = np.random.default_rng(seed)
    gold = rng.integers(0, k, n)
    direction = rng.standard_normal(p)
    h_ans = rng.standard_normal((n, p)).astype(np.float32)
    h_opt = rng.standard_normal((n, k, p)).astype(np.float32)
    for i in range(n):
        h_opt[i, gold[i]] += 2.0 * direction / np.linalg.norm(direction)
        h_ans[i] += direction / np.linalg.norm(direction)
    z = rng.standard_normal((n, k)).astype(np.float32) * 0.3
    z[np.arange(n), gold] += 0.4  # backbone is right more often than chance, but not by much
    target = np.eye(k, dtype=np.float32)[gold]
    mask = np.ones((n, k), bool)
    perm = np.tile(np.arange(k), (n, 1)).astype(np.int32)
    ids = np.asarray([f"r{i}" for i in range(n)])
    kinds = np.asarray(["choice"] * n)
    path = tmp_path / "shard.npz"
    np.savez(path, h_ans=h_ans.astype(np.float16), h_opt=h_opt.astype(np.float16), z=z, target=target, mask=mask,
             perm=perm, ids=ids, kinds=kinds)
    return Shard(path)


def test_zero_init_head_reproduces_backbone(tmp_path):
    shard = _synthetic_shard(tmp_path)
    head = build_head(shard.h_ans.shape[1], rank=8)
    logits = np.asarray(head(mx.array(shard.h_ans[:5]), mx.array(shard.h_opt[:5]), mx.array(shard.z[:5]),
                             mx.array(shard.mask[:5]), mx.array(shard.kinds[:5])))
    assert np.allclose(logits, shard.z[:5], atol=1e-5)
    assert baseline(mx, shard)["acc"] == evaluate(mx, head, shard)["acc"]


def test_permutation_loss_ignores_padding():
    """A shuffled twin of a padded row must give zero KL at the identity head."""
    from lars.train.residual import perm_loss_fn

    p, k_max = 8, 5
    head = build_head(p, rank=4)
    h_ans = np.zeros((1, p), np.float32)
    h_opt = np.zeros((1, k_max, p), np.float32)
    z_base = np.array([[2.0, 0.5, -1.0, 0.0, 0.0]], np.float32)  # K=3 real options, 2 padded
    z_shuf = np.array([[-1.0, 2.0, 0.5, 0.0, 0.0]], np.float32)  # same options in order [2, 0, 1]
    mask = np.array([[True, True, True, False, False]])
    perm = np.array([[2, 0, 1, -1, -1]], np.int32)
    kind = np.array([1])
    base = (mx.array(h_ans), mx.array(h_opt), mx.array(z_base), mx.array(mask), mx.array(kind), None)
    shuf = (mx.array(h_ans), mx.array(h_opt), mx.array(z_shuf), mx.array(mask), mx.array(kind), None)
    kl = float(perm_loss_fn(mx, head, base, shuf, mx.array(perm)))
    assert abs(kl) < 1e-4, kl


def test_training_improves_on_backbone(tmp_path):
    shard = _synthetic_shard(tmp_path)
    before = baseline(mx, shard)["acc"]
    _, history = train(shard, None, rank=8, epochs=6, batch_size=32, lr=5e-3, perm_weight=0.0)
    after = history[-1]["train"]["acc"]
    assert after > before + 0.15, (before, after)


def test_selection_keeps_the_backbone_when_no_epoch_beats_it(tmp_path):
    """With lr 0 every epoch equals the untrained head, so nothing is selected and the saved head is the identity."""
    shard = _synthetic_shard(tmp_path)
    head, history = train(shard, shard, rank=8, epochs=2, batch_size=32, lr=0.0, perm_weight=0.0)
    assert not any(entry.get("selected") for entry in history)
    assert evaluate(mx, head, shard) == baseline(mx, shard)


def test_a_shard_the_manifest_does_not_describe_is_refused(tmp_path):
    import json

    _synthetic_shard(tmp_path)  # no manifest: accepted
    (tmp_path / "manifest.json").write_text(json.dumps({"shard": {"presentations": 1}}))
    with pytest.raises(ValueError, match="left over"):
        Shard(tmp_path / "shard.npz")
