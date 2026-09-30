import numpy as np
import pytest

from lars.heads import PointerHeadScorer
from lars.spans import char_offsets_to_token_indexes, option_end_char_offsets


def test_option_end_offsets_and_token_mapping():
    prefix = "SYSTEM\n<STATE x>\nhello\n</STATE x>\n\n"
    suffix = "QUESTION: q\nOPTIONS:\nA) billing: money\nB) technical\n<end>Answer:"
    ends = option_end_char_offsets(prefix, suffix, 2)
    full = prefix + suffix
    assert full[ends[0] - 5 : ends[0]] == "money"
    assert full[ends[1] - 9 : ends[1]] == "technical"
    # fake tokens of width 3
    offsets = [(i, min(i + 3, len(full))) for i in range(0, len(full), 3)]
    idx = char_offsets_to_token_indexes(offsets, ends)
    assert all(offsets[i][0] < e for i, e in zip(idx, ends, strict=True))
    assert all(i + 1 == len(offsets) or offsets[i + 1][0] >= e for i, e in zip(idx, ends, strict=True))


def test_zero_head_is_identity_and_bias_moves_yes():
    p, r = 8, 4
    head = PointerHeadScorer(np.zeros((r, p)), np.ones((r, p)), np.zeros(3), np.array([1.5, 0, 0]), np.eye(p))
    z = np.array([0.2, -0.1, 0.4])
    h_ans, h_opt = np.ones(p), np.ones((3, p))
    assert np.allclose(head.adjust(z, h_ans, h_opt, "choice"), z)
    adjusted = head.adjust(z[:2], h_ans, h_opt[:2], "noul")
    assert np.isclose(adjusted[0], z[0] + 1.5) and np.isclose(adjusted[1], z[1])


def test_scorer_matches_mlx_head(tmp_path):
    mx = pytest.importorskip("mlx.core")
    from lars.train.residual import build_head, save_head

    head = build_head(proj_dim=8, rank=4)
    head.q.weight = mx.random.normal((4, 8))
    head.log_s = mx.array([0.3, -0.2, 0.1])
    save_head(head, tmp_path / "head.npz")
    proj = np.eye(8, dtype=np.float32)
    np.save(tmp_path / "projection.npy", proj)
    scorer = PointerHeadScorer.load(tmp_path / "head.npz", tmp_path / "projection.npy")

    rng = np.random.default_rng(0)
    h_ans, h_opt, z = rng.standard_normal(8), rng.standard_normal((3, 8)), rng.standard_normal(3)
    expected = np.asarray(
        head(mx.array(h_ans[None]), mx.array(h_opt[None]), mx.array(z[None]), mx.array(np.ones((1, 3), bool)),
             mx.array(np.array([1])))
    )[0]
    # float32 GPU matmul in MLX versus float64 numpy: agree to ~1e-3
    assert np.allclose(scorer.adjust(z, h_ans, h_opt, "choice"), expected, atol=5e-3)
