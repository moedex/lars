import numpy as np

from lars.primitives import (
    choice_confidence,
    expected_score,
    order_sensitivity,
    score_confidence,
    softmax,
    top_margin,
    total_variation,
)


def test_softmax_sums_to_one_and_temperature_flattens():
    logits = np.array([2.0, 0.0, -1.0])
    p1 = softmax(logits, 1.0)
    p5 = softmax(logits, 5.0)
    assert abs(p1.sum() - 1) < 1e-9
    assert p1.max() > p5.max()
    assert p1.argmax() == p5.argmax()


def test_choice_confidence_matches_published_formula():
    assert choice_confidence(np.array([1.0, 0.0, 0.0])) == 1.0
    assert abs(choice_confidence(np.array([1 / 3, 1 / 3, 1 / 3]))) < 1e-9
    probs = np.array([0.85, 0.15, 0.0])
    assert abs(choice_confidence(probs) - (3 * 0.85 - 1) / 2) < 1e-9


def test_expected_score_is_fractional():
    assert abs(expected_score(np.array([0.0, 0.57, 0.43])) - 1.43) < 1e-9


def test_score_confidence_prefers_adjacent_mass():
    one_hot = np.array([0.0, 1.0, 0.0])
    adjacent = np.array([0.0, 0.5, 0.5])
    split = np.array([0.5, 0.0, 0.5])
    assert score_confidence(one_hot) == 1.0
    assert score_confidence(adjacent) > score_confidence(split)


def test_order_sensitivity_zero_when_identical():
    d = np.array([0.7, 0.2, 0.1])
    assert order_sensitivity([d, d.copy()]) == 0.0
    assert order_sensitivity([d, np.array([0.2, 0.7, 0.1])]) > 0.0


def test_margin_and_tvd():
    assert abs(top_margin(np.array([0.6, 0.3, 0.1])) - 0.3) < 1e-9
    assert abs(total_variation(np.array([1.0, 0.0]), np.array([0.0, 1.0])) - 1.0) < 1e-9
