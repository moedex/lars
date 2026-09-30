import numpy as np

from lars.calibration import (
    Calibrator,
    _platt_targets,
    brier,
    coverage_at_error,
    ece,
    fit_platt,
    fit_temperature,
)


def test_fit_temperature_recovers_flatter_scale_for_overconfident_logits():
    rng = np.random.default_rng(0)
    logits, targets = [], []
    for _ in range(200):
        true = rng.integers(0, 3)
        row = rng.normal(0, 1, 3) * 4  # overconfident
        row[true] += 2.0
        if rng.random() < 0.3:  # 30% label noise makes sharp logits wrong
            true = rng.integers(0, 3)
        logits.append(row)
        targets.append(np.eye(3)[true])
    t = fit_temperature(logits, targets)
    assert t > 1.0


def test_fit_platt_learns_bias():
    rng = np.random.default_rng(1)
    scores = rng.normal(0, 2, 500)
    labels = (scores + 1.5 > 0).astype(float)  # shifted decision boundary
    a, b = fit_platt(scores, labels)
    assert a > 0
    assert b > 0.5


def test_fit_platt_survives_wide_nearly_separable_logits():
    """Raw LM logit differences: tens of units wide, 89% accurate. The fit must not flip or explode."""
    rng = np.random.default_rng(2)
    labels = (rng.random(200) < 0.6).astype(float)
    z = np.where(labels > 0.5, 1.0, -1.0) * rng.uniform(8, 30, 200)
    flip = rng.random(200) < 0.11
    z[flip] = -z[flip]
    a, b = fit_platt(z, labels)
    p = 1 / (1 + np.exp(-(a * z + b)))
    assert a > 0
    assert np.isfinite(a) and np.isfinite(b)
    assert ((p > 0.5) == (labels > 0.5)).mean() >= 0.85
    assert p.max() < 0.999  # no longer pinned to 1.0


def test_ece_and_brier_basic():
    conf = np.array([0.9, 0.9, 0.6, 0.6])
    hit = np.array([1, 1, 0, 1])
    assert 0 <= ece(conf, hit) <= 1
    assert brier([np.array([1.0, 0.0])], [np.array([1.0, 0.0])]) == 0.0


def test_coverage_at_error():
    conf = np.array([0.99, 0.95, 0.7, 0.6])
    hit = np.array([1, 1, 0, 1])
    cov, thr = coverage_at_error(conf, hit, 0.0)
    assert cov == 0.5 and thr == 0.95


def test_calibrator_roundtrip(tmp_path):
    c = Calibrator(temperatures={"noul": 1.7}, platt={"noul": (0.9, 0.2)}, fitted_on="x")
    path = tmp_path / "cal.json"
    c.save(path)
    loaded = Calibrator.load(path)
    assert loaded.temperature_for("noul") == 1.7
    assert loaded.platt_for("noul") == (0.9, 0.2)
    assert loaded.temperature_for("choice") == 1.0


def test_coverage_never_reports_a_threshold_over_its_error_budget():
    # The review's case: two decisions at the same confidence, one wrong.
    assert coverage_at_error(np.array([0.9, 0.9]), np.array([1, 0]), 0.0) == (0.0, 1.0)
    cov, thr = coverage_at_error(np.array([0.95, 0.9, 0.9, 0.8]), np.array([1, 1, 0, 1]), 0.0)
    assert (cov, thr) == (0.25, 0.95)


def test_platt_keeps_soft_targets():
    y = _platt_targets(np.array([1.0, 0.0, 0.6, 1.0]))
    assert y[2] == 0.6
    assert 0.5 < y[0] < 1.0 and 0.0 < y[1] < 0.5
