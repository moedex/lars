"""Evidence fusion: caller-supplied numeric features combined with the model logit by a fitted logistic."""

import json

import numpy as np
import pytest

from lars.backends.mock import MockBackend
from lars.calibration import Calibrator, fit_fusion, fit_logistic, fused_probability
from lars.engine import Engine
from lars.evalset import calibrate, evaluate, read_examples
from lars.schema import SystemOneRequest

QUESTION = {"type": "noul", "instructions": "The session shows something new"}


def test_fit_logistic_recovers_weights_on_raw_scale():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4000, 2)) * [1.0, 10.0] + [0.0, 50.0]
    true_w, true_b = np.array([2.0, -0.3]), 14.0
    y = (rng.random(4000) < 1 / (1 + np.exp(-(x @ true_w + true_b)))).astype(float)
    w, b = fit_logistic(x, y, l2=1e-6)
    assert np.allclose(w, true_w, atol=0.15)
    assert abs(b - true_b) < 1.5


def test_fusion_uses_features_the_model_logit_cannot_see():
    # The logit is noise; the evidence carries the label. Fusion should lean on the evidence.
    rng = np.random.default_rng(1)
    novel = rng.random(400)
    labels = (novel > 0.5).astype(float)
    logits = rng.normal(size=400)
    spec = fit_fusion(logits, [{"novel_share": float(v)} for v in novel], labels)
    assert spec["features"] == ["novel_share"]
    assert abs(spec["weights"][0]) < 0.5 < spec["weights"][1]
    assert fused_probability(spec, 0.0, {"novel_share": 0.95}) > 0.9
    assert fused_probability(spec, 0.0, {"novel_share": 0.05}) < 0.1


def test_engine_applies_fusion_only_when_its_features_are_present():
    calibrator = Calibrator(fusion={"noul": {"features": ["novel_share"], "weights": [0.0, 10.0], "bias": -5.0}})
    engine = Engine(MockBackend(), calibrator=calibrator)
    plain = engine.evaluate(SystemOneRequest(state="s", questions={"q": QUESTION}))
    fused = engine.evaluate(SystemOneRequest(
        state="s", questions={"q": QUESTION}, lars={"features": {"q": {"novel_share": 0.0}}}
    ))
    other = engine.evaluate(SystemOneRequest(
        state="s", questions={"q": QUESTION}, lars={"features": {"q": {"unrelated": 1.0}}}
    ))
    assert fused.answers["q"].noul == pytest.approx(1 / (1 + np.exp(5.0)), abs=1e-4)
    assert other.answers["q"].noul == plain.answers["q"].noul, "missing fitted features fall back to the model"


def test_plain_requests_are_unchanged_and_features_must_target_nouls():
    request = SystemOneRequest(state="s", questions={"q": QUESTION})
    assert request.lars.features == {}
    with pytest.raises(ValueError, match="unknown question"):
        SystemOneRequest(state="s", questions={"q": QUESTION}, lars={"features": {"x": {"f": 1.0}}})
    choice = {"type": "choice", "instructions": "Which?", "criteria": {"a": None, "b": None}}
    with pytest.raises(ValueError, match="must be a noul"):
        SystemOneRequest(state="s", questions={"c": choice}, lars={"features": {"c": {"f": 1.0}}})


def test_calibrate_fits_fusion_from_eval_rows_with_features(tmp_path):
    path = tmp_path / "rows.jsonl"
    with path.open("w") as handle:
        for i in range(60):
            share = (i % 10) / 10
            label = "1" if share >= 0.5 else "0"
            row = {"state": f"session {i}", "question": QUESTION, "label": label, "features": {"novel_share": share}}
            handle.write(json.dumps(row) + "\n")
    examples = list(read_examples(path))
    assert examples[0].features == {"novel_share": 0.0}
    engine = Engine(MockBackend())
    calibrator = calibrate(engine, examples, source=str(path))
    assert calibrator.fusion["noul"]["features"] == ["novel_share"]
    before = evaluate(engine, examples)
    after = evaluate(Engine(MockBackend(), calibrator=calibrator), examples)
    assert after.accuracy > before.accuracy
    saved = tmp_path / "cal.json"
    calibrator.save(saved)
    assert Calibrator.load(saved).fusion == calibrator.fusion
