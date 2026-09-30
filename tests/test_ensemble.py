import numpy as np
import pytest

from lars.backends.mock import MockBackend
from lars.calibration import Calibrator
from lars.engine import Engine, EnsembleEngine, _AdapterView
from lars.schema import SystemOneRequest

STATE = "Help! My payouts have been failing for 3 days. This is the second time I have written in."
QUESTIONS = {
    "department": {"type": "choice", "instructions": "Which team should handle this?",
                   "criteria": {"billing": "Payments, payouts", "technical": "Bugs, outages", "sales": "Pricing"}},
    "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                    "criteria": ["Calm", "Frustrated but civil", "Very angry"]},
    "is_urgent": {"type": "noul", "instructions": "The message conveys urgency"},
    "topics": {"type": "multi", "instructions": "Which topics apply?", "criteria": {"payouts": None, "login": None}},
}


class TwoAdapterMock(MockBackend):
    """Adapter i shifts every logit row by a different fixed pattern."""

    def __init__(self):
        super().__init__()
        self.active = 0

    def use_adapter(self, index):
        self.active = index

    def label_logits(self, prefix, suffixes, labels):
        shift = [0.0, 1.5][self.active]
        return [z + shift * np.arange(len(z))[::-1] for z in super().label_logits(prefix, suffixes, labels)]


def _approx(a, b, tol=2e-4):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_approx(a[k], b[k], tol) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_approx(x, y, tol) for x, y in zip(a, b, strict=True))
    if isinstance(a, float):
        return abs(a - b) <= tol
    return a == b


CALIBRATORS = [Calibrator(), Calibrator(temperatures={"noul": 2.0, "choice": 0.7, "score": 1.3, "multi": 1.0})]


def test_ensemble_is_the_mean_of_its_members():
    backend = TwoAdapterMock()
    request = SystemOneRequest(state=STATE, questions=QUESTIONS)
    ensemble = EnsembleEngine(backend, CALIBRATORS).evaluate(request).answers
    members = [Engine(_AdapterView(backend, i), calibrator=c).evaluate(request).answers
               for i, c in enumerate(CALIBRATORS)]
    assert ensemble["is_urgent"].noul == pytest.approx(np.mean([m["is_urgent"].noul for m in members]), abs=1e-4)
    for qid in ("department", "frustration", "topics"):
        for key, p in ensemble[qid].probabilities.items():
            assert p == pytest.approx(np.mean([m[qid].probabilities[key] for m in members]), abs=1e-4)
    assert members[0]["department"].probabilities != members[1]["department"].probabilities
    keys = QUESTIONS["department"]["criteria"]
    mean = {k: np.mean([m["department"].probabilities[k] for m in members]) for k in keys}
    assert ensemble["department"].choice == max(mean, key=mean.get)


def test_identical_members_reproduce_a_single_engine():
    request = SystemOneRequest(state=STATE, questions=QUESTIONS, lars={"explain": True, "abstain_margin": 0.2})
    single = Engine(MockBackend()).evaluate(request).answers
    backend = TwoAdapterMock()
    backend.use_adapter = lambda index: None  # both members see adapter 0, which is the plain mock
    doubled = EnsembleEngine(backend, [None, None]).evaluate(request)
    for qid, answer in single.items():
        # Members round to 4 decimals before averaging, so recomputed fields can move by 1e-4.
        assert _approx(doubled.answers[qid].model_dump(), answer.model_dump())
    assert doubled.usage.output_tokens == 2 * Engine(MockBackend()).evaluate(request).usage.output_tokens


def test_ensemble_checks_the_model_name_and_applies_constraints_once():
    backend = TwoAdapterMock()
    ensemble = EnsembleEngine(backend, CALIBRATORS)
    with pytest.raises(ValueError, match="not served here"):
        ensemble.evaluate(SystemOneRequest(state=STATE, questions=QUESTIONS, model="other"))
    nouls = {f"q{i}": {"type": "noul", "instructions": f"Statement {i} holds"} for i in range(3)}
    request = SystemOneRequest(state=STATE, questions=nouls,
                               lars={"constraints": [{"kind": "exclusive", "questions": list(nouls)}]})
    assert sum(a.noul for a in ensemble.evaluate(request).answers.values()) <= 1.0
