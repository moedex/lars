"""The Molar Triage example must stay loadable and runnable with the mock backend."""

import runpy
from pathlib import Path

from lars.backends import load_backend
from lars.engine import Engine
from lars.evalset import evaluate, read_examples

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "molar_triage"


def test_builder_writes_consistent_splits(tmp_path, monkeypatch):
    module = runpy.run_path(str(EXAMPLE / "build.py"), run_name="not_main")
    messages = module["MESSAGES"]
    assert len(messages) >= 40
    departments = set(module["DEPARTMENTS"])
    for text, in_pain, department, urgency in messages:
        assert text and isinstance(in_pain, bool)
        assert department in departments
        assert 0 <= urgency < len(module["URGENCY"])


def test_checked_in_files_evaluate_with_mock():
    test = list(read_examples(EXAMPLE / "molar_triage.test.jsonl"))
    calibration = list(read_examples(EXAMPLE / "molar_triage.calibration.jsonl"))
    assert len(test) == len(calibration) == 90
    kinds = {e.question["type"] for e in test}
    assert kinds == {"noul", "choice", "score"}
    result = evaluate(Engine(load_backend("mock")), test)
    assert result.count == 90
    assert 0.0 <= result.accuracy <= 1.0
