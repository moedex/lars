"""evals/remote.py scores a System One server with the suite's metrics and calibration."""

import json
import sys
from pathlib import Path

import httpx
import numpy as np
import pytest

from lars.evalset import Example, calibrate, evaluate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals"))
from remote import PROBABILITY_FLOOR, Refused, RemoteClient, RemoteEngine  # noqa: E402

CHOICE = {"type": "choice", "instructions": "Which?", "criteria": {"a": "A", "b": "B", "c": "C"}}
SCORE = {"type": "score", "instructions": "How much?", "criteria": ["low", "mid", "high"]}
NOUL = {"type": "noul", "instructions": "True?"}


def _server(calls: list[dict]) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        question = body["questions"]["q"]
        if question["type"] == "noul":
            answer = {"type": "noul", "noul": 0.8 if body["state"] == "yes" else 0.3}
        elif question["type"] == "score":
            answer = {"type": "score", "probabilities": {"0": 0.1, "1": 0.7, "2": 0.2}}
        else:
            answer = {"type": "choice", "probabilities": {"a": 0.0, "b": 0.9, "c": 0.1}}
        return httpx.Response(200, json={"answers": {"q": answer}})

    return httpx.MockTransport(handle)


def test_remote_engine_reads_probabilities_as_logits_and_caches_rows():
    calls: list[dict] = []
    client = RemoteClient("http://remote", transport=_server(calls))
    examples = [Example("s", CHOICE, "b"), Example("s", SCORE, "2"), Example("yes", NOUL, "1"),
                Example("no", NOUL, "1")]
    result = evaluate(RemoteEngine(client), examples)
    assert result.accuracy == pytest.approx(0.5)
    keys, logits = RemoteEngine(client).raw_logits("s", "q", CHOICE)
    assert keys == ("a", "b", "c")
    assert logits[0] == pytest.approx(np.log(PROBABILITY_FLOOR))
    _, noul = RemoteEngine(client).raw_logits("yes", "q", NOUL)
    assert noul[0] - noul[1] == pytest.approx(np.log(0.8 / 0.2))
    assert len(calls) == len(examples)


def test_remote_engine_is_calibrated_like_a_local_one():
    client = RemoteClient("http://remote", transport=_server([]))
    rows = [Example("yes", NOUL, "0")] * 4 + [Example("no", NOUL, "0")] * 4
    calibrator = calibrate(RemoteEngine(client), rows)
    p_yes = RemoteEngine(client, calibrator).noul_prob(float(np.log(0.8 / 0.2)))
    assert p_yes < 0.5


def test_remote_engine_rejects_a_different_option_set():
    client = RemoteClient("http://remote", transport=_server([]))
    question = {**CHOICE, "criteria": {"x": "X", "y": "Y"}}
    with pytest.raises(ValueError, match="expected"):
        RemoteEngine(client).raw_logits("s", "q", question)


def test_a_422_is_a_refusal_not_a_crash():
    transport = httpx.MockTransport(lambda request: httpx.Response(422, json={"detail": "too many options"}))
    with pytest.raises(Refused, match="too many options"):
        RemoteEngine(RemoteClient("http://remote", transport=transport)).raw_logits("s", "q", CHOICE)
