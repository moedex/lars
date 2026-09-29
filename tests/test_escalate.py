"""Escalation to a hosted model: threshold, blend, opt-out, failures, and the startup key gate."""

import types

import anyio
import pytest
from fastapi.testclient import TestClient

from moelars.backends.mock import MockBackend
from moelars.engine import Engine
from moelars.escalate import Escalator, resolve_api_key, top_probability
from moelars.schema import SystemOneRequest
from moelars.server import create_app

BODY = {"state": "The build failed: 3 tests errored.", "questions": {
    "failed": {"type": "noul", "instructions": "A test failed"},
    "area": {"type": "choice", "instructions": "Area?", "criteria": {"x": "X", "y": "Y"}},
    "level": {"type": "score", "instructions": "How bad?", "criteria": ["low", "mid", "high"]}}}


class FakeMessages:
    def __init__(self, picks, fail=False):
        self.picks, self.fail, self.prompts = picks, fail, []

    async def create(self, **kwargs):
        self.prompts.append(kwargs)
        if self.fail:
            raise RuntimeError("boom")
        options = kwargs["tools"][0]["input_schema"]["properties"]["answer"]["enum"]
        pick = next(p for p in self.picks if p in options)
        return types.SimpleNamespace(content=[types.SimpleNamespace(type="tool_use", input={"answer": pick})])


def _escalator(picks=("no", "y", "2"), fail=False, below=1.01):
    messages = FakeMessages(picks, fail)
    return Escalator("k", below=below, client=types.SimpleNamespace(messages=messages)), messages


def _run(escalator, body=BODY):
    request = SystemOneRequest.model_validate(body)
    local = Engine(MockBackend()).evaluate(request)
    return local, anyio.run(escalator.apply, request, local.model_copy(deep=True))


def test_low_confidence_answers_take_the_hosted_pick():
    escalator, messages = _escalator()
    local, out = _run(escalator)
    assert len(messages.prompts) == 3 and messages.prompts[0]["tool_choice"] == {"type": "tool", "name": "answer"}
    assert out.answers["failed"].noul < 0.5 < local.answers["failed"].noul  # hosted "no" wins at weight 0.8
    assert out.answers["area"].choice == "y" and out.answers["area"].escalated == {"model": escalator.model,
                                                                                   "answer": "y"}
    assert abs(sum(out.answers["level"].probabilities.values()) - 1) < 1e-3 and out.answers["level"].score > 1.5


def test_threshold_opt_out_and_failures_keep_local_answers():
    escalator, messages = _escalator(below=0.0)
    local, out = _run(escalator)
    assert not messages.prompts and out == local
    escalator, messages = _escalator()
    local, out = _run(escalator, {**BODY, "moelars": {"escalate": False}})
    assert not messages.prompts and out == local
    escalator, _ = _escalator(fail=True)
    local, out = _run(escalator)
    assert out.answers["area"].probabilities == local.answers["area"].probabilities
    assert out.answers["area"].escalated["error"] == "RuntimeError" and escalator.errors == 3
    assert top_probability(local.answers["area"]) == max(local.answers["area"].probabilities.values())


def test_key_is_resolved_at_startup_and_absent_key_means_no_escalation(tmp_path, monkeypatch):
    from moelars.cli import _escalator_from_args, build_parser

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("MOELARS_ANTHROPIC_KEY_FILE", str(tmp_path / "missing"))
    assert resolve_api_key() is None
    args = build_parser().parse_args(["serve", "--escalate"])
    escalator, note = _escalator_from_args(args)
    assert escalator is None and "no API key" in note
    monkeypatch.delenv("MOELARS_API_KEY", raising=False)
    client = TestClient(create_app(Engine(MockBackend()), escalator=None, escalation_note=note))
    assert client.get("/v1/status").json()["escalation"]["enabled"] is False
    assert "escalated" not in str(client.post("/v1/systemone", json=BODY).json())
    (tmp_path / "key").write_text("sk-test\n")
    monkeypatch.setenv("MOELARS_ANTHROPIC_KEY_FILE", str(tmp_path / "key"))
    assert resolve_api_key() == "sk-test"
    pytest.importorskip("anthropic")
    escalator, note = _escalator_from_args(args)
    assert escalator is not None and note.startswith("on")


def test_server_escalates_after_the_local_pass(monkeypatch):
    monkeypatch.delenv("MOELARS_API_KEY", raising=False)
    escalator, messages = _escalator()
    client = TestClient(create_app(Engine(MockBackend()), escalator=escalator))
    body = client.post("/v1/systemone", json=BODY).json()
    assert body["answers"]["area"]["choice"] == "y" and len(messages.prompts) == 3
    assert client.get("/v1/status").json()["escalation"]["calls"] == 3
