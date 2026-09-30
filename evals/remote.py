"""Score another System One server with the suite's own metrics: `run_suite.py --endpoint URL`.

`RemoteEngine` stands in for `lars.engine.Engine` in `evaluate` and `calibrate`. It sends
one question per request and returns the log of the server's probabilities as logits, so
the per-config temperature and Platt fits apply to the remote model exactly as they do to
a local one. A noul's P(yes) becomes the logit pair (logit(p), 0). Responses are cached
per row, because the suite scores each row several times (raw, calibrated, dumped).
Probabilities are floored at `PROBABILITY_FLOOR` before the log: servers round them, and
a rounded zero would otherwise be an infinite logit.
"""

from __future__ import annotations

import json

import httpx
import numpy as np

from lars.calibration import Calibrator
from lars.primitives import logit, sigmoid

PROBABILITY_FLOOR = 1e-6


class Refused(Exception):
    """The server declined a question it cannot answer (HTTP 422), such as too many options."""


class RemoteClient:
    def __init__(self, url: str, api_key: str | None = None, model: str | None = None, timeout: float = 120.0,
                 transport: httpx.BaseTransport | None = None) -> None:
        self.url = url.rstrip("/") + "/v1/systemone"
        self.model = model
        headers = {"authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.Client(headers=headers, timeout=timeout, transport=transport)
        self._cache: dict[str, dict] = {}

    def answer(self, state: object, question: dict) -> dict:
        body: dict = {"state": state, "questions": {"q": question}}
        if self.model:
            body["model"] = self.model
        key = json.dumps(body, sort_keys=True, default=str)
        if key not in self._cache:
            response = self._http.post(self.url, json=body)
            if response.status_code == 422:
                raise Refused(response.json().get("detail", response.text))
            response.raise_for_status()
            self._cache[key] = response.json()["answers"]["q"]
        return self._cache[key]


def _option_keys(question: dict) -> tuple[str, ...]:
    criteria = question.get("criteria") or {}
    if isinstance(criteria, dict):
        return tuple(criteria)
    return tuple(str(i) for i in range(len(criteria)))


class RemoteEngine:
    def __init__(self, client: RemoteClient, calibrator: Calibrator | None = None) -> None:
        self.client = client
        self.calibrator = calibrator or Calibrator()

    def raw_logits(self, state: object, question_id: str, question: object) -> tuple[tuple[str, ...], np.ndarray]:
        question = question.model_dump(exclude_none=True) if hasattr(question, "model_dump") else question
        answer = self.client.answer(state, question)  # type: ignore[arg-type]
        if question["type"] == "noul":  # type: ignore[index]
            return ("yes", "no"), np.asarray([logit(float(answer["noul"]), PROBABILITY_FLOOR), 0.0])
        keys = _option_keys(question)  # type: ignore[arg-type]
        probabilities = answer["probabilities"]
        if set(probabilities) != set(keys):
            raise ValueError(f"server answered options {sorted(probabilities)[:5]}..., expected {keys[:5]}...")
        return keys, np.log(np.maximum([float(probabilities[k]) for k in keys], PROBABILITY_FLOOR))

    def noul_prob(self, yes_logit: float, kind: str = "noul", features: dict[str, float] | None = None) -> float:
        platt = self.calibrator.platt_for(kind)
        if platt is None:
            return sigmoid(yes_logit / self.calibrator.temperature_for(kind))
        a, b = platt
        return sigmoid(a * yes_logit + b)
