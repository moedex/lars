"""Send low-confidence answers to a hosted model: `serve --escalate`.

moe-LARS answers every question locally. When an answer's top probability is below a
threshold, the question is also put to a hosted Claude model (Haiku by default), which
picks one option through a forced tool call. The served probabilities become
`weight * one_hot(hosted pick) + (1 - weight) * local`, so the hosted pick wins the argmax
at the default weight but the local distribution still shapes the rest. The answer carries
`escalated: {model, answer}` so a caller can tell. Multi questions are not escalated.

Escalation is per decision. A named calibrator (`serve --calibration-dir`) carries
`"escalate": {"below": p, "weight": w}` only where a measurement showed the hosted model helps
that decision (`evals/escalation_check.py`); a question answered with that calibrator uses it.
Everything else uses the server default, `--escalate-below`, which is 0 (never): escalating
every low-confidence answer lowered jev-bench accuracy at every threshold (0.767 local, 0.750 at
0.8), because the hosted model was right less often than moe-LARS on the rows moe-LARS was unsure
of; it helped only knowledge-heavy tasks (evals/RESULTS.md, 2026-09-29). A request can set
`moelars.escalate_below` itself, or stay local with `moelars.escalate: false`.

The API key is resolved once, at startup (`resolve_api_key`). Without one, no escalator is
built and nothing is ever routed out; the server says so at startup and in /v1/status.
Escalation sends the state and question text to the hosted API, so it is opt-in: the
server needs `--escalate`, and a request can still keep itself local with
`moelars.escalate: false`. A hosted call that fails leaves the local answer unchanged and
records the error class in `escalated`. Hosted calls run concurrently and outside the local
inference queue.

Requires `pip install moelars[escalate]`.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from moelars.schema import ChoiceAnswer, NoulAnswer, ScoreAnswer, SystemOneRequest, SystemOneResponse

DEFAULT_MODEL = "claude-haiku-4-5"
DEFAULT_BELOW = 0.0
DEFAULT_WEIGHT = 0.8
KEY_FILE = Path.home() / ".config" / "moelars" / "anthropic-key"
SYSTEM = ("You answer one typed question about a piece of content. Read the content and the question, "
          "then call the answer tool with the single option that fits best. Use only what the content "
          "says, plus general knowledge where the question needs it.")


def resolve_api_key() -> str | None:
    """ANTHROPIC_API_KEY, else the file MOELARS_ANTHROPIC_KEY_FILE names, else ~/.config/moelars/anthropic-key."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    path = Path(os.environ.get("MOELARS_ANTHROPIC_KEY_FILE") or KEY_FILE).expanduser()
    if path.exists():
        return path.read_text().strip() or None
    return None


def top_probability(answer: Any) -> float | None:
    if isinstance(answer, NoulAnswer):
        return max(answer.noul, 1.0 - answer.noul)
    if isinstance(answer, (ChoiceAnswer, ScoreAnswer)):
        return max(answer.probabilities.values())
    return None


def _options(question: Any) -> dict[str, str]:
    """Option key -> description text, in the order the tool offers them."""
    if question.type == "noul":
        return {"yes": "the statement holds", "no": "it does not"}
    criteria = question.criteria
    if isinstance(criteria, dict):
        return {k: v if isinstance(v, str) else json.dumps(v) for k, v in criteria.items()}
    return {str(i): c if isinstance(c, str) else json.dumps(c) for i, c in enumerate(criteria)}


def _prompt(state: Any, question: Any) -> str:
    content = state if isinstance(state, str) else json.dumps(state, indent=1)
    instructions = question.instructions if isinstance(question.instructions, str) else json.dumps(
        question.instructions)
    kind = {"noul": "Is this true?", "choice": "Pick one option.", "score": "Pick one level; lowest first."}
    lines = "\n".join(f"- {k}: {v}" for k, v in _options(question).items())
    return (f"<content>\n{content}\n</content>\n\n{kind[question.type]}\n{instructions or ''}\n\n"
            f"Options:\n{lines}")


class Escalator:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, below: float = DEFAULT_BELOW,
                 weight: float = DEFAULT_WEIGHT, client: Any | None = None,
                 decisions: dict[str, dict[str, float]] | None = None) -> None:
        """`decisions` maps a named calibrator to its escalation spec, {"below": p, "weight": w}."""
        if client is None:
            import anthropic

            client = anthropic.AsyncAnthropic(api_key=api_key, max_retries=2, timeout=30.0)
        self.client = client
        self.model = model
        self.below = below
        self.weight = weight
        self.decisions = dict(decisions or {})
        self.calls = 0
        self.errors = 0

    def status(self) -> dict[str, Any]:
        return {"model": self.model, "default_below": self.below, "weight": self.weight,
                "decisions": self.decisions, "calls": self.calls, "errors": self.errors}

    def policy(self, request: SystemOneRequest, qid: str) -> tuple[float, float]:
        """(threshold, weight) for one question: the request's override, else its decision's spec,
        else the server default."""
        options = request.moelars
        spec = self.decisions.get(options.calibrators.get(qid, options.calibrator) or "") or {}
        below = options.escalate_below if options.escalate_below is not None else spec.get("below", self.below)
        return float(below), float(spec.get("weight", self.weight))

    async def pick(self, state: Any, question: Any) -> str:
        options = list(_options(question))
        tool = {"name": "answer", "description": "Record the chosen option.",
                "input_schema": {"type": "object", "properties": {"answer": {"type": "string", "enum": options}},
                                 "required": ["answer"], "additionalProperties": False}}
        response = await self.client.messages.create(
            model=self.model, max_tokens=256, system=SYSTEM, tools=[tool],
            tool_choice={"type": "tool", "name": "answer"},
            messages=[{"role": "user", "content": _prompt(state, question)}])
        chosen = next((b.input.get("answer") for b in response.content if b.type == "tool_use"), None)
        if chosen not in options:
            raise ValueError(f"hosted model answered {chosen!r}")
        return chosen

    def _blend(self, answer: Any, chosen: str, w: float) -> Any:
        if isinstance(answer, NoulAnswer):
            target = 1.0 if chosen == "yes" else 0.0
            return answer.model_copy(update={"noul": round(w * target + (1 - w) * answer.noul, 4)})
        probs = {k: round(w * (k == chosen) + (1 - w) * p, 4) for k, p in answer.probabilities.items()}
        update: dict[str, Any] = {"probabilities": probs, "confidence": round(max(probs.values()), 4)}
        if isinstance(answer, ChoiceAnswer):
            update["choice"] = max(probs, key=probs.get)
        else:
            update["score"] = round(sum(int(k) * p for k, p in probs.items()), 4)
        return answer.model_copy(update=update)

    async def apply(self, request: SystemOneRequest, response: SystemOneResponse) -> SystemOneResponse:
        options = request.moelars
        if options.escalate is False:
            return response
        due = [qid for qid, a in response.answers.items()
               if (p := top_probability(a)) is not None and p < self.policy(request, qid)[0]]

        async def one(qid: str) -> None:
            self.calls += 1
            answer = response.answers[qid]
            try:
                chosen = await self.pick(request.state, request.questions[qid])
            except Exception as error:  # noqa: BLE001 - any failure keeps the local answer
                self.errors += 1
                response.answers[qid] = answer.model_copy(
                    update={"escalated": {"model": self.model, "error": type(error).__name__}})
                return
            blended = self._blend(answer, chosen, self.policy(request, qid)[1])
            response.answers[qid] = blended.model_copy(
                update={"escalated": {"model": self.model, "answer": chosen}})

        await asyncio.gather(*(one(qid) for qid in due))
        return response
