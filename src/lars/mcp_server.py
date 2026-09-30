"""MCP tools over LARS, for coding agents: `serve --mcp` mounts them at /mcp.

The tools only need two async callables: `decide` takes a System One request (a dict) and
returns its response, and `status` describes the server. Inside `lars serve` they run
the loaded engine through the server's queue; in `lars mcp-bridge` they forward to a
running server over HTTP. So the tool surface is defined once, here.

Requires `pip install lars-engine[mcp]`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, Field

Decide = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
Status = Callable[[], Awaitable[dict[str, Any]]]

INSTRUCTIONS = """\
LARS answers typed questions about a piece of text with calibrated probabilities, in one
model pass per question (about 0.3 s each; the first call after an idle spell also loads the
model, a few seconds). It never generates text, so answers are always one of the options you
give. Good for triage and gating: does this log show a failure, is this diff risky, which area
does this issue belong to, is this message urgent. Weak on world knowledge (it is a 3B-active
model), so do not ask it facts the text does not contain. Probabilities were calibrated on
classification benchmarks; treat them as well-ordered confidence, and check high-stakes calls.
Inputs are capped (the error says by how much when a text is too long): send the relevant
excerpt, not a whole repository. For a decision that has its own calibrator (listed by
lars_status under "calibrators"), pass its name as `decision`; its probabilities are then
calibrated on that decision's own labeled history. If that calibrator was fitted with numeric
evidence (such as ci_failure's history rates), pass the same names to lars_check as
`features`; without all of them the plain calibration is used."""


class CheckResult(BaseModel):
    p_yes: float = Field(description="Probability that the claim holds for the text")
    model: str


class ClassifyResult(BaseModel):
    choice: str = Field(description="The most probable option")
    probabilities: dict[str, float] = Field(description="Probability of every option; they sum to 1")
    confidence: float = Field(description="How clearly the top option wins, 0 to 1")
    model: str


def _with_decision(request: dict[str, Any], decision: str | None,
                   features: dict[str, dict[str, float]] | None = None) -> dict[str, Any]:
    extensions: dict[str, Any] = {}
    if decision:
        extensions["calibrator"] = decision
    if features:
        extensions["features"] = features
    return {**request, "lars": extensions} if extensions else request


def build_mcp(decide: Decide, status: Status) -> Any:
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError

    server = MCPServer(name="lars", instructions=INSTRUCTIONS)

    async def run(request: dict[str, Any]) -> dict[str, Any]:
        try:
            return await decide(request)
        except ToolError:
            raise
        except ValueError as error:  # schema, budget and model-name errors: the caller can fix these
            raise ToolError(str(error)) from error

    @server.tool(name="lars_check", title="Check a claim against a text")
    async def check(text: str, claim: str, decision: str | None = None,
                    features: dict[str, float] | None = None) -> CheckResult:
        """Probability that `claim` is true of `text`, for example text = a CI log and
        claim = "The build failed because of a failing test". `decision` names a calibrator
        from lars_status; use the claim wording that calibrator was fitted with.
        `features` is numeric evidence by name, fused with the model when the calibrator was
        fitted on exactly those names (all of them must be given)."""
        response = await run(_with_decision(
            {"state": text, "questions": {"q": {"type": "noul", "instructions": claim}}}, decision,
            {"q": features} if features else None))
        return CheckResult(p_yes=response["answers"]["q"]["noul"], model=response["model"])

    @server.tool(name="lars_classify", title="Pick one option for a text")
    async def classify(text: str, question: str, options: dict[str, str],
                       decision: str | None = None) -> ClassifyResult:
        """Which one of `options` fits `text` best. `options` maps each option's name to a short
        description, e.g. {"frontend": "UI, CSS, browser code", "backend": "APIs, database"}.
        `decision` names a calibrator from lars_status, as for lars_check."""
        if len(options) < 2:
            raise ToolError("give at least two options")
        response = await run(_with_decision({"state": text, "questions": {
            "q": {"type": "choice", "instructions": question, "criteria": options}}}, decision))
        answer = response["answers"]["q"]
        return ClassifyResult(choice=answer["choice"], probabilities=answer["probabilities"],
                              confidence=answer["confidence"], model=response["model"])

    @server.tool(name="lars_decide", title="Answer several typed questions about one state")
    async def decide_tool(state: Any, questions: dict[str, dict[str, Any]],
                          lars: dict[str, Any] | None = None) -> dict[str, Any]:
        """The full System One request: one `state` (text or JSON) and named `questions`, each
        {"type": "noul" | "choice" | "score" | "multi", "instructions": ..., "criteria": ...}.
        noul: a yes/no claim, no criteria. choice: criteria maps option names to descriptions.
        score: criteria is a list of ordered levels, lowest first. multi: like choice, but any
        number of options may apply. `lars` takes optional extensions (permutations,
        abstain_margin, constraints, explain). Returns the System One response."""
        request: dict[str, Any] = {"state": state, "questions": questions}
        if lars:
            request["lars"] = lars
        return await run(request)

    @server.tool(name="lars_status", title="LARS server status")
    async def status_tool() -> dict[str, Any]:
        """Which model and adapter are served, whether the model is loaded now, how many
        requests are waiting, and the server version."""
        return await status()

    return server
