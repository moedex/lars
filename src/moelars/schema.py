"""Wire schema.

Request and response models match the System One HTTP API (`POST /v1/systemone`,
`GET /v1/models`) field for field, so the official client SDKs parse moe-LARS
responses unchanged. moe-LARS extensions live under the `moelars` request key and as
optional extra answer fields, which the SDKs ignore.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

JSONContent = str | dict[str, Any] | list[Any] | None

MAX_CHOICE_OPTIONS = 255
MIN_CHOICE_OPTIONS = 2
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10
MAX_MULTI_OPTIONS = 255


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- questions


class NoulCriteria(_Strict):
    """Optional descriptions of the yes and no outcomes."""

    true: JSONContent = None
    false: JSONContent = None


class NoulQuestion(_Strict):
    """A yes/no question answered with P(yes)."""

    type: Literal["noul"]
    instructions: JSONContent = None
    criteria: NoulCriteria | None = None


class ChoiceQuestion(_Strict):
    """Pick one option from a keyed set."""

    type: Literal["choice"]
    instructions: JSONContent = None
    criteria: dict[str, JSONContent]

    @field_validator("criteria")
    @classmethod
    def _bounds(cls, value: dict[str, JSONContent]) -> dict[str, JSONContent]:
        if not MIN_CHOICE_OPTIONS <= len(value) <= MAX_CHOICE_OPTIONS:
            raise ValueError(f"choice criteria must have {MIN_CHOICE_OPTIONS}-{MAX_CHOICE_OPTIONS} options")
        return value


class ScoreQuestion(_Strict):
    """Place the state on an ordered rubric."""

    type: Literal["score"]
    instructions: JSONContent = None
    criteria: list[JSONContent]

    @field_validator("criteria")
    @classmethod
    def _bounds(cls, value: list[JSONContent]) -> list[JSONContent]:
        if not MIN_SCORE_LEVELS <= len(value) <= MAX_SCORE_LEVELS:
            raise ValueError(f"score criteria must have {MIN_SCORE_LEVELS}-{MAX_SCORE_LEVELS} levels")
        return value


class MultiQuestion(_Strict):
    """moe-LARS extension: independent P(applies) per option."""

    type: Literal["multi"]
    instructions: JSONContent = None
    criteria: dict[str, JSONContent]

    @field_validator("criteria")
    @classmethod
    def _bounds(cls, value: dict[str, JSONContent]) -> dict[str, JSONContent]:
        if not 1 <= len(value) <= MAX_MULTI_OPTIONS:
            raise ValueError(f"multi criteria must have 1-{MAX_MULTI_OPTIONS} options")
        return value


Question = Annotated[
    NoulQuestion | ChoiceQuestion | ScoreQuestion | MultiQuestion,
    Field(discriminator="type"),
]


# --------------------------------------------------------------------------- extensions


class Constraint(_Strict):
    """A structural relation between noul questions that answers must respect.

    - `complement`: exactly two nouls where one is the negation of the other. P1 + P2 is
      forced to 1.
    - `exclusive`: at most one of the listed nouls is true. Probabilities are rescaled
      when their sum exceeds 1.
    """

    kind: Literal["complement", "exclusive"]
    questions: list[str] = Field(min_length=2)


class MoelarsOptions(_Strict):
    """Per-request moe-LARS extensions. All default off, so plain System One requests are unchanged."""

    permutations: int = Field(0, ge=0, le=16, description="Extra option orderings averaged for choice answers")
    explain: bool = Field(False, description="Return evidence spans by leave-one-out ablation of the state")
    abstain_margin: float | None = Field(
        None, ge=0.0, le=1.0, description="Mark an answer abstained when top1 - top2 is below this"
    )
    constraints: list[Constraint] = Field(default_factory=list)
    features: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="Named numeric evidence per noul question, fused with the model by a fitted calibration",
    )
    calibrator: str | None = Field(
        None, description="Named calibrator (one fitted for this decision) for every question in the request"
    )
    calibrators: dict[str, str] = Field(
        default_factory=dict, description="Named calibrator per question id; overrides `calibrator`"
    )


class SystemOneRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    state: JSONContent
    model: str = "moelars-latest"
    questions: dict[str, Question] = Field(min_length=1)
    moelars: MoelarsOptions = Field(default_factory=MoelarsOptions)

    @model_validator(mode="after")
    def _constraints_reference_nouls(self) -> SystemOneRequest:
        for qid in self.moelars.features:
            if qid not in self.questions:
                raise ValueError(f"features reference unknown question {qid!r}")
            if self.questions[qid].type != "noul":
                raise ValueError(f"features question {qid!r} must be a noul")
        for constraint in self.moelars.constraints:
            for qid in constraint.questions:
                if qid not in self.questions:
                    raise ValueError(f"constraint references unknown question {qid!r}")
                if self.questions[qid].type != "noul":
                    raise ValueError(f"constraint question {qid!r} must be a noul")
            if constraint.kind == "complement" and len(constraint.questions) != 2:
                raise ValueError("complement constraints take exactly two questions")
        return self


# --------------------------------------------------------------------------- answers


class Evidence(BaseModel):
    span: str
    effect: float = Field(description="Total variation distance the answer moved when this span was removed")


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float
    abstain: bool | None = None
    evidence: list[Evidence] | None = None


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float
    order_sensitivity: float | None = None
    abstain: bool | None = None
    evidence: list[Evidence] | None = None


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    legend: dict[str, Any]
    probabilities: dict[str, float]
    confidence: float
    abstain: bool | None = None
    evidence: list[Evidence] | None = None


class MultiAnswer(BaseModel):
    type: Literal["multi"] = "multi"
    probabilities: dict[str, float]
    selected: list[str]
    evidence: list[Evidence] | None = None


Answer = Annotated[NoulAnswer | ChoiceAnswer | ScoreAnswer | MultiAnswer, Field(discriminator="type")]


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class SystemOneResponse(BaseModel):
    model: str
    answers: dict[str, Answer]
    usage: Usage


class ModelCard(BaseModel):
    name: str
    description: str
    release_date: str


class ListModelsResponse(BaseModel):
    models: list[ModelCard]


class ErrorBody(BaseModel):
    message: str
    error_type: str
