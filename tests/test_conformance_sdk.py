"""Run the official System One Python SDK against LARS in-process.

Skipped when the SDK or its HTTP library is unavailable. This is the contract that
matters: if these pass, every SDK-based integration works with a base URL swap.

The SDK's async client is used because httpx2's ASGI transport is async-only.
"""

import asyncio

import pytest

typesafe_sdk = pytest.importorskip("typesafe_sdk")
httpx2 = pytest.importorskip("httpx2")

from lars.backends.mock import MockBackend  # noqa: E402
from lars.engine import Engine  # noqa: E402
from lars.server import create_app  # noqa: E402


@pytest.fixture
def sdk_client(monkeypatch):
    monkeypatch.delenv("LARS_API_KEY", raising=False)
    transport_cls = getattr(httpx2, "ASGITransport", None)
    if transport_cls is None:
        pytest.skip("httpx2 has no ASGITransport")
    app = create_app(Engine(MockBackend()))
    return typesafe_sdk.AsyncTypeSafeClient(
        api_key="test", base_url="http://lars.test", transport=transport_cls(app=app)
    )


def test_official_sdk_parses_all_three_primitives(sdk_client):
    result = asyncio.run(
        sdk_client.system_one(
            state="Help! My payouts have been failing for 3 days.",
            questions={
                "is_urgent": typesafe_sdk.Noul(instructions="Does this convey urgency?"),
                "department": typesafe_sdk.Choice(
                    instructions="Which team must handle this?",
                    criteria={"billing": "Payments, payouts, refunds", "technical": "Bugs, outages"},
                ),
                "frustration": typesafe_sdk.Score(
                    instructions="How frustrated is the customer?",
                    criteria=["Calm", "Frustrated", "Very angry"],
                ),
            },
        )
    )
    assert 0 <= result.nouls["is_urgent"].noul <= 1
    assert result.choices["department"].choice in {"billing", "technical"}
    assert abs(sum(result.choices["department"].probabilities.values()) - 1) < 1e-3
    assert 0 <= result.choices["department"].confidence <= 1
    assert result.scores["frustration"].legend[1] == "Frustrated"
    assert 0 <= result.scores["frustration"].score <= 2
    assert result.model.startswith("lars-")
    assert result.usage.input_tokens is not None


def test_official_sdk_lists_models(sdk_client):
    models = asyncio.run(sdk_client.models.list())
    assert any(m.name == "lars-latest" for m in models.models)


def test_official_sdk_surfaces_validation_errors(sdk_client):
    with pytest.raises(typesafe_sdk.TypeSafeUnprocessableEntityError):
        asyncio.run(
            sdk_client.system_one(
                state="x",
                questions={"bad": {"type": "score", "criteria": ["only one level"]}},  # type: ignore[dict-item]
            )
        )
