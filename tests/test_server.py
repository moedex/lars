import pytest
from fastapi.testclient import TestClient

from lars.backends.mock import MockBackend
from lars.engine import Engine
from lars.server import create_app

BODY = {
    "model": "lars-latest",
    "state": "I was charged twice for my subscription.",
    "questions": {
        "refund": {"type": "noul", "instructions": "Is the customer asking for money back?"},
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {"billing": "Charges and refunds", "technical": "Bugs and outages"},
        },
    },
}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("LARS_API_KEY", raising=False)
    return TestClient(create_app(Engine(MockBackend())))


def test_systemone_wire_shape(client):
    response = client.post("/v1/systemone", json=BODY)
    assert response.status_code == 200
    data = response.json()
    assert set(data) == {"model", "answers", "usage"}
    assert data["answers"]["refund"].keys() == {"type", "noul"}
    assert set(data["answers"]["department"]) == {"type", "choice", "probabilities", "confidence"}
    assert set(data["usage"]) == {"input_tokens", "output_tokens"}
    assert response.headers["x-typesafe-request-id"]


def test_extensions_are_opt_in(client):
    body = {**BODY, "lars": {"permutations": 2, "abstain_margin": 0.1}}
    data = client.post("/v1/systemone", json=body).json()
    assert "order_sensitivity" in data["answers"]["department"]
    assert "abstain" in data["answers"]["department"]


def test_validation_error_shape(client):
    bad = {**BODY, "questions": {"x": {"type": "verdict", "instructions": "?"}}}
    response = client.post("/v1/systemone", json=bad)
    assert response.status_code == 422
    assert set(response.json()) == {"message", "error_type"}
    assert response.json()["error_type"] == "invalid_request"


def test_models_endpoint(client):
    data = client.get("/v1/models").json()
    names = [m["name"] for m in data["models"]]
    assert "lars-latest" in names
    assert all({"name", "description", "release_date"} <= set(m) for m in data["models"])


def test_auth_when_key_configured(monkeypatch):
    monkeypatch.setenv("LARS_API_KEY", "secret")
    client = TestClient(create_app(Engine(MockBackend())))
    assert client.post("/v1/systemone", json=BODY).status_code == 401
    ok = client.post("/v1/systemone", json=BODY, headers={"Authorization": "Bearer secret"})
    assert ok.status_code == 200
    assert client.get("/healthz").status_code == 200


def test_request_over_the_row_budget_is_refused_before_inference(monkeypatch):
    monkeypatch.delenv("LARS_API_KEY", raising=False)
    engine = Engine(MockBackend(), max_rows=4)
    engine._score = lambda *a: pytest.fail("the model ran")  # type: ignore[method-assign]
    body = {**BODY, "lars": {"permutations": 4}}  # 2 base rows + 4 permutation rows
    response = TestClient(create_app(engine)).post("/v1/systemone", json=body)
    assert response.status_code == 422
    assert response.json()["error_type"] == "invalid_request"
    assert "6 model rows" in response.json()["message"]


def test_request_over_the_token_budget_is_refused(monkeypatch):
    monkeypatch.delenv("LARS_API_KEY", raising=False)
    client = TestClient(create_app(Engine(MockBackend(), max_input_tokens=50)))
    response = client.post("/v1/systemone", json={**BODY, "state": "word " * 200})
    assert response.status_code == 422
    assert "input tokens" in response.json()["message"]


def test_body_over_the_limit_is_413_with_request_ids(monkeypatch):
    monkeypatch.delenv("LARS_API_KEY", raising=False)
    client = TestClient(create_app(Engine(MockBackend()), max_body_bytes=200))
    response = client.post("/v1/systemone", json={**BODY, "state": "x" * 500})
    assert response.status_code == 413
    assert response.json()["error_type"] == "invalid_request"
    assert response.headers["x-typesafe-request-id"] == response.headers["x-lars-request-id"]


def test_auth_failure_carries_the_sdk_request_id(monkeypatch):
    monkeypatch.setenv("LARS_API_KEY", "secret")
    response = TestClient(create_app(Engine(MockBackend()))).post("/v1/systemone", json=BODY)
    assert response.status_code == 401
    assert response.headers["x-typesafe-request-id"]


def test_slow_inference_does_not_block_health_checks(monkeypatch):
    """Inference runs off the event loop: /healthz answers while a request is still being scored."""
    import threading
    import time

    import anyio
    import httpx

    monkeypatch.delenv("LARS_API_KEY", raising=False)
    engine = Engine(MockBackend())
    release = threading.Event()
    evaluate = engine.evaluate

    def slow(request):
        release.wait(5)
        return evaluate(request)

    engine.evaluate = slow  # type: ignore[method-assign]
    app = create_app(engine)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            async with anyio.create_task_group() as tg:
                results = {}

                async def post():
                    results["post"] = await client.post("/v1/systemone", json=BODY)

                tg.start_soon(post)
                await anyio.sleep(0.05)
                started = time.perf_counter()
                health = await client.get("/healthz")
                results["health_s"] = time.perf_counter() - started
                assert "post" not in results  # the slow request is still in flight
                release.set()
            return health, results

    health, results = anyio.run(scenario)
    assert health.status_code == 200 and results["health_s"] < 1.0
    assert results["post"].status_code == 200


@pytest.mark.parametrize("model", ["lars-latest", "jev-latest"])
def test_default_aliases_are_served(client, model):
    assert client.post("/v1/systemone", json={**BODY, "model": model}).status_code == 200


def test_loaded_model_id_is_served(client):
    model_id = client.get("/healthz").json()["model"]
    assert client.post("/v1/systemone", json={**BODY, "model": model_id}).status_code == 200


def test_unknown_model_is_refused(client):
    response = client.post("/v1/systemone", json={**BODY, "model": "totally-unknown"})
    assert response.status_code == 422
    assert response.json()["error_type"] == "invalid_request"
    assert "totally-unknown" in response.json()["message"]
