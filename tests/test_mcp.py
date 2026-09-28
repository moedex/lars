"""MCP at /mcp, the stdio bridge, idle unload, and the launchd plist, all on the mock backend."""

import json
import os
import socket
import stat
import threading
import time

import anyio
import httpx
import pytest

pytest.importorskip("mcp")

import httpx2  # noqa: E402  (the MCP SDK's HTTP client)
import uvicorn  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from mcp import Client  # noqa: E402
from mcp.client.streamable_http import streamable_http_client  # noqa: E402

from moelars.backends.mock import MockBackend  # noqa: E402
from moelars.bridge import make_bridge  # noqa: E402
from moelars.engine import Engine  # noqa: E402
from moelars.lazy import LazyEngine, parse_duration  # noqa: E402
from moelars.server import create_app  # noqa: E402

KEY = "test-key"
TEXT = "Traceback (most recent call last): AssertionError in test_payments.py; 1 failed, 88 passed"
OPTIONS = {"backend": "APIs, database, payments", "frontend": "UI, CSS, browser code"}


@pytest.fixture
def server(monkeypatch):
    """A real server on a free port: MCP over Streamable HTTP needs a live HTTP endpoint."""
    monkeypatch.setenv("MOELARS_API_KEY", KEY)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    config = uvicorn.Config(create_app(Engine(MockBackend()), mcp=True), host="127.0.0.1", port=port,
                            log_level="warning")
    running = uvicorn.Server(config)
    thread = threading.Thread(target=running.run, daemon=True)
    thread.start()
    for _ in range(100):
        if running.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    running.should_exit = True
    thread.join(5)


def _mcp_client(url: str, key: str | None = KEY) -> Client:
    headers = {"authorization": f"Bearer {key}"} if key else {}
    return Client(streamable_http_client(f"{url}/mcp", http_client=httpx2.AsyncClient(headers=headers)))


async def _call_all(client: Client) -> dict:
    async with client:
        names = sorted(tool.name for tool in (await client.list_tools()).tools)
        check = await client.call_tool("moelars_check", {"text": TEXT, "claim": "A test failed"})
        classify = await client.call_tool("moelars_classify", {"text": TEXT, "question": "Which area?",
                                                               "options": OPTIONS})
        decide = await client.call_tool("moelars_decide", {"state": TEXT, "questions": {
            "failed": {"type": "noul", "instructions": "A test failed"},
            "area": {"type": "choice", "instructions": "Which area?", "criteria": OPTIONS}}})
        status = await client.call_tool("moelars_status", {})
        bad = await client.call_tool("moelars_decide", {"state": TEXT, "questions": {
            "q": {"type": "choice", "instructions": "?", "criteria": {}}}})
    return {"names": names, "check": check, "classify": classify, "decide": decide, "status": status, "bad": bad}


def _expected(url: str) -> dict:
    body = {"state": TEXT, "questions": {"failed": {"type": "noul", "instructions": "A test failed"},
                                         "area": {"type": "choice", "instructions": "Which area?",
                                                  "criteria": OPTIONS}}}
    return httpx.post(f"{url}/v1/systemone", json=body, headers={"authorization": f"Bearer {KEY}"}).json()


def _assert_tools_match_http(results: dict, expected: dict) -> None:
    assert results["names"] == ["moelars_check", "moelars_classify", "moelars_decide", "moelars_status"]
    assert results["check"].structured_content["p_yes"] == pytest.approx(expected["answers"]["failed"]["noul"])
    classify = results["classify"].structured_content
    assert classify["choice"] == expected["answers"]["area"]["choice"]
    assert classify["probabilities"] == pytest.approx(expected["answers"]["area"]["probabilities"])
    decided = results["decide"].structured_content
    decided = decided.get("result", decided)
    assert decided["answers"]["area"]["choice"] == expected["answers"]["area"]["choice"]
    status = results["status"].structured_content
    assert status.get("result", status)["loaded"] is True
    assert results["bad"].is_error


def test_mcp_tools_answer_like_the_http_api(server):
    results = anyio.run(_call_all, _mcp_client(server))
    _assert_tools_match_http(results, _expected(server))


def test_mcp_needs_the_key_and_a_local_origin(server):
    initialize = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                  "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                             "clientInfo": {"name": "t", "version": "0"}}}
    headers = {"content-type": "application/json", "accept": "application/json, text/event-stream"}
    assert httpx.post(f"{server}/mcp", json=initialize, headers=headers).status_code == 401
    evil = {**headers, "authorization": f"Bearer {KEY}", "origin": "http://evil.example"}
    assert httpx.post(f"{server}/mcp", json=initialize, headers=evil).status_code == 403
    assert httpx.post(f"{server}/v1/systemone", json={}, headers={"origin": "http://evil.example"}).status_code == 403
    local = {**headers, "authorization": f"Bearer {KEY}", "origin": "http://localhost:3000"}
    assert httpx.post(f"{server}/mcp", json=initialize, headers=local).status_code == 200


def test_the_bridge_forwards_to_the_server_and_reports_it_down(server):
    results = anyio.run(_call_all, Client(make_bridge(server, KEY)))
    _assert_tools_match_http(results, _expected(server))

    async def down():
        async with Client(make_bridge("http://127.0.0.1:9", KEY)) as client:
            return await client.call_tool("moelars_check", {"text": "x", "claim": "y"})

    result = anyio.run(down)
    assert result.is_error and "not reachable" in result.content[0].text

    async def wrong_key():
        async with Client(make_bridge(server, "nope")) as client:
            return await client.call_tool("moelars_check", {"text": "x", "claim": "y"})

    result = anyio.run(wrong_key)
    assert result.is_error and "API key" in result.content[0].text


def test_lazy_engine_loads_on_use_and_unloads_when_idle():
    now = [0.0]
    built = []
    holder = LazyEngine(lambda: built.append(1) or Engine(MockBackend()), idle_unload=60, clock=lambda: now[0])
    assert not holder.loaded and holder.model_id is None
    holder.get()
    assert holder.loaded and holder.loads == 1 and holder.model_id
    now[0] = 59
    assert not holder.maybe_unload() and holder.loaded
    now[0] = 120
    assert holder.maybe_unload() and not holder.loaded
    holder.get()
    assert holder.loads == 2 and len(built) == 2
    assert [parse_duration(x) for x in ("900", "90s", "15m", "1h", "0")] == [900, 90, 900, 3600, 0]


def test_server_sweeps_an_idle_model_and_reloads_it(monkeypatch):
    monkeypatch.delenv("MOELARS_API_KEY", raising=False)
    holder = LazyEngine(lambda: Engine(MockBackend()), idle_unload=0.05)
    body = {"state": "x", "questions": {"q": {"type": "noul", "instructions": "y"}}}
    with TestClient(create_app(holder, sweep_seconds=0.02)) as client:
        assert client.get("/healthz").json()["loaded"] is False  # nothing loads at startup
        assert client.post("/v1/systemone", json=body).status_code == 200
        assert client.get("/v1/status").json()["loaded"] is True
        time.sleep(0.3)
        assert client.get("/healthz").json()["loaded"] is False
        assert client.post("/v1/systemone", json=body).status_code == 200
        assert client.get("/v1/status").json()["loads"] == 2


def test_key_file_is_read_per_request(tmp_path, monkeypatch):
    monkeypatch.delenv("MOELARS_API_KEY", raising=False)
    key_file = tmp_path / "key"
    key_file.write_text("from-file\n")
    monkeypatch.setenv("MOELARS_API_KEY_FILE", str(key_file))
    client = TestClient(create_app(Engine(MockBackend())))
    assert client.get("/v1/status").status_code == 401
    assert client.get("/v1/status", headers={"authorization": "Bearer from-file"}).status_code == 200
    assert client.get("/healthz").status_code == 200  # health stays open for launchd and scripts


def test_service_plist_and_key(tmp_path, monkeypatch):
    from moelars import service
    from moelars.cli import build_parser

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    monkeypatch.chdir(tmp_path)
    args = build_parser().parse_args(["service", "install", "--backend", "mlx", "--model", "org/model",
                                      "--adapter", "adapter", "--calibration", "org/repo/cal.json"])
    argv = service.serve_argv(args)
    assert argv[:2] == ["serve", "--mcp"] and "15m" in argv
    assert argv[argv.index("--adapter") + 1] == str(adapter.resolve())  # launchd starts the agent in /
    assert argv[argv.index("--model") + 1] == "org/model"
    key_file = service.ensure_api_key(tmp_path / "cfg" / "api-key")
    plist = service.build_plist(["python", "-m", "moelars", *argv], api_key_file=key_file, log_dir=tmp_path)
    assert plist["KeepAlive"] == {"SuccessfulExit": False} and plist["RunAtLoad"] is True
    assert plist["EnvironmentVariables"]["HF_HUB_OFFLINE"] == "1"
    key = key_file.read_text().strip()
    assert len(key) >= 32 and key not in json.dumps(plist)
    assert stat.S_IMODE(os.stat(key_file).st_mode) == 0o600
    assert service.ensure_api_key(key_file).read_text().strip() == key  # an existing key is kept


def test_named_calibrator_dir_reaches_the_mcp_tools(tmp_path, monkeypatch):
    from moelars.calibration import Calibrator
    from moelars.cli import load_named_calibrators

    monkeypatch.delenv("MOELARS_API_KEY", raising=False)
    Calibrator(platt={"noul": (4.0, 0.0)}).save(tmp_path / "ci_failure.json")
    engine = Engine(MockBackend(), named_calibrators=load_named_calibrators(str(tmp_path)))
    app = create_app(engine, mcp=True)

    async def calls():
        http = httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app))
        async with Client(streamable_http_client("http://127.0.0.1:8600/mcp", http_client=http)) as c:
            claim = {"text": TEXT, "claim": "A test failed"}
            plain = await c.call_tool("moelars_check", claim)
            named = await c.call_tool("moelars_check", {**claim, "decision": "ci_failure"})
            bad = await c.call_tool("moelars_check", {**claim, "decision": "nope"})
            status = await c.call_tool("moelars_status", {})
        return plain, named, bad, status

    with TestClient(app):  # runs the lifespan, which starts the MCP session manager
        plain, named, bad, status = anyio.run(calls)
    assert named.structured_content["p_yes"] != plain.structured_content["p_yes"]
    assert bad.is_error and "ci_failure" in bad.content[0].text
    got = status.structured_content
    assert got.get("result", got)["calibrators"] == ["ci_failure"]
