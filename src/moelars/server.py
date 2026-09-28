"""HTTP server, wire-compatible with the System One API.

- POST /v1/systemone
- GET  /v1/models
- GET  /v1/status   model, loaded or not, queue depth
- GET  /healthz     no auth, never loads the model
- /mcp              MCP over Streamable HTTP, with `mcp=True` (`moelars.mcp_server`)

Errors use the `{message, error_type}` shape. When an API key is configured
(MOELARS_API_KEY, or MOELARS_API_KEY_FILE naming a file that holds it), /v1/ and /mcp
requests must carry `Authorization: Bearer <key>`. A request whose `Origin` header is not
a localhost origin is refused, so a web page open in a browser cannot reach the server
through DNS rebinding. The response carries both `x-moelars-request-id` and
`x-typesafe-request-id`, since the client SDKs read the latter for their `request_id`
property.

Inference runs on a worker thread, one request at a time: the model is shared state, and
the event loop stays free for health checks and queued requests. HTTP and MCP requests share
that one queue. The engine is a `LazyEngine`: with an idle limit it is dropped after that
long without a request and rebuilt by the next one, both on the same worker. Bodies over
`max_body_bytes` are refused with 413 before parsing; the engine's row and token budgets
refuse oversized requests with 422 before any model pass.
"""

from __future__ import annotations

import contextlib
import os
import re
import uuid
from pathlib import Path
from typing import Any

import anyio
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from moelars import __version__
from moelars.engine import Engine
from moelars.lazy import LazyEngine
from moelars.schema import ErrorBody, ListModelsResponse, SystemOneRequest

LOCAL_ORIGIN = re.compile(r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$")
PROTECTED = ("/v1/", "/mcp")
SWEEP_SECONDS = 30.0


def configured_api_key() -> str | None:
    """MOELARS_API_KEY, else the contents of the file MOELARS_API_KEY_FILE names, else None."""
    key = os.environ.get("MOELARS_API_KEY")
    if key:
        return key
    path = os.environ.get("MOELARS_API_KEY_FILE")
    if path and Path(path).exists():
        return Path(path).read_text().strip() or None
    return None


def _error(status: int, message: str, error_type: str) -> JSONResponse:
    return JSONResponse(status_code=status, content=ErrorBody(message=message, error_type=error_type).model_dump())


def _with_request_id(response: JSONResponse, request_id: str) -> JSONResponse:
    response.headers["x-moelars-request-id"] = request_id
    response.headers["x-typesafe-request-id"] = request_id
    return response


def _format_validation(error: RequestValidationError) -> str:
    parts = []
    for item in error.errors():
        location = ".".join(str(x) for x in item.get("loc", ()) if x != "body")
        parts.append(f"{location}: {item.get('msg')}" if location else str(item.get("msg")))
    return "; ".join(parts) or "invalid request"


MAX_BODY_BYTES = 1_000_000


def create_app(engine: Engine | LazyEngine, max_body_bytes: int = MAX_BODY_BYTES, mcp: bool = False,
               sweep_seconds: float = SWEEP_SECONDS) -> FastAPI:
    holder = engine if isinstance(engine, LazyEngine) else LazyEngine(lambda: engine, engine=engine)
    inference = anyio.CapacityLimiter(1)

    async def on_worker(fn: Any, *args: Any) -> Any:
        return await anyio.to_thread.run_sync(lambda: fn(holder.get(), *args), limiter=inference)

    async def status() -> dict[str, Any]:
        return {"model": holder.model_id, "loaded": holder.loaded, "loads": holder.loads,
                "idle_unload_s": holder.idle_unload, "idle_s": round(holder.idle_for(), 1),
                "waiting": inference.statistics().tasks_waiting, "version": holder.version or __version__,
                "calibrators": holder.calibrator_names}

    mcp_server = None
    if mcp:
        from moelars.mcp_server import build_mcp

        async def decide(body: dict[str, Any]) -> dict[str, Any]:
            request = SystemOneRequest.model_validate(body)
            response = await on_worker(lambda e, r: e.evaluate(r), request)
            return response.model_dump(exclude_none=True)

        mcp_server = build_mcp(decide, status)
        mcp_app = mcp_server.streamable_http_app(streamable_http_path="/mcp", json_response=True, stateless_http=True)

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI):
        async with contextlib.AsyncExitStack() as stack:
            if mcp_server is not None:
                await stack.enter_async_context(mcp_server.session_manager.run())
            async with anyio.create_task_group() as tasks:
                if holder.idle_unload:
                    async def sweep() -> None:
                        while True:
                            await anyio.sleep(sweep_seconds)
                            await anyio.to_thread.run_sync(holder.maybe_unload, limiter=inference)

                    tasks.start_soon(sweep)
                yield
                tasks.cancel_scope.cancel()

    app = FastAPI(title="moe-LARS", version=holder.version or __version__, docs_url="/docs", lifespan=lifespan)
    app.state.engine = holder
    if mcp_server is not None:
        app.router.routes.extend(mcp_app.routes)

    @app.middleware("http")
    async def request_id_and_auth(request: Request, call_next):
        request_id = str(uuid.uuid4())
        origin = request.headers.get("origin")
        if origin and not LOCAL_ORIGIN.match(origin):
            return _with_request_id(_error(403, f"Origin {origin!r} is not allowed", "permission_error"), request_id)
        expected = configured_api_key()
        if expected and request.url.path.startswith(PROTECTED):
            header = request.headers.get("authorization", "")
            if header != f"Bearer {expected}":
                return _with_request_id(_error(401, "Missing or invalid API key", "authentication_error"), request_id)
        length = request.headers.get("content-length")
        if length is not None and length.isdigit() and int(length) > max_body_bytes:
            return _with_request_id(_error(413, f"Request body over {max_body_bytes} bytes", "invalid_request"),
                                    request_id)
        return _with_request_id(await call_next(request), request_id)

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_: Request, error: RequestValidationError):
        return _error(422, _format_validation(error), "invalid_request")

    @app.exception_handler(ValueError)
    async def value_error_handler(_: Request, error: ValueError):
        return _error(422, str(error), "invalid_request")

    @app.get("/healthz")
    async def healthz():
        return {"ok": True, "model": holder.model_id, "loaded": holder.loaded}

    @app.get("/v1/status")
    async def get_status():
        return await status()

    @app.get("/v1/models", response_model=ListModelsResponse)
    async def list_models():
        return ListModelsResponse(models=await on_worker(lambda e: e.models()))

    @app.post("/v1/systemone")
    async def system_one(request: SystemOneRequest):
        response = await on_worker(lambda e, r: e.evaluate(r), request)
        return JSONResponse(content=response.model_dump(exclude_none=True))

    return app
