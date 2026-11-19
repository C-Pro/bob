"""FastAPI application and OpenAI-compatible HTTP endpoints for bob-s1."""
from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
import hmac
import json
import logging
import time
from typing import Any, Callable
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
import uvicorn

from .calibration import CalibrationError, CalibrationRegistry
from .config import MODEL_MAX_POSITION_EMBEDDINGS, SidecarConfig
from .images import ImageValidationError, configure_pillow_limits
from .loader import CheckpointOverlayLoader, LoadedModelBundle, ModelLoader
from .schemas import (
    APIError,
    materialize_chat_completions_request,
    materialize_embeddings_request,
    parse_chat_completions_structure,
    parse_embeddings_structure,
)
from .worker import InferenceWorker, QueuedJob

logger = logging.getLogger("bob_s1.app")


def _error_response(
    status_code: int,
    message: str,
    *,
    error_type: str = "invalid_request_error",
    param: str | None = None,
    code: str | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": param,
                "code": code,
            }
        },
    )


async def _read_json_body(request: Request, max_body_bytes: int) -> dict[str, Any]:
    raw_len = request.headers.get("content-length")
    if raw_len is not None:
        try:
            content_len = int(raw_len)
        except ValueError as exc:
            raise APIError(400, "Invalid Content-Length header") from exc
        if content_len < 0:
            raise APIError(400, "Invalid negative Content-Length header")
        if content_len > max_body_bytes:
            raise APIError(
                413,
                f"Request body size {content_len} exceeds BOB_S1_MAX_BODY_BYTES ({max_body_bytes})",
                code="payload_too_large",
            )

    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > max_body_bytes:
            raise APIError(
                413,
                f"Request body exceeds BOB_S1_MAX_BODY_BYTES ({max_body_bytes})",
                code="payload_too_large",
            )
        chunks.append(chunk)

    raw_bytes = b"".join(chunks)
    if not raw_bytes:
        raise APIError(400, "Request body must not be empty")
    try:
        text = raw_bytes.decode("utf-8")
        parsed = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise APIError(400, f"Malformed JSON request body: {exc}") from exc
    if not isinstance(parsed, dict):
        raise APIError(400, "Request body must be a JSON object")
    return parsed


def _check_auth(request: Request, api_key: str | None) -> None:
    if api_key is None:
        return
    auth_header = request.headers.get("authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise APIError(
            401,
            "Missing or invalid Bearer authentication token",
            error_type="authentication_error",
            code="invalid_api_key",
        )
    supplied = auth_header[len("Bearer ") :].strip()
    if not hmac.compare_digest(supplied.encode("utf-8"), api_key.encode("utf-8")):
        raise APIError(
            401,
            "Invalid Bearer authentication token",
            error_type="authentication_error",
            code="invalid_api_key",
        )


async def _wait_for_http_disconnect(request: Request) -> None:
    """Wait until ASGI delivers an http.disconnect event."""
    while True:
        message = await request.receive()
        if message.get("type") == "http.disconnect":
            return


async def _run_with_disconnect_guard(
    request: Request,
    worker: InferenceWorker,
    decode_fn: Callable[[], Any],
    infer_fn: Callable[[Any], Any],
) -> tuple[Any, bool]:
    """Reserve admission before image decode, keep lease until decode exits, and cancel queued work on disconnect."""
    lease = worker.reserve_slot()
    decode_task: asyncio.Task[Any] | None = None
    try:
        if await request.is_disconnected():
            lease.release()
            return None, True
        decode_task = asyncio.create_task(asyncio.to_thread(decode_fn))
        try:
            materialized = await asyncio.shield(decode_task)
        except asyncio.CancelledError:
            # Keep the admission reservation until the running decode thread actually finishes.
            decode_task.add_done_callback(lambda _t: lease.release())
            raise
        if await request.is_disconnected():
            lease.release()
            return None, True
        job = lease.promote_to_queue(lambda: infer_fn(materialized))
    except BaseException:
        if decode_task is None or decode_task.done():
            lease.release()
        raise

    result_task = asyncio.ensure_future(asyncio.shield(asyncio.wrap_future(job.future)))
    disconnect_task = asyncio.create_task(_wait_for_http_disconnect(request))
    try:
        done, _ = await asyncio.wait(
            {result_task, disconnect_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if result_task in done:
            return result_task.result(), False

        # Client disconnected: cancel only if still queued; if already running on device, wait for it to finish safely.
        cancelled_before_start = job.try_cancel_if_queued()
        if cancelled_before_start:
            result_task.cancel()
            try:
                await result_task
            except (asyncio.CancelledError, BaseException):
                pass
            return None, True

        try:
            await result_task
        except BaseException:
            pass
        return None, True
    except asyncio.CancelledError:
        job.try_cancel_if_queued()
        raise
    finally:
        if not disconnect_task.done():
            disconnect_task.cancel()
            try:
                await disconnect_task
            except (asyncio.CancelledError, BaseException):
                pass


def create_app(
    cfg: SidecarConfig | None = None,
    *,
    bundle: LoadedModelBundle | None = None,
    loader: ModelLoader | None = None,
    registry: CalibrationRegistry | None = None,
) -> FastAPI:
    """Create a configured FastAPI application for bob-s1 inference."""
    resolved_cfg = cfg if cfg is not None else SidecarConfig.from_env()
    resolved_registry = registry if registry is not None else CalibrationRegistry.from_file()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_pillow_limits(resolved_cfg.max_image_pixels)
        if bundle is not None:
            loaded_bundle = bundle
        else:
            active_loader = loader if loader is not None else CheckpointOverlayLoader(resolved_cfg)
            loaded_bundle = active_loader.load()
        if not loaded_bundle.warmup_ok:
            raise RuntimeError("Loaded model bundle did not pass startup warmup")
        worker = InferenceWorker(loaded_bundle, resolved_cfg, resolved_registry)
        app.state.bundle = loaded_bundle
        app.state.worker = worker
        logger.info(
            "bob-s1 ready model_id=%s checkpoint_id=%s device=%s dtype=%s attn=%s max_seq_len=%d",
            loaded_bundle.model_id,
            loaded_bundle.checkpoint_id,
            loaded_bundle.device,
            loaded_bundle.dtype,
            loaded_bundle.attn_implementation,
            resolved_cfg.max_seq_len,
        )
        try:
            yield
        finally:
            worker.close()
            app.state.worker = None

    app = FastAPI(title="bob-s1 Inference Sidecar", version="0.1.0", lifespan=lifespan)
    app.state.config = resolved_cfg
    app.state.bundle = bundle
    app.state.worker = None

    @app.exception_handler(APIError)
    async def api_error_handler(_request: Request, exc: APIError) -> JSONResponse:
        return _error_response(
            exc.status_code,
            exc.message,
            error_type=exc.error_type,
            param=exc.param,
            code=exc.code,
        )

    @app.exception_handler(ImageValidationError)
    async def image_error_handler(_request: Request, exc: ImageValidationError) -> JSONResponse:
        return _error_response(400, exc.message, param=exc.param, code=exc.code)

    @app.exception_handler(CalibrationError)
    async def calibration_error_handler(_request: Request, exc: CalibrationError) -> JSONResponse:
        return _error_response(400, exc.message, param=exc.param, code=exc.code)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
        return _error_response(400, f"Invalid request: {exc.errors()[0]['msg']}")

    @app.exception_handler(Exception)
    async def unhandled_error_handler(_request: Request, _exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error during request processing")
        return _error_response(
            500,
            "Internal server error",
            error_type="server_error",
            code="internal_error",
        )

    def _require_worker() -> tuple[LoadedModelBundle, InferenceWorker]:
        active_bundle: LoadedModelBundle | None = app.state.bundle
        active_worker: InferenceWorker | None = app.state.worker
        if active_bundle is None or active_worker is None or not active_bundle.warmup_ok:
            raise APIError(
                503,
                "Model is not ready",
                error_type="service_unavailable_error",
                code="model_not_ready",
            )
        return active_bundle, active_worker

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        active_bundle, _ = _require_worker()
        return JSONResponse(
            status_code=200,
            content={
                "status": "ok",
                "model_id": active_bundle.model_id,
                "checkpoint_id": active_bundle.checkpoint_id,
                "device": active_bundle.device,
                "dtype": active_bundle.dtype,
                "attn_implementation": active_bundle.attn_implementation,
                "max_seq_len": resolved_cfg.max_seq_len,
            },
        )

    @app.get("/v1/models")
    async def list_models(request: Request) -> JSONResponse:
        _check_auth(request, resolved_cfg.api_key)
        active_bundle, _ = _require_worker()
        return JSONResponse(
            status_code=200,
            content={
                "object": "list",
                "data": [
                    {
                        "id": active_bundle.model_id,
                        "object": "model",
                        "created": 0,
                        "owned_by": "bob",
                        "max_position_embeddings": MODEL_MAX_POSITION_EMBEDDINGS,
                        "max_seq_len": resolved_cfg.max_seq_len,
                        "embedding_dimensions": [128, 256, 384],
                        "capabilities": {
                            "multimodal_embeddings": True,
                            "decisions": ["bob-s1.decision.v1"],
                        },
                    }
                ],
            },
        )

    @app.post("/v1/embeddings")
    async def create_embeddings(request: Request) -> Response:
        _check_auth(request, resolved_cfg.api_key)
        _, active_worker = _require_worker()
        payload = await _read_json_body(request, resolved_cfg.max_body_bytes)
        raw_req = parse_embeddings_structure(payload, resolved_cfg)
        result, disconnected = await _run_with_disconnect_guard(
            request,
            active_worker,
            lambda: materialize_embeddings_request(raw_req, resolved_cfg),
            active_worker.run_embeddings,
        )
        if disconnected:
            return Response(status_code=499)
        logger.info(
            "embeddings items=%d dims=%d enc=%s trunc=%s max_seq=%d prompt_tokens=%d",
            len(raw_req.items),
            raw_req.dimensions,
            raw_req.encoding_format,
            raw_req.truncation,
            raw_req.effective_max_seq_len,
            result["usage"]["prompt_tokens"],
        )
        return JSONResponse(status_code=200, content=result)

    @app.post("/v1/chat/completions")
    async def create_chat_completions(request: Request) -> Response:
        _check_auth(request, resolved_cfg.api_key)
        _, active_worker = _require_worker()
        payload = await _read_json_body(request, resolved_cfg.max_body_bytes)
        raw_req = parse_chat_completions_structure(payload, resolved_cfg)
        output, disconnected = await _run_with_disconnect_guard(
            request,
            active_worker,
            lambda: materialize_chat_completions_request(raw_req, resolved_cfg),
            active_worker.run_chat_decisions,
        )
        if disconnected:
            return Response(status_code=499)
        decision_body, prompt_tokens = output
        resp = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": raw_req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(decision_body, separators=(",", ":")),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 0,
                "total_tokens": prompt_tokens,
            },
        }
        logger.info(
            "chat_decisions count=%d images=%d prefix_mode=%s max_seq=%d joint_emb=%s prompt_tokens=%d",
            len(raw_req.decisions),
            len(raw_req.image_refs),
            raw_req.prefix_mode,
            raw_req.effective_max_seq_len,
            raw_req.embedding_dimensions,
            prompt_tokens,
        )
        return JSONResponse(status_code=200, content=resp)

    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run temporary OpenAI-compatible bob-s1 inference sidecar")
    parser.add_argument("--host", default=None, help="Bind host (overrides BOB_S1_HOST, default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="Bind port (overrides BOB_S1_PORT, default 8090)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = SidecarConfig.from_env()
    host = args.host or cfg.host
    port = args.port or cfg.port
    uvicorn.run(create_app(cfg), host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
