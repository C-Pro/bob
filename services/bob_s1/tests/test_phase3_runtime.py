"""Phase 3 regression tests: strict schema validation, admission/overload, ASGI disconnect, shutdown, and preparation metadata."""
from __future__ import annotations

import asyncio
import base64
from io import BytesIO
import json
from pathlib import Path
import threading
import time

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from bob_s1.app import create_app
from bob_s1.calibration import CalibrationRegistry
from bob_s1.config import SidecarConfig
from bob_s1.images import ImageValidationError, decode_data_uri_images_batch
from bob_s1.loader import LoadedModelBundle, import_jev_model_module, verify_model_runtime
from bob_s1.schemas import APIError
from bob_s1.worker import InferenceWorker


def _make_bundle(max_seq_len: int = 4096):
    repo_root = Path(__file__).resolve().parents[3]
    jev_model = import_jev_model_module(repo_root)
    reg = CalibrationRegistry.from_file()
    model = jev_model.synthetic_model(seed=42, with_lora=True, include_vision=True)
    prep = verify_model_runtime(model, device="cpu", max_seq_len=min(512, max_seq_len))
    return (
        LoadedModelBundle(
            model=model,
            model_id="bob-s1-0.8b",
            checkpoint_id=reg.checkpoint_id,
            checkpoint_sha256=reg.checkpoint_sha256,
            device="cpu",
            dtype="fp32",
            attn_implementation="eager",
            max_seq_len=max_seq_len,
            preprocess_metadata=prep,
            warmup_ok=True,
        ),
        reg,
    )


def _png_data_uri(size: tuple[int, int] = (8, 8), color: tuple[int, int, int] = (255, 0, 0)) -> str:
    buf = BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def test_malformed_inputs_and_auth_return_400_401_never_500():
    bundle, reg = _make_bundle()
    cfg = SidecarConfig.from_env({"BOB_S1_API_KEY": "secret-token"})
    auth = {"Authorization": "Bearer secret-token"}
    app = create_app(cfg, bundle=bundle, registry=reg)

    # Non-ASCII Authorization header via raw ASGI -> 401 (never 500 TypeError in compare_digest)
    async def check_non_ascii_auth():
        async with app.router.lifespan_context(app):
            statuses = []
            chunks = []
            body = json.dumps({"model": "bob-s1-0.8b", "input": "hi"}).encode("utf-8")
            sent = False

            async def receive():
                nonlocal sent
                if not sent:
                    sent = True
                    return {"type": "http.request", "body": body, "more_body": False}
                return {"type": "http.disconnect"}

            async def send(msg):
                if msg["type"] == "http.response.start":
                    statuses.append(msg["status"])
                elif msg["type"] == "http.response.body":
                    chunks.append(msg.get("body", b""))

            scope = {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/v1/embeddings",
                "raw_path": b"/v1/embeddings",
                "query_string": b"",
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"authorization", "Bearer пароль-токен".encode("utf-8")),
                ],
                "client": ("127.0.0.1", 12345),
                "server": ("127.0.0.1", 8090),
            }
            await app(scope, receive, send)
            assert statuses[0] == 401
            err = json.loads(b"".join(chunks))
            assert err["error"]["type"] == "authentication_error"

    asyncio.run(check_non_ascii_auth())

    with TestClient(app) as client:

        # Invalid UTF-8 body -> 400
        r_utf8 = client.post(
            "/v1/embeddings",
            headers={**auth, "Content-Type": "application/json"},
            content=b"\xff\xfe\xfd",
        )
        assert r_utf8.status_code == 400

        # Deeply nested JSON triggering RecursionError -> 400
        deep_json = ("{\"a\":" * 1500) + "1" + ("}" * 1500)
        r_deep = client.post(
            "/v1/embeddings",
            headers={**auth, "Content-Type": "application/json"},
            content=deep_json.encode("utf-8"),
        )
        assert r_deep.status_code == 400

        # Deeply nested inner JSON in chat completions -> 400
        r_deep_inner = client.post(
            "/v1/chat/completions",
            headers=auth,
            json={
                "model": "bob-s1-0.8b",
                "messages": [{"role": "user", "content": deep_json}],
            },
        )
        assert r_deep_inner.status_code == 400

        # Unhashable encoding_format [] or {} -> 400 (never 500 TypeError)
        for bad_enc in ([], {}, 123, None):
            r_enc = client.post(
                "/v1/embeddings",
                headers=auth,
                json={"model": "bob-s1-0.8b", "input": "hi", "encoding_format": bad_enc},
            )
            assert r_enc.status_code == 400


def test_strict_schema_rejects_coercion_and_explicit_nulls():
    bundle, reg = _make_bundle()
    cfg = SidecarConfig.from_env({})
    with TestClient(create_app(cfg, bundle=bundle, registry=reg)) as client:
        bad_inner_payloads = [
            # String coerced to bool
            {
                "protocol": "bob-s1.decision.v1",
                "state": "x",
                "decisions": [{"id": "d1", "task": "compaction", "apply_calibration": "false"}],
            },
            # String coerced to int
            {
                "protocol": "bob-s1.decision.v1",
                "state": "x",
                "embedding": {"dimensions": "128"},
                "decisions": [{"id": "d1", "task": "compaction"}],
            },
            # Bool coerced to float
            {
                "protocol": "bob-s1.decision.v1",
                "state": "x",
                "decisions": [
                    {
                        "id": "d1",
                        "question": "Q?",
                        "options": [
                            {"id": "a", "description": "A", "value": True},
                            {"id": "b", "description": "B", "value": 1.0},
                        ],
                    }
                ],
            },
            # Explicit null for option.value
            {
                "protocol": "bob-s1.decision.v1",
                "state": "x",
                "decisions": [
                    {
                        "id": "d1",
                        "question": "Q?",
                        "options": [
                            {"id": "a", "description": "A", "value": None},
                            {"id": "b", "description": "B", "value": None},
                        ],
                    }
                ],
            },
            # Explicit null for embedding
            {
                "protocol": "bob-s1.decision.v1",
                "state": "x",
                "embedding": None,
                "decisions": [{"id": "d1", "task": "compaction"}],
            },
            # Explicit null for max_seq_len
            {
                "protocol": "bob-s1.decision.v1",
                "state": "x",
                "max_seq_len": None,
                "decisions": [{"id": "d1", "task": "compaction"}],
            },
        ]
        for inner in bad_inner_payloads:
            r = client.post(
                "/v1/chat/completions",
                json={
                    "model": "bob-s1-0.8b",
                    "messages": [{"role": "user", "content": json.dumps(inner)}],
                },
            )
            assert r.status_code == 400, f"Expected 400 for {inner!r}, got {r.status_code}: {r.text}"

        # Non-bool response_format.json_schema.strict
        schema_path = Path(__file__).resolve().parents[1] / "contracts/decision.response.schema.json"
        canon_schema = json.loads(schema_path.read_text(encoding="utf-8"))
        r_strict = client.post(
            "/v1/chat/completions",
            json={
                "model": "bob-s1-0.8b",
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "bob_s1_decision_v1", "strict": "true", "schema": canon_schema},
                },
                "messages": [
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "protocol": "bob-s1.decision.v1",
                                "state": "x",
                                "decisions": [{"id": "d1", "task": "compaction"}],
                            }
                        ),
                    }
                ],
            },
        )
        assert r_strict.status_code == 400


def test_aggregate_image_budget_and_admission_before_decode(monkeypatch):
    # Aggregate pixel budget blocks multiple smaller PNGs exceeding total cap
    img_uri = _png_data_uri(size=(20, 20))  # 400 pixels each
    with pytest.raises(ImageValidationError, match="Aggregate decoded image pixels"):
        decode_data_uri_images_batch(
            [(img_uri, "p0"), (img_uri, "p1"), (img_uri, "p2")],
            max_bytes=1024 * 1024,
            max_pixels=1000,
            max_total_pixels=900,
        )

    # Queue saturation (429) rejects BEFORE decoding images
    bundle, reg = _make_bundle()
    cfg = SidecarConfig.from_env({"BOB_S1_MAX_QUEUE_DEPTH": "1"})
    decode_calls = 0

    import bob_s1.app as app_mod

    orig_materialize = app_mod.materialize_embeddings_request

    def counting_materialize(raw, c):
        nonlocal decode_calls
        decode_calls += 1
        return orig_materialize(raw, c)

    monkeypatch.setattr(app_mod, "materialize_embeddings_request", counting_materialize)

    # Gate worker on a slow first job so the queue fills at depth=1
    gate = threading.Event()
    started = threading.Event()
    orig_run_emb = InferenceWorker.run_embeddings

    def gated_run_emb(self, req):
        started.set()
        gate.wait(timeout=5.0)
        return orig_run_emb(self, req)

    monkeypatch.setattr(InferenceWorker, "run_embeddings", gated_run_emb)

    async def exercise():
        import httpx

        app = create_app(cfg, bundle=bundle, registry=reg)
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
                t1 = asyncio.create_task(ac.post("/v1/embeddings", json={"model": "bob-s1-0.8b", "input": "first"}))
                # Wait until first job is dequeued and running on worker thread
                for _ in range(100):
                    if started.is_set():
                        break
                    await asyncio.sleep(0.01)
                assert started.is_set()
                # Second job fills queue depth=1
                t2 = asyncio.create_task(ac.post("/v1/embeddings", json={"model": "bob-s1-0.8b", "input": "second"}))
                await asyncio.sleep(0.05)
                calls_before_overload = decode_calls
                # Third job with compressed image MUST receive 429 WITHOUT decoding image
                r3 = await ac.post(
                    "/v1/embeddings",
                    json={
                        "model": "bob-s1-0.8b",
                        "input": [{"content": [{"type": "image_url", "image_url": {"url": img_uri}}]}],
                    },
                )
                assert r3.status_code == 429
                assert r3.json()["error"]["code"] == "queue_full"
                assert decode_calls == calls_before_overload
                gate.set()
                r1, r2 = await asyncio.gather(t1, t2)
                assert r1.status_code == 200
                assert r2.status_code == 200

    asyncio.run(exercise())


def test_asgi_disconnect_cancels_queued_job_without_running_gpu(monkeypatch):
    bundle, reg = _make_bundle()
    cfg = SidecarConfig.from_env({"BOB_S1_MAX_QUEUE_DEPTH": "4"})

    executed_inputs: list[str] = []
    gate = threading.Event()
    started_first = threading.Event()
    orig_run_emb = InferenceWorker.run_embeddings

    def tracked_run_emb(self, req):
        txt = req.items[0].text
        executed_inputs.append(txt)
        if txt == "active_job":
            started_first.set()
            gate.wait(timeout=5.0)
        return orig_run_emb(self, req)

    monkeypatch.setattr(InferenceWorker, "run_embeddings", tracked_run_emb)

    async def exercise():
        app = create_app(cfg, bundle=bundle, registry=reg)
        async with app.router.lifespan_context(app):
            # 1. Start active job via ASGI call
            async def run_asgi(body_bytes: bytes, disconnect_after: float | None = None):
                sent_body = False
                statuses: list[int] = []

                async def receive():
                    nonlocal sent_body
                    if not sent_body:
                        sent_body = True
                        return {"type": "http.request", "body": body_bytes, "more_body": False}
                    if disconnect_after is not None:
                        await asyncio.sleep(disconnect_after)
                        return {"type": "http.disconnect"}
                    await asyncio.sleep(5.0)
                    return {"type": "http.disconnect"}

                async def send(msg):
                    if msg["type"] == "http.response.start":
                        statuses.append(msg["status"])

                scope = {
                    "type": "http",
                    "asgi": {"version": "3.0"},
                    "http_version": "1.1",
                    "method": "POST",
                    "scheme": "http",
                    "path": "/v1/embeddings",
                    "raw_path": b"/v1/embeddings",
                    "query_string": b"",
                    "headers": [(b"content-type", b"application/json")],
                    "client": ("127.0.0.1", 12345),
                    "server": ("127.0.0.1", 8090),
                }
                await app(scope, receive, send)
                return statuses[0]

            t_active = asyncio.create_task(
                run_asgi(json.dumps({"model": "bob-s1-0.8b", "input": "active_job"}).encode())
            )
            for _ in range(100):
                if started_first.is_set():
                    break
                await asyncio.sleep(0.01)
            assert started_first.is_set()

            # 2. Enqueue second job that disconnects while waiting in queue
            t_disconnected = asyncio.create_task(
                run_asgi(
                    json.dumps({"model": "bob-s1-0.8b", "input": "queued_then_disconnected"}).encode(),
                    disconnect_after=0.05,
                )
            )
            status_disc = await t_disconnected
            assert status_disc == 499

            # 3. Release active job and verify queued_then_disconnected never executed
            gate.set()
            status_active = await t_active
            assert status_active == 200
            await asyncio.sleep(0.05)
            assert executed_inputs == ["active_job"]

    asyncio.run(exercise())


def test_worker_shutdown_resolves_queued_futures_and_finishes_active():
    bundle, reg = _make_bundle()
    cfg = SidecarConfig.from_env({"BOB_S1_MAX_QUEUE_DEPTH": "4"})
    worker = InferenceWorker(bundle, cfg, reg)

    started_active = threading.Event()
    gate = threading.Event()

    def active_fn():
        started_active.set()
        gate.wait(timeout=5.0)
        return "active_done"

    async def exercise():
        t_active = asyncio.create_task(worker.submit(active_fn))
        for _ in range(100):
            if started_active.is_set():
                break
            await asyncio.sleep(0.01)
        assert started_active.is_set()

        t_queued = asyncio.create_task(worker.submit(lambda: "should_not_run"))
        await asyncio.sleep(0.05)

        close_thread = threading.Thread(target=worker.close, kwargs={"timeout": 5.0})
        close_thread.start()

        # Queued job must immediately fail with 503 APIError (not hang)
        with pytest.raises(APIError) as exc_info:
            await asyncio.wait_for(t_queued, timeout=2.0)
        assert exc_info.value.status_code == 503

        # Active forward completes cleanly
        gate.set()
        assert await asyncio.wait_for(t_active, timeout=2.0) == "active_done"
        close_thread.join(timeout=5.0)
        assert not close_thread.is_alive()

    asyncio.run(exercise())


def test_per_question_preparation_nulls_both_when_suffix_differs():
    bundle, reg = _make_bundle()
    cfg = SidecarConfig.from_env({})
    with TestClient(create_app(cfg, bundle=bundle, registry=reg)) as client:
        # Short state: prefix_tokens is equal for both decisions, but suffix_tokens differs (compaction vs loop_guard)
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "bob-s1-0.8b",
                "messages": [
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "protocol": "bob-s1.decision.v1",
                                "state": "short state",
                                "prefix_mode": "per_question",
                                "decisions": [
                                    {"id": "d1", "task": "compaction"},
                                    {"id": "d2", "task": "loop_guard"},
                                ],
                            }
                        ),
                    }
                ],
            },
        )
        assert r.status_code == 200
        body = json.loads(r.json()["choices"][0]["message"]["content"])
        assert body["decisions"][0]["prefix_tokens"] == body["decisions"][1]["prefix_tokens"]
        assert body["decisions"][0]["suffix_tokens"] != body["decisions"][1]["suffix_tokens"]
        assert body["preparation"]["prefix_tokens"] is None
        assert body["preparation"]["suffix_reserve"] is None
        assert body["preparation"]["truncated"] is False


def test_chat_enforces_max_request_images():
    bundle, reg = _make_bundle()
    cfg = SidecarConfig.from_env({"BOB_S1_MAX_IMAGES": "8", "BOB_S1_MAX_REQUEST_IMAGES": "1"})
    img_uri = _png_data_uri(size=(4, 4))
    with TestClient(create_app(cfg, bundle=bundle, registry=reg)) as client:
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "bob-s1-0.8b",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": img_uri}},
                            {"type": "image_url", "image_url": {"url": img_uri}},
                            {
                                "type": "text",
                                "text": json.dumps(
                                    {
                                        "protocol": "bob-s1.decision.v1",
                                        "state": "two images",
                                        "decisions": [{"id": "d1", "task": "compaction"}],
                                    }
                                ),
                            },
                        ],
                    }
                ],
            },
        )
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "too_many_images"


def test_cancelled_decode_keeps_admission_lease_until_thread_exits(monkeypatch):
    bundle, reg = _make_bundle()
    cfg = SidecarConfig.from_env({"BOB_S1_MAX_QUEUE_DEPTH": "1"})
    decode_entered = threading.Event()
    release_decode = threading.Event()

    import bob_s1.app as app_mod

    orig_materialize = app_mod.materialize_embeddings_request

    def blocked_materialize(raw, c):
        if raw.items[0].text == "slow_decode":
            decode_entered.set()
            release_decode.wait(timeout=5.0)
        return orig_materialize(raw, c)

    monkeypatch.setattr(app_mod, "materialize_embeddings_request", blocked_materialize)

    async def exercise():
        import httpx

        app = create_app(cfg, bundle=bundle, registry=reg)
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
                t1 = asyncio.create_task(ac.post("/v1/embeddings", json={"model": "bob-s1-0.8b", "input": "slow_decode"}))
                for _ in range(100):
                    if decode_entered.is_set():
                        break
                    await asyncio.sleep(0.01)
                assert decode_entered.is_set()

                # Cancel the HTTP task while its decoder thread is still blocked
                t1.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await t1

                # Admission slot MUST remain held until the decode thread exits
                r_overload = await ac.post("/v1/embeddings", json={"model": "bob-s1-0.8b", "input": "fast"})
                assert r_overload.status_code == 429
                assert r_overload.json()["error"]["code"] == "queue_full"

                # Release decoder thread and allow done_callback to run on loop
                release_decode.set()
                await asyncio.sleep(0.05)

                r_ok = await ac.post("/v1/embeddings", json={"model": "bob-s1-0.8b", "input": "fast_after"})
                assert r_ok.status_code == 200

    asyncio.run(exercise())


def test_shutdown_vs_disconnect_race_completes_close_cleanly():
    from bob_s1.worker import QueuedJob

    # Direct atomic race test on QueuedJob
    j_cancel_first = QueuedJob(fn=lambda: None)
    assert j_cancel_first.try_cancel_if_queued() is True
    assert j_cancel_first.fail_if_queued(APIError(503, "shutdown")) is False
    assert j_cancel_first.future.cancelled() is True

    j_fail_first = QueuedJob(fn=lambda: None)
    assert j_fail_first.fail_if_queued(APIError(503, "shutdown")) is True
    assert j_fail_first.try_cancel_if_queued() is True
    assert isinstance(j_fail_first.future.exception(), APIError)

    # Full worker test: 1 active job, 1 disconnected queued job, 1 remaining queued job
    bundle, reg = _make_bundle()
    cfg = SidecarConfig.from_env({"BOB_S1_MAX_QUEUE_DEPTH": "4"})
    worker = InferenceWorker(bundle, cfg, reg)

    started_active = threading.Event()
    gate = threading.Event()

    def active_fn():
        started_active.set()
        gate.wait(timeout=5.0)
        return "active_done"

    async def exercise():
        t_active = asyncio.create_task(worker.submit(active_fn))
        for _ in range(100):
            if started_active.is_set():
                break
            await asyncio.sleep(0.01)
        assert started_active.is_set()

        lease1 = worker.reserve_slot()
        job_disc = lease1.promote_to_queue(lambda: "disc")
        lease2 = worker.reserve_slot()
        job_keep = lease2.promote_to_queue(lambda: "keep")

        # Cancel job_disc concurrently right as close() drains the queue
        job_disc.try_cancel_if_queued()
        close_thread = threading.Thread(target=worker.close, kwargs={"timeout": 5.0})
        close_thread.start()

        with pytest.raises(APIError) as exc_info:
            await asyncio.wait_for(asyncio.wrap_future(job_keep.future), timeout=2.0)
        assert exc_info.value.status_code == 503

        gate.set()
        assert await asyncio.wait_for(t_active, timeout=2.0) == "active_done"
        close_thread.join(timeout=5.0)
        assert not close_thread.is_alive()

    asyncio.run(exercise())
