"""Real-checkpoint numerical parity tests against cached bob-s1-0.8b weights."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator
import pytest

from bob_s1.app import create_app
from bob_s1.calibration import CalibrationRegistry
from bob_s1.config import SidecarConfig
from bob_s1.loader import CheckpointOverlayLoader, import_jev_model_module

CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "contracts"


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)


def _check_prerequisites_or_skip(cfg: SidecarConfig) -> None:
    require_parity = os.environ.get("BOB_S1_REQUIRE_PARITY", "").strip().lower() in ("1", "true", "yes")
    missing_reason: str | None = None

    if not cfg.checkpoint_path.is_file():
        missing_reason = f"Real checkpoint not present at {cfg.checkpoint_path}"
    else:
        jev_model = import_jev_model_module(cfg.repo_root)
        cache_dir = str(cfg.hf_cache_dir) if cfg.hf_cache_dir is not None else None
        try:
            jev_model.resolve_cached_snapshot(
                jev_model.BASE_MODEL, jev_model.BASE_MODEL_REVISION, cache_dir=cache_dir, local_files_only=True
            )
            jev_model.resolve_cached_snapshot(
                jev_model.KEV_ADAPTER, jev_model.KEV_ADAPTER_REVISION, cache_dir=cache_dir, local_files_only=True
            )
        except OSError as exc:
            missing_reason = f"Cached Hugging Face snapshots unavailable: {exc}"

    if missing_reason is not None:
        if require_parity:
            pytest.fail(f"BOB_S1_REQUIRE_PARITY=1 but prerequisites missing: {missing_reason}")
        pytest.skip(missing_reason)


def test_real_bob_s1_http_parity_against_frozen_reference():
    env = dict(os.environ)
    env["BOB_S1_MAX_SEQ_LEN"] = "4096"
    cfg = SidecarConfig.from_env(env)
    _check_prerequisites_or_skip(cfg)

    # Once prerequisites exist, load errors (digest mismatch, corrupt tensors, warmup failure) MUST fail the test.
    bundle = CheckpointOverlayLoader(cfg).load()
    reg = CalibrationRegistry.from_file()

    parity = json.loads((CONTRACTS_DIR / "real_checkpoint_parity_fixtures.json").read_text(encoding="utf-8"))
    conformance = json.loads((CONTRACTS_DIR / "conformance_fixtures.json").read_text(encoding="utf-8"))
    tol = conformance["tolerances"]
    schemas = {p.name: json.loads(p.read_text(encoding="utf-8")) for p in CONTRACTS_DIR.glob("*.schema.json")}

    assert bundle.checkpoint_id == parity["checkpoint_id"]
    assert bundle.checkpoint_sha256 == parity["checkpoint_sha256"]

    emb_tol = float(tol["embedding_cosine_min_bf16"])
    mm_tol = float(tol["multimodal_embedding_cosine_min_bf16"])
    logit_tol = float(tol["decision_logit_max_abs_diff_bf16"])
    runtime_tag = f"device={bundle.device} dtype={bundle.dtype} attn={bundle.attn_implementation}"

    with TestClient(create_app(cfg, bundle=bundle, registry=reg)) as client:
        # 1. Text embeddings parity across 128/256/384
        for dim in (128, 256, 384):
            r_emb = client.post(
                "/v1/embeddings",
                json={"model": bundle.model_id, "input": parity["text_sample"], "dimensions": dim},
            )
            assert r_emb.status_code == 200
            body = r_emb.json()
            Draft202012Validator(schemas["embeddings.response.schema.json"]).validate(body)
            vec = body["data"][0]["embedding"]
            cos = _cosine(vec, parity["text_embeddings"][str(dim)])
            assert cos >= emb_tol, f"[{runtime_tag}] Text {dim}d cosine {cos} < {emb_tol}"

        # 2. Multimodal embeddings parity + red/blue visual sensitivity
        r_mm = client.post(
            "/v1/embeddings",
            json={
                "model": bundle.model_id,
                "input": [
                    {
                        "content": [
                            {"type": "image_url", "image_url": {"url": parity["assets"]["red_32x32_png"]}},
                            {"type": "text", "text": parity["multimodal_text"]},
                        ]
                    },
                    {
                        "content": [
                            {"type": "image_url", "image_url": {"url": parity["assets"]["blue_32x32_png"]}},
                            {"type": "text", "text": parity["multimodal_text"]},
                        ]
                    },
                ],
                "dimensions": 384,
            },
        )
        assert r_mm.status_code == 200
        mm_body = r_mm.json()
        Draft202012Validator(schemas["embeddings.response.schema.json"]).validate(mm_body)
        vec_red = mm_body["data"][0]["embedding"]
        vec_blue = mm_body["data"][1]["embedding"]
        assert _cosine(vec_red, parity["multimodal_red_384"]) >= mm_tol, runtime_tag
        assert _cosine(vec_blue, parity["multimodal_blue_384"]) >= mm_tol, runtime_tag
        assert _cosine(vec_red, vec_blue) < 0.999

        # 3. Built-in decision parity (calibrated compaction & outcome_prefix; uncalibrated loop_guard)
        for task_name in ("compaction", "outcome_prefix", "loop_guard"):
            r_dec = client.post(
                "/v1/chat/completions",
                json={
                    "model": bundle.model_id,
                    "messages": [
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "protocol": "bob-s1.decision.v1",
                                    "state": parity["text_sample"],
                                    "decisions": [{"id": task_name, "task": task_name}],
                                }
                            ),
                        }
                    ],
                },
            )
            assert r_dec.status_code == 200
            resp_json = r_dec.json()
            Draft202012Validator(schemas["chat.completions.response.schema.json"]).validate(resp_json)
            inner = json.loads(resp_json["choices"][0]["message"]["content"])
            Draft202012Validator(schemas["decision.response.schema.json"]).validate(inner)
            d0 = inner["decisions"][0]
            got_logits = [s["logit"] for s in d0["scores"]]
            ref_logits = parity["decision_logits"][task_name]
            max_diff = max(abs(a - b) for a, b in zip(got_logits, ref_logits))
            assert max_diff <= logit_tol, f"[{runtime_tag}] {task_name} max_diff={max_diff} > {logit_tol}"
            if task_name in ("compaction", "outcome_prefix"):
                assert d0["probability_kind"] == "calibrated_affine"
                assert d0["calibration"]["applied"] is True
            else:
                assert d0["probability_kind"] == "uncalibrated_softmax"
                assert d0["calibration"]["applied"] is False

        # 4. Joint single-pass shared_prefix multimodal decision + 128d embedding parity
        r_joint = client.post(
            "/v1/chat/completions",
            json={
                "model": bundle.model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": parity["assets"]["red_32x32_png"]}},
                            {
                                "type": "text",
                                "text": json.dumps(
                                    {
                                        "protocol": "bob-s1.decision.v1",
                                        "state": parity["multimodal_text"],
                                        "prefix_mode": "shared_prefix",
                                        "embedding": {"dimensions": 128, "encoding_format": "float"},
                                        "decisions": [
                                            {"id": "c1", "task": "compaction"},
                                            {"id": "c2", "task": "loop_guard"},
                                        ],
                                    }
                                ),
                            },
                        ],
                    }
                ],
            },
        )
        assert r_joint.status_code == 200
        joint_inner = json.loads(r_joint.json()["choices"][0]["message"]["content"])
        Draft202012Validator(schemas["decision.response.schema.json"]).validate(joint_inner)
        assert _cosine(joint_inner["embedding"]["vector"], parity["joint_shared_prefix"]["embedding_128"]) >= mm_tol, runtime_tag
        c1_logits = [s["logit"] for s in joint_inner["decisions"][0]["scores"]]
        c2_logits = [s["logit"] for s in joint_inner["decisions"][1]["scores"]]
        assert max(abs(a - b) for a, b in zip(c1_logits, parity["joint_shared_prefix"]["compaction_logits"])) <= logit_tol, runtime_tag
        assert max(abs(a - b) for a, b in zip(c2_logits, parity["joint_shared_prefix"]["loop_guard_logits"])) <= logit_tol, runtime_tag
