"""Phase 4 comprehensive conformance, calibration, truncation, and DeltaNet isolation suite."""
from __future__ import annotations

import base64
from io import BytesIO
import json
import math
from pathlib import Path
import struct

from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator
from PIL import Image
import pytest

from bob_s1.app import create_app
from bob_s1.calibration import BUILTIN_QUESTIONS, CalibrationRegistry, score_decision
from bob_s1.config import SidecarConfig
from bob_s1.loader import LoadedModelBundle, import_jev_model_module, verify_model_runtime

CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "contracts"


def _load_schemas() -> dict[str, dict]:
    return {p.name: json.loads(p.read_text(encoding="utf-8")) for p in sorted(CONTRACTS_DIR.glob("*.schema.json"))}


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


def _decode_base64_f32(b64_str: str) -> list[float]:
    raw = base64.b64decode(b64_str, validate=True)
    assert len(raw) % 4 == 0
    return list(struct.unpack(f"<{len(raw) // 4}f", raw))


def _png_data_uri(color: tuple[int, int, int], size: tuple[int, int] = (16, 16)) -> str:
    buf = BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def test_conformance_fixtures_valid_and_rejection_cases():
    schemas = _load_schemas()
    fixtures = json.loads((CONTRACTS_DIR / "conformance_fixtures.json").read_text(encoding="utf-8"))

    for schema in schemas.values():
        Draft202012Validator.check_schema(schema)

    for neg in fixtures["response_rejection_cases"]:
        errs = list(Draft202012Validator(schemas[neg["schema"]]).iter_errors(neg["payload"]))
        assert errs, f"Expected response schema error for {neg['name']}"

    bundles: dict[int, tuple[LoadedModelBundle, CalibrationRegistry]] = {
        4096: _make_bundle(4096),
        8192: _make_bundle(8192),
    }

    for max_seq in (4096, 8192):
        bundle, reg = bundles[max_seq]
        cfg = SidecarConfig.from_env({"BOB_S1_MAX_SEQ_LEN": str(max_seq)})
        with TestClient(create_app(cfg, bundle=bundle, registry=reg)) as client:
            for case in fixtures["valid_cases"]:
                if case["server_config"]["max_seq_len"] != max_seq:
                    continue

                # Validate request fixture against frozen Draft 2020-12 request schemas
                if case["endpoint"] == "/v1/embeddings":
                    Draft202012Validator(schemas["embeddings.request.schema.json"]).validate(case["request"])
                else:
                    Draft202012Validator(schemas["chat.completions.request.schema.json"]).validate(case["request"])
                    raw_msg = case["request"]["messages"][0]["content"]
                    inner_req = json.loads(raw_msg if isinstance(raw_msg, str) else raw_msg[-1]["text"])
                    Draft202012Validator(schemas["decision.request.schema.json"]).validate(inner_req)

                resp = client.post(case["endpoint"], json=case["request"])
                assert resp.status_code == case["expected"]["status"], f"{case['name']}: {resp.text}"
                body = resp.json()

                if case["endpoint"] == "/v1/embeddings":
                    Draft202012Validator(schemas["embeddings.response.schema.json"]).validate(body)
                    assert len(body["data"]) == case["expected"]["count"]
                    for idx, item in enumerate(body["data"]):
                        assert item["index"] == idx
                        vec = (
                            _decode_base64_f32(item["embedding"])
                            if case["expected"]["encoding_format"] == "base64"
                            else item["embedding"]
                        )
                        assert len(vec) == case["expected"]["dimensions"]
                        norm = math.sqrt(sum(v * v for v in vec))
                        assert norm == pytest.approx(1.0, abs=1e-4)
                else:
                    Draft202012Validator(schemas["chat.completions.response.schema.json"]).validate(body)
                    inner = json.loads(body["choices"][0]["message"]["content"])
                    Draft202012Validator(schemas["decision.response.schema.json"]).validate(inner)
                    exp = case["expected"]
                    if "decision_count" in exp:
                        assert len(inner["decisions"]) == exp["decision_count"]
                    if "effective_max_seq_len" in exp:
                        assert inner["preparation"]["effective_max_seq_len"] == exp["effective_max_seq_len"]
                    if "has_embedding" in exp:
                        assert (inner["embedding"] is not None) is exp["has_embedding"]
                        assert inner["embedding"]["dimensions"] == exp["embedding_dimensions"]
                    d0 = inner["decisions"][0]
                    if "probability_kind" in exp:
                        assert d0["probability_kind"] == exp["probability_kind"]
                    if "calibration_applied" in exp:
                        assert d0["calibration"]["applied"] is exp["calibration_applied"]
                    if "choice_basis" in exp:
                        assert d0["choice_basis"] == exp["choice_basis"]
                    if "option_ids" in exp:
                        assert [s["id"] for s in d0["scores"]] == exp["option_ids"]
                    if "scoring_mode" in exp:
                        assert d0["scoring_mode"] == exp["scoring_mode"]
                    if "has_expected_score_for" in exp:
                        target = next(d for d in inner["decisions"] if d["id"] == exp["has_expected_score_for"])
                        assert target["expected_score"] is not None

            for case in fixtures["rejection_cases"]:
                if case["server_config"]["max_seq_len"] != max_seq:
                    continue
                resp = client.post(case["endpoint"], json=case["request"])
                assert resp.status_code == case["expected"]["status"], f"{case['name']}: {resp.text}"
                err_body = resp.json()
                Draft202012Validator(schemas["error.schema.json"]).validate(err_body)
                if "error_type" in case["expected"]:
                    assert err_body["error"]["type"] == case["expected"]["error_type"]
                if "code" in case["expected"]:
                    assert err_body["error"]["code"] == case["expected"]["code"]


def test_embeddings_float_base64_parity_and_independent_slice_normalization():
    bundle, reg = _make_bundle()
    cfg = SidecarConfig.from_env({})
    text = "Agent ran pytest and checked git status."
    with TestClient(create_app(cfg, bundle=bundle, registry=reg)) as client:
        vecs: dict[int, list[float]] = {}
        for dim in (128, 256, 384):
            r_float = client.post(
                "/v1/embeddings",
                json={"model": "bob-s1-0.8b", "input": text, "dimensions": dim, "encoding_format": "float"},
            )
            r_b64 = client.post(
                "/v1/embeddings",
                json={"model": "bob-s1-0.8b", "input": text, "dimensions": dim, "encoding_format": "base64"},
            )
            assert r_float.status_code == 200 and r_b64.status_code == 200
            vf = r_float.json()["data"][0]["embedding"]
            vb = _decode_base64_f32(r_b64.json()["data"][0]["embedding"])
            assert len(vf) == dim and len(vb) == dim
            for a, b in zip(vf, vb):
                assert a == pytest.approx(b, abs=1e-6)
            assert math.sqrt(sum(x * x for x in vf)) == pytest.approx(1.0, abs=1e-5)
            vecs[dim] = vf

        trunc_norm = math.sqrt(sum(x * x for x in vecs[384][:128]))
        assert trunc_norm < 0.99
        assert math.sqrt(sum(x * x for x in vecs[128])) == pytest.approx(1.0, abs=1e-5)


def test_multimodal_image_sensitivity_on_embeddings_and_decisions():
    bundle, reg = _make_bundle()
    cfg = SidecarConfig.from_env({})
    red_uri = _png_data_uri((255, 0, 0))
    blue_uri = _png_data_uri((0, 0, 255))

    with TestClient(create_app(cfg, bundle=bundle, registry=reg)) as client:
        r_emb = client.post(
            "/v1/embeddings",
            json={
                "model": "bob-s1-0.8b",
                "input": [
                    {"content": [{"type": "image_url", "image_url": {"url": red_uri}}, {"type": "text", "text": "screen"}]},
                    {"content": [{"type": "image_url", "image_url": {"url": blue_uri}}, {"type": "text", "text": "screen"}]},
                ],
                "dimensions": 384,
            },
        )
        assert r_emb.status_code == 200
        v_red = r_emb.json()["data"][0]["embedding"]
        v_blue = r_emb.json()["data"][1]["embedding"]
        max_diff = max(abs(a - b) for a, b in zip(v_red, v_blue))
        assert max_diff > 1e-3

        def decide(img_uri: str):
            payload = {
                "model": "bob-s1-0.8b",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": img_uri}},
                            {
                                "type": "text",
                                "text": json.dumps(
                                    {
                                        "protocol": "bob-s1.decision.v1",
                                        "state": "screen",
                                        "decisions": [{"id": "d1", "task": "compaction"}],
                                    }
                                ),
                            },
                        ],
                    }
                ],
            }
            res = client.post("/v1/chat/completions", json=payload)
            assert res.status_code == 200
            return json.loads(res.json()["choices"][0]["message"]["content"])["decisions"][0]

        d_red = decide(red_uri)
        d_blue = decide(blue_uri)
        logit_diff = max(abs(a["logit"] - b["logit"]) for a, b in zip(d_red["scores"], d_blue["scores"]))
        assert logit_diff > 1e-4


def test_elastic_truncation_policies_and_exact_suffix_preservation():
    bundle, reg = _make_bundle(max_seq_len=4096)
    cfg = SidecarConfig.from_env({"BOB_S1_MAX_SEQ_LEN": "4096"})
    long_state = "token " * 400
    q_comp, opts_comp = BUILTIN_QUESTIONS["compaction"]
    expected_suffix_ids, _, _ = bundle.model.question_ids(q_comp, list(opts_comp))
    expected_suffix_len = len(expected_suffix_ids)

    with TestClient(create_app(cfg, bundle=bundle, registry=reg)) as client:
        r_err = client.post(
            "/v1/embeddings",
            json={"model": "bob-s1-0.8b", "input": long_state, "max_seq_len": 256},
        )
        assert r_err.status_code == 400
        assert r_err.json()["error"]["code"] == "context_length_exceeded"

        r_ht = client.post(
            "/v1/embeddings",
            json={"model": "bob-s1-0.8b", "input": long_state, "max_seq_len": 256, "truncation": "head_tail"},
        )
        assert r_ht.status_code == 200
        assert r_ht.json()["usage"]["prompt_tokens"] == 256

        r_dec = client.post(
            "/v1/chat/completions",
            json={
                "model": "bob-s1-0.8b",
                "messages": [
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "protocol": "bob-s1.decision.v1",
                                "state": long_state,
                                "max_seq_len": 256,
                                "truncation": "head_tail",
                                "decisions": [{"id": "d1", "task": "compaction"}],
                            }
                        ),
                    }
                ],
            },
        )
        assert r_dec.status_code == 200
        inner = json.loads(r_dec.json()["choices"][0]["message"]["content"])
        d0 = inner["decisions"][0]
        assert inner["preparation"]["effective_max_seq_len"] == 256
        assert d0["truncated"] is True
        # Exact suffix length preserved intact; state prefix absorbs the full remaining budget
        assert d0["suffix_tokens"] == expected_suffix_len
        assert d0["prefix_tokens"] == 256 - expected_suffix_len


def test_calibration_math_winner_flip_mismatch_disclosures_and_tie_breaking():
    bundle, reg = _make_bundle(max_seq_len=4096)
    cfg = SidecarConfig.from_env({"BOB_S1_MAX_SEQ_LEN": "4096"})

    # 1. Exact tie -> earliest index wins deterministically
    tied = score_decision(
        logits=[0.5, 0.5],
        option_ids=["opt_a", "opt_b"],
        option_values=[0.0, 10.0],
        task=None,
        scoring_mode="joint",
        prefix_mode="per_question",
        effective_max_seq_len=4096,
        checkpoint_id=reg.checkpoint_id,
        checkpoint_sha256=reg.checkpoint_sha256,
        apply_calibration=True,
        require_calibrated=False,
        registry=reg,
    )
    assert tied["choice"] == "opt_a"
    assert tied["choice_basis"] == "raw_argmax"
    assert tied["expected_score"] == pytest.approx(5.0, abs=1e-9)

    # 2. Controlled logits where raw_argmax ('skip') flips to calibrated_argmax ('compact')
    # margin = (-1.0 - 1.0 + 4.25) / 1.895898683669047 = +2.25 / 1.8959 > 0 -> p_compact > 0.5
    flipped = score_decision(
        logits=[1.0, -1.0],
        option_ids=["skip", "compact"],
        option_values=[0.0, 1.0],
        task="compaction",
        scoring_mode="joint",
        prefix_mode="per_question",
        effective_max_seq_len=4096,
        checkpoint_id=reg.checkpoint_id,
        checkpoint_sha256=reg.checkpoint_sha256,
        apply_calibration=True,
        require_calibrated=False,
        registry=reg,
    )
    assert flipped["choice"] == "compact"
    assert flipped["choice_basis"] == "calibrated_argmax"
    assert flipped["probability_kind"] == "calibrated_affine"
    expected_p1 = 1.0 / (1.0 + math.exp(-((-1.0 - 1.0 + 4.25) / 1.895898683669047)))
    assert flipped["scores"][1]["probability"] == pytest.approx(expected_p1, abs=1e-9)
    assert flipped["expected_score"] == pytest.approx(expected_p1, abs=1e-9)

    with TestClient(create_app(cfg, bundle=bundle, registry=reg)) as client:
        # 3. Live endpoint calibrated compaction matches affine formula
        r_cal = client.post(
            "/v1/chat/completions",
            json={
                "model": "bob-s1-0.8b",
                "messages": [
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "protocol": "bob-s1.decision.v1",
                                "state": "trace",
                                "decisions": [{"id": "d1", "task": "compaction"}],
                            }
                        ),
                    }
                ],
            },
        )
        assert r_cal.status_code == 200
        d_cal = json.loads(r_cal.json()["choices"][0]["message"]["content"])["decisions"][0]
        z0 = d_cal["scores"][0]["logit"]
        z1 = d_cal["scores"][1]["logit"]
        cal = d_cal["calibration"]
        live_p1 = 1.0 / (1.0 + math.exp(-((z1 - z0 + cal["bias"]) / cal["temperature"])))
        assert d_cal["scores"][1]["probability"] == pytest.approx(live_p1, abs=1e-9)
        assert d_cal["scores"][0]["probability"] == pytest.approx(1.0 - live_p1, abs=1e-9)

        # 4. shared_prefix mode disables calibration with explicit mismatch_reason
        r_sp = client.post(
            "/v1/chat/completions",
            json={
                "model": "bob-s1-0.8b",
                "messages": [
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "protocol": "bob-s1.decision.v1",
                                "state": "trace",
                                "prefix_mode": "shared_prefix",
                                "decisions": [{"id": "d1", "task": "compaction"}],
                            }
                        ),
                    }
                ],
            },
        )
        assert r_sp.status_code == 200
        d_sp = json.loads(r_sp.json()["choices"][0]["message"]["content"])["decisions"][0]
        assert d_sp["probability_kind"] == "uncalibrated_softmax"
        assert d_sp["calibration"]["applied"] is False
        assert d_sp["calibration"]["mismatch_reason"] == "prefix_mode_not_calibrated"

        # 5. equivariant mode disables calibration and rejects require_calibrated=True
        r_eq = client.post(
            "/v1/chat/completions",
            json={
                "model": "bob-s1-0.8b",
                "messages": [
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "protocol": "bob-s1.decision.v1",
                                "state": "trace",
                                "decisions": [
                                    {
                                        "id": "d1",
                                        "task": "compaction",
                                        "scoring_mode": "equivariant",
                                        "require_calibrated": True,
                                    }
                                ],
                            }
                        ),
                    }
                ],
            },
        )
        assert r_eq.status_code == 400
        assert r_eq.json()["error"]["code"] == "calibration_unavailable"


def test_deltanet_prefix_isolation_across_siblings_shared_prefix_and_requests():
    bundle_a, reg = _make_bundle()
    bundle_fresh, _ = _make_bundle()
    cfg = SidecarConfig.from_env({})
    state_a = "Iter 1: ran pytest and observed failure in auth.py."
    state_b = "Completely different long multimodal trace " * 12
    red_uri = _png_data_uri((255, 0, 0))

    # Fresh isolated baseline for state_a embedding & compaction logits
    with TestClient(create_app(cfg, bundle=bundle_fresh, registry=reg)) as fresh_client:
        fresh_emb = fresh_client.post(
            "/v1/embeddings",
            json={"model": "bob-s1-0.8b", "input": state_a, "dimensions": 128},
        ).json()["data"][0]["embedding"]
        fresh_dec = json.loads(
            fresh_client.post(
                "/v1/chat/completions",
                json={
                    "model": "bob-s1-0.8b",
                    "messages": [
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "protocol": "bob-s1.decision.v1",
                                    "state": state_a,
                                    "decisions": [{"id": "d1", "task": "compaction"}],
                                }
                            ),
                        }
                    ],
                },
            ).json()["choices"][0]["message"]["content"]
        )["decisions"][0]
        fresh_logits = [s["logit"] for s in fresh_dec["scores"]]

    with TestClient(create_app(cfg, bundle=bundle_a, registry=reg)) as client:
        def post_decisions(decisions, prefix_mode="per_question"):
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
                                    "state": state_a,
                                    "prefix_mode": prefix_mode,
                                    "decisions": decisions,
                                }
                            ),
                        }
                    ],
                },
            )
            assert r.status_code == 200
            return json.loads(r.json()["choices"][0]["message"]["content"])["decisions"]

        # 1. per_question sibling isolation
        q_sibling_long = {
            "id": "q_extra",
            "question": "Another sibling question with lots of tokens?",
            "options": [
                {"id": "o1", "description": "Option one description"},
                {"id": "o2", "description": "Option two description"},
            ],
        }
        multi_pq = post_decisions([{"id": "lg", "task": "loop_guard"}, {"id": "d1", "task": "compaction"}, q_sibling_long], "per_question")
        comp_pq = [s["logit"] for s in next(d for d in multi_pq if d["id"] == "d1")["scores"]]
        for a, b in zip(fresh_logits, comp_pq):
            assert a == pytest.approx(b, abs=1e-5)

        # 2. shared_prefix sibling permutation & sibling content change invariance (when max suffix reserve stays constant)
        sp_order_1 = post_decisions([{"id": "d1", "task": "compaction"}, q_sibling_long], "shared_prefix")
        q_sibling_alt = {
            "id": "q_extra",
            "question": "Different text same length? padding tokens here!",
            "options": [
                {"id": "o1", "description": "Alternative one text!"},
                {"id": "o2", "description": "Alternative two text!"},
            ],
        }
        sp_order_2 = post_decisions([q_sibling_alt, {"id": "d1", "task": "compaction"}], "shared_prefix")
        sp_logits_1 = [s["logit"] for s in next(d for d in sp_order_1 if d["id"] == "d1")["scores"]]
        sp_logits_2 = [s["logit"] for s in next(d for d in sp_order_2 if d["id"] == "d1")["scores"]]
        for a, b, f in zip(sp_logits_1, sp_logits_2, fresh_logits):
            assert a == pytest.approx(b, abs=1e-5)
            assert a == pytest.approx(f, abs=1e-5)

        # 3. Cross-request isolation: run heavy multimodal request B, then re-verify state_a embedding & decision numerically
        r_b = client.post(
            "/v1/chat/completions",
            json={
                "model": "bob-s1-0.8b",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": red_uri}},
                            {
                                "type": "text",
                                "text": json.dumps(
                                    {
                                        "protocol": "bob-s1.decision.v1",
                                        "state": state_b,
                                        "prefix_mode": "shared_prefix",
                                        "embedding": {"dimensions": 384},
                                        "decisions": [{"id": "b1", "task": "outcome_prefix"}],
                                    }
                                ),
                            },
                        ],
                    }
                ],
            },
        )
        assert r_b.status_code == 200

        after_emb = client.post(
            "/v1/embeddings",
            json={"model": "bob-s1-0.8b", "input": state_a, "dimensions": 128},
        ).json()["data"][0]["embedding"]
        for a, b in zip(fresh_emb, after_emb):
            assert a == pytest.approx(b, abs=1e-6)

        after_dec = post_decisions([{"id": "d1", "task": "compaction"}], "per_question")[0]
        for a, b in zip(fresh_logits, [s["logit"] for s in after_dec["scores"]]):
            assert a == pytest.approx(b, abs=1e-6)
