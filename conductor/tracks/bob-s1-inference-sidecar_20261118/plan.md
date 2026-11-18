# Implementation Plan: Temporary `bob-s1` Python Inference Sidecar (OpenAI-Compatible)

## Phase 1: Contract Freeze, Spec Reconciliation & Conformance Fixtures (`services/bob_s1/contracts/`)
- [x] Task 1.1: Reconcile [`conductor/tracks/jev-multimodal-decision-embedder_20261114/spec.md`](../jev-multimodal-decision-embedder_20261114/spec.md) with implemented [`eval/jev/model.py`](../../../eval/jev/model.py) and [`eval/BOB_S1.md`](../../../eval/BOB_S1.md) (remove stale `LayerNorm` / `<|state_end|>` mentions, document `services/bob_s1` as the temporary sidecar prior to Phase B native server integration).
- [x] Task 1.2: Define versioned Draft 2020-12 JSON schemas (`chat.completions.{request,response}.schema.json`, `decision.{request,response}.schema.json`, `embeddings.{request,response}.schema.json`, `error.schema.json`) and concrete portable conformance fixtures (`conformance_fixtures.json`) so future `lemond`/`vllm`/`llama.cpp` server patches can be verified against the exact same wire contract.
- [x] Task 1.3: Export exact full-precision calibration constants from [`eval/results/jev_dataset/dev_final_v2_4k/experiment_report.json`](../../../eval/results/jev_dataset/dev_final_v2_4k/experiment_report.json) (`exp_b_v2_4k_emb_onlymm`) into a self-contained serving calibration registry (`calibration_profiles.json`) bound to checkpoint SHA-256 and preparation invariants, decoupled from `eval_experiment.py`.

---

## Phase 2: Service Packaging, Validated Model Loader & Secure Image Decoder (`services/bob_s1/`)
- [x] Task 2.1: Create `services/bob_s1/` package layout (`pyproject.toml`, `uv.lock`, `config.py`, `schemas.py`, `images.py`, `calibration.py`, `loader.py`, `worker.py`, `app.py`) with mutually exclusive `cpu`, `cuda`, and `rocm` Torch installation profiles and pinned `transformers`/`peft` dependencies. (Implemented `pyproject.toml`, `uv.lock`, `config.py`, `images.py`, `loader.py`, `osv-scanner.toml`; remaining runtime modules in Phase 3.)
- [x] Task 2.2: Implement `ModelLoader` interface and `CheckpointOverlayLoader` wrapping `DualHeadDecisionEmbedder` from [`eval/jev/model.py`](../../../eval/jev/model.py) (without importing `train_experiment.py` or `eval_experiment.py`):
  - Support `BOB_S1_CHECKPOINT`, `BOB_S1_HF_CACHE_DIR`, `BOB_S1_DEVICE`, `BOB_S1_DTYPE`, `BOB_S1_ATTN_IMPLEMENTATION` (`eager` default, optional `sdpa`), and `BOB_S1_MAX_SEQ_LEN` (`4096` default, valid `256..262144`).
  - Explicitly verify required LoRA, `PointerHead`, and `W_emb` tensors, checkpoint SHA-256/identity, shapes, finiteness, special token IDs, and vision preprocessing metadata at startup; keep heads FP32; run startup warmup before marking `/healthz` ready.
- [x] Task 2.3: Implement strict inline image decoder (`images.py`):
  - Accept only `data:image/(png|jpeg|webp|gif);base64,...` URIs; reject `http(s)://`, `file://`, and filesystem paths.
  - Validate decoded byte limit, `PIL.Image` verification, single-frame check, and pixel dimensions before passing `PIL.Image` objects to the model (never pass raw strings to `_to_pil_rgb()`).

---

## Phase 3: Worker Scheduler, DeltaNet Prefix Isolation & OpenAI-Compatible Endpoints
- [x] Task 3.1: Implement single-worker inference scheduler (`worker.py`):
  - Dedicated device-owning thread + bounded FIFO queue (`BOB_S1_MAX_QUEUE_DEPTH`) executing unpadded `batch=1` forwards under `torch.inference_mode()` + `model.eval()`.
  - Handle client disconnects/cancellations safely at queue wait boundaries without releasing GPU lock mid-forward; return `429` (`queue_full`) on queue saturation.
  - Enforce exact `effective_max_seq_len` (`BOB_S1_MAX_SEQ_LEN` when omitted; reject explicit `max_seq_len > BOB_S1_MAX_SEQ_LEN` with `400`; accept in-range caps unchanged) across full token budgets (vision + delimiters + state + suffix).
- [x] Task 3.2: Implement embedding, decision, and joint execution paths + calibration engine (`calibration.py`):
  - `/v1/embeddings`: standard `str` / `list[str]` and multimodal `[{"content": [...]}]` (images followed by optional text; reject repeated text, text-before-image, and token arrays); `dimensions` in `{128, 256, 384}` via `m.embeddings()`; `"float"` and little-endian float32 `"base64"`; `"error"` (default) and `"head_tail"` truncation; exact `400` when `text_budget < 1`.
  - `/v1/chat/completions`: structural parsing of outer Chat Completions envelope (`messages` = single `user` message with optional preceding `image_url` parts + JSON text part) + `bob-s1.decision.v1` JSON; explicit rejection of `stream: true`, `tools`, `logprobs: true`, and any sampling controls; `"per_question"` (default, grouping identical prepared prefixes) vs `"shared_prefix"` execution using `encode_prefix()` + `question_from_prefix(..., isolate=True)`; `"joint"` (default) vs `"equivariant"` scoring; `expected_score` requiring all-or-none finite `value`s; optional single-pass joint `embedding` output (requiring `shared_prefix` for multi-decision requests); exact `usage` accounting (`completion_tokens = 0`).
  - Calibration binding: apply `(T, b)` only when `apply_calibration=true`, `checkpoint_id == "bob-s1-0.8b"`, `scoring_mode == "joint"`, `prefix_mode == "per_question"`, `effective_max_seq_len == 4096`, and canonical `QUESTIONS` match; otherwise emit `"uncalibrated_softmax"` with explicit `calibration.mismatch_reason` (or `400` if `require_calibrated=true` / `calibration_conflict`).
- [x] Task 3.3: Implement FastAPI/Uvicorn HTTP server (`app.py`):
  - Endpoints: `GET /healthz`, `GET /v1/models`, `POST /v1/embeddings`, `POST /v1/chat/completions`, plus OpenAI error handler (`error.schema.json`).
  - Loopback-by-default bind (`127.0.0.1`), optional bearer auth (`BOB_S1_API_KEY`), request-size caps (`413` on body limit), and state/image/credential log redaction.

---

## Phase 4: Verification, Parity Fixtures & Documentation
- [x] Task 4.1: Fast CPU unit & API conformance test suite (`services/bob_s1/tests/` using `synthetic_model()`):
  - Validate all Draft 2020-12 contract schemas and execute every case in `conformance_fixtures.json`.
  - `float` vs little-endian `base64` parity and per-slice L2 normalization across `128`, `256`, `384`.
  - Image sensitivity & security rejection tests (SSRF URLs, local paths, `image/jpg` vs `png|jpeg|webp|gif`, oversized/corrupt/animated images, text-before-image rejection).
  - Elastic `max_seq_len` boundary tests (`error` vs `head_tail`, `max_seq_len > server_max` -> `400`, `text_budget < 1` -> `400`, never truncating suffix/options).
  - Calibration math (`p_1 = sigmoid((z_1 - z_0 + b) / T)`), profile mismatch disclosure (`effective_max_seq_len != 4096`, `shared_prefix`, `equivariant`, `apply_calibration=false` + `require_calibrated=true`), `raw_argmax` vs `calibrated_argmax` separation, and `expected_score` all-or-none rule.
  - DeltaNet prefix cache isolation (`isolate=True`, sibling order/content invariance, cross-request isolation) and queue overload/cancellation behavior.
- [x] Task 4.2: Real-checkpoint parity & smoke test (gated/executed against cached `bob-s1-0.8b` on ROCm/CPU):
  - Verify HTTP `/v1/embeddings` and `/v1/chat/completions` outputs match direct `DualHeadDecisionEmbedder` calls within declared numerical tolerance across text, multimodal, joint single-pass, and `compaction`/`outcome_prefix`/`loop_guard` fixtures.
  - Freeze numerical parity snapshot into `services/bob_s1/contracts/` for future `lemond`/`vllm`/`llama.cpp` server tests.
- [x] Task 4.3: Add `services/bob_s1/README.md` documenting startup flags, `uv` profile setup (CPU/CUDA/ROCm), elastic `max_seq_len` tradeoffs, wire examples (`curl` + `go-openai`), run `make check`, and update `conductor/tracks/bob-s1-inference-sidecar_20261118/metadata.json` on completion.
