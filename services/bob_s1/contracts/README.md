# `bob-s1` OpenAI-Compatible Inference Wire Contract (`v1`)

This directory defines the frozen HTTP/JSON contract for `bob-s1-0.8b` serving. The temporary Python sidecar (`services/bob_s1/`) is the reference implementation; future native server patches (`lemond`, `vllm`, `llama.cpp`) should satisfy these exact schemas, semantics, and conformance fixtures.

## Endpoints & Schemas

| Endpoint | Purpose | Schema Files |
| :--- | :--- | :--- |
| `GET /healthz` | Readiness & health probe (`200` only after load + tensor verification + warmup; `503` otherwise) | — |
| `GET /v1/models` | OpenAI model discovery (`bob-s1-0.8b`) | — |
| `POST /v1/embeddings` | Text & multimodal state embeddings (`W_emb`) | `embeddings.request.schema.json`, `embeddings.response.schema.json` |
| `POST /v1/chat/completions` | System-1 `PointerHead` decisions (`bob-s1.decision.v1`) + optional joint `W_emb` | `chat.completions.{request,response}.schema.json`, `decision.{request,response}.schema.json` |
| *(all errors)* | Standard OpenAI error envelope | `error.schema.json` |

## Wire & Semantic Invariants

1. **Per-Slice Matryoshka L2 Normalization:** `dimensions` accepts `128`, `256`, or `384` (default `384`). Vectors are normalized independently via `F.normalize(W_emb(h)[..., :dim], dim=-1)`—never by slicing a pre-normalized `384d` vector.
2. **Base64 Vector Encoding:** `encoding_format: "base64"` encodes IEEE 754 little-endian `float32` bytes (`<f4`), matching `go-openai`'s `Base64Embedding.Decode()`.
3. **Multimodal Ordering & MIME Types:** `prepare_state()` places all image spans before state text (`<|fim_prefix|> [<|vision_start|><|image_pad|>×N<|vision_end|>]* STATE`). Requests must place `image_url` parts before at most one `text` part; repeated text parts or text-before-image interleaving are rejected with `400`. Only `data:image/(png|jpeg|webp|gif);base64,...` URIs are accepted.
4. **Extension Fields & `extra_body`:** OpenAI SDK `extra_body` maps are merged onto the top-level JSON request body on the wire; the server accepts `truncation` and `max_seq_len` at the top level of `/v1/embeddings` and rejects unknown top-level parameters.
5. **Elastic `max_seq_len`:** Server configures hard ceiling `BOB_S1_MAX_SEQ_LEN` (default `4096`, valid `256..262144`). Omitted `max_seq_len` uses `BOB_S1_MAX_SEQ_LEN`; explicit `max_seq_len > BOB_S1_MAX_SEQ_LEN` or `< 1` is rejected with `400`; accepted values are used unchanged as `effective_max_seq_len`. Short inputs execute unpadded (`batch=1`).
6. **Portable Truncation Algorithm (`prepare_state` parity):**
   - `suffix_reserve`: `0` on `/v1/embeddings`; `len(question_ids(q, opts)[0])` for `joint` decisions; `max(len(question_ids(q, [opt])[0]) for opt in opts)` for `equivariant` decisions; `max(decision_reserve(d))` across decisions under `shared_prefix`.
   - `budget = effective_max_seq_len - suffix_reserve`; `text_budget = budget - len(image_tokens)`. If `text_budget < 1` (no room for `<|fim_prefix|>`), always return `400` (`context_length_exceeded`), even under `truncation: "head_tail"`.
   - `keep = text_budget - 1`. When `len(state_text_ids) > keep`, `truncation: "error"` returns `400`, while `truncation: "head_tail"` executes the exact 4-iteration `\n[... {omitted} tokens omitted ...]\n` head/tail split from `eval/jev/model.py:370-390` (omitted count is approximate).
7. **DeltaNet Cache Isolation:** `encode_prefix()` caches are immutable and always cloned per branch (`question_from_prefix(..., isolate=True)`). Cross-request prefix caching is disabled in v1.
8. **Calibration Binding (`calibration_profiles.json`):** Binary affine calibration (`p_1 = sigmoid((z_1 - z_0 + b) / T)`) applies only when `apply_calibration == true`, `checkpoint_id` & `model.pt` digest match `bob-s1-0.8b`, `scoring_mode == "joint"`, `prefix_mode == "per_question"`, `effective_max_seq_len == 4096`, and canonical `QUESTIONS` match. Otherwise responses return `"uncalibrated_softmax"` with explicit `calibration.mismatch_reason` (or `400` if `require_calibrated == true`).
9. **Expected Score:** Either all options in a decision provide finite `value` or none do (partial values return `400`). When provided, `expected_score = sum(p_i * v_i)` uses the returned `probability` values.
10. **Usage Accounting:**
    - `/v1/embeddings`: `prompt_tokens = sum(len(prefix_ids_i))`, `total_tokens = prompt_tokens`.
    - `/v1/chat/completions`: `completion_tokens = 0`; `prompt_tokens =` sum of actual tokens processed across unique prefix prefills and suffix/option branches; `total_tokens = prompt_tokens`.
11. **HTTP Status & Error Code Mapping:**
    - `400` (`invalid_request_error`): schema/validation failure, unsupported params (`stream`, `tools`, `logprobs`, sampling params), `context_length_exceeded`, `calibration_unavailable`, `calibration_conflict`.
    - `401` (`authentication_error`): invalid/missing bearer token when `BOB_S1_API_KEY` is set.
    - `413` (`invalid_request_error`, `code: "payload_too_large"`): HTTP body exceeds `BOB_S1_MAX_BODY_BYTES`.
    - `429` (`rate_limit_error`, `code: "queue_full"`): bounded worker queue full.
    - `503` (`service_unavailable_error`, `code: "model_not_ready"`): model still loading or unhealthy.
12. **Numerical Parity Tolerances (for future native server ports):**
    - `W_emb` embedding cosine similarity against reference PyTorch `bf16` >= `0.999` (text) / `0.995` (multimodal).
    - `PointerHead` raw logit max absolute difference against reference PyTorch `bf16` <= `0.05`.
