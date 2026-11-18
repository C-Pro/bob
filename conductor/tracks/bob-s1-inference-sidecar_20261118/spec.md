# Specification: Temporary `bob-s1` Python Inference Sidecar (OpenAI-Compatible)

## 1. Objective & Scope

Provide a standalone, temporary Python inference sidecar under `services/bob_s1/` serving **`bob-s1-0.8b`** (`DualHeadDecisionEmbedder` in [`eval/jev/model.py`](../../../eval/jev/model.py)) over **OpenAI-compatible HTTP endpoints** for:
1. **Multimodal & Text Matryoshka Embeddings (`POST /v1/embeddings`)** (`W_emb`, independent L2 normalization at `128d`, `256d`, `384d`).
2. **System-1 Decision Model Inference (`POST /v1/chat/completions`)** (`PointerHead`, variable-cardinality option scoring with optional calibrated probabilities).
3. **Single-Pass Joint Output (Optional)** inside `POST /v1/chat/completions` when both decisions and a state embedding are requested from one shared prefix prefill.

### Long-Term Compatibility & Out-of-Scope Boundaries
- **Long-term goal (out of scope):** Patch upstream C++/Python inference servers (`lemond`, `vllm`, `llama.cpp`) to natively execute `bob-s1` (`W_emb`, `PointerHead`, multimodal `mmproj`, and Gated DeltaNet cache cloning).
- **This track's role:** Freeze an exact, unambiguous OpenAI-compatible wire contract plus portable conformance and numerical parity fixtures so future native servers can drop in behind the same client contract without breaking Bob.
- **Deferred to existing tracks:** Go configuration/client wiring (`internal/config`, `internal/llm`) remains in [`conductor/tracks/jev-multimodal-decision-embedder_20261114/plan.md`](../jev-multimodal-decision-embedder_20261114/plan.md) Step 3, and FSM/memory/held-out evaluation remain in Steps 4–6.

### Reconciliation with [`jev-multimodal-decision-embedder_20261114/spec.md`](../jev-multimodal-decision-embedder_20261114/spec.md)
The implemented release model ([`eval/jev/model.py`](../../../eval/jev/model.py), [`eval/BOB_S1.md`](../../../eval/BOB_S1.md)) differs from the earlier Phase 4 draft spec in three concrete ways that this sidecar follows strictly:
1. **`PointerHead` has no `LayerNorm`:** `PointerHead` applies `q = Linear(1024, 256, bias=True)` and `k = Linear(1024, 256, bias=True)` directly to backbone hidden states and scales by `1 / sqrt(256)`.
2. **Delimiter & Image Token Layout:** Prompts use Qwen native FIM/box delimiters (`<|fim_prefix|>`, `<|fim_middle|>`, `<|box_start|>`, `<|box_end|>`, `<|fim_suffix|>`) rather than `<|state_end|>`. `prepare_state()` places all `<|vision_start|><|image_pad|>×N<|vision_end|>` spans before the text tokens.
3. **All-in-One Multimodal Serving:** Stock `llama-server`, `lemond`, and `Ollama` do not currently support multimodal `/embeddings` or `--pooling none` `PointerHead` scoring (`eval/BOB_S1.md` §6.2); PyTorch `DualHeadDecisionEmbedder` is the verified reference runtime.

---

## 2. OpenAI-Compatible API Contract

### 2.1 Health, Discovery & Error Mapping
- `GET /healthz`: Returns `200 OK` only after model loading, checkpoint tensor verification, and startup warmup succeed (`503` `service_unavailable_error` / `model_not_ready` otherwise).
- `GET /v1/models`: Standard OpenAI model list advertising `bob-s1-0.8b` (with metadata indicating `max_position_embeddings = 262144`, configured `BOB_S1_MAX_SEQ_LEN`, and supported `embedding_dimensions = [128, 256, 384]`).
- **Standard OpenAI Error Envelope & Status Codes:** All errors return `{"error": {"message", "type", "param", "code"}}`:
  - `400` (`invalid_request_error`): schema/validation failure, unsupported params (`stream: true`, `tools`, `logprobs: true`, any explicit sampling parameter), invalid `dimensions`/`encoding_format`, bad image URI/bytes, `context_length_exceeded`, `calibration_unavailable`, `calibration_conflict`.
  - `401` (`authentication_error`): missing/invalid bearer token when `BOB_S1_API_KEY` is configured.
  - `413` (`invalid_request_error`, `code: "payload_too_large"`): HTTP body exceeds `BOB_S1_MAX_BODY_BYTES`.
  - `429` (`rate_limit_error`, `code: "queue_full"`): bounded worker queue is full.
  - `503` (`service_unavailable_error`, `code: "model_not_ready"`): model not yet ready/warmed up.

### 2.2 `POST /v1/embeddings`
Maintains strict compatibility with standard OpenAI SDKs (including `sashabaranov/go-openai`'s `EmbeddingRequest{Input: any, Dimensions, EncodingFormat, ExtraBody}`) while adding one unambiguous multimodal item shape:

- **Accepted `input` forms:**
  1. Single text string: `"input": "state text"`
  2. Batch of text strings: `"input": ["text 1", "text 2"]`
  3. Batch of multimodal objects:
     ```json
     {
       "model": "bob-s1-0.8b",
       "input": [
         {
           "content": [
             {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
             {"type": "text", "text": "State description"}
           ]
         }
       ],
       "dimensions": 384,
       "encoding_format": "float"
     }
     ```
- **Input ordering & rejection rules:**
  - Within a multimodal `content` array, accept zero or more `image_url` parts followed by at most one `text` part (supporting image-only, text-only, and images-before-text). Reject repeated `text` parts or text-before-image interleaving with `400` because `prepare_state()` places all images before text; never silently reorder scattered parts.
  - Accept only `data:image/(png|jpeg|webp|gif);base64,...` URIs.
  - Explicitly reject raw token arrays (`int[]` / `int[][]`) and empty batches with `400`.
- **Dimensions, encoding & response:**
  - `dimensions` defaults to `384` and accepts only `{128, 256, 384}`; reject other values with `400`.
  - Compute the requested slice via `DualHeadDecisionEmbedder.embeddings()` (independently L2-normalized `F.normalize(full[..., :dim], dim=-1)` per slice—never truncate an already normalized `384d` vector).
  - `encoding_format` accepts `"float"` (default) or `"base64"` (little-endian IEEE 754 `float32`, matching `go-openai`'s `Base64Embedding.Decode()`).
  - Return standard OpenAI `object: "list"`, ordered `data: [{object: "embedding", index, embedding}]`, `model`, and `usage` (`prompt_tokens = sum(len(prefix_ids_i))`, `total_tokens = prompt_tokens`).
- **Wire extension fields (top-level JSON; SDK `extra_body` flattens onto top-level JSON):**
  - `truncation`: `"error"` (default on `/v1/embeddings`) or `"head_tail"`.
  - `max_seq_len`: optional per-request integer cap; if omitted uses `BOB_S1_MAX_SEQ_LEN`; if provided outside `1..BOB_S1_MAX_SEQ_LEN` returns `400`.

### 2.3 `POST /v1/chat/completions` (`bob-s1.decision.v1`)
Uses standard Chat Completions as an SDK-compatible transport carrying a versioned JSON decision protocol (never running a conversational chat template or generative decoding):

- **Outer Chat Completions envelope rules (`chat.completions.request.schema.json`):**
  - `model`: `"bob-s1-0.8b"`.
  - `messages`: Must contain **exactly one** message with `role: "user"` (multi-turn or `system`/`assistant`/`tool` messages are rejected with `400`).
    - Message `content` may be a JSON string or an array of content parts: zero or more preceding `{"type": "image_url", "image_url": {"url": "data:image/(png|jpeg|webp|gif);base64,..."}}` parts belonging to the shared state, followed by **exactly one** `{"type": "text", "text": "<JSON>"}` part.
  - `response_format`: Optional; accepted forms are `{"type": "text"}`, `{"type": "json_object"}`, or `{"type": "json_schema", "json_schema": {"name": "bob_s1_decision_v1", "schema": ...}}` where `schema` matches `decision.response.schema.json` (arbitrary custom JSON schemas are rejected with `400`).
  - **Explicit rejections (`400`):** `stream: true`, `tools`, `tool_choice`, `logprobs: true`, or **any** explicit sampling/generation parameter (`temperature`, `top_p`, `n`, `stop`, `max_tokens`, `max_completion_tokens`, `presence_penalty`, `frequency_penalty`, `logit_bias`, `seed`). Chat `temperature` must never exist or alter `PointerHead` calibration.
- **Inner Decision Request JSON (`bob-s1.decision.v1`):**
  - Each decision item in `decisions[]` must have a unique non-empty `id` and either:
    1. a built-in `task` (`"compaction" | "outcome_prefix" | "loop_guard"`), resolved server-side from `QUESTIONS` (contradictory `question` or `options` overrides are rejected with `400`), OR
    2. both `question` and `options` (`2..255` items with unique non-empty `id`s and descriptions).
  - **Option `value` & `expected_score`:** Either **all** options in a decision carry finite numeric `value` or **none** carry `value` (partial values return `400`). When all options carry `value`, `expected_score = sum(p_i * v_i)` is computed over the returned `probability` entries; otherwise `expected_score = null`.
  - **Multi-decision joint embedding rule:** If `embedding` is present and `len(decisions) > 1`, `prefix_mode` MUST be `"shared_prefix"` (`400` otherwise).
- **Usage accounting on `/v1/chat/completions`:**
  - `usage.completion_tokens = 0` (no autoregressive token generation).
  - `usage.prompt_tokens =` total tokens actually processed across distinct prefix prefills and suffix/option branches (`usage.total_tokens = usage.prompt_tokens`).

#### Example A: Calibrated `per_question` Decision (`compaction`)
Request (`messages[0].content`):
```json
{
  "protocol": "bob-s1.decision.v1",
  "state": "Iter 1..5: read compaction.go, ran go test ./internal/fsm/... -> PASS",
  "decisions": [
    {
      "id": "d1",
      "task": "compaction",
      "scoring_mode": "joint",
      "apply_calibration": true
    }
  ],
  "prefix_mode": "per_question",
  "truncation": "head_tail"
}
```
Response (`choices[0].message.content`, with `BOB_S1_MAX_SEQ_LEN=4096`):
```json
{
  "protocol": "bob-s1.decision.v1",
  "decisions": [
    {
      "id": "d1",
      "task": "compaction",
      "scoring_mode": "joint",
      "choice": "compact",
      "choice_basis": "calibrated_argmax",
      "scores": [
        {"id": "skip", "logit": -1.2, "probability": 0.041553},
        {"id": "compact", "logit": 0.5, "probability": 0.958447}
      ],
      "probability_kind": "calibrated_affine",
      "expected_score": null,
      "calibration": {
        "profile": "bob-s1-0.8b/compaction/cal_v2_4k",
        "applied": true,
        "mismatch_reason": null,
        "temperature": 1.895898683669047,
        "bias": 4.25,
        "balanced_threshold": 0.8964872561924658,
        "risk_threshold_5pct": 0.8393841862136965
      },
      "prefix_tokens": 24,
      "suffix_tokens": 34,
      "truncated": false
    }
  ],
  "embedding": null,
  "preparation": {
    "prefix_mode": "per_question",
    "truncation_policy": "head_tail",
    "effective_max_seq_len": 4096,
    "suffix_reserve": 34,
    "prefix_tokens": 24,
    "truncated": false
  }
}
```
*(Check: `sigmoid((0.5 - (-1.2) + 4.25) / 1.895898683669047) = sigmoid(5.95 / 1.895898683669047) ≈ 0.958447`.)*

#### Example B: Uncalibrated Multi-Decision + Joint Embedding (`shared_prefix`)
Request (`messages[0].content` after a preceding `image_url` part):
```json
{
  "protocol": "bob-s1.decision.v1",
  "state": "Screenshot attached.",
  "decisions": [
    {
      "id": "c1",
      "question": "Is the panel red?",
      "options": [
        {"id": "no", "description": "Not red", "value": 0.0},
        {"id": "yes", "description": "Red", "value": 1.0}
      ]
    },
    {
      "id": "c2",
      "task": "loop_guard"
    }
  ],
  "embedding": {"dimensions": 128, "encoding_format": "float"},
  "prefix_mode": "shared_prefix",
  "truncation": "head_tail"
}
```
Response (`choices[0].message.content`):
```json
{
  "protocol": "bob-s1.decision.v1",
  "decisions": [
    {
      "id": "c1",
      "task": null,
      "scoring_mode": "joint",
      "choice": "yes",
      "choice_basis": "raw_argmax",
      "scores": [
        {"id": "no", "logit": -0.4, "probability": 0.231475},
        {"id": "yes", "logit": 0.8, "probability": 0.768525}
      ],
      "probability_kind": "uncalibrated_softmax",
      "expected_score": 0.768525,
      "calibration": {
        "profile": null,
        "applied": false,
        "mismatch_reason": "no_calibration_profile_for_custom_task",
        "temperature": null,
        "bias": null,
        "balanced_threshold": null,
        "risk_threshold_5pct": null
      },
      "prefix_tokens": 12,
      "suffix_tokens": 22,
      "truncated": false
    },
    {
      "id": "c2",
      "task": "loop_guard",
      "scoring_mode": "joint",
      "choice": "none",
      "choice_basis": "raw_argmax",
      "scores": [
        {"id": "none", "logit": 1.1, "probability": 0.6512},
        {"id": "intercept", "logit": 0.2, "probability": 0.2646},
        {"id": "force_synthesis", "logit": -0.945, "probability": 0.0842}
      ],
      "probability_kind": "uncalibrated_softmax",
      "expected_score": null,
      "calibration": {
        "profile": null,
        "applied": false,
        "mismatch_reason": "no_calibration_profile_for_task",
        "temperature": null,
        "bias": null,
        "balanced_threshold": null,
        "risk_threshold_5pct": null
      },
      "prefix_tokens": 12,
      "suffix_tokens": 48,
      "truncated": false
    }
  ],
  "embedding": {
    "dimensions": 128,
    "encoding_format": "float",
    "vector": [0.012, -0.045]
  },
  "preparation": {
    "prefix_mode": "shared_prefix",
    "truncation_policy": "head_tail",
    "effective_max_seq_len": 4096,
    "suffix_reserve": 48,
    "prefix_tokens": 12,
    "truncated": false
  }
}
```
*(Note: `embedding.vector` above is abbreviated for display; live responses return all 128 floats.)*

---

## 3. Elastic Context Budget, Prefix Preparation & Calibration Rules

### 3.1 Elastic `max_seq_len`
- **Model capability vs calibrated operating point:** `qwen3_5_text` supports `max_position_embeddings = 262,144` and `eval/jev/model.py` operates with **unpadded `batch=1`** forwards (short inputs only consume compute/VRAM proportional to actual token count).
- **Configuration & overflow handling:**
  - Server-side hard ceiling `BOB_S1_MAX_SEQ_LEN` (default `4096`, valid range `256..262144`).
  - If request `max_seq_len` is omitted, `effective_max_seq_len = BOB_S1_MAX_SEQ_LEN`.
  - If request `max_seq_len` is supplied, it must satisfy `1 <= max_seq_len <= BOB_S1_MAX_SEQ_LEN` (else `400` `invalid_request_error`), and the accepted cap is used unchanged as `effective_max_seq_len`.
  - Configurable `BOB_S1_ATTN_IMPLEMENTATION` (`"eager"` default for exact reference parity; `"sdpa"` optional for scaling `>4k` contexts on the 6 full-attention layers without `O(L^2)` eager attention score materialization).

### 3.2 Portable Truncation & Reserve Semantics (`prepare_state` parity)
1. **Reserve calculation:**
   - `POST /v1/embeddings`: `suffix_reserve = 0`.
   - `scoring_mode == "joint"`: `suffix_reserve` for decision `d` = `len(question_ids(d.question, d.options)[0])`.
   - `scoring_mode == "equivariant"`: `suffix_reserve` for decision `d` = `max(len(question_ids(d.question, [opt])[0]) for opt in d.options)` (matching `model.py:529`). Response `suffix_tokens` reports that max single-option suffix length.
   - Under `prefix_mode == "shared_prefix"` (including mixed `joint`/`equivariant` requests): shared `suffix_reserve = max(decision_reserve(d) for d in decisions)`.
2. **Prefix construction & overflow:**
   - All image spans (`<|vision_start|><|image_pad|>×N<|vision_end|>`) are preserved intact; never silently truncate questions/options or alter image resize bounds (`min_pixels=16384`, `max_pixels=98304`).
   - `budget = effective_max_seq_len - suffix_reserve`; `text_budget = budget - len(image_tokens)`.
   - If `text_budget < 1` (not even room for the `<|fim_prefix|>` token), **always** return `400` (`context_length_exceeded`), even when `truncation == "head_tail"`.
   - `keep = text_budget - 1`. If `len(state_text_ids) > keep`:
     - Under `truncation == "error"`: return `400` (`context_length_exceeded`).
     - Under `truncation == "head_tail"`: apply the exact 4-iteration omission marker loop from `eval/jev/model.py:370-390` (`\n[... {omitted} tokens omitted ...]\n`, note omitted count is approximate).
3. **Top-level `preparation` reporting:**
   - In `shared_prefix` mode (or single-decision `per_question`), `preparation.prefix_tokens` and `preparation.suffix_reserve` hold the exact integers.
   - In multi-decision `per_question` mode, if all decisions share identical `prefix_tokens`/`suffix_tokens`, top-level `preparation` reports them; otherwise `preparation.prefix_tokens` and `preparation.suffix_reserve` are `null`, `preparation.truncated = any(d.truncated)`, and every decision item carries its own exact `prefix_tokens`, `suffix_tokens`, and `truncated`.
   - Standalone `/v1/embeddings` (`suffix_reserve = 0`) and joint `/v1/chat/completions` embeddings are numerically identical only when their prepared prefix token sequences match (i.e. neither truncated under different reserves).

### 3.3 Calibration Provenance & Strict Binding
- **Provenance (`calibration_profiles.json`):** Exact full-precision constants from [`eval/results/jev_dataset/dev_final_v2_4k/experiment_report.json`](../../../eval/results/jev_dataset/dev_final_v2_4k/experiment_report.json) entry `models.exp_b_v2_4k_emb_onlymm.decisions`, bound to `model.pt` SHA-256 (`a0cfc5bb5fd214f76acdb2774cc99e2be35f4432c5cc85547ae9b410b731397c`), `BASE_MODEL_REVISION` (`dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`), and `KEV_ADAPTER_REVISION` (`bf75a6a8848ea6960ff2ed108d9ed44c2941174f`).
- **Formula & thresholds:** `p_1 = sigmoid((z_1 - z_0 + b) / T)`, `p_0 = 1 - p_1`. `balanced_threshold` applies to `p_1` (positive-class probability); `risk_policy.threshold` applies to `max(p_0, p_1)` (argmax confidence) as advisory metadata. `choice` selects the earliest index attaining maximum probability across the returned `scores[].probability` values.
- **Unconditional eligibility rule:** A calibration profile applies **only** when ALL hold:
  1. `apply_calibration == true`
  2. `task` has a registered profile (`compaction` or `outcome_prefix`)
  3. `scoring_mode == "joint"`
  4. `prefix_mode == "per_question"`
  5. `effective_max_seq_len == 4096`
  6. Loaded checkpoint matches the profile's bound digest/identity.
- **Mismatch handling:**
  - `apply_calibration == false` and `require_calibrated == true` -> `400` (`calibration_conflict`).
  - Any other failure of conditions (2)–(6) when `require_calibrated == true` -> `400` (`calibration_unavailable`).
  - When `require_calibrated == false`, return `probability_kind: "uncalibrated_softmax"`, `choice_basis: "raw_argmax"`, and `calibration.applied: false` with explicit `mismatch_reason`. Never reuse `model.temperature` (`~2.35`, retained donor value).

---

## 4. Runtime Concurrency, Packaging & Security

- **Execution model:** Single Uvicorn process with one dedicated model-owning worker thread and bounded FIFO queue (`BOB_S1_MAX_QUEUE_DEPTH`, returning `429` `queue_full` on saturation). Runs under `torch.inference_mode()` and `model.eval()`. Disconnected HTTP requests cancel cleanly at queue wait boundaries without releasing device ownership mid-forward.
- **DeltaNet isolation:** `encode_prefix()` cache is treated as immutable; every branch executes `question_from_prefix(..., isolate=True)` (`isolate=False` is never exposed). No cross-request prefix cache sharing in v1.
- **Startup & loader:** Wrap `DualHeadDecisionEmbedder` behind a small `ModelLoader` interface (`checkpoint_overlay` implemented first, `merged_hf` extensible later). Validate every expected LoRA/`PointerHead`/`W_emb` tensor, special token ID, finiteness, and vision preprocessing config at startup; fail hard rather than falling back to donor/random weights.
- **Packaging (`services/bob_s1/`):** Dedicated `pyproject.toml` + `uv.lock` pinning `transformers`, `peft`, `fastapi`, `uvicorn`, `pydantic`, `pillow`, `numpy`, with explicit `cpu`, `cuda`, and `rocm` installation profiles. Never import `eval/jev/train_experiment.py` or `eval/jev/eval_experiment.py`.
- **Security boundaries:**
  - Bind `127.0.0.1` by default; support optional `BOB_S1_API_KEY` bearer authentication (`401` on mismatch).
  - Accept only bounded `data:image/(png|jpeg|webp|gif);base64,...` URIs; reject `http://`, `https://`, `file://`, and raw filesystem paths.
  - Decode and validate image format, byte size, pixel count (`Image.MAX_IMAGE_PIXELS` + explicit cap), and single-frame check before passing `PIL.Image` instances to the model.
  - Enforce request limits (`BOB_S1_MAX_BATCH_SIZE`, `BOB_S1_MAX_DECISIONS`, `BOB_S1_MAX_OPTIONS`, `BOB_S1_MAX_IMAGES`, `BOB_S1_MAX_IMAGE_BYTES`, `BOB_S1_MAX_BODY_BYTES`).
  - Redact state text, base64 images, and credentials from logs.
