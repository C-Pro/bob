# `bob-s1` Python Inference Sidecar (`services/bob_s1`)

Temporary OpenAI-compatible Python inference sidecar for [`bob-s1-0.8b`](../../eval/BOB_S1.md) (`DualHeadDecisionEmbedder` in [`eval/jev/model.py`](../../eval/jev/model.py)), providing:
- **`POST /v1/embeddings`**: Text and multimodal (`image_url` + `text`) Matryoshka state embeddings (`128d`, `256d`, `384d` independently L2-normalized) in `float` or little-endian `base64` formats.
- **`POST /v1/chat/completions`**: System-1 `PointerHead` decision inference (`bob-s1.decision.v1`) over variable-cardinality options (`compaction`, `outcome_prefix`, `loop_guard`, or custom questions) with optional single-pass joint `W_emb` embedding output.
- **`GET /healthz` & `GET /v1/models`**: Readiness gating (post checkpoint validation and warmup) and OpenAI model discovery.

> **Long-Term Note:** This sidecar freezes the wire contract and portable conformance fixtures under [`contracts/`](contracts/README.md) so future native server patches (`lemond`, `vllm`, `llama.cpp`) can implement the exact same OpenAI-compatible protocol.

---

## Installation (`uv`)

`pyproject.toml` defines mutually exclusive `cpu`, `cuda`, and `rocm` PyTorch installation profiles:

```bash
cd services/bob_s1

# AMD ROCm 6.4 (matches eval/BOB_S1.md Radeon 780M environment)
uv sync --extra rocm

# NVIDIA CUDA 12.8
uv sync --extra cuda

# CPU-only
uv sync --extra cpu
```

---

## Running the Sidecar

Ensure the pinned HF snapshots (`Qwen/Qwen3.5-0.8B-Base`, `jaredpalmer/kev-0.8b`) and `eval/jev/checkpoints/bob-s1-0.8b/model.pt` are present locally:

```bash
cd services/bob_s1
.venv/bin/bob-s1-sidecar --host 127.0.0.1 --port 8090
```

### Environment Variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `BOB_S1_HOST` | `127.0.0.1` | Bind address (loopback by default). |
| `BOB_S1_PORT` | `8090` | Listen port. |
| `BOB_S1_API_KEY` | *(unset)* | Optional Bearer auth token for `/v1/*` endpoints. |
| `BOB_S1_CHECKPOINT` | `eval/jev/checkpoints/bob-s1-0.8b/model.pt` | Path to `bob-s1` checkpoint `.pt`. |
| `BOB_S1_EXPECTED_SHA256` | `a0cfc5bb5fd214f76...` | Expected SHA-256 digest (`""` disables pinning). |
| `BOB_S1_HF_CACHE_DIR` | *(default HF cache)* | Optional local Hugging Face cache directory. |
| `BOB_S1_DEVICE` | `auto` | `auto`, `cpu`, `cuda`, or `cuda:N`. |
| `BOB_S1_DTYPE` | `auto` | `auto` (`bf16` on GPU, `fp32` on CPU), `bf16`, `fp32`, `fp16`. Heads stay `fp32`. |
| `BOB_S1_ATTN_IMPLEMENTATION` | `eager` | `eager` (exact parity) or `sdpa` (lower memory at long contexts). |
| `BOB_S1_MAX_SEQ_LEN` | `4096` | Elastic server ceiling (`256..262144`). |
| `BOB_S1_MAX_QUEUE_DEPTH` | `16` | Bounded queue depth before returning `429 queue_full`. |
| `BOB_S1_MAX_BATCH_SIZE` | `16` | Maximum items per `/v1/embeddings` batch. |
| `BOB_S1_MAX_DECISIONS` | `16` | Maximum decisions per `/v1/chat/completions` call. |
| `BOB_S1_MAX_OPTIONS` | `255` | Maximum options per decision (`2..255`). |
| `BOB_S1_MAX_IMAGES` | `8` | Maximum `image_url` parts per item/message. |
| `BOB_S1_MAX_REQUEST_IMAGES` | `16` | Aggregate `image_url` limit across an entire request. |
| `BOB_S1_MAX_IMAGE_BYTES` | `10485760` | Maximum decoded bytes per image (`10 MiB`). |
| `BOB_S1_MAX_IMAGE_PIXELS` | `16777216` | Maximum pixels per image (`4096x4096`). |
| `BOB_S1_MAX_REQUEST_IMAGE_PIXELS` | `16777216` | Aggregate decoded pixel budget per request. |
| `BOB_S1_MAX_BODY_BYTES` | `16777216` | Maximum HTTP request body size (`16 MiB`). |

---

## Elastic `max_seq_len` & Calibration Tradeoffs

- `DualHeadDecisionEmbedder` runs **unpadded `batch=1`** forwards, so short inputs pay compute/VRAM only for their actual token length regardless of `BOB_S1_MAX_SEQ_LEN`.
- Callers can set `BOB_S1_MAX_SEQ_LEN=8192` (or higher, up to `262144`) and optionally pass a per-request `max_seq_len <= BOB_S1_MAX_SEQ_LEN`.
- Because `bob-s1-0.8b` calibration profiles (`compaction`, `outcome_prefix`) were fitted at `effective_max_seq_len=4096` under `joint` scoring and `per_question` preparation, requests with `effective_max_seq_len != 4096`, `shared_prefix`, or `equivariant` return `uncalibrated_softmax` with an explicit `calibration.mismatch_reason` (or `400 calibration_unavailable` when `require_calibrated=true`).

---

## Wire Examples

### 1. Multimodal Embedding (`curl`)

```bash
curl -s http://127.0.0.1:8090/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bob-s1-0.8b",
    "input": [{
      "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAIAAAAmkwkpAAAAEElEQVR4nGP4z8AARwzEcQCukw/x0F8jngAAAABJRU5ErkJggg=="}},
        {"type": "text", "text": "Terminal build output screenshot"}
      ]
    }],
    "dimensions": 256,
    "encoding_format": "float"
  }'
```

### 2. Decision + Optional Joint Embedding (`curl`)

```bash
curl -s http://127.0.0.1:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bob-s1-0.8b",
    "messages": [{
      "role": "user",
      "content": "{\"protocol\":\"bob-s1.decision.v1\",\"state\":\"Iter 1..4: tests pass\",\"decisions\":[{\"id\":\"d1\",\"task\":\"compaction\"}]}"
    }],
    "response_format": {"type": "json_object"},
    "stream": false
  }'
```

### 3. `go-openai` Client Snippet (Multimodal `/v1/embeddings`)

```go
type mmPart struct {
	Type     string               `json:"type"`
	Text     string               `json:"text,omitempty"`
	ImageURL *openai.ChatMessageImageURL `json:"image_url,omitempty"`
}
type mmItem struct {
	Content []mmPart `json:"content"`
}

resp, err := client.CreateEmbeddings(ctx, openai.EmbeddingRequest{
	Model:          "bob-s1-0.8b",
	Dimensions:     256,
	EncodingFormat: openai.EmbeddingEncodingFormatFloat,
	Input: []mmItem{{
		Content: []mmPart{
			{Type: "image_url", ImageURL: &openai.ChatMessageImageURL{URL: dataURI}},
			{Type: "text", Text: "State description"},
		},
	}},
})
```

---

## Running Tests

```bash
cd services/bob_s1
# Fast unit, security, concurrency, and schema conformance suite
.venv/bin/python -m pytest -v tests/test_phase2_loader_and_images.py tests/test_phase3_runtime.py tests/test_phase4_conformance_and_invariants.py

# Real-checkpoint parity test (requires cached HF snapshots + bob-s1-0.8b/model.pt)
.venv/bin/python -m pytest -v tests/test_real_checkpoint_parity.py
```
