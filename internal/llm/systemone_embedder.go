package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strings"

	"bob/internal/config"

	"github.com/liliang-cn/cortexdb/v2/pkg/cortexdb"
	openai "github.com/sashabaranov/go-openai"
)

var _ cortexdb.Embedder = (*SystemOneEmbedder)(nil)

type embeddingImagesContextKey struct{}

// WithEmbeddingImages attaches inline image data URIs to ctx for single-RPC multimodal indexing
// inside CortexDB SaveMemory calls.
func WithEmbeddingImages(ctx context.Context, dataURIs []string) context.Context {
	if ctx == nil {
		ctx = context.Background()
	}
	if len(dataURIs) == 0 {
		return ctx
	}
	copied := append([]string(nil), dataURIs...)
	return context.WithValue(ctx, embeddingImagesContextKey{}, copied)
}

func embeddingImagesFromContext(ctx context.Context) []string {
	if ctx == nil {
		return nil
	}
	raw, ok := ctx.Value(embeddingImagesContextKey{}).([]string)
	if !ok || len(raw) == 0 {
		return nil
	}
	return raw
}

// SystemOneEmbedder implements cortexdb.Embedder backed by bob-s1 /v1/embeddings.
type SystemOneEmbedder struct {
	client          *SystemOneClient
	model           string
	dims            int
	maxSeqLen       int
	fallbackOnError bool
}

// NewSystemOneEmbedder constructs a SystemOneEmbedder using client and Matryoshka dims (128, 256, or 384).
// fallbackOnError defaults to true if omitted.
func NewSystemOneEmbedder(client *SystemOneClient, dims int, maxSeqLen int, fallbackOnError ...bool) (*SystemOneEmbedder, error) {
	if client == nil {
		return nil, errors.New("system1 client is required")
	}
	if dims == 0 {
		dims = client.EmbeddingDims()
	}
	if dims != 128 && dims != 256 && dims != 384 {
		return nil, fmt.Errorf("system1 embedding dims must be 128, 256, or 384, got %d", dims)
	}
	if maxSeqLen < 0 || (maxSeqLen > 0 && (maxSeqLen < config.MinSystemOneMaxSeqLen || maxSeqLen > config.MaxSystemOneMaxSeqLen)) {
		return nil, fmt.Errorf("system1 max_seq_len must be 0 or between %d and %d, got %d",
			config.MinSystemOneMaxSeqLen, config.MaxSystemOneMaxSeqLen, maxSeqLen)
	}

	fallback := true
	if len(fallbackOnError) > 0 {
		fallback = fallbackOnError[0]
	}

	return &SystemOneEmbedder{
		client:          client,
		model:           client.Model(),
		dims:            dims,
		maxSeqLen:       maxSeqLen,
		fallbackOnError: fallback,
	}, nil
}

// Dim returns the configured Matryoshka dimension immediately without making an RPC.
func (e *SystemOneEmbedder) Dim() int {
	return e.dims
}

// Model returns the configured System-1 model identifier.
func (e *SystemOneEmbedder) Model() string {
	return e.model
}

// Embed computes the embedding for text, honoring any multimodal images attached via WithEmbeddingImages.
func (e *SystemOneEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	if strings.TrimSpace(text) == "" {
		return nil, errors.New("empty text provided")
	}
	if imgs := embeddingImagesFromContext(ctx); len(imgs) > 0 {
		return e.EmbedMultimodal(ctx, text, imgs)
	}
	vecs, err := e.EmbedBatch(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	if len(vecs) != 1 {
		return nil, fmt.Errorf("expected 1 embedding vector, got %d", len(vecs))
	}
	return vecs[0], nil
}

// EmbedBatch computes embeddings for texts in bounded sub-batches of at most 16 items.
func (e *SystemOneEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	for i, text := range texts {
		if strings.TrimSpace(text) == "" {
			return nil, fmt.Errorf("empty text provided at index %d", i)
		}
	}

	out := make([][]float32, 0, len(texts))
	for start := 0; start < len(texts); start += SystemOneMaxBatchSize {
		end := min(start+SystemOneMaxBatchSize, len(texts))
		vecs, err := e.embedTextChunk(ctx, texts[start:end])
		if err != nil {
			return nil, err
		}
		out = append(out, vecs...)
	}
	return out, nil
}

// EmbedMultimodal computes a joint image+text embedding with cap enforcement and optional text-only fallback.
func (e *SystemOneEmbedder) EmbedMultimodal(ctx context.Context, text string, imageDataURIs []string) ([]float32, error) {
	if strings.TrimSpace(text) == "" {
		return nil, errors.New("empty text provided")
	}

	validImgs, invalidCount, truncatedCount := sanitizeSystemOneImages(imageDataURIs, SystemOneMaxImagesPerItem)
	if invalidCount > 0 || truncatedCount > 0 {
		slog.Warn("filtered multimodal embedding images before bob-s1 request",
			"kept", len(validImgs),
			"invalidDropped", invalidCount,
			"capDropped", truncatedCount,
		)
	}
	if len(validImgs) == 0 {
		if !e.fallbackOnError && invalidCount > 0 {
			return nil, fmt.Errorf("all %d multimodal image URI(s) were invalid in strict mode", invalidCount)
		}
		return e.embedSingleText(ctx, text)
	}

	parts := make([]map[string]any, 0, len(validImgs)+1)
	for _, uri := range validImgs {
		parts = append(parts, map[string]any{
			"type": "image_url",
			"image_url": map[string]any{
				"url": uri,
			},
		})
	}
	parts = append(parts, map[string]any{
		"type": "text",
		"text": text,
	})

	input := []map[string]any{
		{"content": parts},
	}
	vecs, err := e.sendEmbeddingsRequest(ctx, input, 1)
	if err != nil {
		if e.fallbackOnError && shouldFallbackMultimodalError(err) {
			slog.Warn("multimodal bob-s1 embedding failed, falling back to text-only embedding",
				"imageCount", len(validImgs),
				"error", err,
			)
			return e.embedSingleText(ctx, text)
		}
		return nil, err
	}
	return vecs[0], nil
}

func (e *SystemOneEmbedder) embedSingleText(ctx context.Context, text string) ([]float32, error) {
	vecs, err := e.embedTextChunk(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	return vecs[0], nil
}

func (e *SystemOneEmbedder) embedTextChunk(ctx context.Context, texts []string) ([][]float32, error) {
	return e.sendEmbeddingsRequest(ctx, texts, len(texts))
}

func (e *SystemOneEmbedder) sendEmbeddingsRequest(ctx context.Context, input any, expectedCount int) ([][]float32, error) {
	extraBody := map[string]any{
		"truncation": SystemOneTruncationHeadTail,
	}
	if e.maxSeqLen > 0 {
		extraBody["max_seq_len"] = e.maxSeqLen
	}

	req := openai.EmbeddingRequest{
		Input:          input,
		Model:          openai.EmbeddingModel(e.model),
		EncodingFormat: openai.EmbeddingEncodingFormatBase64,
		Dimensions:     e.dims,
		ExtraBody:      extraBody,
	}

	reqBytes, err := marshalEmbeddingRequestWithExtraBody(req)
	if err != nil {
		return nil, fmt.Errorf("marshal bob-s1 embeddings request: %w", err)
	}

	respBytes, err := e.client.doPostJSONWithRetry(ctx, e.client.openAIBaseURL+"/embeddings", reqBytes)
	if err != nil {
		return nil, err
	}

	return e.decodeAndValidateBase64Response(respBytes, expectedCount)
}

func marshalEmbeddingRequestWithExtraBody(req openai.EmbeddingRequest) ([]byte, error) {
	extra := req.ExtraBody
	req.ExtraBody = nil
	baseBytes, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	if len(extra) == 0 {
		return baseBytes, nil
	}
	var body map[string]any
	if err := json.Unmarshal(baseBytes, &body); err != nil {
		return nil, err
	}
	for k, v := range extra {
		body[k] = v
	}
	return json.Marshal(body)
}

type systemOneEmbeddingsResponseRaw struct {
	Object string `json:"object"`
	Model  string `json:"model"`
	Data   []struct {
		Object    string `json:"object"`
		Index     int    `json:"index"`
		Embedding string `json:"embedding"`
	} `json:"data"`
}

func (e *SystemOneEmbedder) decodeAndValidateBase64Response(respBytes []byte, expectedCount int) ([][]float32, error) {
	var parsed systemOneEmbeddingsResponseRaw
	if err := json.Unmarshal(respBytes, &parsed); err != nil {
		return nil, fmt.Errorf("decode bob-s1 embeddings response: %w", err)
	}
	if len(parsed.Data) != expectedCount {
		return nil, fmt.Errorf("expected %d embedding items, got %d", expectedCount, len(parsed.Data))
	}

	seen := make([]bool, expectedCount)
	out := make([][]float32, expectedCount)
	for i, item := range parsed.Data {
		if item.Index < 0 || item.Index >= expectedCount {
			return nil, fmt.Errorf("embedding item %d has out-of-range index %d (expected 0..%d)", i, item.Index, expectedCount-1)
		}
		if seen[item.Index] {
			return nil, fmt.Errorf("duplicate embedding index %d", item.Index)
		}
		seen[item.Index] = true

		vec, err := decodeStrictBase64Float32(item.Embedding, e.dims)
		if err != nil {
			return nil, fmt.Errorf("embedding item index %d: %w", item.Index, err)
		}
		out[item.Index] = vec
	}
	return out, nil
}

func sanitizeSystemOneImages(rawURIs []string, capPerItem int) (valid []string, invalidCount int, capDropped int) {
	if len(rawURIs) == 0 {
		return nil, 0, 0
	}
	for _, raw := range rawURIs {
		trimmed := strings.TrimSpace(raw)
		if !IsValidSystemOneImageDataURI(trimmed) {
			invalidCount++
			continue
		}
		if capPerItem > 0 && len(valid) >= capPerItem {
			capDropped++
			continue
		}
		valid = append(valid, trimmed)
	}
	return valid, invalidCount, capDropped
}

func shouldFallbackMultimodalError(err error) bool {
	var apiErr *SystemOneAPIError
	if !errors.As(err, &apiErr) {
		return false
	}
	switch apiErr.StatusCode {
	case http.StatusBadRequest, http.StatusRequestEntityTooLarge:
		return true
	default:
		return false
	}
}
