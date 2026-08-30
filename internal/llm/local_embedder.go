package llm

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"bob/internal/config"

	"go-embed/pkg/engine"
)

// DefaultLocalModel is the default pure Go CPU embedding model.
const DefaultLocalModel = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

// LocalEmbedder wraps the go-embed inference engine to provide pure-Go embeddings
// conforming to cortexdb's Embedder interface.
type LocalEmbedder struct {
	engine    *engine.Engine
	modelName string
	dim       int
	mu        sync.RWMutex
	closed    bool
}

// NewLocalEmbedder initializes the go-embed inference engine using the provided configuration.
func NewLocalEmbedder(cfg *config.Config, extraOpts ...engine.Option) (*LocalEmbedder, error) {
	if cfg == nil {
		return nil, errors.New("config is nil")
	}

	modelName := DefaultLocalModel
	modelsDir := filepath.Join(cfg.DataDir, "models", modelName)
	if err := os.MkdirAll(modelsDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create models directory %s: %w", modelsDir, err)
	}

	prec := engine.PrecisionBF16
	switch strings.ToLower(strings.TrimSpace(cfg.EmbeddingPrecision)) {
	case "int8":
		prec = engine.PrecisionINT8
	case "fp32":
		prec = engine.PrecisionFP32
	case "bf16", "":
		prec = engine.PrecisionBF16
	}

	opts := []engine.Option{
		engine.WithDataDir(modelsDir),
		engine.WithModelName(modelName),
		engine.WithPrecision(prec),
	}
	opts = append(opts, extraOpts...)

	eng, err := engine.NewEngine(opts...)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize go-embed engine: %w", err)
	}

	return &LocalEmbedder{
		engine:    eng,
		modelName: modelName,
		dim:       384, // Standard hidden dimension for MiniLM-L12-v2 and E5-small
	}, nil
}

// NewLocalEmbedderWithEngine constructs a LocalEmbedder wrapping an existing engine instance.
func NewLocalEmbedderWithEngine(eng *engine.Engine, modelName string) *LocalEmbedder {
	if modelName == "" {
		modelName = DefaultLocalModel
	}
	return &LocalEmbedder{
		engine:    eng,
		modelName: modelName,
		dim:       384,
	}
}

// Embed converts a single text into a normalized float32 vector using sliding window aggregation.
func (l *LocalEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	if ctx != nil && ctx.Err() != nil {
		return nil, ctx.Err()
	}
	if strings.TrimSpace(text) == "" {
		return nil, errors.New("empty text provided")
	}

	l.mu.RLock()
	if l.closed {
		l.mu.RUnlock()
		return nil, errors.New("local embedder is closed")
	}
	if l.engine == nil {
		l.mu.RUnlock()
		return nil, errors.New("local embedding engine not initialized")
	}
	rawSlices, err := l.engine.Embed(text)
	l.mu.RUnlock()

	if err != nil {
		return nil, fmt.Errorf("local embedding failed: %w", err)
	}

	return AggregateMultiWindowVectors(rawSlices)
}

// EmbedBatch converts multiple texts into normalized vectors concurrently.
func (l *LocalEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if ctx != nil && ctx.Err() != nil {
		return nil, ctx.Err()
	}
	if len(texts) == 0 {
		return nil, nil
	}

	l.mu.RLock()
	if l.closed {
		l.mu.RUnlock()
		return nil, errors.New("local embedder is closed")
	}
	if l.engine == nil {
		l.mu.RUnlock()
		return nil, errors.New("local embedding engine not initialized")
	}
	rawBatch, err := l.engine.EmbedBatch(texts)
	l.mu.RUnlock()

	if err != nil {
		return nil, fmt.Errorf("local batch embedding failed: %w", err)
	}
	if len(rawBatch) != len(texts) {
		return nil, fmt.Errorf("expected %d batch embeddings, got %d", len(texts), len(rawBatch))
	}

	results := make([][]float32, len(texts))
	for i, rawSlices := range rawBatch {
		vec, err := AggregateMultiWindowVectors(rawSlices)
		if err != nil {
			return nil, fmt.Errorf("failed to aggregate vectors for text %d: %w", i, err)
		}
		results[i] = vec
	}

	if len(results) > 0 && len(results[0]) > 0 {
		l.mu.Lock()
		l.dim = len(results[0])
		l.mu.Unlock()
	}

	return results, nil
}

// Dim returns the vector dimensionality produced by the local model.
func (l *LocalEmbedder) Dim() int {
	l.mu.RLock()
	defer l.mu.RUnlock()
	if l.dim > 0 {
		return l.dim
	}
	return 384
}

// Model returns the local embedding model name.
func (l *LocalEmbedder) Model() string {
	return l.modelName
}

// Close releases any memory mapped resources held by the underlying engine safely.
func (l *LocalEmbedder) Close() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.closed {
		return nil
	}
	l.closed = true
	if l.engine != nil {
		return l.engine.Close()
	}
	return nil
}

// AggregateMultiWindowVectors aggregates multiple sliding window vectors for long text
// into a single representative L2-normalized vector using mean pooling.
func AggregateMultiWindowVectors(rawSlices [][]float32) ([]float32, error) {
	if len(rawSlices) == 0 {
		return nil, errors.New("no embedding slices returned")
	}
	if len(rawSlices) == 1 {
		if len(rawSlices[0]) == 0 {
			return nil, errors.New("empty embedding vector")
		}
		return rawSlices[0], nil
	}

	dim := len(rawSlices[0])
	if dim == 0 {
		return nil, errors.New("empty embedding vector in multi-slice output")
	}

	avg := make([]float32, dim)
	for _, slice := range rawSlices {
		if len(slice) != dim {
			return nil, fmt.Errorf("inconsistent slice dimension: expected %d, got %d", dim, len(slice))
		}
		for d := 0; d < dim; d++ {
			avg[d] += slice[d]
		}
	}

	n := float32(len(rawSlices))
	var sumSq float32
	for d := 0; d < dim; d++ {
		avg[d] /= n
		sumSq += avg[d] * avg[d]
	}

	norm := float32(math.Sqrt(math.Max(float64(sumSq), 1e-12)))
	for d := 0; d < dim; d++ {
		avg[d] /= norm
	}

	return avg, nil
}
