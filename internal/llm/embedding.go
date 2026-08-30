package llm

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	openai "github.com/sashabaranov/go-openai"
)

// CreateEmbeddings generates vector embeddings for the provided texts using the specified model.
func (c *Client) CreateEmbeddings(ctx context.Context, texts []string, model string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	if model == "" {
		if c.cfg != nil && c.cfg.EmbeddingModel != "" {
			model = c.cfg.EmbeddingModel
		} else {
			return nil, errors.New("embedding model is not configured")
		}
	}

	req := openai.EmbeddingRequest{
		Input: texts,
		Model: openai.EmbeddingModel(model),
	}

	var lastErr error
	for attempt := 0; attempt <= c.maxRetries; attempt++ {
		if attempt > 0 {
			delay := c.baseDelay * (1 << (attempt - 1))
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(delay):
			}
		}

		resp, err := c.apiClient.CreateEmbeddings(ctx, req)
		if err == nil {
			if len(resp.Data) == 0 {
				return nil, errors.New("no embeddings returned by API")
			}

			results := make([][]float32, len(texts))
			for _, item := range resp.Data {
				if item.Index >= 0 && item.Index < len(results) {
					results[item.Index] = item.Embedding
				}
			}
			return results, nil
		}

		lastErr = err
		if !isRetryableError(err) {
			return nil, fmt.Errorf("embedding API error: %w", err)
		}
	}

	return nil, fmt.Errorf("embedding request failed after %d retries; last error: %w", c.maxRetries, lastErr)
}

// Embedder provides text-to-vector embedding generation conforming to the CortexDB Embedder interface.
type Embedder struct {
	client *Client
	model  string
	mu     sync.RWMutex
	dim    int
}

// NewEmbedder creates an Embedder using the given client and model.
func NewEmbedder(client *Client, model string) *Embedder {
	if model == "" && client != nil && client.cfg != nil {
		model = client.cfg.EmbeddingModel
	}
	return &Embedder{
		client: client,
		model:  model,
	}
}

// Embed converts a single text string into a vector by delegating to EmbedBatch.
func (e *Embedder) Embed(ctx context.Context, text string) ([]float32, error) {
	if e == nil {
		return nil, errors.New("embedder is nil")
	}
	if strings.TrimSpace(text) == "" {
		return nil, errors.New("empty text provided")
	}
	vecs, err := e.EmbedBatch(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	if len(vecs) == 0 || len(vecs[0]) == 0 {
		return nil, errors.New("no embedding returned")
	}
	return vecs[0], nil
}

// EmbedBatch converts multiple texts into vectors using the OpenAI-compatible batch API client.
func (e *Embedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if e == nil {
		return nil, errors.New("embedder is nil")
	}
	if len(texts) == 0 {
		return nil, nil
	}
	if e.client == nil {
		return nil, errors.New("embedding client not configured")
	}
	vecs, err := e.client.CreateEmbeddings(ctx, texts, e.model)
	if err != nil {
		return nil, err
	}
	if len(vecs) > 0 && len(vecs[0]) > 0 {
		e.mu.Lock()
		e.dim = len(vecs[0])
		e.mu.Unlock()
	}
	return vecs, nil
}

// Dim returns the known vector dimension of the embedder (or 0 if not yet determined).
func (e *Embedder) Dim() int {
	if e == nil {
		return 0
	}
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.dim
}

// Model returns the configured embedding model name.
func (e *Embedder) Model() string {
	if e == nil {
		return ""
	}
	return e.model
}
