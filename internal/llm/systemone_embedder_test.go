package llm

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func encodeTestFloat32Base64(vec []float32) string {
	buf := make([]byte, len(vec)*4)
	for i, v := range vec {
		binary.LittleEndian.PutUint32(buf[i*4:(i+1)*4], math.Float32bits(v))
	}
	return base64.StdEncoding.EncodeToString(buf)
}

func makeTestFloatVector(dims int, seed float32) []float32 {
	v := make([]float32, dims)
	for i := range v {
		v[i] = seed + float32(i)*0.001
	}
	return v
}

func TestSystemOneEmbedder_DimWithoutRPCAndSubBatching(t *testing.T) {
	var callCount atomic.Int32
	var batchSizes []int

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/v1/embeddings", r.URL.Path)
		assert.Equal(t, "Bearer s1-secret", r.Header.Get("Authorization"))
		callCount.Add(1)

		raw, err := io.ReadAll(r.Body)
		require.NoError(t, err)

		var reqMap map[string]any
		require.NoError(t, json.Unmarshal(raw, &reqMap))
		assertNoExplicitNullsInMap(t, reqMap, "embeddings")
		assert.Equal(t, "bob-s1-0.8b", reqMap["model"])
		assert.Equal(t, "base64", reqMap["encoding_format"])
		assert.Equal(t, float64(256), reqMap["dimensions"])
		assert.Equal(t, "head_tail", reqMap["truncation"])
		assert.Equal(t, float64(4096), reqMap["max_seq_len"])

		inputs, ok := reqMap["input"].([]any)
		require.True(t, ok)
		require.LessOrEqual(t, len(inputs), SystemOneMaxBatchSize)
		batchSizes = append(batchSizes, len(inputs))

		// Emit items in reverse order to verify Index reassembly.
		data := make([]map[string]any, 0, len(inputs))
		for idx := len(inputs) - 1; idx >= 0; idx-- {
			txt := inputs[idx].(string)
			seed := float32(len(txt))
			data = append(data, map[string]any{
				"object":    "embedding",
				"index":     idx,
				"embedding": encodeTestFloat32Base64(makeTestFloatVector(256, seed)),
			})
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"object": "list",
			"model":  "bob-s1-0.8b",
			"data":   data,
			"usage":  map[string]any{"prompt_tokens": 100, "total_tokens": 100},
		})
	}))
	defer srv.Close()

	cfg := testSystemOneConfig(srv.URL)
	client, err := NewSystemOneClient(cfg, srv.Client())
	require.NoError(t, err)

	emb, err := NewSystemOneEmbedder(client, 256, 4096)
	require.NoError(t, err)

	// Dim() must return without triggering any HTTP calls.
	assert.Equal(t, 256, emb.Dim())
	assert.Equal(t, int32(0), callCount.Load())

	// Empty batch is a no-op
	empty, err := emb.EmbedBatch(context.Background(), nil)
	require.NoError(t, err)
	assert.Nil(t, empty)
	assert.Equal(t, int32(0), callCount.Load())

	// 35 texts should split cleanly into 16 + 16 + 3 batches
	texts := make([]string, 35)
	for i := range texts {
		texts[i] = fmt.Sprintf("chunk-%02d", i)
	}
	// Make lengths distinct to verify reverse-index reassembly
	texts[0] = "a"
	texts[15] = "sixteen-chars-xx"

	vecs, err := emb.EmbedBatch(context.Background(), texts)
	require.NoError(t, err)
	require.Len(t, vecs, 35)
	assert.Equal(t, []int{16, 16, 3}, batchSizes)
	assert.InDelta(t, float32(len(texts[0])), vecs[0][0], 1e-6)
	assert.InDelta(t, float32(len(texts[15])), vecs[15][0], 1e-6)
}

func TestSystemOneEmbedder_StrictResponseValidation(t *testing.T) {
	dims := 128

	cases := []struct {
		name string
		data []map[string]any
	}{
		{
			name: "wrong decoded byte length (not multiple of 4 / trailing byte)",
			data: []map[string]any{
				{"object": "embedding", "index": 0, "embedding": base64.StdEncoding.EncodeToString(make([]byte, dims*4+1))},
			},
		},
		{
			name: "wrong dims multiple",
			data: []map[string]any{
				{"object": "embedding", "index": 0, "embedding": encodeTestFloat32Base64(make([]float32, 64))},
			},
		},
		{
			name: "non-finite NaN element",
			data: []map[string]any{
				{"object": "embedding", "index": 0, "embedding": encodeTestFloat32Base64(func() []float32 {
					v := makeTestFloatVector(dims, 1)
					v[7] = float32(math.NaN())
					return v
				}())},
			},
		},
		{
			name: "duplicate index",
			data: []map[string]any{
				{"object": "embedding", "index": 0, "embedding": encodeTestFloat32Base64(makeTestFloatVector(dims, 1))},
				{"object": "embedding", "index": 0, "embedding": encodeTestFloat32Base64(makeTestFloatVector(dims, 2))},
			},
		},
		{
			name: "wrong item count",
			data: []map[string]any{},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_ = json.NewEncoder(w).Encode(map[string]any{
					"object": "list",
					"model":  "bob-s1-0.8b",
					"data":   tc.data,
				})
			}))
			defer srv.Close()

			client, err := NewSystemOneClient(testSystemOneConfig(srv.URL), srv.Client())
			require.NoError(t, err)
			emb, err := NewSystemOneEmbedder(client, dims, 0)
			require.NoError(t, err)

			if tc.name == "duplicate index" {
				_, err = emb.EmbedBatch(context.Background(), []string{"a", "b"})
			} else {
				_, err = emb.Embed(context.Background(), "a")
			}
			require.Error(t, err)
		})
	}
}

func TestSystemOneEmbedder_MultimodalAndFallback(t *testing.T) {
	validPNG := "data:image/png;base64," + base64.StdEncoding.EncodeToString([]byte("png"))

	t.Run("WithEmbeddingImages sends multimodal content parts ordered before text and caps at 8", func(t *testing.T) {
		var sawMultimodal atomic.Bool
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			raw, _ := io.ReadAll(r.Body)
			var reqMap map[string]any
			require.NoError(t, json.Unmarshal(raw, &reqMap))
			assert.NotContains(t, reqMap, "max_seq_len", "max_seq_len=0 must be omitted")

			inputs := reqMap["input"].([]any)
			require.Len(t, inputs, 1)
			item := inputs[0].(map[string]any)
			parts := item["content"].([]any)
			// 10 valid + 1 invalid -> capped at 8 images + 1 text = 9 parts
			require.Len(t, parts, 9)
			for i := 0; i < 8; i++ {
				p := parts[i].(map[string]any)
				assert.Equal(t, "image_url", p["type"])
				imgObj := p["image_url"].(map[string]any)
				assert.Len(t, imgObj, 1)
			}
			last := parts[8].(map[string]any)
			assert.Equal(t, "text", last["type"])
			assert.Equal(t, "chunk with images", last["text"])
			sawMultimodal.Store(true)

			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{
				"object": "list",
				"data": []map[string]any{
					{"object": "embedding", "index": 0, "embedding": encodeTestFloat32Base64(makeTestFloatVector(384, 2))},
				},
			})
		}))
		defer srv.Close()

		client, err := NewSystemOneClient(testSystemOneConfig(srv.URL), srv.Client())
		require.NoError(t, err)
		emb, err := NewSystemOneEmbedder(client, 384, 0)
		require.NoError(t, err)

		imgs := make([]string, 0, 11)
		imgs = append(imgs, "https://not-inline.example/x.png")
		for i := 0; i < 10; i++ {
			imgs = append(imgs, validPNG)
		}

		ctx := WithEmbeddingImages(context.Background(), imgs)
		vec, err := emb.Embed(ctx, "chunk with images")
		require.NoError(t, err)
		require.Len(t, vec, 384)
		assert.True(t, sawMultimodal.Load())
	})

	t.Run("multimodal 400 falls back to text-only when fallback_on_error=true, fails when false", func(t *testing.T) {
		var calls atomic.Int32
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			calls.Add(1)
			raw, _ := io.ReadAll(r.Body)
			var reqMap map[string]any
			require.NoError(t, json.Unmarshal(raw, &reqMap))
			inputs := reqMap["input"].([]any)
			if _, isTextOnly := inputs[0].(string); isTextOnly {
				w.Header().Set("Content-Type", "application/json")
				_ = json.NewEncoder(w).Encode(map[string]any{
					"object": "list",
					"data": []map[string]any{
						{"object": "embedding", "index": 0, "embedding": encodeTestFloat32Base64(makeTestFloatVector(384, 5))},
					},
				})
				return
			}
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"error":{"message":"bad image","type":"invalid_request_error","code":"invalid_image"}}`))
		}))
		defer srv.Close()

		client, err := NewSystemOneClient(testSystemOneConfig(srv.URL), srv.Client())
		require.NoError(t, err)

		// Fallback enabled
		embFallback, err := NewSystemOneEmbedder(client, 384, 0, true)
		require.NoError(t, err)
		vec, err := embFallback.Embed(WithEmbeddingImages(context.Background(), []string{validPNG}), "text")
		require.NoError(t, err)
		assert.InDelta(t, 5.0, vec[0], 1e-6)
		assert.Equal(t, int32(2), calls.Load())

		// Strict mode: no fallback
		calls.Store(0)
		embStrict, err := NewSystemOneEmbedder(client, 384, 0, false)
		require.NoError(t, err)
		_, err = embStrict.Embed(WithEmbeddingImages(context.Background(), []string{validPNG}), "text")
		require.Error(t, err)
		assert.Equal(t, int32(1), calls.Load())
	})
}
