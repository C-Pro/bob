package llm

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"sync/atomic"
	"testing"
	"time"

	"bob/internal/config"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func testSystemOneConfig(baseURL string) *config.Config {
	return &config.Config{
		SystemOneBaseURL:       baseURL,
		SystemOneAPIKey:        "s1-secret",
		SystemOneModel:         "bob-s1-0.8b",
		SystemOneTimeout:       5 * time.Second,
		SystemOneEmbeddingDims: 384,
	}
}

func assertNoExplicitNullsInMap(t *testing.T, obj map[string]any, contextPath string) {
	t.Helper()
	for k, v := range obj {
		keyPath := contextPath + "." + k
		assert.NotNilf(t, v, "explicit null at %s violates bob-s1 _StrictModel", keyPath)
		switch typed := v.(type) {
		case map[string]any:
			assertNoExplicitNullsInMap(t, typed, keyPath)
		case []any:
			for i, item := range typed {
				if sub, ok := item.(map[string]any); ok {
					assertNoExplicitNullsInMap(t, sub, keyPath)
				} else {
					assert.NotNilf(t, item, "explicit null array element at %s[%d]", keyPath, i)
				}
			}
		}
	}
}

func assertValidOuterChatEnvelope(t *testing.T, rawBody []byte) (map[string]any, map[string]any, []any) {
	t.Helper()
	var outer map[string]any
	require.NoError(t, json.Unmarshal(rawBody, &outer))
	assertNoExplicitNullsInMap(t, outer, "outer")

	allowedOuter := map[string]bool{
		"model": true, "messages": true, "response_format": true, "stream": true, "logprobs": true, "user": true,
	}
	for k := range outer {
		assert.Truef(t, allowedOuter[k], "unexpected/forbidden outer chat completion param %q", k)
	}

	msgs, ok := outer["messages"].([]any)
	require.True(t, ok)
	require.Len(t, msgs, 1)
	msg, ok := msgs[0].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "user", msg["role"])

	var protoText string
	var parts []any
	switch content := msg["content"].(type) {
	case string:
		protoText = content
	case []any:
		parts = content
		seenText := false
		for i, rawPart := range parts {
			part, ok := rawPart.(map[string]any)
			require.True(t, ok)
			switch part["type"] {
			case "image_url":
				assert.False(t, seenText, "image_url at index %d appeared after text", i)
				imgObj, ok := part["image_url"].(map[string]any)
				require.True(t, ok)
				assert.Len(t, imgObj, 1, "image_url must contain only url (no detail)")
				_, hasURL := imgObj["url"].(string)
				assert.True(t, hasURL)
			case "text":
				assert.False(t, seenText, "duplicate text part")
				var txtOK bool
				protoText, txtOK = part["text"].(string)
				require.True(t, txtOK)
				seenText = true
			default:
				t.Fatalf("unexpected content part type %v", part["type"])
			}
		}
		require.True(t, seenText, "multimodal message must end with a text part")
	default:
		t.Fatalf("unexpected messages[0].content type %T", msg["content"])
	}

	var proto map[string]any
	require.NoError(t, json.Unmarshal([]byte(protoText), &proto))
	assertNoExplicitNullsInMap(t, proto, "decision_proto")
	assert.Equal(t, SystemOneProtocolV1, proto["protocol"])
	return outer, proto, parts
}

func TestNewSystemOneClient_URLAndValidation(t *testing.T) {
	_, err := NewSystemOneClient(nil)
	require.Error(t, err)

	_, err = NewSystemOneClient(&config.Config{})
	require.Error(t, err)

	_, err = NewSystemOneClient(&config.Config{SystemOneBaseURL: "ftp://bad:8090"})
	require.Error(t, err)

	c1, err := NewSystemOneClient(testSystemOneConfig("http://127.0.0.1:8090"))
	require.NoError(t, err)
	assert.Equal(t, "http://127.0.0.1:8090/v1", c1.OpenAIBaseURL())
	assert.Equal(t, "http://127.0.0.1:8090/healthz", c1.HealthURL())
	assert.Equal(t, "bob-s1-0.8b", c1.Model())
	assert.Equal(t, 384, c1.EmbeddingDims())

	c2, err := NewSystemOneClient(testSystemOneConfig("http://127.0.0.1:8090/proxy/bob-s1/v1/"))
	require.NoError(t, err)
	assert.Equal(t, "http://127.0.0.1:8090/proxy/bob-s1/v1", c2.OpenAIBaseURL())
	assert.Equal(t, "http://127.0.0.1:8090/proxy/bob-s1/healthz", c2.HealthURL())
}

func TestSystemOneClient_Health(t *testing.T) {
	var ready atomic.Bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/healthz", r.URL.Path)
		assert.Equal(t, http.MethodGet, r.Method)
		w.Header().Set("Content-Type", "application/json")
		if !ready.Load() {
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = w.Write([]byte(`{"error":{"message":"Model is not ready","type":"service_unavailable_error","code":"model_not_ready"}}`))
			return
		}
		_, _ = w.Write([]byte(`{
			"status":"ok",
			"model_id":"bob-s1-0.8b",
			"checkpoint_id":"ckpt-1",
			"device":"cuda:0",
			"dtype":"bfloat16",
			"attn_implementation":"sdpa",
			"max_seq_len":8192
		}`))
	}))
	defer srv.Close()

	client, err := NewSystemOneClient(testSystemOneConfig(srv.URL+"/v1"), srv.Client())
	require.NoError(t, err)

	_, err = client.Health(context.Background())
	require.Error(t, err)
	var apiErr *SystemOneAPIError
	require.ErrorAs(t, err, &apiErr)
	assert.Equal(t, 503, apiErr.StatusCode)
	assert.Equal(t, "model_not_ready", apiErr.Code)

	ready.Store(true)
	h, err := client.Health(context.Background())
	require.NoError(t, err)
	assert.Equal(t, "ok", h.Status)
	assert.Equal(t, "bob-s1-0.8b", h.ModelID)
	assert.Equal(t, "ckpt-1", h.CheckpointID)
	assert.Equal(t, 8192, h.MaxSeqLen)
}

func TestSystemOneClient_DecideCompaction_SchemaAndCalibration(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/v1/chat/completions", r.URL.Path)
		assert.Equal(t, "Bearer s1-secret", r.Header.Get("Authorization"))

		rawBody := make([]byte, r.ContentLength)
		_, _ = r.Body.Read(rawBody)
		outer, proto, parts := assertValidOuterChatEnvelope(t, rawBody)
		assert.Nil(t, parts)
		assert.Equal(t, "bob-s1-0.8b", outer["model"])
		assert.Equal(t, float64(4096), proto["max_seq_len"])
		assert.Equal(t, "per_question", proto["prefix_mode"])
		assert.Equal(t, "head_tail", proto["truncation"])

		decisions, ok := proto["decisions"].([]any)
		require.True(t, ok)
		require.Len(t, decisions, 1)
		dec := decisions[0].(map[string]any)
		assert.Equal(t, "compaction", dec["task"])
		assert.Equal(t, "joint", dec["scoring_mode"])
		assert.Equal(t, true, dec["apply_calibration"])
		assert.Equal(t, true, dec["require_calibrated"])

		innerResp := `{
			"protocol":"bob-s1.decision.v1",
			"decisions":[{
				"id":"compaction",
				"task":"compaction",
				"scoring_mode":"joint",
				"choice":"skip",
				"choice_basis":"calibrated_argmax",
				"scores":[
					{"id":"skip","logit":-1.2,"probability":0.87},
					{"id":"compact","logit":0.3,"probability":0.13}
				],
				"probability_kind":"calibrated_affine",
				"expected_score":null,
				"calibration":{
					"profile":"cal_v2_4k",
					"applied":true,
					"mismatch_reason":null,
					"temperature":1.895898683669047,
					"bias":4.25,
					"balanced_threshold":0.8964872561924658,
					"risk_threshold_5pct":0.8393841862136965
				},
				"prefix_tokens":412,
				"suffix_tokens":28,
				"truncated":false
			}],
			"embedding":null,
			"preparation":{
				"prefix_mode":"per_question",
				"truncation_policy":"head_tail",
				"effective_max_seq_len":4096,
				"suffix_reserve":28,
				"prefix_tokens":412,
				"truncated":false
			}
		}`
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id":      "chatcmpl-1",
			"object":  "chat.completion",
			"created": 123,
			"model":   "bob-s1-0.8b",
			"choices": []map[string]any{
				{"index": 0, "message": map[string]any{"role": "assistant", "content": innerResp}, "finish_reason": "stop"},
			},
			"usage": map[string]any{"prompt_tokens": 440, "completion_tokens": 0, "total_tokens": 440},
		})
	}))
	defer srv.Close()

	cfg := testSystemOneConfig(srv.URL)
	cfg.SystemOneMaxSeqLen = 8192 // DecideCompaction must still clamp/override to 4096 when 0 passed
	client, err := NewSystemOneClient(cfg, srv.Client())
	require.NoError(t, err)

	entry, resp, err := client.DecideCompaction(context.Background(), "Goal: fix bug", 0)
	require.NoError(t, err)
	require.NotNil(t, entry)
	require.NotNil(t, resp)

	assert.Equal(t, "skip", entry.Choice)
	assert.Equal(t, "calibrated_affine", entry.ProbabilityKind)
	assert.True(t, entry.Calibration.Applied)
	require.NotNil(t, entry.Calibration.Profile)
	assert.Equal(t, "cal_v2_4k", *entry.Calibration.Profile)
	require.NotNil(t, entry.Calibration.RiskThreshold5Pct)
	assert.InDelta(t, 0.8393841862136965, *entry.Calibration.RiskThreshold5Pct, 1e-12)
	pSkip, ok := entry.Probability("skip")
	require.True(t, ok)
	assert.InDelta(t, 0.87, pSkip, 1e-9)
	assert.InDelta(t, 0.87, entry.MaxProbability(), 1e-9)
	assert.Equal(t, 440, resp.PromptTokens)
}

func TestSystemOneClient_MultimodalImagesAndHelpers(t *testing.T) {
	img1 := "data:image/png;base64," + base64.StdEncoding.EncodeToString([]byte("png-1"))
	img2 := "data:image/jpeg;base64," + base64.StdEncoding.EncodeToString([]byte("jpg-2"))

	// Prepare 128-dim base64 joint embedding
	vecBytes := make([]byte, 128*4)
	binary.LittleEndian.PutUint32(vecBytes[0:4], math.Float32bits(0.5))
	vecB64 := base64.StdEncoding.EncodeToString(vecBytes)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rawBody := make([]byte, r.ContentLength)
		_, _ = r.Body.Read(rawBody)
		_, proto, parts := assertValidOuterChatEnvelope(t, rawBody)

		if len(parts) > 0 {
			require.Len(t, parts, 3)
			assert.Equal(t, "image_url", parts[0].(map[string]any)["type"])
			assert.Equal(t, "image_url", parts[1].(map[string]any)["type"])
			assert.Equal(t, "text", parts[2].(map[string]any)["type"])
		}

		decs := proto["decisions"].([]any)
		dec := decs[0].(map[string]any)
		id := dec["id"].(string)
		choice := "completed"
		switch id {
		case "loop_guard":
			choice = "none"
		case "outcome_prefix":
			choice = "passed"
		}

		innerObj := map[string]any{
			"protocol": SystemOneProtocolV1,
			"decisions": []map[string]any{
				{
					"id":               id,
					"task":             dec["task"],
					"scoring_mode":     "joint",
					"choice":           choice,
					"choice_basis":     "uncalibrated_argmax",
					"scores":           []map[string]any{{"id": choice, "logit": 1.1, "probability": 0.9}},
					"probability_kind": "uncalibrated_softmax",
					"expected_score":   nil,
					"calibration": map[string]any{
						"profile": nil, "applied": false, "mismatch_reason": "no_profile",
					},
					"prefix_tokens": 50,
					"suffix_tokens": 12,
					"truncated":     false,
				},
			},
			"embedding": nil,
			"preparation": map[string]any{
				"prefix_mode": "per_question", "truncation_policy": "head_tail",
				"effective_max_seq_len": 8192, "suffix_reserve": 12, "prefix_tokens": 50, "truncated": false,
			},
		}
		if proto["embedding"] != nil {
			innerObj["embedding"] = map[string]any{
				"dimensions":      128,
				"encoding_format": "base64",
				"vector":          vecB64,
			}
		}
		innerBytes, _ := json.Marshal(innerObj)

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{"index": 0, "message": map[string]any{"role": "assistant", "content": string(innerBytes)}},
			},
		})
	}))
	defer srv.Close()

	client, err := NewSystemOneClient(testSystemOneConfig(srv.URL), srv.Client())
	require.NoError(t, err)

	lg, _, err := client.DecideLoopGuard(context.Background(), "state", []string{img1, img2})
	require.NoError(t, err)
	assert.Equal(t, "none", lg.Choice)

	op, _, err := client.DecideOutcomePrefix(context.Background(), "state", false)
	require.NoError(t, err)
	assert.Equal(t, "passed", op.Choice)

	custom, _, err := client.DecideCustom(context.Background(), "episode_status", "Classify status", []DecisionOptionSpec{
		{ID: "completed", Description: "done"},
		{ID: "ongoing", Description: "wip"},
	}, "ru text", []string{img1, img2})
	require.NoError(t, err)
	assert.Equal(t, "completed", custom.Choice)

	resp, err := client.Decide(context.Background(), DecisionRequest{
		State: "state",
		Decisions: []DecisionSpec{
			{ID: "custom_joint", Question: "Q", Options: []DecisionOptionSpec{{ID: "completed", Description: "y"}, {ID: "no", Description: "n"}}},
		},
		Embedding: &JointEmbeddingSpec{Dimensions: 128, EncodingFormat: "base64"},
	})
	require.NoError(t, err)
	require.NotNil(t, resp.Embedding)
	require.Len(t, resp.Embedding.Vector, 128)
	assert.InDelta(t, 0.5, resp.Embedding.Vector[0], 1e-6)
}

func TestSystemOneClient_RetryAndNonRetryErrors(t *testing.T) {
	t.Run("retries 429 and 503 up to success", func(t *testing.T) {
		var attempts atomic.Int32
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			n := attempts.Add(1)
			w.Header().Set("Content-Type", "application/json")
			if n == 1 {
				w.WriteHeader(http.StatusTooManyRequests)
				_, _ = w.Write([]byte(`{"error":{"message":"Queue full","type":"rate_limit_error","code":"queue_full"}}`))
				return
			}
			if n == 2 {
				w.WriteHeader(http.StatusServiceUnavailable)
				_, _ = w.Write([]byte(`{"error":{"message":"Warming up","type":"service_unavailable_error","code":"model_not_ready"}}`))
				return
			}
			_, _ = w.Write([]byte(`{
				"choices":[{"index":0,"message":{"role":"assistant","content":"{\"protocol\":\"bob-s1.decision.v1\",\"decisions\":[{\"id\":\"loop_guard\",\"scoring_mode\":\"joint\",\"choice\":\"none\",\"choice_basis\":\"uncalibrated_argmax\",\"scores\":[],\"probability_kind\":\"uncalibrated_softmax\",\"calibration\":{\"applied\":false},\"prefix_tokens\":10,\"suffix_tokens\":5,\"truncated\":false}],\"preparation\":{\"prefix_mode\":\"per_question\",\"truncation_policy\":\"head_tail\",\"effective_max_seq_len\":8192,\"truncated\":false}}"}}]
			}`))
		}))
		defer srv.Close()

		client, err := NewSystemOneClient(testSystemOneConfig(srv.URL), srv.Client())
		require.NoError(t, err)
		client.retryBaseDelay = time.Millisecond

		entry, _, err := client.DecideLoopGuard(context.Background(), "s", nil)
		require.NoError(t, err)
		assert.Equal(t, "none", entry.Choice)
		assert.Equal(t, int32(3), attempts.Load())
	})

	for _, tc := range []struct {
		name   string
		status int
		code   string
	}{
		{name: "400 invalid_request", status: http.StatusBadRequest, code: "calibration_unavailable"},
		{name: "401 auth_error", status: http.StatusUnauthorized, code: "invalid_api_key"},
		{name: "413 payload_too_large", status: http.StatusRequestEntityTooLarge, code: "payload_too_large"},
	} {
		t.Run("never retries "+tc.name, func(t *testing.T) {
			var attempts atomic.Int32
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				attempts.Add(1)
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(tc.status)
				_ = json.NewEncoder(w).Encode(map[string]any{
					"error": map[string]any{"message": "bad", "type": "error", "code": tc.code},
				})
			}))
			defer srv.Close()

			client, err := NewSystemOneClient(testSystemOneConfig(srv.URL), srv.Client())
			require.NoError(t, err)
			client.retryBaseDelay = time.Millisecond

			_, _, err = client.DecideCompaction(context.Background(), "s", 4096)
			require.Error(t, err)
			var apiErr *SystemOneAPIError
			require.ErrorAs(t, err, &apiErr)
			assert.Equal(t, tc.status, apiErr.StatusCode)
			assert.Equal(t, tc.code, apiErr.Code)
			assert.Equal(t, int32(1), attempts.Load())
		})
	}

	t.Run("invalid image data URI rejected client-side", func(t *testing.T) {
		client, err := NewSystemOneClient(testSystemOneConfig("http://127.0.0.1:8090"))
		require.NoError(t, err)
		_, _, err = client.DecideLoopGuard(context.Background(), "s", []string{"https://example.com/a.png"})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "data:image")
	})

	t.Run("respects context cancellation during backoff", func(t *testing.T) {
		var attempts atomic.Int32
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			attempts.Add(1)
			w.WriteHeader(http.StatusServiceUnavailable)
		}))
		defer srv.Close()

		client, err := NewSystemOneClient(testSystemOneConfig(srv.URL), srv.Client())
		require.NoError(t, err)
		client.retryBaseDelay = 200 * time.Millisecond

		ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
		defer cancel()
		_, _, err = client.DecideLoopGuard(ctx, "s", nil)
		require.Error(t, err)
		assert.True(t, errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled))
	})
}

func TestSystemOneLiveSmoke(t *testing.T) {
	liveURL := os.Getenv("BOB_S1_TEST_URL")
	if liveURL == "" {
		t.Skip("set BOB_S1_TEST_URL to run live bob-s1 smoke test")
	}

	cfg := testSystemOneConfig(liveURL)
	cfg.SystemOneAPIKey = os.Getenv("BOB_S1_TEST_API_KEY")
	client, err := NewSystemOneClient(cfg)
	require.NoError(t, err)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	h, err := client.Health(ctx)
	require.NoError(t, err)
	assert.Equal(t, "ok", h.Status)

	entry, _, err := client.DecideCompaction(ctx, "Goal: verify live bob-s1 sidecar", 4096)
	require.NoError(t, err)
	assert.True(t, entry.Calibration.Applied)

	emb, err := NewSystemOneEmbedder(client, 384, 0)
	require.NoError(t, err)
	vec, err := emb.Embed(ctx, "live smoke test embedding")
	require.NoError(t, err)
	assert.Len(t, vec, 384)
}
