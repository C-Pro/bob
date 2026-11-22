package llm

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"math/big"
	"net"
	"net/http"
	"regexp"
	"strings"
	"time"

	"bob/internal/config"
)

const (
	// SystemOneProtocolV1 is the decision protocol version supported by the bob-s1 sidecar.
	SystemOneProtocolV1 = "bob-s1.decision.v1"

	SystemOneTaskCompaction    = "compaction"
	SystemOneTaskOutcomePrefix = "outcome_prefix"
	SystemOneTaskLoopGuard     = "loop_guard"

	SystemOneScoringJoint       = "joint"
	SystemOneScoringEquivariant = "equivariant"

	SystemOnePrefixPerQuestion = "per_question"
	SystemOnePrefixShared      = "shared_prefix"

	SystemOneTruncationHeadTail = "head_tail"
	SystemOneTruncationError    = "error"

	// SystemOneCalibratedSeqLen is the training/calibration sequence budget (4k tokens).
	SystemOneCalibratedSeqLen = 4096

	// Sidecar resource and batch caps.
	SystemOneMaxBatchSize        = 16
	SystemOneMaxImagesPerItem    = 8
	SystemOneMaxImagesPerRequest = 16
	SystemOneMaxDecisions        = 16

	defaultSystemOneMaxAttempts    = 3
	defaultSystemOneRetryBaseDelay = 50 * time.Millisecond
	maxSystemOneErrorBodyBytes     = 64 * 1024
)

var validSystemOneImageDataURIRegex = regexp.MustCompile(`^data:image/(png|jpeg|webp|gif);base64,[A-Za-z0-9+/=\r\n]+$`)

// IsValidSystemOneImageDataURI reports whether uri matches bob-s1's accepted inline image data URI format.
func IsValidSystemOneImageDataURI(uri string) bool {
	trimmed := strings.TrimSpace(uri)
	if trimmed == "" {
		return false
	}
	return validSystemOneImageDataURIRegex.MatchString(trimmed)
}

// SystemOneAPIError represents a structured error response returned by the bob-s1 sidecar.
type SystemOneAPIError struct {
	StatusCode int
	Message    string
	Type       string
	Param      string
	Code       string
}

func (e *SystemOneAPIError) Error() string {
	if e == nil {
		return ""
	}
	if e.Code != "" {
		return fmt.Sprintf("bob-s1 HTTP %d (%s): %s", e.StatusCode, e.Code, e.Message)
	}
	return fmt.Sprintf("bob-s1 HTTP %d: %s", e.StatusCode, e.Message)
}

// SystemOneHealth represents the GET /healthz readiness payload from bob-s1.
type SystemOneHealth struct {
	Status             string `json:"status"`
	ModelID            string `json:"model_id"`
	CheckpointID       string `json:"checkpoint_id"`
	Device             string `json:"device"`
	DType              string `json:"dtype"`
	AttnImplementation string `json:"attn_implementation"`
	MaxSeqLen          int    `json:"max_seq_len"`
}

// DecisionOptionSpec defines a candidate choice inside a bob-s1.decision.v1 request.
type DecisionOptionSpec struct {
	ID          string   `json:"id"`
	Description string   `json:"description"`
	Value       *float64 `json:"value,omitempty"`
}

// DecisionSpec defines a single built-in or custom decision within a DecisionRequest.
type DecisionSpec struct {
	ID                string               `json:"id"`
	Task              string               `json:"task,omitempty"`
	Question          string               `json:"question,omitempty"`
	Options           []DecisionOptionSpec `json:"options,omitempty"`
	ScoringMode       string               `json:"scoring_mode,omitempty"`
	ApplyCalibration  *bool                `json:"apply_calibration,omitempty"`
	RequireCalibrated *bool                `json:"require_calibrated,omitempty"`
}

// JointEmbeddingSpec requests an optional joint prefix embedding alongside a decision.
type JointEmbeddingSpec struct {
	Dimensions     int    `json:"dimensions,omitempty"`
	EncodingFormat string `json:"encoding_format,omitempty"`
}

// DecisionRequest represents the bob-s1.decision.v1 payload and optional attached images.
type DecisionRequest struct {
	Protocol   string              `json:"protocol"`
	State      string              `json:"state"`
	Decisions  []DecisionSpec      `json:"decisions"`
	Embedding  *JointEmbeddingSpec `json:"embedding,omitempty"`
	PrefixMode string              `json:"prefix_mode,omitempty"`
	Truncation string              `json:"truncation,omitempty"`
	MaxSeqLen  *int                `json:"max_seq_len,omitempty"`

	Images []string `json:"-"`
	User   string   `json:"-"`
}

// DecisionScoreEntry contains raw logit and probability for one option.
type DecisionScoreEntry struct {
	ID          string  `json:"id"`
	Logit       float64 `json:"logit"`
	Probability float64 `json:"probability"`
}

// DecisionCalibrationMeta describes calibration profile application for a decision.
type DecisionCalibrationMeta struct {
	Profile           *string  `json:"profile"`
	Applied           bool     `json:"applied"`
	MismatchReason    *string  `json:"mismatch_reason"`
	Temperature       *float64 `json:"temperature"`
	Bias              *float64 `json:"bias"`
	BalancedThreshold *float64 `json:"balanced_threshold"`
	RiskThreshold5Pct *float64 `json:"risk_threshold_5pct"`
	SelectivePolicy   bool     `json:"selective_policy,omitempty"`
}

// DecisionResultEntry holds the scored result for one decision item.
type DecisionResultEntry struct {
	ID              string                  `json:"id"`
	Task            *string                 `json:"task"`
	ScoringMode     string                  `json:"scoring_mode"`
	Choice          string                  `json:"choice"`
	ChoiceBasis     string                  `json:"choice_basis"`
	Scores          []DecisionScoreEntry    `json:"scores"`
	ProbabilityKind string                  `json:"probability_kind"`
	ExpectedScore   *float64                `json:"expected_score"`
	Calibration     DecisionCalibrationMeta `json:"calibration"`
	PrefixTokens    int                     `json:"prefix_tokens"`
	SuffixTokens    int                     `json:"suffix_tokens"`
	Truncated       bool                    `json:"truncated"`
}

// Probability returns the score probability for optionID.
func (e *DecisionResultEntry) Probability(optionID string) (float64, bool) {
	if e == nil {
		return 0, false
	}
	for _, s := range e.Scores {
		if s.ID == optionID {
			return s.Probability, true
		}
	}
	return 0, false
}

// MaxProbability returns the maximum probability across scored options.
func (e *DecisionResultEntry) MaxProbability() float64 {
	if e == nil || len(e.Scores) == 0 {
		return 0
	}
	best := e.Scores[0].Probability
	for _, s := range e.Scores[1:] {
		if s.Probability > best {
			best = s.Probability
		}
	}
	return best
}

// JointEmbeddingResult holds an optional joint embedding returned from /v1/chat/completions.
type JointEmbeddingResult struct {
	Dimensions     int             `json:"dimensions"`
	EncodingFormat string          `json:"encoding_format"`
	RawVector      json.RawMessage `json:"vector"`
	Vector         []float32       `json:"-"`
}

// DecisionPreparation summarizes sequence preparation parameters applied by bob-s1.
type DecisionPreparation struct {
	PrefixMode         string `json:"prefix_mode"`
	TruncationPolicy   string `json:"truncation_policy"`
	EffectiveMaxSeqLen int    `json:"effective_max_seq_len"`
	SuffixReserve      *int   `json:"suffix_reserve"`
	PrefixTokens       *int   `json:"prefix_tokens"`
	Truncated          bool   `json:"truncated"`
}

// DecisionResponse represents the parsed bob-s1.decision.v1 JSON payload from /v1/chat/completions.
type DecisionResponse struct {
	Protocol     string                `json:"protocol"`
	Decisions    []DecisionResultEntry `json:"decisions"`
	Embedding    *JointEmbeddingResult `json:"embedding"`
	Preparation  DecisionPreparation   `json:"preparation"`
	PromptTokens int                   `json:"-"`
	Latency      time.Duration         `json:"-"`
}

// Decision looks up a decision entry by ID.
func (r *DecisionResponse) Decision(id string) (*DecisionResultEntry, bool) {
	if r == nil {
		return nil, false
	}
	for i := range r.Decisions {
		if r.Decisions[i].ID == id {
			return &r.Decisions[i], true
		}
	}
	return nil, false
}

type systemOneChatResponseFormat struct {
	Type string `json:"type"`
}

type systemOneChatImageURL struct {
	URL string `json:"url"`
}

type systemOneChatContentPart struct {
	Type     string                 `json:"type"`
	ImageURL *systemOneChatImageURL `json:"image_url,omitempty"`
	Text     string                 `json:"text,omitempty"`
}

type systemOneChatMessage struct {
	Role    string `json:"role"`
	Content any    `json:"content"`
}

type systemOneChatRequest struct {
	Model          string                       `json:"model"`
	Messages       []systemOneChatMessage       `json:"messages"`
	ResponseFormat *systemOneChatResponseFormat `json:"response_format,omitempty"`
	User           string                       `json:"user,omitempty"`
}

type systemOneChatCompletionEnvelope struct {
	Choices []struct {
		Index   int `json:"index"`
		Message struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
	Usage struct {
		PromptTokens int `json:"prompt_tokens"`
	} `json:"usage"`
}

type systemOneErrorEnvelope struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Param   string `json:"param"`
		Code    string `json:"code"`
	} `json:"error"`
}

// SystemOneClient communicates with the external bob-s1 inference sidecar without geminiTransport hooks.
type SystemOneClient struct {
	openAIBaseURL string
	healthURL     string
	apiKey        string
	model         string
	timeout       time.Duration
	maxSeqLen     int
	embeddingDims int

	httpClient     *http.Client
	maxAttempts    int
	retryBaseDelay time.Duration
}

// NewSystemOneClient builds a dedicated SystemOneClient from config.
// When httpClient is provided, a shallow copy is configured with the System-1 timeout if needed.
func NewSystemOneClient(cfg *config.Config, httpClient ...*http.Client) (*SystemOneClient, error) {
	if cfg == nil {
		return nil, errors.New("config is required")
	}
	if !cfg.SystemOneEnabled() {
		return nil, errors.New("system1.base_url is not configured")
	}

	openAIBaseURL, healthURL, err := cfg.SystemOneURLs()
	if err != nil {
		return nil, err
	}

	timeout := cfg.SystemOneTimeout
	if timeout <= 0 {
		timeout = config.DefaultSystemOneTimeout
	}

	var hc *http.Client
	if len(httpClient) > 0 && httpClient[0] != nil {
		copied := *httpClient[0]
		if copied.Timeout <= 0 {
			copied.Timeout = timeout
		}
		hc = &copied
	} else {
		hc = &http.Client{
			Timeout: timeout,
			Transport: &http.Transport{
				DialContext:           (&net.Dialer{Timeout: 10 * time.Second}).DialContext,
				ResponseHeaderTimeout: timeout,
				MaxIdleConns:          32,
				IdleConnTimeout:       90 * time.Second,
			},
		}
	}

	model := strings.TrimSpace(cfg.SystemOneModel)
	if model == "" {
		model = config.DefaultSystemOneModel
	}
	dims := cfg.SystemOneEmbeddingDims
	if dims == 0 {
		dims = config.DefaultSystemOneEmbeddingDims
	}

	return &SystemOneClient{
		openAIBaseURL:  openAIBaseURL,
		healthURL:      healthURL,
		apiKey:         cfg.SystemOneAPIKey,
		model:          model,
		timeout:        timeout,
		maxSeqLen:      cfg.SystemOneMaxSeqLen,
		embeddingDims:  dims,
		httpClient:     hc,
		maxAttempts:    defaultSystemOneMaxAttempts,
		retryBaseDelay: defaultSystemOneRetryBaseDelay,
	}, nil
}

// OpenAIBaseURL returns the normalized /v1 base URL.
func (c *SystemOneClient) OpenAIBaseURL() string {
	return c.openAIBaseURL
}

// HealthURL returns the normalized /healthz URL.
func (c *SystemOneClient) HealthURL() string {
	return c.healthURL
}

// Model returns the configured System-1 model identifier.
func (c *SystemOneClient) Model() string {
	return c.model
}

// MaxSeqLen returns the configured System-1 max sequence length override (0 when omitted).
func (c *SystemOneClient) MaxSeqLen() int {
	return c.maxSeqLen
}

// EmbeddingDims returns the configured System-1 embedding dimensionality.
func (c *SystemOneClient) EmbeddingDims() int {
	return c.embeddingDims
}

// Health queries GET /healthz and validates readiness.
func (c *SystemOneClient) Health(ctx context.Context) (*SystemOneHealth, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.healthURL, nil)
	if err != nil {
		return nil, fmt.Errorf("create healthz request: %w", err)
	}
	req.Header.Set("Accept", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("bob-s1 healthz request failed: %w", err)
	}
	defer func() {
		_ = resp.Body.Close()
	}()

	body, err := io.ReadAll(io.LimitReader(resp.Body, maxSystemOneErrorBodyBytes))
	if err != nil {
		return nil, fmt.Errorf("read bob-s1 healthz response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, parseSystemOneHTTPError(resp.StatusCode, body)
	}

	var health SystemOneHealth
	if err := json.Unmarshal(body, &health); err != nil {
		return nil, fmt.Errorf("decode bob-s1 healthz response: %w", err)
	}
	if health.Status != "ok" {
		return &health, fmt.Errorf("bob-s1 unhealthy status: %q", health.Status)
	}
	return &health, nil
}

// Decide executes a bob-s1.decision.v1 request against /v1/chat/completions.
func (c *SystemOneClient) Decide(ctx context.Context, req DecisionRequest) (*DecisionResponse, error) {
	if len(req.Decisions) == 0 {
		return nil, errors.New("at least one decision is required")
	}
	if len(req.Decisions) > SystemOneMaxDecisions {
		return nil, fmt.Errorf("decision count %d exceeds maximum %d", len(req.Decisions), SystemOneMaxDecisions)
	}
	if len(req.Images) > SystemOneMaxImagesPerItem {
		return nil, fmt.Errorf("image count %d exceeds maximum %d", len(req.Images), SystemOneMaxImagesPerItem)
	}
	for i, img := range req.Images {
		if !IsValidSystemOneImageDataURI(img) {
			return nil, fmt.Errorf("images[%d] must be an inline data:image/(png|jpeg|webp|gif);base64 URI", i)
		}
	}

	protoReq := req
	if protoReq.Protocol == "" {
		protoReq.Protocol = SystemOneProtocolV1
	}
	if protoReq.PrefixMode == "" {
		protoReq.PrefixMode = SystemOnePrefixPerQuestion
	}
	if protoReq.Truncation == "" {
		protoReq.Truncation = SystemOneTruncationHeadTail
	}
	if protoReq.MaxSeqLen == nil && c.maxSeqLen > 0 {
		msl := c.maxSeqLen
		protoReq.MaxSeqLen = &msl
	}

	protoBytes, err := json.Marshal(protoReq)
	if err != nil {
		return nil, fmt.Errorf("marshal decision protocol payload: %w", err)
	}

	var content any
	if len(req.Images) == 0 {
		content = string(protoBytes)
	} else {
		parts := make([]systemOneChatContentPart, 0, len(req.Images)+1)
		for _, img := range req.Images {
			parts = append(parts, systemOneChatContentPart{
				Type:     "image_url",
				ImageURL: &systemOneChatImageURL{URL: strings.TrimSpace(img)},
			})
		}
		parts = append(parts, systemOneChatContentPart{
			Type: "text",
			Text: string(protoBytes),
		})
		content = parts
	}

	outerReq := systemOneChatRequest{
		Model: c.model,
		Messages: []systemOneChatMessage{
			{
				Role:    "user",
				Content: content,
			},
		},
		ResponseFormat: &systemOneChatResponseFormat{Type: "json_object"},
		User:           strings.TrimSpace(req.User),
	}

	outerBytes, err := json.Marshal(outerReq)
	if err != nil {
		return nil, fmt.Errorf("marshal bob-s1 chat request: %w", err)
	}

	start := time.Now()
	respBody, err := c.doPostJSONWithRetry(ctx, c.openAIBaseURL+"/chat/completions", outerBytes)
	if err != nil {
		return nil, err
	}
	latency := time.Since(start)

	var envelope systemOneChatCompletionEnvelope
	if err := json.Unmarshal(respBody, &envelope); err != nil {
		return nil, fmt.Errorf("decode bob-s1 chat completion envelope: %w", err)
	}
	if len(envelope.Choices) == 0 {
		return nil, errors.New("bob-s1 chat completion returned empty choices")
	}

	var parsed DecisionResponse
	if err := json.Unmarshal([]byte(envelope.Choices[0].Message.Content), &parsed); err != nil {
		return nil, fmt.Errorf("decode bob-s1 decision payload: %w", err)
	}
	if parsed.Protocol != SystemOneProtocolV1 {
		return nil, fmt.Errorf("unexpected bob-s1 decision protocol %q", parsed.Protocol)
	}
	if len(parsed.Decisions) != len(req.Decisions) {
		return nil, fmt.Errorf("expected %d decisions in bob-s1 response, got %d", len(req.Decisions), len(parsed.Decisions))
	}

	if parsed.Embedding != nil {
		if err := decodeJointEmbeddingResult(parsed.Embedding); err != nil {
			return nil, fmt.Errorf("invalid joint embedding in bob-s1 decision response: %w", err)
		}
	}

	parsed.PromptTokens = envelope.Usage.PromptTokens
	parsed.Latency = latency
	return &parsed, nil
}

// DecideCompaction runs the calibrated compaction binary decision at max_seq_len <= 4096.
func (c *SystemOneClient) DecideCompaction(ctx context.Context, state string, maxSeqLen int) (*DecisionResultEntry, *DecisionResponse, error) {
	if maxSeqLen <= 0 {
		maxSeqLen = SystemOneCalibratedSeqLen
	}
	applyCal := true
	reqCal := true
	return c.decideSingle(ctx, DecisionRequest{
		Protocol: SystemOneProtocolV1,
		State:    state,
		Decisions: []DecisionSpec{
			{
				ID:                "compaction",
				Task:              SystemOneTaskCompaction,
				ScoringMode:       SystemOneScoringJoint,
				ApplyCalibration:  &applyCal,
				RequireCalibrated: &reqCal,
			},
		},
		PrefixMode: SystemOnePrefixPerQuestion,
		Truncation: SystemOneTruncationHeadTail,
		MaxSeqLen:  &maxSeqLen,
	})
}

// DecideLoopGuard runs the semantic loop_guard 3-way classification decision.
func (c *SystemOneClient) DecideLoopGuard(ctx context.Context, state string, images []string) (*DecisionResultEntry, *DecisionResponse, error) {
	applyCal := false
	reqCal := false
	return c.decideSingle(ctx, DecisionRequest{
		Protocol: SystemOneProtocolV1,
		State:    state,
		Images:   images,
		Decisions: []DecisionSpec{
			{
				ID:                "loop_guard",
				Task:              SystemOneTaskLoopGuard,
				ScoringMode:       SystemOneScoringJoint,
				ApplyCalibration:  &applyCal,
				RequireCalibrated: &reqCal,
			},
		},
		PrefixMode: SystemOnePrefixPerQuestion,
		Truncation: SystemOneTruncationHeadTail,
	})
}

// DecideOutcomePrefix runs the outcome_prefix decision helper (not wired into active FSM control flow).
func (c *SystemOneClient) DecideOutcomePrefix(ctx context.Context, state string, requireCalibrated bool) (*DecisionResultEntry, *DecisionResponse, error) {
	applyCal := true
	var maxSeqPtr *int
	if requireCalibrated || c.maxSeqLen <= 0 {
		msl := SystemOneCalibratedSeqLen
		maxSeqPtr = &msl
	}
	return c.decideSingle(ctx, DecisionRequest{
		Protocol: SystemOneProtocolV1,
		State:    state,
		Decisions: []DecisionSpec{
			{
				ID:                "outcome_prefix",
				Task:              SystemOneTaskOutcomePrefix,
				ScoringMode:       SystemOneScoringJoint,
				ApplyCalibration:  &applyCal,
				RequireCalibrated: &requireCalibrated,
			},
		},
		PrefixMode: SystemOnePrefixPerQuestion,
		Truncation: SystemOneTruncationHeadTail,
		MaxSeqLen:  maxSeqPtr,
	})
}

// DecideCustom executes a custom uncalibrated question/options decision query.
func (c *SystemOneClient) DecideCustom(ctx context.Context, id, question string, options []DecisionOptionSpec, state string, images []string) (*DecisionResultEntry, *DecisionResponse, error) {
	id = strings.TrimSpace(id)
	if id == "" {
		id = "custom"
	}
	if strings.TrimSpace(question) == "" {
		return nil, nil, errors.New("question must not be empty")
	}
	if len(options) < 2 {
		return nil, nil, fmt.Errorf("custom decision %q requires at least 2 options", id)
	}
	applyCal := false
	reqCal := false
	return c.decideSingle(ctx, DecisionRequest{
		Protocol: SystemOneProtocolV1,
		State:    state,
		Images:   images,
		Decisions: []DecisionSpec{
			{
				ID:                id,
				Question:          question,
				Options:           options,
				ScoringMode:       SystemOneScoringJoint,
				ApplyCalibration:  &applyCal,
				RequireCalibrated: &reqCal,
			},
		},
		PrefixMode: SystemOnePrefixPerQuestion,
		Truncation: SystemOneTruncationHeadTail,
	})
}

func (c *SystemOneClient) decideSingle(ctx context.Context, req DecisionRequest) (*DecisionResultEntry, *DecisionResponse, error) {
	if len(req.Decisions) != 1 {
		return nil, nil, fmt.Errorf("expected exactly 1 decision, got %d", len(req.Decisions))
	}
	resp, err := c.Decide(ctx, req)
	if err != nil {
		return nil, nil, err
	}
	entry, ok := resp.Decision(req.Decisions[0].ID)
	if !ok {
		return nil, resp, fmt.Errorf("bob-s1 response missing expected decision id %q", req.Decisions[0].ID)
	}
	return entry, resp, nil
}

func (c *SystemOneClient) doPostJSONWithRetry(ctx context.Context, endpoint string, payload []byte) ([]byte, error) {
	attempts := c.maxAttempts
	if attempts < 1 {
		attempts = 1
	}

	var lastErr error
	for attempt := 0; attempt < attempts; attempt++ {
		if ctx != nil && ctx.Err() != nil {
			return nil, ctx.Err()
		}

		body, err := c.doPostJSONOnce(ctx, endpoint, payload)
		if err == nil {
			return body, nil
		}
		lastErr = err

		if !shouldRetrySystemOneError(ctx, err) || attempt == attempts-1 {
			return nil, err
		}

		delay := c.retryDelay(attempt)
		if err := sleepWithContext(ctx, delay); err != nil {
			return nil, err
		}
	}
	return nil, lastErr
}

func (c *SystemOneClient) doPostJSONOnce(ctx context.Context, endpoint string, payload []byte) ([]byte, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("create bob-s1 POST request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("bob-s1 POST %s failed: %w", endpoint, err)
	}
	defer func() {
		_ = resp.Body.Close()
	}()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read bob-s1 response: %w", err)
	}
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		return nil, parseSystemOneHTTPError(resp.StatusCode, body)
	}
	return body, nil
}

func (c *SystemOneClient) retryDelay(attempt int) time.Duration {
	base := c.retryBaseDelay
	if base <= 0 {
		base = defaultSystemOneRetryBaseDelay
	}
	delay := base * time.Duration(1<<min(attempt, 5))
	jitterSpan := delay / 4
	if jitterSpan <= 0 {
		return delay
	}
	n, err := rand.Int(rand.Reader, big.NewInt(int64(jitterSpan)))
	if err != nil {
		return delay
	}
	return delay + time.Duration(n.Int64())
}

func sleepWithContext(ctx context.Context, d time.Duration) error {
	if d <= 0 {
		return nil
	}
	timer := time.NewTimer(d)
	defer timer.Stop()
	if ctx == nil {
		<-timer.C
		return nil
	}
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

func parseSystemOneHTTPError(statusCode int, body []byte) error {
	var env systemOneErrorEnvelope
	if len(body) > 0 && json.Unmarshal(body, &env) == nil && env.Error.Message != "" {
		return &SystemOneAPIError{
			StatusCode: statusCode,
			Message:    env.Error.Message,
			Type:       env.Error.Type,
			Param:      env.Error.Param,
			Code:       env.Error.Code,
		}
	}
	msg := strings.TrimSpace(string(body))
	if len(msg) > 512 {
		msg = msg[:512] + "..."
	}
	if msg == "" {
		msg = http.StatusText(statusCode)
	}
	return &SystemOneAPIError{
		StatusCode: statusCode,
		Message:    msg,
	}
}

func shouldRetrySystemOneError(ctx context.Context, err error) bool {
	if err == nil {
		return false
	}
	if ctx != nil && ctx.Err() != nil {
		return false
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return false
	}

	var apiErr *SystemOneAPIError
	if errors.As(err, &apiErr) {
		switch apiErr.StatusCode {
		case http.StatusTooManyRequests,
			http.StatusBadGateway,
			http.StatusServiceUnavailable,
			http.StatusGatewayTimeout:
			return true
		default:
			return false
		}
	}

	var netErr net.Error
	if errors.As(err, &netErr) {
		return true
	}
	return errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF)
}

func decodeJointEmbeddingResult(res *JointEmbeddingResult) error {
	if res == nil {
		return nil
	}
	if res.Dimensions <= 0 {
		return fmt.Errorf("invalid joint embedding dimensions %d", res.Dimensions)
	}
	switch res.EncodingFormat {
	case "base64":
		var b64 string
		if err := json.Unmarshal(res.RawVector, &b64); err != nil {
			return fmt.Errorf("expected base64 string vector: %w", err)
		}
		vec, err := decodeStrictBase64Float32(b64, res.Dimensions)
		if err != nil {
			return err
		}
		res.Vector = vec
		return nil
	case "float", "":
		var floats []float64
		if err := json.Unmarshal(res.RawVector, &floats); err != nil {
			return fmt.Errorf("expected float array vector: %w", err)
		}
		if len(floats) != res.Dimensions {
			return fmt.Errorf("expected %d embedding floats, got %d", res.Dimensions, len(floats))
		}
		vec := make([]float32, len(floats))
		for i, v := range floats {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return fmt.Errorf("embedding element %d is not finite", i)
			}
			vec[i] = float32(v)
		}
		res.Vector = vec
		return nil
	default:
		return fmt.Errorf("unsupported joint embedding encoding_format %q", res.EncodingFormat)
	}
}

func decodeStrictBase64Float32(b64 string, expectedDims int) ([]float32, error) {
	raw, err := base64.StdEncoding.DecodeString(b64)
	if err != nil {
		return nil, fmt.Errorf("decode base64 embedding vector: %w", err)
	}
	expectedBytes := expectedDims * 4
	if len(raw) != expectedBytes {
		return nil, fmt.Errorf("expected %d bytes for %d float32 values, got %d bytes", expectedBytes, expectedDims, len(raw))
	}
	vec := make([]float32, expectedDims)
	for i := 0; i < expectedDims; i++ {
		bits := binary.LittleEndian.Uint32(raw[i*4 : (i+1)*4])
		val := math.Float32frombits(bits)
		if math.IsNaN(float64(val)) || math.IsInf(float64(val), 0) {
			return nil, fmt.Errorf("decoded embedding element %d is not finite", i)
		}
		vec[i] = val
	}
	return vec, nil
}
