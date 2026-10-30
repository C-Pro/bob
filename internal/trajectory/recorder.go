package trajectory

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"time"

	openai "github.com/sashabaranov/go-openai"
)

// newEventID generates a secure hex identifier for outbox events.
func newEventID() string {
	b := make([]byte, 16)
	if _, err := rand.Read(b); err != nil {
		return fmt.Sprintf("ev-%d", time.Now().UnixNano())
	}
	return hex.EncodeToString(b)
}

// MakeTurnStartEvent constructs an immutable turn_start outbox event.
func MakeTurnStartEvent(trajID, runID string, gen int, payload TurnStartPayload) (*Event, error) {
	if gen <= 0 {
		gen = 1
	}
	payload.Generation = gen
	if payload.RunID == "" {
		payload.RunID = runID
	}
	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal turn_start payload: %w", err)
	}
	return &Event{
		ID:           newEventID(),
		TrajectoryID: trajID,
		RunID:        runID,
		Generation:   gen,
		Iteration:    0,
		Phase:        "start",
		StepID:       "",
		Attempt:      1,
		EventType:    "turn_start",
		Payload:      payloadBytes,
		CreatedAt:    time.Now().UTC(),
	}, nil
}

// MakeToolAttemptEvent constructs an immutable tool_attempt outbox event.
func MakeToolAttemptEvent(trajID, runID string, gen, iteration int, phase, stepID string, attempt int, step Step) (*Event, error) {
	if gen <= 0 {
		gen = 1
	}
	if phase == "" {
		phase = "loop"
	}
	if attempt <= 0 {
		attempt = 1
	}
	payloadBytes, err := json.Marshal(step)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal tool_attempt step: %w", err)
	}
	return &Event{
		ID:           newEventID(),
		TrajectoryID: trajID,
		RunID:        runID,
		Generation:   gen,
		Iteration:    iteration,
		Phase:        phase,
		StepID:       stepID,
		Attempt:      attempt,
		EventType:    "tool_attempt",
		Payload:      payloadBytes,
		CreatedAt:    time.Now().UTC(),
	}, nil
}

// LLMResponsePayload stores assistant generation output and reasoning traces.
type LLMResponsePayload struct {
	Content          string            `json:"content"`
	ReasoningContent string            `json:"reasoning_content,omitempty"`
	ToolCalls        []openai.ToolCall `json:"tool_calls,omitempty"`
}

// MakeLLMResponseEvent constructs an immutable llm_response outbox event.
func MakeLLMResponseEvent(trajID, runID string, gen, iteration int, phase string, content, reasoning string, toolCalls []openai.ToolCall) (*Event, error) {
	if gen <= 0 {
		gen = 1
	}
	if phase == "" {
		phase = "loop"
	}
	payload := LLMResponsePayload{
		Content:          content,
		ReasoningContent: reasoning,
		ToolCalls:        toolCalls,
	}
	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal llm_response payload: %w", err)
	}
	return &Event{
		ID:           newEventID(),
		TrajectoryID: trajID,
		RunID:        runID,
		Generation:   gen,
		Iteration:    iteration,
		Phase:        phase,
		StepID:       "",
		Attempt:      1,
		EventType:    "llm_response",
		Payload:      payloadBytes,
		CreatedAt:    time.Now().UTC(),
	}, nil
}

// MakeTurnTerminalEvent constructs an immutable turn_terminal outbox event.
func MakeTurnTerminalEvent(trajID, runID string, gen int, payload TurnTerminalPayload, phase ...string) (*Event, error) {
	if gen <= 0 {
		gen = 1
	}
	p := "terminal"
	if len(phase) > 0 && phase[0] != "" {
		p = phase[0]
	}
	payload.Generation = gen
	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal turn_terminal payload: %w", err)
	}
	return &Event{
		ID:           newEventID(),
		TrajectoryID: trajID,
		RunID:        runID,
		Generation:   gen,
		Iteration:    0,
		Phase:        p,
		StepID:       "",
		Attempt:      1,
		EventType:    "turn_terminal",
		Payload:      payloadBytes,
		CreatedAt:    time.Now().UTC(),
	}, nil
}

// ParseExitCode scans tool output text for structured "Exit Code: <int>" lines.
func ParseExitCode(content string) *int {
	lines := strings.Split(content, "\n")
	for _, l := range lines {
		trimmed := strings.TrimSpace(l)
		if strings.HasPrefix(trimmed, "Exit Code:") {
			valStr := strings.TrimSpace(strings.TrimPrefix(trimmed, "Exit Code:"))
			if code, err := strconv.Atoi(valStr); err == nil {
				return &code
			}
		}
	}
	return nil
}

// DerivePrivacyLevel assigns privacy classification and export approval.
// Townhall chats default to public/approved unless private memories or secrets are indicated.
// Direct Messages default to private/unapproved.
func DerivePrivacyLevel(isDM bool, userInput, taskState string, activeSkills []string) (string, bool) {
	if isDM {
		return "private", false
	}
	// Check for sensitive markers in input or task state
	combined := strings.ToLower(userInput + " " + taskState)
	for _, kw := range []string{"api_key", "password", "secret", "token", "credential", "private"} {
		if strings.Contains(combined, kw) {
			return "private", false
		}
	}
	return "public", true
}

// BuildTrajectoryProjection constructs a canonical Trajectory record from turn start, terminal payload, and steps.
func BuildTrajectoryProjection(start TurnStartPayload, terminal TurnTerminalPayload, steps []Step, finalOutput string, delay time.Duration) *Trajectory {
	now := time.Now().UTC()
	var succCount, failCount int
	for _, s := range steps {
		if s.ExitCode == nil || *s.ExitCode == 0 {
			succCount++
		} else {
			failCount++
		}
	}

	budgetExhausted := terminal.Reason == "budget_exhausted"
	breakdown := EvalBreakdown{
		TotalToolCalls:      len(steps),
		SuccessfulToolCalls: succCount,
		FailedToolCalls:     failCount,
		BudgetExhausted:     budgetExhausted,
		FSMFailed:           terminal.Failed,
		TerminationReason:   terminal.Reason,
		RewardVersion:       RewardVersionV1,
	}
	if len(steps) > 0 {
		breakdown.ToolSuccessRate = float64(succCount) / float64(len(steps))
	}

	rEval := CalculateReward(breakdown)

	privacy := start.PrivacyLevel
	exportApproved := start.ExportApproved
	if privacy == "" {
		privacy = "private"
		exportApproved = false
	}

	gen := terminal.Generation
	if gen <= 0 {
		gen = start.Generation
	}
	if gen <= 0 {
		gen = 1
	}

	output := finalOutput
	if output == "" && terminal.FinalReply != "" {
		output = terminal.FinalReply
	}

	return &Trajectory{
		ID:             start.RunID,
		RunID:          start.RunID,
		FrontendID:     start.FrontendID,
		SessionID:      start.SessionID,
		ScopeID:        start.ScopeID,
		ActorID:        start.ActorID,
		Model:          start.Model,
		ExecutionKind:  start.ExecutionKind,
		PrivacyLevel:   privacy,
		ExportApproved: exportApproved,
		Generation:     gen,
		CreatedAt:      now,
		CompletedAt:    now,
		UserInput:      start.Input,
		TaskState:      start.TaskState,
		ActiveSkills:   start.ActiveSkills,
		Steps:          steps,
		FinalOutput:    output,
		Reward:         rEval,
		Breakdown:      breakdown,
		Revision:       1,
	}
}
