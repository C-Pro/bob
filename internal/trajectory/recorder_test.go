package trajectory

import (
	"encoding/json"
	"testing"
	"time"

	openai "github.com/sashabaranov/go-openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRecorder_EventConstructors(t *testing.T) {
	// Turn start
	startPayload := TurnStartPayload{
		RunID:         "run-1",
		FrontendID:    "besedka",
		SessionID:     "123",
		ScopeID:       "123",
		ActorID:       "user-456",
		Model:         "gpt-4o",
		ExecutionKind: "interactive",
		PrivacyLevel:  "public",
		Input:         "What is the weather?",
	}
	startEv, err := MakeTurnStartEvent("traj-1", "run-1", 1, startPayload)
	require.NoError(t, err)
	assert.Equal(t, "turn_start", startEv.EventType)
	assert.Equal(t, "traj-1", startEv.TrajectoryID)
	assert.Equal(t, "run-1", startEv.RunID)
	assert.Equal(t, 1, startEv.Generation)
	assert.Equal(t, "start", startEv.Phase)

	// Tool attempt
	exitZero := 0
	step := Step{
		Index:       0,
		Iteration:   1,
		Attempt:     1,
		Action:      "weather_fetch",
		ActionArgs:  json.RawMessage(`{"city":"Paris"}`),
		Observation: `{"temp":"20C"}`,
		ExitCode:    &exitZero,
		DurationMs:  120,
	}
	stepEv, err := MakeToolAttemptEvent("traj-1", "run-1", 1, 1, "loop", "step-1", 1, step)
	require.NoError(t, err)
	assert.Equal(t, "tool_attempt", stepEv.EventType)
	assert.Equal(t, "step-1", stepEv.StepID)
	assert.Equal(t, 1, stepEv.Attempt)

	// LLM response
	llmEv, err := MakeLLMResponseEvent("traj-1", "run-1", 1, 1, "loop", "Here is the weather", "Checking Paris...", []openai.ToolCall{
		{
			ID:   "call_1",
			Type: openai.ToolTypeFunction,
			Function: openai.FunctionCall{
				Name:      "weather_fetch",
				Arguments: `{"city":"Paris"}`,
			},
		},
	})
	require.NoError(t, err)
	assert.Equal(t, "llm_response", llmEv.EventType)
	var llmPayload LLMResponsePayload
	err = json.Unmarshal(llmEv.Payload, &llmPayload)
	require.NoError(t, err)
	assert.Equal(t, "Here is the weather", llmPayload.Content)
	assert.Equal(t, "Checking Paris...", llmPayload.ReasoningContent)
	assert.Len(t, llmPayload.ToolCalls, 1)

	// Turn terminal
	termPayload := TurnTerminalPayload{
		Generation: 1,
		Reason:     "completed",
		Failed:     false,
		FinalReply: "The weather in Paris is 20C.",
	}
	termEv, err := MakeTurnTerminalEvent("traj-1", "run-1", 1, termPayload)
	require.NoError(t, err)
	assert.Equal(t, "turn_terminal", termEv.EventType)
	assert.Equal(t, "terminal", termEv.Phase)
}

func TestRecorder_ParseExitCode(t *testing.T) {
	assert.Nil(t, ParseExitCode("Just ordinary text"))

	code := ParseExitCode("Exit Code: 0\nDuration: 10ms\n[stdout]\nHello")
	require.NotNil(t, code)
	assert.Equal(t, 0, *code)

	codeFail := ParseExitCode("Exit Code: 127\nDuration: 5ms\n[stderr]\nCommand not found")
	require.NotNil(t, codeFail)
	assert.Equal(t, 127, *codeFail)
}

func TestRecorder_DerivePrivacyLevel(t *testing.T) {
	// DMs are always private
	p, approved := DerivePrivacyLevel(true, "hello world", "", nil)
	assert.Equal(t, "private", p)
	assert.False(t, approved)

	// Townhall without secrets is public
	p2, approved2 := DerivePrivacyLevel(false, "what is go 1.26?", "", nil)
	assert.Equal(t, "public", p2)
	assert.True(t, approved2)

	// Townhall with secret is private
	p3, approved3 := DerivePrivacyLevel(false, "my api_key is 123", "", nil)
	assert.Equal(t, "private", p3)
	assert.False(t, approved3)
}

func TestRecorder_BuildTrajectoryProjection(t *testing.T) {
	start := TurnStartPayload{
		RunID:          "run-123",
		FrontendID:     "besedka",
		SessionID:      "townhall",
		ScopeID:        "townhall",
		ActorID:        "user-1",
		Model:          "gpt-4o",
		ExecutionKind:  "interactive",
		PrivacyLevel:   "public",
		ExportApproved: true,
		Generation:     1,
		Input:          "Check weather",
	}
	terminal := TurnTerminalPayload{
		Generation: 1,
		Reason:     "completed",
		Failed:     false,
		FinalReply: "It's sunny",
	}
	exitZero := 0
	steps := []Step{
		{Index: 0, Action: "weather", ExitCode: &exitZero, Observation: "sunny"},
	}

	traj := BuildTrajectoryProjection(start, terminal, steps, "It's sunny", 15*time.Minute)
	require.NotNil(t, traj)
	assert.Equal(t, "run-123", traj.ID)
	assert.Equal(t, "run-123", traj.RunID)
	assert.Equal(t, "public", traj.PrivacyLevel)
	assert.True(t, traj.ExportApproved)
	assert.Equal(t, 1, traj.Generation)
	assert.Equal(t, 0.4, traj.Reward) // 1 tool succeeding, normal completed turn
	assert.Equal(t, 1, traj.Breakdown.TotalToolCalls)
	assert.Equal(t, 1, traj.Breakdown.SuccessfulToolCalls)
	assert.Equal(t, 0, traj.Breakdown.FailedToolCalls)
}
