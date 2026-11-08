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

	// Compaction event
	compPayload := CompactionEventPayload{
		Version:           1,
		ParentVersion:     0,
		CompactedUpToIter: 3,
		ShouldCompact:     true,
		RubricVerdict:     "probed_compact",
		Reason:            "Subtask completed and verified",
		Model:             "gpt-4o",
		InputBytes:        12000,
		OutputBytes:       850,
		StateJSON:         json.RawMessage(`{"version":1,"current_subgoal":"finish report"}`),
	}
	compEv, err := MakeCompactionEvent("traj-1", "run-1", 0, 3, "", compPayload)
	require.NoError(t, err)
	assert.Equal(t, "compaction", compEv.EventType)
	assert.Equal(t, "compaction", compEv.Phase)
	assert.Equal(t, 1, compEv.Generation)
	assert.Equal(t, "compact_v1_i3", compEv.StepID)
	assert.Equal(t, 3, compEv.Iteration)
	var decodedComp CompactionEventPayload
	require.NoError(t, json.Unmarshal(compEv.Payload, &decodedComp))
	assert.Equal(t, 1, decodedComp.Version)
	assert.True(t, decodedComp.ShouldCompact)
	assert.Equal(t, "probed_compact", decodedComp.RubricVerdict)
	assert.Equal(t, "Subtask completed and verified", decodedComp.Reason)

	// Probe skip event has distinct default StepID
	skipEv, err := MakeCompactionEvent("traj-1", "run-1", 1, 3, "", CompactionEventPayload{
		ParentVersion: 0,
		ShouldCompact: false,
		RubricVerdict: "probed_skip",
		Reason:        "Verification still in progress",
	})
	require.NoError(t, err)
	assert.Equal(t, "probe_skip_i3", skipEv.StepID)
	assert.NotEqual(t, compEv.StepID, skipEv.StepID)

	// Explicit StepID is preserved
	explicitEv, err := MakeCompactionEvent("traj-1", "run-1", 2, 4, "custom_step_id", CompactionEventPayload{
		ShouldCompact: false,
		RubricVerdict: "probed_skip",
	})
	require.NoError(t, err)
	assert.Equal(t, "custom_step_id", explicitEv.StepID)
	assert.Equal(t, 2, explicitEv.Generation)

	// Invalid StateJSON is rejected
	_, err = MakeCompactionEvent("traj-1", "run-1", 1, 3, "", CompactionEventPayload{
		Version:       1,
		ShouldCompact: true,
		RubricVerdict: "probed_compact",
		StateJSON:     json.RawMessage(`{invalid json`),
	})
	require.Error(t, err)
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
