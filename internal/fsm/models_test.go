package fsm

import (
	"encoding/json"
	"testing"

	openai "github.com/sashabaranov/go-openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestEncodeDecodeMessages(t *testing.T) {
	msgs := []openai.ChatCompletionMessage{
		{
			Role:    openai.ChatMessageRoleSystem,
			Content: "You are a helpful assistant.",
		},
		{
			Role:    openai.ChatMessageRoleUser,
			Content: "What is the weather?",
		},
		{
			Role: openai.ChatMessageRoleAssistant,
			ToolCalls: []openai.ToolCall{
				{
					ID:   "call_123",
					Type: openai.ToolTypeFunction,
					Function: openai.FunctionCall{
						Name:      "web_search",
						Arguments: `{"query":"weather"}`,
					},
				},
			},
		},
		{
			Role:       openai.ChatMessageRoleTool,
			Content:    `{"result":"Sunny 22C"}`,
			ToolCallID: "call_123",
		},
	}

	encoded, err := EncodeMessages(msgs)
	require.NoError(t, err)
	assert.NotEmpty(t, encoded)

	decoded, err := DecodeMessages(encoded)
	require.NoError(t, err)
	require.Len(t, decoded, len(msgs))
	assert.Equal(t, msgs[0].Role, decoded[0].Role)
	assert.Equal(t, msgs[0].Content, decoded[0].Content)
	assert.Equal(t, msgs[2].ToolCalls[0].Function.Name, decoded[2].ToolCalls[0].Function.Name)
	assert.Equal(t, msgs[3].ToolCallID, decoded[3].ToolCallID)

	// Empty decode handling
	emptyDecoded, err := DecodeMessages("")
	require.NoError(t, err)
	assert.Nil(t, emptyDecoded)

	emptyDecodedObj, err := DecodeMessages("{}")
	require.NoError(t, err)
	assert.Nil(t, emptyDecodedObj)

	// Invalid json
	_, err = DecodeMessages("invalid json")
	require.Error(t, err)
}

func TestRunStatus_IsTerminal(t *testing.T) {
	assert.False(t, RunStatusPending.IsTerminal())
	assert.False(t, RunStatusRunning.IsTerminal())
	assert.False(t, RunStatusWaiting.IsTerminal())
	assert.True(t, RunStatusCompleted.IsTerminal())
	assert.True(t, RunStatusFailed.IsTerminal())
	assert.True(t, RunStatusTerminated.IsTerminal())
}

func TestStepStatus_IsTerminal(t *testing.T) {
	assert.False(t, StepStatusPending.IsTerminal())
	assert.False(t, StepStatusRunning.IsTerminal())
	assert.True(t, StepStatusCompleted.IsTerminal())
	assert.True(t, StepStatusFailed.IsTerminal())
	assert.True(t, StepStatusTimedOut.IsTerminal())
	assert.True(t, StepStatusSkipped.IsTerminal())
}

func TestNewStepFromToolCall(t *testing.T) {
	tc := openai.ToolCall{
		ID:   "call_abc",
		Type: openai.ToolTypeFunction,
		Function: openai.FunctionCall{
			Name:      "web_search",
			Arguments: `{"query":"test"}`,
		},
	}

	step := NewStepFromToolCall("run_1", 2, 0, tc, ExecutionModeParallel)
	assert.Equal(t, "run_1_i2_s0", step.ID)
	assert.Equal(t, "run_1", step.RunID)
	assert.Equal(t, 2, step.Iteration)
	assert.Equal(t, 0, step.StepIndex)
	assert.Equal(t, "web_search", step.ToolName)
	assert.Equal(t, "call_abc", step.ToolCallID)
	assert.Equal(t, `{"query":"test"}`, step.ArgsJSON)
	assert.Equal(t, ExecutionModeParallel, step.ExecutionMode)
	assert.Equal(t, StepStatusPending, step.Status)
	assert.Equal(t, 0, step.Attempt)
	assert.Equal(t, 3, step.MaxAttempts)
	assert.Equal(t, 30, step.TimeoutSeconds)

	// Propose tools have MaxAttempts = 1 to prevent non-idempotent retries after timeout
	tcMem := openai.ToolCall{Function: openai.FunctionCall{Name: "propose_memory"}}
	stepMem := NewStepFromToolCall("run_1", 2, 1, tcMem, ExecutionModeSequential)
	assert.Equal(t, 1, stepMem.MaxAttempts)

	tcSkill := openai.ToolCall{Function: openai.FunctionCall{Name: "propose_skill"}}
	stepSkill := NewStepFromToolCall("run_1", 2, 2, tcSkill, ExecutionModeSequential)
	assert.Equal(t, 1, stepSkill.MaxAttempts)

	// Default timeout for exec tool is 60s
	tcDefaultExec := openai.ToolCall{
		ID: "call_exec_def",
		Function: openai.FunctionCall{
			Name:      "execute",
			Arguments: `{"command":"go test ./..."}`,
		},
	}
	stepDefaultExec := NewStepFromToolCall("run_1", 2, 3, tcDefaultExec, ExecutionModeSequential)
	assert.Equal(t, 60, stepDefaultExec.TimeoutSeconds)

	// Custom timeout honored
	tcCustomTimeout := openai.ToolCall{
		ID: "call_exec",
		Function: openai.FunctionCall{
			Name:      "execute",
			Arguments: `{"command":"go test ./...","timeout_seconds":120}`,
		},
	}
	stepCustom := NewStepFromToolCall("run_1", 2, 4, tcCustomTimeout, ExecutionModeSequential)
	assert.Equal(t, 120, stepCustom.TimeoutSeconds)

	// Clamped to 600 max
	tcOverTimeout := openai.ToolCall{
		ID: "call_exec",
		Function: openai.FunctionCall{
			Name:      "execute",
			Arguments: `{"command":"go test ./...","timeout_seconds":2500}`,
		},
	}
	stepOver := NewStepFromToolCall("run_1", 2, 5, tcOverTimeout, ExecutionModeSequential)
	assert.Equal(t, 600, stepOver.TimeoutSeconds)

	// Negative/zero falls back to default 60
	tcNegativeTimeout := openai.ToolCall{
		ID: "call_exec",
		Function: openai.FunctionCall{
			Name:      "execute",
			Arguments: `{"command":"go test ./...","timeout_seconds":-10}`,
		},
	}
	stepNeg := NewStepFromToolCall("run_1", 2, 6, tcNegativeTimeout, ExecutionModeSequential)
	assert.Equal(t, 60, stepNeg.TimeoutSeconds)

	// Non-exec tool ignores timeout_seconds argument and stays at default 30
	tcNonExec := openai.ToolCall{
		ID: "call_web",
		Function: openai.FunctionCall{
			Name:      "web_search",
			Arguments: `{"query":"test","timeout_seconds":120}`,
		},
	}
	stepNonExec := NewStepFromToolCall("run_1", 2, 6, tcNonExec, ExecutionModeParallel)
	assert.Equal(t, 30, stepNonExec.TimeoutSeconds)
}

func TestCompactedTaskStateSerialization(t *testing.T) {
	probe := CompactionProbeResult{
		ShouldCompact: true,
		Reason:        "Subtask 1 finished and verified",
		State: &CompactedTaskState{
			Version:              1,
			ParentVersion:        0,
			RunID:                "run_123",
			CompactedUpToIter:    3,
			CompactedStepIDs:     []string{"run_123_i1_s0", "run_123_i2_s0"},
			OriginalObjective:    "Investigate database schema and add compactions table",
			ImmutableConstraints: []string{"Never delete raw fsm_steps"},
			VerifiedFindings: []VerifiedFinding{
				{
					Summary:       "fsm uses fsm_schema_version in per-chat SQLite DB",
					SourceStepIDs: []string{"run_123_i1_s0"},
				},
			},
			DecisionsAndRejected: []string{"Use fsm_compactions table instead of mutating context_json"},
			CurrentSubgoal:       "Implement store CRUD for compactions",
			NextIntendedActions:  []string{"Add SaveCompaction and GetLatestCompaction"},
			RubricVerdict:        CompactionVerdictProbedCompact,
			Model:                "gpt-4o",
			InputBytes:           8192,
			OutputBytes:          512,
			CreatedAt:            1700000000,
		},
	}

	raw, err := json.Marshal(probe)
	require.NoError(t, err)

	var decoded CompactionProbeResult
	require.NoError(t, json.Unmarshal(raw, &decoded))
	assert.True(t, decoded.ShouldCompact)
	assert.Equal(t, "Subtask 1 finished and verified", decoded.Reason)
	require.NotNil(t, decoded.State)
	assert.Equal(t, 1, decoded.State.Version)
	assert.Equal(t, "run_123", decoded.State.RunID)
	assert.Equal(t, 3, decoded.State.CompactedUpToIter)
	assert.Len(t, decoded.State.VerifiedFindings, 1)
	assert.Equal(t, "run_123_i1_s0", decoded.State.VerifiedFindings[0].SourceStepIDs[0])
}

func TestCompactedTaskStateFlexibleStringSlices(t *testing.T) {
	// Test that CompactedTaskState can unmarshal string fields when the LLM produces single strings
	// instead of JSON arrays for slice fields.
	rawJSON := `{
		"should_compact": true,
		"reason": "checkpoint reached",
		"state": {
			"version": 1,
			"original_objective": "Test objective",
			"immutable_constraints": "Single constraint string",
			"decisions_and_rejected": "Single decision",
			"current_subgoal": "Subgoal 1",
			"next_intended_actions": "Single next action",
			"unresolved_blockers": "Single blocker string",
			"produced_artifacts": "Single artifact",
			"active_skill_versions": "v1.0",
			"active_memory_versions": "m1.0"
		}
	}`

	var probe CompactionProbeResult
	err := json.Unmarshal([]byte(rawJSON), &probe)
	require.NoError(t, err)
	require.NotNil(t, probe.State)
	assert.Equal(t, []string{"Single constraint string"}, probe.State.ImmutableConstraints)
	assert.Equal(t, []string{"Single decision"}, probe.State.DecisionsAndRejected)
	assert.Equal(t, []string{"Single next action"}, probe.State.NextIntendedActions)
	assert.Equal(t, []string{"Single blocker string"}, probe.State.UnresolvedBlockers)
	assert.Equal(t, []string{"Single artifact"}, probe.State.ProducedArtifacts)
	assert.Equal(t, []string{"v1.0"}, probe.State.ActiveSkillVersions)
	assert.Equal(t, []string{"m1.0"}, probe.State.ActiveMemoryVersions)

	// Test with null and empty string fields
	emptyJSON := `{
		"should_compact": false,
		"reason": "not yet",
		"state": {
			"unresolved_blockers": "",
			"next_intended_actions": null
		}
	}`
	var emptyProbe CompactionProbeResult
	err = json.Unmarshal([]byte(emptyJSON), &emptyProbe)
	require.NoError(t, err)
	require.NotNil(t, emptyProbe.State)
	assert.Nil(t, emptyProbe.State.UnresolvedBlockers)
	assert.Nil(t, emptyProbe.State.NextIntendedActions)
}
