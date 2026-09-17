package fsm

import (
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
}
