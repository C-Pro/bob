package agentapi_test

import (
	"encoding/json"
	"testing"

	"bob/internal/agentapi"

	openai "github.com/sashabaranov/go-openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSessionRef_Serialization(t *testing.T) {
	ref := agentapi.SessionRef{
		FrontendID: "besedka",
		SessionID:  "chat-123",
		ScopeID:    "dm",
	}

	data, err := json.Marshal(ref)
	require.NoError(t, err)

	var decoded agentapi.SessionRef
	err = json.Unmarshal(data, &decoded)
	require.NoError(t, err)

	assert.Equal(t, ref, decoded)
}

func TestRunDescriptor_Serialization(t *testing.T) {
	desc := agentapi.RunDescriptor{
		RunID: "run-456",
		Session: agentapi.SessionRef{
			FrontendID: "besedka",
			SessionID:  "chat-123",
			ScopeID:    "townhall",
		},
		Actor: agentapi.Actor{
			ID: "user-789",
		},
		Kind:          agentapi.Interactive,
		Model:         "gpt-4o",
		MaxIterations: 10,
	}

	data, err := json.Marshal(desc)
	require.NoError(t, err)

	var decoded agentapi.RunDescriptor
	err = json.Unmarshal(data, &decoded)
	require.NoError(t, err)

	assert.Equal(t, desc, decoded)
}

func TestTurnAndResult_Serialization(t *testing.T) {
	actionPayload := map[string]string{"foo": "bar"}
	payloadBytes, err := json.Marshal(actionPayload)
	require.NoError(t, err)

	turn := agentapi.Turn{
		Run: agentapi.RunDescriptor{
			RunID: "run-1",
			Session: agentapi.SessionRef{
				FrontendID: "cli",
				SessionID:  "local",
				ScopeID:    "stdout",
			},
			Actor:         agentapi.Actor{ID: "local-user"},
			Kind:          agentapi.Interactive,
			Model:         "gpt-4o",
			MaxIterations: 5,
		},
		SystemPrompt: "You are a helpful assistant.",
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleUser, Content: "Hello"},
		},
	}

	turnData, err := json.Marshal(turn)
	require.NoError(t, err)
	var decodedTurn agentapi.Turn
	require.NoError(t, json.Unmarshal(turnData, &decodedTurn))
	assert.Equal(t, turn.Run.RunID, decodedTurn.Run.RunID)
	assert.Equal(t, turn.SystemPrompt, decodedTurn.SystemPrompt)
	assert.Len(t, decodedTurn.Messages, 1)

	res := agentapi.Result{
		RunID:      "run-1",
		Content:    "Hello back!",
		Iterations: 1,
		Attachments: []agentapi.Attachment{
			{
				ID:       "att-1",
				Type:     agentapi.AttachmentFile,
				Name:     "notes.txt",
				MIMEType: "text/plain",
			},
		},
		Actions: []agentapi.Action{
			{
				ID:      "act-1",
				Type:    agentapi.ActionApprovalRequest,
				Content: "Requesting approval",
				Payload: payloadBytes,
			},
		},
	}

	resData, err := json.Marshal(res)
	require.NoError(t, err)
	var decodedRes agentapi.Result
	require.NoError(t, json.Unmarshal(resData, &decodedRes))
	assert.Equal(t, res.RunID, decodedRes.RunID)
	assert.Equal(t, res.Content, decodedRes.Content)
	assert.Len(t, decodedRes.Attachments, 1)
	assert.Equal(t, agentapi.AttachmentFile, decodedRes.Attachments[0].Type)
	assert.Len(t, decodedRes.Actions, 1)
	assert.Equal(t, agentapi.ActionApprovalRequest, decodedRes.Actions[0].Type)
}

func TestProgressEvent_Serialization(t *testing.T) {
	step := agentapi.StepProgress{
		ID:        "step-1",
		Iteration: 1,
		StepIndex: 0,
		Call: agentapi.ToolCall{
			ID:        "call-1",
			Name:      "sandbox_exec",
			Arguments: `{"command":"ls"}`,
		},
		Status:  agentapi.StepRunning,
		Attempt: 1,
	}

	event := agentapi.ProgressEvent{
		RunID:  "run-1",
		Type:   agentapi.ProgressStepUpdated,
		Step:   &step,
		Status: agentapi.RunRunning,
	}

	data, err := json.Marshal(event)
	require.NoError(t, err)

	var decoded agentapi.ProgressEvent
	require.NoError(t, json.Unmarshal(data, &decoded))
	assert.Equal(t, event.RunID, decoded.RunID)
	assert.Equal(t, event.Type, decoded.Type)
	require.NotNil(t, decoded.Step)
	assert.Equal(t, "sandbox_exec", decoded.Step.Call.Name)
}
