package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"bob/internal/agentapi"
	"bob/internal/models"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func hasToolNamed(defs []agentapi.ToolDefinition, name string) bool {
	for _, d := range defs {
		if d.Schema.Function != nil && d.Schema.Function.Name == name {
			return true
		}
	}
	return false
}

func hasOpenAIToolNamed(defs []agentapi.ToolDefinition, name string) bool {
	return hasToolNamed(defs, name)
}

func TestAskUser_ToolDefinitionAndExposure(t *testing.T) {
	reg := NewRegistry(nil, nil)

	// Static ToolDefinitions() remains 4 (CLI/default scope must never gain ask_user)
	assert.Len(t, reg.ToolDefinitions(), 4)
	assert.False(t, IsReadOnlyTool(AskUserToolName))

	// Non-interactive sessions (Townhall & DM) must NOT expose ask_user
	thNonInteractive := reg.ToolDefinitionsForSession(ChatSessionContext{
		ChatID:                "townhall",
		UserID:                "alice",
		IsDM:                  false,
		SupportsInteractiveUI: false,
	})
	assert.Len(t, thNonInteractive, 4)

	dmNonInteractive := reg.ToolDefinitionsForSession(ChatSessionContext{
		ChatID:                "dm_alice",
		UserID:                "alice",
		IsDM:                  true,
		SupportsInteractiveUI: false,
	})
	for _, tool := range dmNonInteractive {
		assert.NotEqual(t, AskUserToolName, tool.Function.Name)
	}

	// Scheduled runs NEVER expose ask_user even if SupportsInteractiveUI is somehow set
	scheduledDefs := reg.ToolDefinitionsForSession(ChatSessionContext{
		ChatID:                "dm_alice",
		UserID:                "alice",
		IsDM:                  true,
		IsScheduled:           true,
		SupportsInteractiveUI: true,
	})
	for _, tool := range scheduledDefs {
		assert.NotEqual(t, AskUserToolName, tool.Function.Name)
	}

	// Interactive Townhall & DM expose ask_user
	for _, tc := range []ChatSessionContext{
		{ChatID: "townhall", UserID: "alice", IsDM: false, SupportsInteractiveUI: true},
		{ChatID: "dm_alice", UserID: "alice", IsDM: true, SupportsInteractiveUI: true},
	} {
		defs := reg.ToolDefinitionsForSession(tc)
		found := false
		for _, tool := range defs {
			if tool.Function != nil && tool.Function.Name == AskUserToolName {
				found = true
				break
			}
		}
		assert.True(t, found, "expected ask_user in %+v", tc)
	}

	// Verify static slice was not mutated by appending
	assert.Len(t, reg.ToolDefinitions(), 4)

	// Verify SessionToolset.Definitions propagates SupportsInteractiveUI cleanly
	ts := reg.SessionToolset(ChatSessionContext{
		ChatID:                "townhall",
		UserID:                "alice",
		IsDM:                  false,
		SupportsInteractiveUI: true,
	})
	tsDefs, err := ts.Definitions(context.Background())
	require.NoError(t, err)
	assert.True(t, hasOpenAIToolNamed(tsDefs, AskUserToolName))
}

func TestExecuteAskUser_SuccessAndActions(t *testing.T) {
	reg := NewRegistry(nil, nil)

	sess := NewChatSessionContext("townhall", "alice", false)
	sess.SupportsInteractiveUI = true
	ctx := WithChatSession(context.Background(), sess)

	args := `{
		"message": "Choose rollout target and options:",
		"elements": [
			{"id": "us_east", "type": "radio", "label": "US East", "description": "Primary region", "default": true},
			{"id": "eu_west", "type": "radio", "label": "EU West"},
			{"id": "notify", "type": "checkbox", "label": "Notify team", "default": false},
			{"id": "deploy", "type": "button", "label": "Deploy Now", "style": "primary"}
		]
	}`

	res, err := reg.Execute(ctx, AskUserToolName, args)
	require.NoError(t, err)
	assert.Equal(t, `{"status":"prompt_queued","elements":4}`, res)

	acts := sess.Actions.All()
	require.Len(t, acts, 1)
	assert.Equal(t, agentapi.ActionInteractivePrompt, acts[0].Type)
	assert.Equal(t, "Choose rollout target and options:", acts[0].Content)

	var payload agentapi.InteractivePromptPayload
	require.NoError(t, json.Unmarshal(acts[0].Payload, &payload))
	assert.Equal(t, "Choose rollout target and options:", payload.Message)
	require.Len(t, payload.Elements, 4)
	assert.Equal(t, "us_east", payload.Elements[0].ID)
	assert.True(t, payload.Elements[0].Default)
	assert.Equal(t, agentapi.UIElementButton, payload.Elements[3].Type)
	assert.Equal(t, agentapi.UIElementStylePrimary, payload.Elements[3].Style)

	// Also verify SessionToolset.Execute returns ActionInteractivePrompt in ToolResult.Actions
	ts := reg.SessionToolset(ChatSessionContext{
		ChatID:                "dm_alice",
		UserID:                "alice",
		IsDM:                  true,
		SupportsInteractiveUI: true,
	})
	tsRes, err := ts.Execute(context.Background(), agentapi.ToolCall{
		Name:      AskUserToolName,
		Arguments: `{"message":"Proceed?","elements":[{"id":"yes","type":"button","label":"Yes","style":"primary"}]}`,
	})
	require.NoError(t, err)
	assert.Equal(t, `{"status":"prompt_queued","elements":1}`, tsRes.Content)
	require.Len(t, tsRes.Actions, 1)
	assert.Equal(t, agentapi.ActionInteractivePrompt, tsRes.Actions[0].Type)
}

func TestExecuteAskUser_ValidationAndSecurity(t *testing.T) {
	reg := NewRegistry(nil, nil)
	validArgs := `{"message":"Pick one","elements":[{"id":"opt_1","type":"button","label":"One"}]}`

	// Non-interactive session rejected
	nonInt := NewChatSessionContext("dm_alice", "alice", true)
	nonInt.SupportsInteractiveUI = false
	_, err := reg.Execute(WithChatSession(context.Background(), nonInt), AskUserToolName, validArgs)
	assert.ErrorContains(t, err, "only available in interactive UI sessions")

	// Scheduled session rejected even if SupportsInteractiveUI=true
	schedSess := NewChatSessionContext("dm_alice", "alice", true)
	schedSess.SupportsInteractiveUI = true
	schedSess.IsScheduled = true
	_, err = reg.Execute(WithChatSession(context.Background(), schedSess), AskUserToolName, validArgs)
	assert.ErrorContains(t, err, "only available in interactive UI sessions")

	// Nil Actions collector rejected
	nilActSess := ChatSessionContext{
		ChatID:                "dm_alice",
		UserID:                "alice",
		IsDM:                  true,
		SupportsInteractiveUI: true,
	}
	_, err = reg.Execute(WithChatSession(context.Background(), nilActSess), AskUserToolName, validArgs)
	assert.ErrorContains(t, err, "not configured")

	intSess := NewChatSessionContext("townhall", "alice", false)
	intSess.SupportsInteractiveUI = true
	ctx := WithChatSession(context.Background(), intSess)

	tooMany := make([]string, models.MaxUIElements+1)
	for i := range tooMany {
		tooMany[i] = fmt.Sprintf(`{"id":"opt_%d","type":"button","label":"Option %d"}`, i, i)
	}

	badPayloads := []struct {
		name    string
		args    string
		errText string
	}{
		{
			name:    "empty message",
			args:    `{"message":"  ","elements":[{"id":"a","type":"button","label":"A"}]}`,
			errText: "message cannot be empty",
		},
		{
			name:    "empty elements",
			args:    `{"message":"Hi","elements":[]}`,
			errText: "elements cannot be empty",
		},
		{
			name:    "reserved bob prefix",
			args:    `{"message":"Confirm","elements":[{"id":"bob:sandbox:approve:sbx-1","type":"button","label":"Approve"}]}`,
			errText: "reserved prefix",
		},
		{
			name:    "whitespace padded reserved bob prefix",
			args:    `{"message":"Confirm","elements":[{"id":"  bob:mem:approve:mem_1@1  ","type":"button","label":"Approve"}]}`,
			errText: "reserved prefix",
		},
		{
			name:    "button default true rejected",
			args:    `{"message":"Confirm","elements":[{"id":"ok","type":"button","label":"OK","default":true}]}`,
			errText: "cannot have default state",
		},
		{
			name:    "multiple default radios rejected",
			args:    `{"message":"Pick","elements":[{"id":"r1","type":"radio","label":"R1","default":true},{"id":"r2","type":"radio","label":"R2","default":true}]}`,
			errText: "at most one radio",
		},
		{
			name:    "duplicate ids rejected",
			args:    `{"message":"Pick","elements":[{"id":"dup","type":"button","label":"One"},{"id":"dup","type":"button","label":"Two"}]}`,
			errText: "duplicate",
		},
		{
			name:    "too many elements rejected",
			args:    fmt.Sprintf(`{"message":"Pick","elements":[%s]}`, strings.Join(tooMany, ",")),
			errText: "50",
		},
		{
			name:    "invalid style rejected",
			args:    `{"message":"Pick","elements":[{"id":"b1","type":"button","label":"B1","style":"neon"}]}`,
			errText: "style",
		},
	}

	for _, tc := range badPayloads {
		t.Run(tc.name, func(t *testing.T) {
			_, err := reg.Execute(ctx, AskUserToolName, tc.args)
			require.Error(t, err)
			assert.Contains(t, err.Error(), tc.errText)
		})
	}
	assert.Empty(t, intSess.Actions.All())
}
