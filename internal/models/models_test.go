package models

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestJSONSerialization(t *testing.T) {
	clientMsg := ClientMessage{
		Type:    ClientMessageTypeSend,
		ChatID:  "townhall",
		Content: "Hello world",
	}

	data, err := json.Marshal(clientMsg)
	require.NoError(t, err)

	var decoded ClientMessage
	err = json.Unmarshal(data, &decoded)
	require.NoError(t, err)
	assert.Equal(t, clientMsg.Type, decoded.Type)
	assert.Equal(t, clientMsg.ChatID, decoded.ChatID)
	assert.Equal(t, clientMsg.Content, decoded.Content)

	serverMsgJSON := `{
		"type": "messages",
		"chatId": "townhall",
		"messages": [
			{
				"seq": 1,
				"timestamp": 1700000000,
				"chatId": "townhall",
				"userId": "user-123",
				"content": "Test message"
			}
		]
	}`

	var sMsg ServerMessage
	err = json.Unmarshal([]byte(serverMsgJSON), &sMsg)
	require.NoError(t, err)
	assert.Equal(t, ServerMessageTypeMessages, sMsg.Type)
	assert.Equal(t, "townhall", sMsg.ChatID)
	require.Len(t, sMsg.Messages, 1)
	assert.Equal(t, "Test message", sMsg.Messages[0].Content)

	// Test Ping and Pong messages
	pingJSON := `{"type":"ping"}`
	var pingMsg ServerMessage
	err = json.Unmarshal([]byte(pingJSON), &pingMsg)
	require.NoError(t, err)
	assert.Equal(t, ServerMessageTypePing, pingMsg.Type)

	pongMsg := ClientMessage{Type: ClientMessageTypePong}
	pongData, err := json.Marshal(pongMsg)
	require.NoError(t, err)
	assert.JSONEq(t, `{"type":"pong"}`, string(pongData))
}

func TestLocationJSONSerialization(t *testing.T) {
	locMsg := ClientMessage{
		Type: ClientMessageTypeLocation,
		Location: &Location{
			Lat: 37.7749,
			Lng: -122.4194,
		},
	}

	data, err := json.Marshal(locMsg)
	require.NoError(t, err)

	expectedJSON := `{"type":"location","location":{"lat":37.7749,"lng":-122.4194}}`
	assert.JSONEq(t, expectedJSON, string(data))

	var decoded ClientMessage
	err = json.Unmarshal(data, &decoded)
	require.NoError(t, err)
	assert.Equal(t, ClientMessageTypeLocation, decoded.Type)
	require.NotNil(t, decoded.Location)
	assert.Equal(t, 37.7749, decoded.Location.Lat)
	assert.Equal(t, -122.4194, decoded.Location.Lng)
}

func TestUserHelpers(t *testing.T) {
	u1 := User{ID: "u1", DisplayName: "Alice Smith", UserName: "alice", Name: "Alice"}
	assert.Equal(t, "Alice Smith", u1.GetDisplayName())
	assert.Equal(t, "alice", u1.GetUserName())

	u2 := User{ID: "u2", Name: "Bob"}
	assert.Equal(t, "Bob", u2.GetDisplayName())
	assert.Equal(t, "Bob", u2.GetUserName())

	u3 := User{ID: "u3"}
	assert.Equal(t, "", u3.GetDisplayName())
	assert.Equal(t, "", u3.GetUserName())

	u4 := User{ID: "u4", UserName: "charlie"}
	assert.Equal(t, "charlie", u4.GetDisplayName())
	assert.Equal(t, "charlie", u4.GetUserName())
}

func TestUserTimeZoneAndLanguageSerialization(t *testing.T) {
	raw := `{
		"id": "u1",
		"displayName": "Alice",
		"timeZone": "America/New_York",
		"preferredLanguage": "en-US"
	}`
	var u User
	err := json.Unmarshal([]byte(raw), &u)
	require.NoError(t, err)
	assert.Equal(t, "u1", u.ID)
	assert.Equal(t, "Alice", u.GetDisplayName())
	assert.Equal(t, "America/New_York", u.TimeZone)
	assert.Equal(t, "en-US", u.PreferredLanguage)

	marshaled, err := json.Marshal(u)
	require.NoError(t, err)
	assert.Contains(t, string(marshaled), `"timeZone":"America/New_York"`)
	assert.Contains(t, string(marshaled), `"preferredLanguage":"en-US"`)
}

func TestProgressMessageSerialization(t *testing.T) {
	// Root progress message
	rootJSON := `{
		"seq": 100,
		"timestamp": 1700000000,
		"chatId": "townhall",
		"userId": "bot-1",
		"type": "progress",
		"progress": {
			"cardStatus": "running",
			"title": "Executing task...",
			"steps": [
				{
					"id": "step-1",
					"title": "Search web",
					"description": "Searching for go documentation",
					"status": "running"
				}
			]
		}
	}`

	var rootMsg Message
	err := json.Unmarshal([]byte(rootJSON), &rootMsg)
	require.NoError(t, err)
	assert.Equal(t, MessageTypeProgress, rootMsg.Type)
	require.NotNil(t, rootMsg.Progress)
	assert.Equal(t, ProgressStatusRunning, rootMsg.Progress.CardStatus)
	assert.Equal(t, "Executing task...", rootMsg.Progress.Title)
	require.Len(t, rootMsg.Progress.Steps, 1)
	assert.Equal(t, "step-1", rootMsg.Progress.Steps[0].ID)
	assert.Equal(t, ProgressStatusRunning, rootMsg.Progress.Steps[0].Status)

	// Child step message
	childJSON := `{
		"seq": 101,
		"timestamp": 1700000005,
		"chatId": "townhall",
		"userId": "bot-1",
		"type": "progress",
		"progress": {
			"parentSeq": 100,
			"step": {
				"id": "step-1",
				"title": "Search web",
				"status": "completed"
			}
		}
	}`

	var childMsg Message
	err = json.Unmarshal([]byte(childJSON), &childMsg)
	require.NoError(t, err)
	assert.Equal(t, int64(100), childMsg.Progress.ParentSeq)
	require.NotNil(t, childMsg.Progress.Step)
	assert.Equal(t, ProgressStatusCompleted, childMsg.Progress.Step.Status)

	// Status-only child message
	statusOnlyJSON := `{
		"type": "progress",
		"progress": {
			"parentSeq": 100,
			"cardStatus": "completed"
		}
	}`

	var statusMsg Message
	err = json.Unmarshal([]byte(statusOnlyJSON), &statusMsg)
	require.NoError(t, err)
	assert.Equal(t, int64(100), statusMsg.Progress.ParentSeq)
	assert.Equal(t, ProgressStatusCompleted, statusMsg.Progress.CardStatus)
	assert.Nil(t, statusMsg.Progress.Step)

	// ClientMessage with progress
	clientMsg := ClientMessage{
		Type:        ClientMessageTypeSend,
		ChatID:      "dm_1",
		MessageType: MessageTypeProgress,
		Progress: &ProgressData{
			Title:      "Initial work",
			CardStatus: ProgressStatusRunning,
		},
	}
	clientData, err := json.Marshal(clientMsg)
	require.NoError(t, err)
	assert.Contains(t, string(clientData), `"messageType":"progress"`)
	assert.Contains(t, string(clientData), `"title":"Initial work"`)
}

func TestUIMessageSerialization(t *testing.T) {
	legacyMsg := Message{
		Seq:       1,
		Timestamp: 1700000000,
		ChatID:    "townhall",
		UserID:    "user-1",
		Content:   "Plain message",
	}
	legacyBytes, err := json.Marshal(legacyMsg)
	require.NoError(t, err)
	assert.NotContains(t, string(legacyBytes), `"ui"`)
	assert.NotContains(t, string(legacyBytes), `"uiResponse"`)

	promptMsgJSON := `{
		"seq": 10,
		"timestamp": 1700000010,
		"chatId": "townhall",
		"userId": "bot-1",
		"type": "text",
		"content": "Deploy **v1.2.3**?",
		"ui": {
			"elements": [
				{
					"id": "notify",
					"type": "checkbox",
					"label": "Notify team",
					"description": "Post in #ops",
					"default": true
				},
				{
					"id": "approve",
					"type": "button",
					"label": "Approve",
					"style": "primary"
				},
				{
					"id": "deny",
					"type": "button",
					"label": "Deny",
					"style": "danger"
				}
			],
			"answered": true,
			"selectedIds": ["notify", "approve"],
			"responderId": "alice"
		}
	}`

	var promptMsg Message
	require.NoError(t, json.Unmarshal([]byte(promptMsgJSON), &promptMsg))
	assert.Equal(t, MessageTypeText, promptMsg.Type)
	require.NotNil(t, promptMsg.UI)
	assert.True(t, promptMsg.UI.Answered)
	assert.Equal(t, []string{"notify", "approve"}, promptMsg.UI.SelectedIDs)
	assert.Equal(t, "alice", promptMsg.UI.ResponderID)
	require.Len(t, promptMsg.UI.Elements, 3)
	assert.Equal(t, UIElementTypeCheckbox, promptMsg.UI.Elements[0].Type)
	assert.True(t, promptMsg.UI.Elements[0].Default)
	assert.Equal(t, UIElementStylePrimary, promptMsg.UI.Elements[1].Style)
	assert.Equal(t, UIElementStyleDanger, promptMsg.UI.Elements[2].Style)

	respMsgJSON := `{
		"seq": 11,
		"timestamp": 1700000015,
		"chatId": "townhall",
		"userId": "alice",
		"type": "ui_response",
		"content": "@bob Selected: Approve",
		"uiResponse": {
			"parentSeq": 10,
			"selectedIds": ["notify", "approve"]
		}
	}`

	var respMsg Message
	require.NoError(t, json.Unmarshal([]byte(respMsgJSON), &respMsg))
	assert.Equal(t, MessageTypeUIResponse, respMsg.Type)
	require.NotNil(t, respMsg.UIResponse)
	assert.Equal(t, int64(10), respMsg.UIResponse.ParentSeq)
	assert.Equal(t, []string{"notify", "approve"}, respMsg.UIResponse.SelectedIDs)

	clientPrompt := ClientMessage{
		Type:        ClientMessageTypeSend,
		ChatID:      "dm_1",
		Content:     "Choose region:",
		MessageType: MessageTypeText,
		UI: &UIData{
			Elements: []UIElement{
				{ID: "us", Type: UIElementTypeRadio, Label: "US", Default: true},
				{ID: "eu", Type: UIElementTypeRadio, Label: "EU"},
			},
		},
	}
	clientBytes, err := json.Marshal(clientPrompt)
	require.NoError(t, err)
	var decodedClient ClientMessage
	require.NoError(t, json.Unmarshal(clientBytes, &decodedClient))
	assert.Equal(t, MessageTypeText, decodedClient.MessageType)
	require.NotNil(t, decodedClient.UI)
	require.Len(t, decodedClient.UI.Elements, 2)
	assert.True(t, decodedClient.UI.Elements[0].Default)
}

func TestValidateAndSanitizeUIData(t *testing.T) {
	tooMany := make([]UIElement, MaxUIElements+1)
	for i := range tooMany {
		tooMany[i] = UIElement{
			ID:    strings.Repeat("a", 2) + string(rune('a'+i%26)) + string(rune('0'+i/26)),
			Type:  UIElementTypeButton,
			Label: "Opt",
		}
	}

	tests := []struct {
		name    string
		ui      *UIData
		wantErr string
	}{
		{
			name:    "nil ui",
			ui:      nil,
			wantErr: "cannot be nil",
		},
		{
			name:    "empty elements",
			ui:      &UIData{Elements: []UIElement{}},
			wantErr: "invalid ui elements count",
		},
		{
			name:    "too many elements",
			ui:      &UIData{Elements: tooMany},
			wantErr: "invalid ui elements count",
		},
		{
			name: "empty id",
			ui: &UIData{Elements: []UIElement{
				{ID: "   ", Type: UIElementTypeButton, Label: "OK"},
			}},
			wantErr: "ID cannot be empty",
		},
		{
			name: "id too long",
			ui: &UIData{Elements: []UIElement{
				{ID: strings.Repeat("x", MaxUIElementIDBytes+1), Type: UIElementTypeButton, Label: "OK"},
			}},
			wantErr: "exceeds 64 bytes",
		},
		{
			name: "duplicate id after trim",
			ui: &UIData{Elements: []UIElement{
				{ID: "ok", Type: UIElementTypeButton, Label: "OK"},
				{ID: " ok ", Type: UIElementTypeButton, Label: "Also OK"},
			}},
			wantErr: "duplicate ui element ID",
		},
		{
			name: "empty label",
			ui: &UIData{Elements: []UIElement{
				{ID: "ok", Type: UIElementTypeButton, Label: " \t "},
			}},
			wantErr: "label cannot be empty",
		},
		{
			name: "label too long",
			ui: &UIData{Elements: []UIElement{
				{ID: "ok", Type: UIElementTypeButton, Label: strings.Repeat("a", MaxUIElementLabelBytes+1)},
			}},
			wantErr: "label exceeds 255 bytes",
		},
		{
			name: "description too long",
			ui: &UIData{Elements: []UIElement{
				{ID: "ok", Type: UIElementTypeButton, Label: "OK", Description: strings.Repeat("d", MaxUIElementDescBytes+1)},
			}},
			wantErr: "description exceeds 1024 bytes",
		},
		{
			name: "invalid type",
			ui: &UIData{Elements: []UIElement{
				{ID: "ok", Type: UIElementType("slider"), Label: "Slide"},
			}},
			wantErr: "invalid ui element type",
		},
		{
			name: "invalid style",
			ui: &UIData{Elements: []UIElement{
				{ID: "ok", Type: UIElementTypeButton, Label: "OK", Style: UIElementStyle("neon")},
			}},
			wantErr: "invalid ui element style",
		},
		{
			name: "button with default true",
			ui: &UIData{Elements: []UIElement{
				{ID: "ok", Type: UIElementTypeButton, Label: "OK", Default: true},
			}},
			wantErr: "button element cannot have default state",
		},
		{
			name: "multiple default radios",
			ui: &UIData{Elements: []UIElement{
				{ID: "r1", Type: UIElementTypeRadio, Label: "One", Default: true},
				{ID: "r2", Type: UIElementTypeRadio, Label: "Two", Default: true},
			}},
			wantErr: "at most one radio element can have default state",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateUIData(tt.ui)
			require.Error(t, err)
			assert.Contains(t, err.Error(), tt.wantErr)

			sanitized, sErr := SanitizeUIDataForSend(tt.ui)
			assert.Error(t, sErr)
			assert.Nil(t, sanitized)
		})
	}

	input := &UIData{
		Elements: []UIElement{
			{ID: " cb1 ", Type: UIElementTypeCheckbox, Label: " Notify ", Description: " Send alert ", Default: true},
			{ID: " r1 ", Type: UIElementTypeRadio, Label: " US ", Default: true},
			{ID: " btn ", Type: UIElementTypeButton, Label: " Go ", Style: UIElementStylePrimary},
		},
		Answered:    true,
		SelectedIDs: []string{"cb1"},
		ResponderID: "alice",
	}

	sanitized, err := SanitizeUIDataForSend(input)
	require.NoError(t, err)
	require.NotNil(t, sanitized)

	assert.False(t, sanitized.Answered)
	assert.Nil(t, sanitized.SelectedIDs)
	assert.Empty(t, sanitized.ResponderID)
	assert.Equal(t, "cb1", sanitized.Elements[0].ID)
	assert.Equal(t, "Notify", sanitized.Elements[0].Label)
	assert.Equal(t, "Send alert", sanitized.Elements[0].Description)
	assert.Equal(t, "btn", sanitized.Elements[2].ID)

	// Input must remain unmutated
	assert.True(t, input.Answered)
	assert.Equal(t, []string{"cb1"}, input.SelectedIDs)
	assert.Equal(t, "alice", input.ResponderID)
	assert.Equal(t, " cb1 ", input.Elements[0].ID)
}

func TestUIContextHelpers(t *testing.T) {
	assert.True(t, HasReservedUIActionPrefix("bob:sandbox_approve"))
	assert.True(t, HasReservedUIActionPrefix("  BOB:memory_reject:mem_1@1 "))
	assert.False(t, HasReservedUIActionPrefix("approve"))
	assert.False(t, HasReservedUIActionPrefix("custom_bob:test"))

	assert.Equal(t, "", FormatUIForContext(nil))
	assert.Equal(t, "", FormatUIForContext(&UIData{}))

	unanswered := &UIData{
		Elements: []UIElement{
			{ID: "notify", Type: UIElementTypeCheckbox, Label: "Notify Slack", Default: true},
			{ID: "approve", Type: UIElementTypeButton, Label: "Approve", Description: "Deploy now", Style: UIElementStylePrimary},
		},
	}
	answeredCopy := &UIData{
		Elements:    unanswered.Elements,
		Answered:    true,
		SelectedIDs: []string{"notify", "approve"},
		ResponderID: "alice",
	}

	formattedUnanswered := FormatUIForContext(unanswered)
	formattedAnswered := FormatUIForContext(answeredCopy)
	assert.Equal(t, formattedUnanswered, formattedAnswered)
	assert.Equal(t, `[Interactive UI options: checkbox "Notify Slack" (id=notify, default); button "Approve" (id=approve): Deploy now]`, formattedUnanswered)

	assert.Equal(t, "hello", FormatUIResponseForContext(" hello ", nil))
	assert.Equal(t, "[UI response to seq 42 | selected_ids: none]", FormatUIResponseForContext("", &UIResponseData{ParentSeq: 42}))
	assert.Equal(
		t,
		"@bob Selected: Approve\n[UI response to seq 42 | selected_ids: notify, approve]",
		FormatUIResponseForContext(" @bob Selected: Approve ", &UIResponseData{ParentSeq: 42, SelectedIDs: []string{"notify", "approve"}}),
	)
}


