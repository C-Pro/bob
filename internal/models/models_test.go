package models

import (
	"encoding/json"
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


