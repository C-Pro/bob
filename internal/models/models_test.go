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
}

func TestUserHelpers(t *testing.T) {
	u1 := User{ID: "u1", DisplayName: "Alice Smith", UserName: "alice", Name: "Alice"}
	assert.Equal(t, "Alice Smith", u1.GetDisplayName())
	assert.Equal(t, "alice", u1.GetUserName())

	u2 := User{ID: "u2", Name: "Bob"}
	assert.Equal(t, "Bob", u2.GetDisplayName())
	assert.Equal(t, "Bob", u2.GetUserName())

	u3 := User{ID: "u3"}
	assert.Equal(t, "u3", u3.GetDisplayName())
	assert.Equal(t, "u3", u3.GetUserName())
}
