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
