package models

// Message represents a chat message.
type Message struct {
	Seq         int64        `json:"seq"`
	Timestamp   int64        `json:"timestamp"` // Unix timestamp (seconds)
	ChatID      string       `json:"chatId"`
	UserID      string       `json:"userId"`
	Content     string       `json:"content"`
	RawContent  string       `json:"rawContent,omitempty"`
	Attachments []Attachment `json:"attachments,omitempty"`
}

// ClientMessage represents a message sent from the bot to the Besedka server.
type ClientMessage struct {
	Type        ClientMessageType `json:"type"`
	ChatID      string            `json:"chatId,omitempty"`
	Content     string            `json:"content,omitempty"`
	Attachments []Attachment      `json:"attachments,omitempty"`
}

// ServerMessage represents an incoming event frame from the Besedka server.
type ServerMessage struct {
	Type     ServerMessageType `json:"type"`
	ChatID   string            `json:"chatId,omitempty"`
	Messages []Message         `json:"messages,omitempty"`
	User     User              `json:"user,omitempty"`
}

// User represents a user or bot in the system.
type User struct {
	ID          string `json:"id"`
	UserName    string `json:"userName"`
	DisplayName string `json:"displayName"`
	AvatarURL   string `json:"avatarUrl,omitempty"`
}

type AttachmentType string

const (
	AttachmentTypeImage AttachmentType = "image"
	AttachmentTypeFile  AttachmentType = "file"
)

type Attachment struct {
	Type     AttachmentType `json:"type"`
	Name     string         `json:"name"`
	MimeType string         `json:"mimeType"`
	FileID   string         `json:"fileId"`
}

type ClientMessageType string

const (
	ClientMessageTypeSend ClientMessageType = "send"
)

type ServerMessageType string

const (
	ServerMessageTypeMessages ServerMessageType = "messages"
)
