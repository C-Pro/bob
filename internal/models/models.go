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

// Location represents geographical coordinates.
type Location struct {
	Lat float64 `json:"lat"`
	Lng float64 `json:"lng"`
}

// ClientMessage represents a message sent from the bot to the Besedka server.
type ClientMessage struct {
	Type        ClientMessageType `json:"type"`
	ChatID      string            `json:"chatId,omitempty"`
	Content     string            `json:"content,omitempty"`
	Attachments []Attachment      `json:"attachments,omitempty"`
	Location    *Location         `json:"location,omitempty"`
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
	UserName    string `json:"userName,omitempty"`
	Name        string `json:"name,omitempty"`
	DisplayName string `json:"displayName,omitempty"`
	AvatarURL   string `json:"avatarUrl,omitempty"`
}

// GetDisplayName returns the user's DisplayName or Name, falling back to UserName if empty.
func (u User) GetDisplayName() string {
	if u.DisplayName != "" {
		return u.DisplayName
	}
	if u.Name != "" {
		return u.Name
	}
	if u.UserName != "" {
		return u.UserName
	}
	return ""
}

// GetUserName returns the username handle if available.
func (u User) GetUserName() string {
	if u.UserName != "" {
		return u.UserName
	}
	if u.Name != "" {
		return u.Name
	}
	return ""
}

// Chat represents a Besedka chat or channel.
type Chat struct {
	ID           string   `json:"id"`
	Name         string   `json:"name,omitempty"`
	Type         string   `json:"type,omitempty"` // "townhall", "dm", etc.
	LastSeq      int      `json:"lastSeq,omitempty"`
	IsDM         bool     `json:"isDm,omitempty"`
	LastSeenSeq  int64    `json:"lastSeenSeq,omitempty"`
	UserIDs      []string `json:"userIds,omitempty"`
	TargetUserID string   `json:"targetUserId,omitempty"`
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
	ClientMessageTypeSend     ClientMessageType = "send"
	ClientMessageTypeLocation ClientMessageType = "location"
)

type ServerMessageType string

const (
	ServerMessageTypeMessages ServerMessageType = "messages"
)
