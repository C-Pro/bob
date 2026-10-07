package agentapi

import "context"

// AttachmentType classifies the format of an attachment.
type AttachmentType string

const (
	AttachmentImage AttachmentType = "image"
	AttachmentFile  AttachmentType = "file"
)

// Attachment represents an opaque reference to a staged or persisted file/image attachment.
type Attachment struct {
	ID       string         `json:"id"`
	Type     AttachmentType `json:"type"`
	Name     string         `json:"name"`
	MIMEType string         `json:"mime_type"`
}

// AttachmentContent contains raw bytes and metadata for uploading or downloading an attachment.
type AttachmentContent struct {
	Type     AttachmentType `json:"type"`
	Name     string         `json:"name"`
	MIMEType string         `json:"mime_type"`
	Data     []byte         `json:"-"`
}

// AttachmentHandler facilitates downloading and uploading attachments for a specific frontend.
type AttachmentHandler interface {
	Download(context.Context, Attachment) (AttachmentContent, error)
	Upload(context.Context, AttachmentContent) (Attachment, error)
}
