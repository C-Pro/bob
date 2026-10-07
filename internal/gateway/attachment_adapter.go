package gateway

import (
	"context"
	"errors"

	"bob/internal/agentapi"
	"bob/internal/models"
	"bob/internal/tools"
)

// AttachmentToAgentAPI converts a Besedka models.Attachment to an agentapi.Attachment.
func AttachmentToAgentAPI(att models.Attachment) agentapi.Attachment {
	var attType agentapi.AttachmentType
	switch att.Type {
	case models.AttachmentTypeImage:
		attType = agentapi.AttachmentImage
	default:
		attType = agentapi.AttachmentFile
	}
	return agentapi.Attachment{
		ID:       att.FileID,
		Type:     attType,
		Name:     att.Name,
		MIMEType: att.MimeType,
	}
}

// AttachmentFromAgentAPI converts an agentapi.Attachment to a Besedka models.Attachment.
func AttachmentFromAgentAPI(att agentapi.Attachment) models.Attachment {
	var attType models.AttachmentType
	switch att.Type {
	case agentapi.AttachmentImage:
		attType = models.AttachmentTypeImage
	default:
		attType = models.AttachmentTypeFile
	}
	return models.Attachment{
		Type:     attType,
		Name:     att.Name,
		MimeType: att.MIMEType,
		FileID:   att.ID,
	}
}

// AttachmentsToAgentAPI converts a slice of Besedka models.Attachment to []agentapi.Attachment.
func AttachmentsToAgentAPI(atts []models.Attachment) []agentapi.Attachment {
	if atts == nil {
		return nil
	}
	res := make([]agentapi.Attachment, len(atts))
	for i, a := range atts {
		res[i] = AttachmentToAgentAPI(a)
	}
	return res
}

// AttachmentsFromAgentAPI converts a slice of agentapi.Attachment to []models.Attachment.
func AttachmentsFromAgentAPI(atts []agentapi.Attachment) []models.Attachment {
	if atts == nil {
		return nil
	}
	res := make([]models.Attachment, len(atts))
	for i, a := range atts {
		res[i] = AttachmentFromAgentAPI(a)
	}
	return res
}

// Aliases for clear directionality
var ToBesedkaAttachments = AttachmentsFromAgentAPI
var FromBesedkaAttachments = AttachmentsToAgentAPI

// GatewayAttachmentAdapter adapts Gateway attachment operations to agentapi.AttachmentHandler.
type GatewayAttachmentAdapter struct {
	client tools.AttachmentClient
	chatID string
}

var _ agentapi.AttachmentHandler = (*GatewayAttachmentAdapter)(nil)

// NewGatewayAttachmentAdapter creates a new GatewayAttachmentAdapter.
func NewGatewayAttachmentAdapter(client tools.AttachmentClient, chatID string) *GatewayAttachmentAdapter {
	return &GatewayAttachmentAdapter{
		client: client,
		chatID: chatID,
	}
}

func (a *GatewayAttachmentAdapter) Download(ctx context.Context, att agentapi.Attachment) (agentapi.AttachmentContent, error) {
	if a.client == nil {
		return agentapi.AttachmentContent{}, errors.New("attachment client not configured")
	}
	data, mimeType, err := a.client.DownloadAttachment(ctx, att.ID)
	if err != nil {
		return agentapi.AttachmentContent{}, err
	}
	return agentapi.AttachmentContent{
		Type:     att.Type,
		Name:     att.Name,
		MIMEType: mimeType,
		Data:     data,
	}, nil
}

func (a *GatewayAttachmentAdapter) Upload(ctx context.Context, content agentapi.AttachmentContent) (agentapi.Attachment, error) {
	if a.client == nil {
		return agentapi.Attachment{}, errors.New("attachment client not configured")
	}
	var fileID string
	var err error
	if content.Type == agentapi.AttachmentImage {
		fileID, err = a.client.UploadImage(ctx, content.Data, content.Name, content.MIMEType)
	} else {
		fileID, err = a.client.UploadFile(ctx, content.Data, content.Name, content.MIMEType)
	}
	if err != nil {
		return agentapi.Attachment{}, err
	}
	return agentapi.Attachment{
		ID:       fileID,
		Type:     content.Type,
		Name:     content.Name,
		MIMEType: content.MIMEType,
	}, nil
}
