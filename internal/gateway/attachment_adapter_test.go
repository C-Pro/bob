package gateway_test

import (
	"testing"

	"bob/internal/agentapi"
	"bob/internal/gateway"
	"bob/internal/models"

	"github.com/stretchr/testify/assert"
)

func TestAttachmentConversion_RoundTrip(t *testing.T) {
	origModels := []models.Attachment{
		{
			Type:     models.AttachmentTypeImage,
			Name:     "photo.png",
			MimeType: "image/png",
			FileID:   "file-101",
		},
		{
			Type:     models.AttachmentTypeFile,
			Name:     "doc.pdf",
			MimeType: "application/pdf",
			FileID:   "file-102",
		},
	}

	agentAPIAtts := gateway.AttachmentsToAgentAPI(origModels)
	assert.Len(t, agentAPIAtts, 2)

	assert.Equal(t, "file-101", agentAPIAtts[0].ID)
	assert.Equal(t, agentapi.AttachmentImage, agentAPIAtts[0].Type)
	assert.Equal(t, "photo.png", agentAPIAtts[0].Name)
	assert.Equal(t, "image/png", agentAPIAtts[0].MIMEType)

	assert.Equal(t, "file-102", agentAPIAtts[1].ID)
	assert.Equal(t, agentapi.AttachmentFile, agentAPIAtts[1].Type)
	assert.Equal(t, "doc.pdf", agentAPIAtts[1].Name)
	assert.Equal(t, "application/pdf", agentAPIAtts[1].MIMEType)

	roundTrip := gateway.AttachmentsFromAgentAPI(agentAPIAtts)
	assert.Equal(t, origModels, roundTrip)
}

func TestAttachmentConversion_NilHandling(t *testing.T) {
	assert.Nil(t, gateway.AttachmentsToAgentAPI(nil))
	assert.Nil(t, gateway.AttachmentsFromAgentAPI(nil))
}

func TestAttachmentConversion_DefaultFallbacks(t *testing.T) {
	// Unknown model type defaults to file
	m := models.Attachment{
		Type:     "unknown",
		Name:     "custom.dat",
		MimeType: "application/octet-stream",
		FileID:   "file-custom",
	}
	apiAtt := gateway.AttachmentToAgentAPI(m)
	assert.Equal(t, agentapi.AttachmentFile, apiAtt.Type)

	// Unknown agentapi type defaults to file
	customAPIAtt := agentapi.Attachment{
		ID:       "file-other",
		Type:     "unknown_type",
		Name:     "other.dat",
		MIMEType: "text/plain",
	}
	mBack := gateway.AttachmentFromAgentAPI(customAPIAtt)
	assert.Equal(t, models.AttachmentTypeFile, mBack.Type)
}
