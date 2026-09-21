package tools

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"syscall"

	"bob/internal/config"
	"bob/internal/models"
	"bob/internal/sandbox"

	openai "github.com/sashabaranov/go-openai"
)

// AttachmentClient defines the client interface for downloading and uploading chat attachments.
type AttachmentClient interface {
	DownloadAttachment(ctx context.Context, fileID string) ([]byte, string, error)
	UploadFile(ctx context.Context, data []byte, filename, mimeType string) (string, error)
	UploadImage(ctx context.Context, data []byte, filename, mimeType string) (string, error)
}

// StagedAttachmentCollector collects attachments staged during tool execution for a chat response.
type StagedAttachmentCollector struct {
	mu    sync.Mutex
	items []models.Attachment
}

// NewStagedAttachmentCollector creates a new empty attachment collector.
func NewStagedAttachmentCollector() *StagedAttachmentCollector {
	return &StagedAttachmentCollector{}
}

// MaxStagedAttachments defines the maximum number of attachments that can be staged per response.
const MaxStagedAttachments = 10

// Add appends an attachment in a thread-safe manner. Returns an error if the staging limit is reached.
func (c *StagedAttachmentCollector) Add(att models.Attachment) error {
	if c == nil {
		return errors.New("attachment collector is nil")
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if len(c.items) >= MaxStagedAttachments {
		return fmt.Errorf("cannot stage more than %d attachments per response", MaxStagedAttachments)
	}
	c.items = append(c.items, att)
	return nil
}

// All returns a copy of all staged attachments.
func (c *StagedAttachmentCollector) All() []models.Attachment {
	if c == nil {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	res := make([]models.Attachment, len(c.items))
	copy(res, c.items)
	return res
}

// SandboxDownloadAttachmentArgs defines arguments for sandbox_download_attachment tool.
type SandboxDownloadAttachmentArgs struct {
	FileID          string `json:"file_id"`
	DestinationPath string `json:"destination_path"`
}

// SandboxUploadAttachmentArgs defines arguments for sandbox_upload_attachment tool.
type SandboxUploadAttachmentArgs struct {
	SourcePath string `json:"source_path"`
	Name       string `json:"name,omitempty"`
}

func formatAttachmentLimit(bytes int64) string {
	if bytes <= 0 {
		return "0B"
	}
	const (
		mib = 1024 * 1024
		kib = 1024
	)
	if bytes%mib == 0 {
		return fmt.Sprintf("%dMB", bytes/mib)
	}
	if bytes >= mib {
		val := float64(bytes) / float64(mib)
		return strings.TrimRight(strings.TrimRight(fmt.Sprintf("%.1f", val), "0"), ".") + "MB"
	}
	if bytes%kib == 0 {
		return fmt.Sprintf("%dKB", bytes/kib)
	}
	if bytes >= kib {
		val := float64(bytes) / float64(kib)
		return strings.TrimRight(strings.TrimRight(fmt.Sprintf("%.1f", val), "0"), ".") + "KB"
	}
	return fmt.Sprintf("%dB", bytes)
}

// attachmentToolDefinitions returns the OpenAI tool definitions for sandbox attachment operations.
func (r *Registry) attachmentToolDefinitions() []openai.Tool {
	maxSizeBytes := r.maxAttachmentSizeBytes
	if maxSizeBytes <= 0 {
		maxSizeBytes = config.DefaultMaxAttachmentSizeBytes
	}
	genericLimitStr := formatAttachmentLimit(maxSizeBytes)

	const maxImageLimit = 10 * 1024 * 1024
	effectiveImageLimit := maxSizeBytes
	if effectiveImageLimit > maxImageLimit {
		effectiveImageLimit = maxImageLimit
	}
	imageLimitStr := formatAttachmentLimit(effectiveImageLimit)

	downloadSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"file_id": map[string]interface{}{
				"type":        "string",
				"description": "The file ID of the attachment to download from chat messages into the sandbox.",
			},
			"destination_path": map[string]interface{}{
				"type":        "string",
				"description": "Relative file path inside the sandbox workspace where the downloaded file will be saved (e.g. 'data/report.csv' or 'input.txt').",
			},
		},
		"required": []string{"file_id", "destination_path"},
	}

	uploadSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"source_path": map[string]interface{}{
				"type":        "string",
				"description": "Relative file path inside the sandbox workspace of the file to upload (e.g. 'output/results.png' or 'chart.pdf').",
			},
			"name": map[string]interface{}{
				"type":        "string",
				"description": "Optional display filename for the attachment in chat (defaults to the basename of source_path).",
			},
		},
		"required": []string{"source_path"},
	}

	return []openai.Tool{
		{
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{
				Name:        "sandbox_download_attachment",
				Description: "Download a file or image attachment from the chat conversation into your sandbox workspace. Only available in 1-on-1 direct messages with an active sandbox.",
				Parameters:  downloadSchema,
			},
		},
		{
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{
				Name:        "sandbox_upload_attachment",
				Description: fmt.Sprintf("Upload a file from your sandbox workspace to be attached to your response message in chat. Max file size is %s (%s for images). Only available in 1-on-1 direct messages with an active sandbox.", genericLimitStr, imageLimitStr),
				Parameters:  uploadSchema,
			},
		},
	}
}

func validateWorkspaceRelPath(relPath string) (string, error) {
	trimmedRel := strings.TrimSpace(relPath)
	if trimmedRel == "" {
		return "", errors.New("path cannot be empty")
	}
	if strings.ContainsRune(relPath, '\x00') {
		return "", fmt.Errorf("path cannot contain null bytes: %q", relPath)
	}

	cleanRel := filepath.Clean(trimmedRel)
	if filepath.IsAbs(cleanRel) || strings.HasPrefix(cleanRel, "/") || strings.HasPrefix(cleanRel, "\\") {
		return "", fmt.Errorf("path must be relative to sandbox workspace, got absolute: %s", relPath)
	}
	if cleanRel == "." || cleanRel == ".." || strings.HasPrefix(cleanRel, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("path cannot traverse outside sandbox workspace: %s", relPath)
	}

	return cleanRel, nil
}

func (r *Registry) executeSandboxDownloadAttachment(ctx context.Context, argsJSON string) (string, error) {
	if r.sandboxManager == nil {
		return "", errors.New("sandbox execution is not configured on this server")
	}
	if r.attachmentClient == nil {
		return "", errors.New("attachment client is not configured on this server")
	}

	session, ok := ChatSessionFromContext(ctx)
	if !ok || !session.IsDM {
		return "", errors.New("sandbox attachment tools are strictly available in 1-on-1 direct messages")
	}
	if session.UserID == "" {
		return "", errors.New("cannot determine user ID from chat session")
	}

	sbx, ok := r.sandboxManager.GetStatus(session.UserID)
	if !ok || sbx == nil || sbx.Status != sandbox.StatusRunning {
		return "", errors.New("no running sandbox found; request or wait for sandbox creation first")
	}

	var args SandboxDownloadAttachmentArgs
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		return "", fmt.Errorf("failed to parse sandbox_download_attachment arguments: %w", err)
	}

	fileID := strings.TrimSpace(args.FileID)
	if fileID == "" {
		return "", errors.New("file_id cannot be empty")
	}

	cleanRel, err := validateWorkspaceRelPath(args.DestinationPath)
	if err != nil {
		return "", err
	}

	if strings.TrimSpace(sbx.WorkspaceDir) == "" {
		return "", errors.New("workspace directory cannot be empty")
	}
	absWorkspace, err := filepath.Abs(sbx.WorkspaceDir)
	if err != nil {
		return "", fmt.Errorf("failed to resolve workspace directory: %w", err)
	}

	root, err := os.OpenRoot(absWorkspace)
	if err != nil {
		return "", fmt.Errorf("failed to open sandbox workspace root: %w", err)
	}
	defer func() { _ = root.Close() }()

	parentDir := filepath.Dir(cleanRel)
	if parentDir != "." {
		if err := root.MkdirAll(parentDir, 0o755); err != nil {
			return "", fmt.Errorf("failed to create parent directories for %s: %w", args.DestinationPath, err)
		}
	}

	// Check if destination path exists and whether it is a symlink or directory
	if fi, err := root.Lstat(cleanRel); err == nil {
		if fi.Mode()&os.ModeSymlink != 0 {
			return "", fmt.Errorf("destination path %q is a symbolic link", args.DestinationPath)
		}
		if fi.IsDir() {
			return "", fmt.Errorf("destination path %q is a directory", args.DestinationPath)
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return "", fmt.Errorf("failed to inspect destination path %q: %w", args.DestinationPath, err)
	}

	if session.Progress != nil {
		session.Progress.SetCurrent(fmt.Sprintf("Downloading attachment %s", fileID))
	}

	data, mimeType, err := r.attachmentClient.DownloadAttachment(ctx, fileID)
	if err != nil {
		return "", fmt.Errorf("failed to download attachment %s: %w", fileID, err)
	}

	f, err := root.OpenFile(cleanRel, os.O_WRONLY|os.O_CREATE|os.O_TRUNC|syscall.O_NONBLOCK|syscall.O_NOFOLLOW, 0o644)
	if err != nil {
		return "", fmt.Errorf("failed to create destination file %s: %w", args.DestinationPath, err)
	}
	defer func() { _ = f.Close() }()

	fi, err := f.Stat()
	if err != nil {
		return "", fmt.Errorf("failed to inspect destination file %s: %w", args.DestinationPath, err)
	}
	if !fi.Mode().IsRegular() {
		return "", fmt.Errorf("destination path %q is not a regular file", args.DestinationPath)
	}

	if _, err := f.Write(data); err != nil {
		return "", fmt.Errorf("failed to write attachment to %s: %w", args.DestinationPath, err)
	}

	fileName := filepath.Base(cleanRel)
	resPayload := map[string]interface{}{
		"file_id":          fileID,
		"name":             fileName,
		"destination_path": cleanRel,
		"size_bytes":       len(data),
		"mime_type":        mimeType,
		"status":           "downloaded",
	}
	respBytes, err := json.Marshal(resPayload)
	if err != nil {
		return "", fmt.Errorf("failed to encode download result: %w", err)
	}
	return string(respBytes), nil
}

func (r *Registry) executeSandboxUploadAttachment(ctx context.Context, argsJSON string) (string, error) {
	if r.sandboxManager == nil {
		return "", errors.New("sandbox execution is not configured on this server")
	}
	if r.attachmentClient == nil {
		return "", errors.New("attachment client is not configured on this server")
	}

	session, ok := ChatSessionFromContext(ctx)
	if !ok || !session.IsDM {
		return "", errors.New("sandbox attachment tools are strictly available in 1-on-1 direct messages")
	}
	if session.UserID == "" {
		return "", errors.New("cannot determine user ID from chat session")
	}
	if session.StagedAttachments == nil {
		return "", errors.New("attachment staging collector is not configured for this chat session")
	}

	sbx, ok := r.sandboxManager.GetStatus(session.UserID)
	if !ok || sbx == nil || sbx.Status != sandbox.StatusRunning {
		return "", errors.New("no running sandbox found; request or wait for sandbox creation first")
	}

	var args SandboxUploadAttachmentArgs
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		return "", fmt.Errorf("failed to parse sandbox_upload_attachment arguments: %w", err)
	}

	cleanRel, err := validateWorkspaceRelPath(args.SourcePath)
	if err != nil {
		return "", err
	}

	if strings.TrimSpace(sbx.WorkspaceDir) == "" {
		return "", errors.New("workspace directory cannot be empty")
	}
	absWorkspace, err := filepath.Abs(sbx.WorkspaceDir)
	if err != nil {
		return "", fmt.Errorf("failed to resolve workspace directory: %w", err)
	}

	root, err := os.OpenRoot(absWorkspace)
	if err != nil {
		return "", fmt.Errorf("failed to open sandbox workspace root: %w", err)
	}
	defer func() { _ = root.Close() }()

	// Open with O_NONBLOCK to prevent blocking indefinitely if the path is a FIFO
	f, err := root.OpenFile(cleanRel, os.O_RDONLY|syscall.O_NONBLOCK, 0)
	if err != nil {
		return "", fmt.Errorf("source file %q not found: %w", args.SourcePath, err)
	}
	defer func() { _ = f.Close() }()

	fi, err := f.Stat()
	if err != nil {
		return "", fmt.Errorf("failed to inspect source file %q: %w", args.SourcePath, err)
	}
	if !fi.Mode().IsRegular() {
		return "", fmt.Errorf("source path %q is not a regular file", args.SourcePath)
	}

	maxLimit := r.maxAttachmentSizeBytes
	if maxLimit <= 0 {
		maxLimit = config.DefaultMaxAttachmentSizeBytes
	}
	if fi.Size() > maxLimit {
		return "", fmt.Errorf("file size (%d bytes) exceeds maximum attachment limit (%d bytes)", fi.Size(), maxLimit)
	}

	limit := maxLimit
	if limit < math.MaxInt64 {
		limit++
	}
	lr := io.LimitReader(f, limit)
	data, err := io.ReadAll(lr)
	if err != nil {
		return "", fmt.Errorf("failed to read file %q: %w", args.SourcePath, err)
	}
	if int64(len(data)) > maxLimit {
		return "", fmt.Errorf("file size (%d bytes) exceeds maximum attachment limit (%d bytes)", len(data), maxLimit)
	}

	name := strings.TrimSpace(args.Name)
	if name == "" {
		name = filepath.Base(cleanRel)
	}

	var mimeType string
	if len(data) > 0 {
		mimeType = http.DetectContentType(data)
	} else {
		mimeType = "application/octet-stream"
	}

	isImage := strings.HasPrefix(strings.ToLower(mimeType), "image/")
	const maxImageUploadSize = 10 * 1024 * 1024

	var fileID string
	var attType models.AttachmentType

	if session.Progress != nil {
		session.Progress.SetCurrent(fmt.Sprintf("Uploading attachment %s", name))
	}

	if isImage && int64(len(data)) <= maxImageUploadSize {
		id, uploadErr := r.attachmentClient.UploadImage(ctx, data, name, mimeType)
		if uploadErr == nil {
			fileID = id
			attType = models.AttachmentTypeImage
		} else {
			id, fileErr := r.attachmentClient.UploadFile(ctx, data, name, mimeType)
			if fileErr != nil {
				return "", fmt.Errorf("failed to upload image as image (%v) and as file: %w", uploadErr, fileErr)
			}
			fileID = id
			attType = models.AttachmentTypeFile
		}
	} else {
		id, fileErr := r.attachmentClient.UploadFile(ctx, data, name, mimeType)
		if fileErr != nil {
			return "", fmt.Errorf("failed to upload file %s: %w", name, fileErr)
		}
		fileID = id
		attType = models.AttachmentTypeFile
	}

	if err := session.StageAttachment(models.Attachment{
		Type:     attType,
		Name:     name,
		MimeType: mimeType,
		FileID:   fileID,
	}); err != nil {
		return "", fmt.Errorf("failed to stage attachment: %w", err)
	}

	resPayload := map[string]interface{}{
		"file_id":     fileID,
		"name":        name,
		"source_path": cleanRel,
		"size_bytes":  len(data),
		"mime_type":   mimeType,
		"type":        string(attType),
		"status":      "staged",
	}
	respBytes, err := json.Marshal(resPayload)
	if err != nil {
		return "", fmt.Errorf("failed to encode upload result: %w", err)
	}
	return string(respBytes), nil
}
