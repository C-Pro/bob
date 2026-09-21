package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"bob/internal/config"
	"bob/internal/models"
	"bob/internal/sandbox"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type mockAttachmentClient struct {
	mu               sync.Mutex
	downloadFunc     func(ctx context.Context, fileID string) ([]byte, string, error)
	uploadFileFunc   func(ctx context.Context, data []byte, filename, mimeType string) (string, error)
	uploadImageFunc  func(ctx context.Context, data []byte, filename, mimeType string) (string, error)
	downloadCalls    []string
	uploadFileCalls  []struct{ Data []byte; Filename, MimeType string }
	uploadImageCalls []struct{ Data []byte; Filename, MimeType string }
}

func (m *mockAttachmentClient) DownloadAttachment(ctx context.Context, fileID string) ([]byte, string, error) {
	m.mu.Lock()
	m.downloadCalls = append(m.downloadCalls, fileID)
	m.mu.Unlock()
	if m.downloadFunc != nil {
		return m.downloadFunc(ctx, fileID)
	}
	return []byte("test attachment content"), "text/plain; charset=utf-8", nil
}

func (m *mockAttachmentClient) UploadFile(ctx context.Context, data []byte, filename, mimeType string) (string, error) {
	m.mu.Lock()
	m.uploadFileCalls = append(m.uploadFileCalls, struct {
		Data               []byte
		Filename, MimeType string
	}{Data: data, Filename: filename, MimeType: mimeType})
	m.mu.Unlock()
	if m.uploadFileFunc != nil {
		return m.uploadFileFunc(ctx, data, filename, mimeType)
	}
	return "upl_file_123", nil
}

func (m *mockAttachmentClient) UploadImage(ctx context.Context, data []byte, filename, mimeType string) (string, error) {
	m.mu.Lock()
	m.uploadImageCalls = append(m.uploadImageCalls, struct {
		Data               []byte
		Filename, MimeType string
	}{Data: data, Filename: filename, MimeType: mimeType})
	m.mu.Unlock()
	if m.uploadImageFunc != nil {
		return m.uploadImageFunc(ctx, data, filename, mimeType)
	}
	return "upl_img_456", nil
}

func setupTestSandboxAndRegistry(t *testing.T, client AttachmentClient) (*sandbox.Manager, *Registry, string) {
	t.Helper()
	tempDir := t.TempDir()
	cfg := &config.Config{
		DataDir:                   tempDir,
		SandboxEnabled:            true,
		SandboxDrivers:            []string{"bwrap"},
		SandboxMaxLifetime:        30 * time.Minute,
		SandboxDefaultExecTimeout: 1 * time.Minute,
		SandboxMaxExecTimeout:     10 * time.Minute,
		MaxAttachmentSizeBytes:    25 * 1024 * 1024,
	}

	mockDriver := &mockSandboxDriver{available: true}
	mgr := sandbox.NewManager(cfg.SandboxConfig(), []sandbox.Driver{mockDriver})
	t.Cleanup(func() { _ = mgr.Close() })

	reg := NewRegistry(nil, nil, mgr)
	reg.SetAttachmentClient(client)
	reg.SetMaxAttachmentSize(cfg.MaxAttachmentSizeBytes)

	return mgr, reg, tempDir
}

func TestValidateWorkspaceRelPath(t *testing.T) {
	tests := []struct {
		name      string
		rel       string
		wantErr   bool
		errSubstr string
	}{
		{
			name:      "empty rel path",
			rel:       "   ",
			wantErr:   true,
			errSubstr: "path cannot be empty",
		},
		{
			name:      "null byte injection",
			rel:       "data.csv\x00.evil",
			wantErr:   true,
			errSubstr: "cannot contain null bytes",
		},
		{
			name:      "absolute path unix",
			rel:       "/etc/passwd",
			wantErr:   true,
			errSubstr: "path must be relative",
		},
		{
			name:      "dot path",
			rel:       ".",
			wantErr:   true,
			errSubstr: "cannot traverse outside",
		},
		{
			name:      "dot dot path",
			rel:       "..",
			wantErr:   true,
			errSubstr: "cannot traverse outside",
		},
		{
			name:      "traversal outside workspace",
			rel:       "../secret.txt",
			wantErr:   true,
			errSubstr: "cannot traverse outside",
		},
		{
			name:      "nested traversal outside workspace",
			rel:       "foo/bar/../../../escape.txt",
			wantErr:   true,
			errSubstr: "cannot traverse outside",
		},
		{
			name:    "valid root file",
			rel:     "data.csv",
			wantErr: false,
		},
		{
			name:    "valid nested path",
			rel:     "sub/folder/data.json",
			wantErr: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := validateWorkspaceRelPath(tc.rel)
			if tc.wantErr {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tc.errSubstr)
			} else {
				require.NoError(t, err)
				assert.Equal(t, filepath.Clean(strings.TrimSpace(tc.rel)), got)
			}
		})
	}
}

func TestAttachmentToolDefinitions_Formatting(t *testing.T) {
	reg := NewRegistry(nil, nil, nil)

	// 1. 50MB (generic 50MB, images capped at 10MB)
	reg.SetMaxAttachmentSize(50 * 1024 * 1024)
	defs := reg.attachmentToolDefinitions()
	require.Len(t, defs, 2)
	assert.Equal(t, "sandbox_download_attachment", defs[0].Function.Name)
	assert.Equal(t, "sandbox_upload_attachment", defs[1].Function.Name)
	assert.Contains(t, defs[1].Function.Description, "50MB (10MB for images)")

	// 2. Default (25MB generic, 10MB images)
	reg.SetMaxAttachmentSize(0)
	defsDefault := reg.attachmentToolDefinitions()
	assert.Contains(t, defsDefault[1].Function.Description, "25MB (10MB for images)")

	// 3. Below 10MB (5MB generic, 5MB images)
	reg.SetMaxAttachmentSize(5 * 1024 * 1024)
	defs5MB := reg.attachmentToolDefinitions()
	assert.Contains(t, defs5MB[1].Function.Description, "5MB (5MB for images)")

	// 4. Sub-MiB (500KB generic, 500KB images)
	reg.SetMaxAttachmentSize(500 * 1024)
	defs500KB := reg.attachmentToolDefinitions()
	assert.Contains(t, defs500KB[1].Function.Description, "500KB (500KB for images)")

	// 5. Fractional MiB (1.5MB)
	reg.SetMaxAttachmentSize(1536 * 1024)
	defs1_5MB := reg.attachmentToolDefinitions()
	assert.Contains(t, defs1_5MB[1].Function.Description, "1.5MB (1.5MB for images)")
}

func TestExecuteSandboxDownloadAttachment_Validation(t *testing.T) {
	mockClient := &mockAttachmentClient{}
	mgr, reg, _ := setupTestSandboxAndRegistry(t, mockClient)
	ctx := context.Background()

	// 1. Missing sandbox manager
	regNoSbx := NewRegistry(nil, nil, nil)
	regNoSbx.SetAttachmentClient(mockClient)
	_, err := regNoSbx.Execute(ctx, "sandbox_download_attachment", `{"file_id":"f1","destination_path":"a.txt"}`)
	assert.ErrorContains(t, err, "sandbox execution is not configured")

	// 2. Missing attachment client
	regNoClient := NewRegistry(nil, nil, mgr)
	_, err = regNoClient.Execute(ctx, "sandbox_download_attachment", `{"file_id":"f1","destination_path":"a.txt"}`)
	assert.ErrorContains(t, err, "attachment client is not configured")

	// 3. No session in context
	_, err = reg.Execute(ctx, "sandbox_download_attachment", `{"file_id":"f1","destination_path":"a.txt"}`)
	assert.ErrorContains(t, err, "strictly available in 1-on-1 direct messages")

	// 4. Townhall chat (non-DM)
	thCtx := WithChatSession(ctx, ChatSessionContext{
		ChatID: "townhall",
		IsDM:   false,
		UserID: "u1",
	})
	_, err = reg.Execute(thCtx, "sandbox_download_attachment", `{"file_id":"f1","destination_path":"a.txt"}`)
	assert.ErrorContains(t, err, "strictly available in 1-on-1 direct messages")

	// 5. DM without UserID
	noUserCtx := WithChatSession(ctx, ChatSessionContext{
		ChatID: "dm1",
		IsDM:   true,
	})
	_, err = reg.Execute(noUserCtx, "sandbox_download_attachment", `{"file_id":"f1","destination_path":"a.txt"}`)
	assert.ErrorContains(t, err, "cannot determine user ID")

	// 6. User without active running sandbox
	dmCtx := WithChatSession(ctx, ChatSessionContext{
		ChatID: "dm1",
		IsDM:   true,
		UserID: "u1",
	})
	_, err = reg.Execute(dmCtx, "sandbox_download_attachment", `{"file_id":"f1","destination_path":"a.txt"}`)
	assert.ErrorContains(t, err, "no running sandbox found")

	// Request sandbox but don't approve it yet
	_, err = mgr.RequestSandbox(ctx, "u1", "dm1", sandbox.RequestParams{
		Driver:      sandbox.DriverBwrap,
		NetworkMode: sandbox.NetworkNone,
	})
	require.NoError(t, err)

	_, err = reg.Execute(dmCtx, "sandbox_download_attachment", `{"file_id":"f1","destination_path":"a.txt"}`)
	assert.ErrorContains(t, err, "no running sandbox found")

	// Now approve sandbox
	_, err = mgr.ApproveSandbox(ctx, "u1")
	require.NoError(t, err)

	// 7. Invalid JSON arguments
	_, err = reg.Execute(dmCtx, "sandbox_download_attachment", `invalid json`)
	assert.ErrorContains(t, err, "failed to parse sandbox_download_attachment arguments")

	// 8. Empty file_id
	_, err = reg.Execute(dmCtx, "sandbox_download_attachment", `{"file_id":"  ","destination_path":"a.txt"}`)
	assert.ErrorContains(t, err, "file_id cannot be empty")

	// 9. Path traversal destination
	_, err = reg.Execute(dmCtx, "sandbox_download_attachment", `{"file_id":"f1","destination_path":"../escape.txt"}`)
	assert.ErrorContains(t, err, "cannot traverse outside sandbox workspace")
}

func TestExecuteSandboxDownloadAttachment_SuccessAndClientError(t *testing.T) {
	mockClient := &mockAttachmentClient{}
	mgr, reg, _ := setupTestSandboxAndRegistry(t, mockClient)
	ctx := context.Background()

	_, err := mgr.RequestSandbox(ctx, "u1", "dm1", sandbox.RequestParams{
		Driver:      sandbox.DriverBwrap,
		NetworkMode: sandbox.NetworkNone,
	})
	require.NoError(t, err)
	sbx, err := mgr.ApproveSandbox(ctx, "u1")
	require.NoError(t, err)

	collector := NewStagedAttachmentCollector()
	dmCtx := WithChatSession(ctx, ChatSessionContext{
		ChatID:            "dm1",
		IsDM:              true,
		UserID:            "u1",
		StagedAttachments: collector,
	})

	// Client returns error
	mockClient.downloadFunc = func(ctx context.Context, fileID string) ([]byte, string, error) {
		return nil, "", errors.New("404 attachment not found")
	}
	_, err = reg.Execute(dmCtx, "sandbox_download_attachment", `{"file_id":"nonexistent","destination_path":"data/doc.txt"}`)
	assert.ErrorContains(t, err, "failed to download attachment nonexistent: 404 attachment not found")

	// Client returns success
	expectedContent := []byte("hello attachment from server")
	mockClient.downloadFunc = func(ctx context.Context, fileID string) ([]byte, string, error) {
		return expectedContent, "text/plain; charset=utf-8", nil
	}

	res, err := reg.Execute(dmCtx, "sandbox_download_attachment", `{
		"file_id": "att_12345",
		"destination_path": "nested/dir/downloaded.txt"
	}`)
	require.NoError(t, err)

	var payload map[string]interface{}
	err = json.Unmarshal([]byte(res), &payload)
	require.NoError(t, err)

	assert.Equal(t, "att_12345", payload["file_id"])
	assert.Equal(t, "downloaded.txt", payload["name"])
	assert.Equal(t, "nested/dir/downloaded.txt", payload["destination_path"])
	assert.Equal(t, float64(len(expectedContent)), payload["size_bytes"])
	assert.Equal(t, "text/plain; charset=utf-8", payload["mime_type"])
	assert.Equal(t, "downloaded", payload["status"])

	// Verify file was written to disk inside sandbox workspace
	savedPath := filepath.Join(sbx.WorkspaceDir, "nested/dir/downloaded.txt")
	diskData, err := os.ReadFile(savedPath)
	require.NoError(t, err)
	assert.Equal(t, expectedContent, diskData)
}

func TestExecuteSandboxDownloadAttachment_SymlinkEscapes(t *testing.T) {
	mockClient := &mockAttachmentClient{}
	mgr, reg, _ := setupTestSandboxAndRegistry(t, mockClient)
	ctx := context.Background()

	_, err := mgr.RequestSandbox(ctx, "u1", "dm1", sandbox.RequestParams{
		Driver:      sandbox.DriverBwrap,
		NetworkMode: sandbox.NetworkNone,
	})
	require.NoError(t, err)
	sbx, err := mgr.ApproveSandbox(ctx, "u1")
	require.NoError(t, err)

	dmCtx := WithChatSession(ctx, NewChatSessionContext("dm1", "u1", true))

	// 1. Parent directory is a symlink pointing outside the workspace
	outsideDir := t.TempDir()
	linkParent := filepath.Join(sbx.WorkspaceDir, "outside_dir_link")
	require.NoError(t, os.Symlink(outsideDir, linkParent))

	_, err = reg.Execute(dmCtx, "sandbox_download_attachment", `{
		"file_id": "att_ext_dir",
		"destination_path": "outside_dir_link/file.txt"
	}`)
	assert.Error(t, err, "download through outside directory symlink must fail")

	// 2. Destination path itself is a symlink pointing outside the workspace
	outsideFile := filepath.Join(outsideDir, "pwned.txt")
	require.NoError(t, os.WriteFile(outsideFile, []byte("original"), 0o644))
	linkFile := filepath.Join(sbx.WorkspaceDir, "symlink_file.txt")
	require.NoError(t, os.Symlink(outsideFile, linkFile))

	_, err = reg.Execute(dmCtx, "sandbox_download_attachment", `{
		"file_id": "att_ext_file",
		"destination_path": "symlink_file.txt"
	}`)
	assert.Error(t, err, "download to existing symlink destination must be rejected")
	assert.Contains(t, err.Error(), "symbolic link")

	// Outside file must remain untouched
	content, readErr := os.ReadFile(outsideFile)
	require.NoError(t, readErr)
	assert.Equal(t, []byte("original"), content)

	// 3. Destination path is an existing FIFO
	fifoPath := filepath.Join(sbx.WorkspaceDir, "dest_fifo")
	if mkfifoErr := syscall.Mkfifo(fifoPath, 0o666); mkfifoErr == nil {
		defer func() { _ = os.Remove(fifoPath) }()
		_, err = reg.Execute(dmCtx, "sandbox_download_attachment", `{
			"file_id": "att_fifo",
			"destination_path": "dest_fifo"
		}`)
		assert.Error(t, err)
	}
}

func TestExecuteSandboxUploadAttachment_Validation(t *testing.T) {
	mockClient := &mockAttachmentClient{}
	mgr, reg, _ := setupTestSandboxAndRegistry(t, mockClient)
	ctx := context.Background()

	// 1. Missing sandbox manager
	regNoSbx := NewRegistry(nil, nil, nil)
	regNoSbx.SetAttachmentClient(mockClient)
	_, err := regNoSbx.Execute(ctx, "sandbox_upload_attachment", `{"source_path":"a.txt"}`)
	assert.ErrorContains(t, err, "sandbox execution is not configured")

	// 2. Missing attachment client
	regNoClient := NewRegistry(nil, nil, mgr)
	_, err = regNoClient.Execute(ctx, "sandbox_upload_attachment", `{"source_path":"a.txt"}`)
	assert.ErrorContains(t, err, "attachment client is not configured")

	// 3. No session
	_, err = reg.Execute(ctx, "sandbox_upload_attachment", `{"source_path":"a.txt"}`)
	assert.ErrorContains(t, err, "strictly available in 1-on-1 direct messages")

	// 4. Townhall
	thCtx := WithChatSession(ctx, ChatSessionContext{
		ChatID: "townhall",
		IsDM:   false,
		UserID: "u1",
	})
	_, err = reg.Execute(thCtx, "sandbox_upload_attachment", `{"source_path":"a.txt"}`)
	assert.ErrorContains(t, err, "strictly available in 1-on-1 direct messages")

	// 5. DM without UserID
	noUserCtx := WithChatSession(ctx, ChatSessionContext{
		ChatID: "dm1",
		IsDM:   true,
	})
	_, err = reg.Execute(noUserCtx, "sandbox_upload_attachment", `{"source_path":"a.txt"}`)
	assert.ErrorContains(t, err, "cannot determine user ID")

	// 6. Running sandbox not found
	collector := NewStagedAttachmentCollector()
	dmCtx := WithChatSession(ctx, ChatSessionContext{
		ChatID:            "dm1",
		IsDM:              true,
		UserID:            "u1",
		StagedAttachments: collector,
	})
	_, err = reg.Execute(dmCtx, "sandbox_upload_attachment", `{"source_path":"a.txt"}`)
	assert.ErrorContains(t, err, "no running sandbox found")

	_, err = mgr.RequestSandbox(ctx, "u1", "dm1", sandbox.RequestParams{
		Driver:      sandbox.DriverBwrap,
		NetworkMode: sandbox.NetworkNone,
	})
	require.NoError(t, err)
	sbx, err := mgr.ApproveSandbox(ctx, "u1")
	require.NoError(t, err)

	// 7. Missing staging collector
	nilCollectorCtx := WithChatSession(ctx, ChatSessionContext{
		ChatID: "dm1",
		IsDM:   true,
		UserID: "u1",
	})
	_, err = reg.Execute(nilCollectorCtx, "sandbox_upload_attachment", `{"source_path":"a.txt"}`)
	assert.ErrorContains(t, err, "attachment staging collector is not configured")

	// 8. Invalid JSON
	_, err = reg.Execute(dmCtx, "sandbox_upload_attachment", `not json`)
	assert.ErrorContains(t, err, "failed to parse sandbox_upload_attachment arguments")

	// 9. Path traversal
	_, err = reg.Execute(dmCtx, "sandbox_upload_attachment", `{"source_path":"../../outside.txt"}`)
	assert.ErrorContains(t, err, "cannot traverse outside sandbox workspace")

	// 10. File not found
	_, err = reg.Execute(dmCtx, "sandbox_upload_attachment", `{"source_path":"missing.txt"}`)
	assert.ErrorContains(t, err, "not found")

	// 11. Path is a directory
	subDir := filepath.Join(sbx.WorkspaceDir, "somedir")
	require.NoError(t, os.MkdirAll(subDir, 0o755))
	_, err = reg.Execute(dmCtx, "sandbox_upload_attachment", `{"source_path":"somedir"}`)
	assert.ErrorContains(t, err, "not a regular file")

	// 12. File exceeds maximum size limit
	reg.SetMaxAttachmentSize(100) // 100 bytes limit
	bigFile := filepath.Join(sbx.WorkspaceDir, "big.txt")
	require.NoError(t, os.WriteFile(bigFile, bytes.Repeat([]byte("A"), 200), 0o644))
	_, err = reg.Execute(dmCtx, "sandbox_upload_attachment", `{"source_path":"big.txt"}`)
	assert.ErrorContains(t, err, "exceeds maximum attachment limit")

	// 13. math.MaxInt64 boundary does not overflow
	reg.SetMaxAttachmentSize(math.MaxInt64)
	mockClient.uploadFileFunc = func(ctx context.Context, data []byte, filename, mimeType string) (string, error) {
		assert.Equal(t, []byte("A"), data)
		return "f_max_int", nil
	}
	tinyFile := filepath.Join(sbx.WorkspaceDir, "tiny.txt")
	require.NoError(t, os.WriteFile(tinyFile, []byte("A"), 0o644))
	outMaxInt, err := reg.Execute(dmCtx, "sandbox_upload_attachment", `{"source_path":"tiny.txt"}`)
	require.NoError(t, err)
	assert.Contains(t, outMaxInt, "f_max_int")
}

func TestExecuteSandboxUploadAttachment_SymlinkEscapes(t *testing.T) {
	mockClient := &mockAttachmentClient{}
	mgr, reg, _ := setupTestSandboxAndRegistry(t, mockClient)
	ctx := context.Background()

	_, err := mgr.RequestSandbox(ctx, "u1", "dm1", sandbox.RequestParams{
		Driver:      sandbox.DriverBwrap,
		NetworkMode: sandbox.NetworkNone,
	})
	require.NoError(t, err)
	sbx, err := mgr.ApproveSandbox(ctx, "u1")
	require.NoError(t, err)

	dmCtx := WithChatSession(ctx, NewChatSessionContext("dm1", "u1", true))

	// 1. Direct symlink to outside sensitive file
	outsideDir := t.TempDir()
	outsideSecret := filepath.Join(outsideDir, "host_secret.txt")
	require.NoError(t, os.WriteFile(outsideSecret, []byte("super-secret-key"), 0o644))

	linkFile := filepath.Join(sbx.WorkspaceDir, "outside_secret_link.txt")
	require.NoError(t, os.Symlink(outsideSecret, linkFile))

	_, err = reg.Execute(dmCtx, "sandbox_upload_attachment", `{"source_path":"outside_secret_link.txt"}`)
	assert.Error(t, err, "uploading external symlink target must fail")

	// 2. Parent directory is a symlink pointing outside
	linkDir := filepath.Join(sbx.WorkspaceDir, "outside_dir")
	require.NoError(t, os.Symlink(outsideDir, linkDir))

	_, err = reg.Execute(dmCtx, "sandbox_upload_attachment", `{"source_path":"outside_dir/host_secret.txt"}`)
	assert.Error(t, err, "uploading through external directory symlink must fail")
}

func TestExecuteSandboxUploadAttachment_NonRegularFiles(t *testing.T) {
	mockClient := &mockAttachmentClient{}
	mgr, reg, _ := setupTestSandboxAndRegistry(t, mockClient)
	ctx := context.Background()

	_, err := mgr.RequestSandbox(ctx, "u1", "dm1", sandbox.RequestParams{
		Driver:      sandbox.DriverBwrap,
		NetworkMode: sandbox.NetworkNone,
	})
	require.NoError(t, err)
	sbx, err := mgr.ApproveSandbox(ctx, "u1")
	require.NoError(t, err)

	dmCtx := WithChatSession(ctx, NewChatSessionContext("dm1", "u1", true))

	// Create a FIFO inside workspace
	fifoPath := filepath.Join(sbx.WorkspaceDir, "test_fifo")
	err = syscall.Mkfifo(fifoPath, 0o666)
	if err == nil {
		defer func() { _ = os.Remove(fifoPath) }()
		_, err = reg.Execute(dmCtx, "sandbox_upload_attachment", `{"source_path":"test_fifo"}`)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "not a regular file")
	}
}

func TestExecuteSandboxUploadAttachment_RoutingAndStaging(t *testing.T) {
	mockClient := &mockAttachmentClient{}
	mgr, reg, _ := setupTestSandboxAndRegistry(t, mockClient)
	ctx := context.Background()

	_, err := mgr.RequestSandbox(ctx, "u1", "dm1", sandbox.RequestParams{
		Driver:      sandbox.DriverBwrap,
		NetworkMode: sandbox.NetworkNone,
	})
	require.NoError(t, err)
	sbx, err := mgr.ApproveSandbox(ctx, "u1")
	require.NoError(t, err)

	collector := NewStagedAttachmentCollector()
	dmCtx := WithChatSession(ctx, ChatSessionContext{
		ChatID:            "dm1",
		IsDM:              true,
		UserID:            "u1",
		StagedAttachments: collector,
	})

	// 1. Generic text/data file -> UploadFile
	textFile := filepath.Join(sbx.WorkspaceDir, "data.csv")
	csvContent := []byte("col1,col2\nval1,val2\n")
	require.NoError(t, os.WriteFile(textFile, csvContent, 0o644))

	mockClient.uploadFileFunc = func(ctx context.Context, data []byte, filename, mimeType string) (string, error) {
		assert.Equal(t, "custom_report.csv", filename)
		assert.Equal(t, csvContent, data)
		return "file_id_csv_1", nil
	}

	out, err := reg.Execute(dmCtx, "sandbox_upload_attachment", `{
		"source_path": "data.csv",
		"name": "custom_report.csv"
	}`)
	require.NoError(t, err)

	var payload map[string]interface{}
	require.NoError(t, json.Unmarshal([]byte(out), &payload))
	assert.Equal(t, "file_id_csv_1", payload["file_id"])
	assert.Equal(t, "custom_report.csv", payload["name"])
	assert.Equal(t, "file", payload["type"])
	assert.Equal(t, "staged", payload["status"])

	staged := collector.All()
	require.Len(t, staged, 1)
	assert.Equal(t, "file_id_csv_1", staged[0].FileID)
	assert.Equal(t, models.AttachmentTypeFile, staged[0].Type)
	assert.Equal(t, "custom_report.csv", staged[0].Name)

	// 2. Small PNG image <= 10MB -> UploadImage
	pngData := []byte{
		0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d,
		0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
		0x08, 0x06, 0x00, 0x00, 0x00, 0x1f, 0x15, 0xc4, 0x89, 0x00, 0x00, 0x00,
		0x0a, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9c, 0x63, 0x00, 0x01, 0x00, 0x00,
		0x05, 0x00, 0x01, 0x0d, 0x0a, 0x2d, 0xb4, 0x00, 0x00, 0x00, 0x00, 0x49,
		0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
	}
	pngPath := filepath.Join(sbx.WorkspaceDir, "chart.png")
	require.NoError(t, os.WriteFile(pngPath, pngData, 0o644))

	mockClient.uploadImageFunc = func(ctx context.Context, data []byte, filename, mimeType string) (string, error) {
		assert.Equal(t, "chart.png", filename)
		assert.Equal(t, "image/png", mimeType)
		return "img_id_png_2", nil
	}

	outImg, err := reg.Execute(dmCtx, "sandbox_upload_attachment", `{"source_path":"chart.png"}`)
	require.NoError(t, err)

	var imgPayload map[string]interface{}
	require.NoError(t, json.Unmarshal([]byte(outImg), &imgPayload))
	assert.Equal(t, "img_id_png_2", imgPayload["file_id"])
	assert.Equal(t, "chart.png", imgPayload["name"])
	assert.Equal(t, "image", imgPayload["type"])
	assert.Equal(t, "staged", imgPayload["status"])

	staged = collector.All()
	require.Len(t, staged, 2)
	assert.Equal(t, "img_id_png_2", staged[1].FileID)
	assert.Equal(t, models.AttachmentTypeImage, staged[1].Type)

	// 3. Small PNG image where UploadImage fails -> fallback to UploadFile
	mockClient.uploadImageFunc = func(ctx context.Context, data []byte, filename, mimeType string) (string, error) {
		return "", errors.New("image processing service unavailable")
	}
	mockClient.uploadFileFunc = func(ctx context.Context, data []byte, filename, mimeType string) (string, error) {
		return "fallback_file_id_3", nil
	}

	outFallback, err := reg.Execute(dmCtx, "sandbox_upload_attachment", `{"source_path":"chart.png","name":"fallback.png"}`)
	require.NoError(t, err)

	var fbPayload map[string]interface{}
	require.NoError(t, json.Unmarshal([]byte(outFallback), &fbPayload))
	assert.Equal(t, "fallback_file_id_3", fbPayload["file_id"])
	assert.Equal(t, "file", fbPayload["type"])

	staged = collector.All()
	require.Len(t, staged, 3)
	assert.Equal(t, "fallback_file_id_3", staged[2].FileID)
	assert.Equal(t, models.AttachmentTypeFile, staged[2].Type)

	// 4. Large Image (> 10MB) -> routes directly to UploadFile
	reg.SetMaxAttachmentSize(30 * 1024 * 1024)
	largeImgPath := filepath.Join(sbx.WorkspaceDir, "large.png")
	largeImgData := append(pngData, bytes.Repeat([]byte{0}, 11*1024*1024)...)
	require.NoError(t, os.WriteFile(largeImgPath, largeImgData, 0o644))

	mockClient.uploadImageCalls = nil
	mockClient.uploadFileFunc = func(ctx context.Context, data []byte, filename, mimeType string) (string, error) {
		return "large_img_as_file_4", nil
	}

	outLarge, err := reg.Execute(dmCtx, "sandbox_upload_attachment", `{"source_path":"large.png"}`)
	require.NoError(t, err)

	var largePayload map[string]interface{}
	require.NoError(t, json.Unmarshal([]byte(outLarge), &largePayload))
	assert.Equal(t, "large_img_as_file_4", largePayload["file_id"])
	assert.Equal(t, "file", largePayload["type"])
	assert.Empty(t, mockClient.uploadImageCalls)
}

func TestStagedAttachmentCollector_Concurrent(t *testing.T) {
	c := NewStagedAttachmentCollector()
	var wg sync.WaitGroup
	count := MaxStagedAttachments

	for i := 0; i < count; i++ {
		wg.Add(2)
		idx := i
		go func() {
			defer wg.Done()
			_ = c.Add(models.Attachment{
				FileID:   "f" + string(rune('a'+idx%26)),
				Name:     "file.txt",
				MimeType: "text/plain",
				Type:     models.AttachmentTypeFile,
			})
		}()
		go func() {
			defer wg.Done()
			_ = c.All()
		}()
	}

	wg.Wait()
	assert.Len(t, c.All(), count)
}

func TestStagedAttachmentCollector_Limit(t *testing.T) {
	c := NewStagedAttachmentCollector()
	for i := 0; i < MaxStagedAttachments; i++ {
		err := c.Add(models.Attachment{
			FileID: fmt.Sprintf("file_%d", i),
			Name:   fmt.Sprintf("file_%d.txt", i),
		})
		require.NoError(t, err)
	}

	// 11th should exceed limit
	err := c.Add(models.Attachment{
		FileID: "file_overflow",
		Name:   "file_overflow.txt",
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "cannot stage more than 10 attachments")
}
