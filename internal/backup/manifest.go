package backup

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"time"

	"bob/internal/objectstore"
)

const (
	ManifestVersion = 1
	ManifestFile    = "manifest.json"
)

// DBBackupEntry holds metadata for a single database's latest backup.
type DBBackupEntry struct {
	Key       string    `json:"key"`
	Timestamp time.Time `json:"timestamp"`
	Size      int64     `json:"size"`
}

// Manifest tracks the latest backup state for all databases in object storage.
type Manifest struct {
	Version   int                      `json:"version"`
	UpdatedAt time.Time                `json:"updated_at"`
	Databases map[string]DBBackupEntry `json:"databases"`
}

// NewManifest creates an initialized empty Manifest.
func NewManifest() *Manifest {
	return &Manifest{
		Version:   ManifestVersion,
		UpdatedAt: time.Now().UTC(),
		Databases: make(map[string]DBBackupEntry),
	}
}

// FetchManifest retrieves and parses the manifest from object storage.
// If the manifest file does not exist, a new empty manifest is returned.
func FetchManifest(ctx context.Context, obj *objectstore.Client, prefix string) (*Manifest, error) {
	if obj == nil {
		return nil, fmt.Errorf("objectstore client is nil")
	}

	key := prefix + ManifestFile
	rc, err := obj.Get(ctx, key)
	if err != nil {
		if errors.Is(err, objectstore.ErrNotFound) {
			return NewManifest(), nil
		}
		return nil, fmt.Errorf("failed to fetch manifest: %w", err)
	}
	defer func() { _ = rc.Close() }()

	data, err := io.ReadAll(rc)
	if err != nil {
		return nil, fmt.Errorf("failed to read manifest payload: %w", err)
	}

	var m Manifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, fmt.Errorf("failed to unmarshal manifest JSON: %w", err)
	}
	if m.Databases == nil {
		m.Databases = make(map[string]DBBackupEntry)
	}

	return &m, nil
}

// PutManifest serializes and uploads the manifest to object storage.
func PutManifest(ctx context.Context, obj *objectstore.Client, prefix string, m *Manifest) error {
	if obj == nil {
		return fmt.Errorf("objectstore client is nil")
	}
	if m == nil {
		return fmt.Errorf("manifest is nil")
	}

	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal manifest: %w", err)
	}

	key := prefix + ManifestFile
	if err := obj.Put(ctx, key, bytes.NewReader(data), int64(len(data))); err != nil {
		return fmt.Errorf("failed to upload manifest: %w", err)
	}

	return nil
}
