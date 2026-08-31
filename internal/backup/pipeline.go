package backup

import (
	"bytes"
	"compress/gzip"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

// EncodeSnapshot reads a SQLite snapshot file, compresses it with gzip,
// encrypts it with AES-256-GCM using a key derived from secret, and returns
// the assembled artifact including the BOBB header.
func EncodeSnapshot(snapshotPath string, secret string) ([]byte, error) {
	data, err := os.ReadFile(snapshotPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read snapshot file: %w", err)
	}

	// 1. Gzip compression
	var gzipped bytes.Buffer
	gw := gzip.NewWriter(&gzipped)
	if _, err := gw.Write(data); err != nil {
		_ = gw.Close()
		return nil, fmt.Errorf("gzip compression failed: %w", err)
	}
	if err := gw.Close(); err != nil {
		return nil, fmt.Errorf("gzip writer close failed: %w", err)
	}

	// 2. Encryption
	crypter, err := NewCrypter([]byte(secret), nil)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize crypter: %w", err)
	}
	ciphertext, err := crypter.Encrypt(gzipped.Bytes())
	if err != nil {
		return nil, fmt.Errorf("encryption failed: %w", err)
	}

	// 3. Assemble Header + Ciphertext
	var artifact bytes.Buffer
	h := Header{
		Version: HeaderVersion1,
		Salt:    crypter.Salt(),
	}
	if err := WriteHeader(&artifact, h, ciphertext); err != nil {
		return nil, fmt.Errorf("failed to write artifact header: %w", err)
	}

	return artifact.Bytes(), nil
}

// DecodeSnapshot parses an artifact, decrypts it using secret, decompresses
// the gzip payload, and writes the resulting SQLite database atomically to destPath.
func DecodeSnapshot(artifactData []byte, secret string, destPath string) error {
	// 1. Read header
	hdr, ciphertext, err := ReadHeader(artifactData)
	if err != nil {
		return fmt.Errorf("invalid backup artifact: %w", err)
	}

	// 2. Decrypt
	crypter, err := NewCrypter([]byte(secret), hdr.Salt)
	if err != nil {
		return fmt.Errorf("failed to derive decryption key: %w", err)
	}
	gzipped, err := crypter.Decrypt(ciphertext)
	if err != nil {
		return fmt.Errorf("failed to decrypt artifact (wrong SECRET?): %w", err)
	}

	// 3. Decompress
	gr, err := gzip.NewReader(bytes.NewReader(gzipped))
	if err != nil {
		return fmt.Errorf("gzip reader init failed: %w", err)
	}
	defer func() { _ = gr.Close() }()

	decompressed, err := io.ReadAll(gr)
	if err != nil {
		return fmt.Errorf("gzip decompression failed: %w", err)
	}

	// 4. Atomic write to destPath
	dir := filepath.Dir(destPath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create destination directory: %w", err)
	}

	tmpFile, err := os.CreateTemp(dir, "restore-*.tmp")
	if err != nil {
		return fmt.Errorf("failed to create temp file: %w", err)
	}
	tmpName := tmpFile.Name()
	defer func() {
		_ = tmpFile.Close()
		_ = os.Remove(tmpName)
	}()

	if _, err := tmpFile.Write(decompressed); err != nil {
		return fmt.Errorf("failed to write decompressed data: %w", err)
	}
	if err := tmpFile.Sync(); err != nil {
		return fmt.Errorf("failed to sync temp file: %w", err)
	}
	if err := tmpFile.Close(); err != nil {
		return fmt.Errorf("failed to close temp file: %w", err)
	}

	if err := os.Rename(tmpName, destPath); err != nil {
		return fmt.Errorf("failed to rename temp file to %s: %w", destPath, err)
	}

	return nil
}
