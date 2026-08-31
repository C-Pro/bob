package backup

import (
	"compress/gzip"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

// EncodeSnapshot reads a SQLite snapshot file, compresses it with gzip,
// encrypts it with AES-256-GCM using a key derived from secret, and streams
// the assembled artifact directly to w via chained writers.
func EncodeSnapshot(snapshotPath string, secret string, w io.Writer) error {
	snapFile, err := os.Open(snapshotPath)
	if err != nil {
		return fmt.Errorf("failed to open snapshot file: %w", err)
	}
	defer func() { _ = snapFile.Close() }()

	crypter, err := NewCrypter([]byte(secret), nil)
	if err != nil {
		return fmt.Errorf("failed to initialize crypter: %w", err)
	}

	// 1. Write Header
	h := Header{
		Version: HeaderVersion1,
		Salt:    crypter.Salt(),
	}
	if err := WriteHeader(w, h); err != nil {
		return fmt.Errorf("failed to write artifact header: %w", err)
	}

	// 2. Chained Writer: EncryptWriter -> GzipWriter
	encWriter, err := crypter.EncryptWriter(w)
	if err != nil {
		return fmt.Errorf("failed to initialize encrypt writer: %w", err)
	}

	gw := gzip.NewWriter(encWriter)
	if _, err := io.Copy(gw, snapFile); err != nil {
		_ = gw.Close()
		_ = encWriter.Close()
		return fmt.Errorf("gzip compression failed: %w", err)
	}
	if err := gw.Close(); err != nil {
		_ = encWriter.Close()
		return fmt.Errorf("gzip writer close failed: %w", err)
	}
	if err := encWriter.Close(); err != nil {
		return fmt.Errorf("encrypt writer close failed: %w", err)
	}

	return nil
}

// DecodeSnapshot parses an artifact from r, decrypts it using secret, decompresses
// the gzip payload, and streams the resulting SQLite database atomically to destPath.
func DecodeSnapshot(r io.Reader, secret string, destPath string) error {
	// 1. Read header from reader
	hdr, err := ReadHeaderFrom(r)
	if err != nil {
		return fmt.Errorf("invalid backup artifact: %w", err)
	}

	// 2. Chained Reader: DecryptReader -> GzipReader
	crypter, err := NewCrypter([]byte(secret), hdr.Salt)
	if err != nil {
		return fmt.Errorf("failed to derive decryption key: %w", err)
	}
	decReader, err := crypter.DecryptReader(r)
	if err != nil {
		return fmt.Errorf("failed to initialize decrypt reader: %w", err)
	}

	gr, err := gzip.NewReader(decReader)
	if err != nil {
		return fmt.Errorf("gzip reader init failed (wrong SECRET?): %w", err)
	}
	defer func() { _ = gr.Close() }()

	// 3. Atomic write to destPath streaming directly from gzip reader
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

	const maxDecompressedBytes = 5 << 30 // 5 GiB safety limit
	if _, err := io.CopyN(tmpFile, gr, maxDecompressedBytes); err != nil && err != io.EOF {
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
