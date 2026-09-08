package sandbox

import (
	"bytes"
	"fmt"
	"sync"
)

// DefaultMaxOutputBytes is the default maximum number of bytes captured per output stream (1 MB).
const DefaultMaxOutputBytes = 1024 * 1024

// BoundedBuffer implements io.Writer and captures at most maxBytes.
// Excess bytes are silently discarded to prevent memory exhaustion, and
// String() appends an explicit truncation marker if output was truncated.
type BoundedBuffer struct {
	mu        sync.Mutex
	buf       bytes.Buffer
	maxBytes  int
	truncated bool
	total     int64
}

// NewBoundedBuffer creates a BoundedBuffer with the specified maximum byte limit.
// If maxBytes <= 0, DefaultMaxOutputBytes is used.
func NewBoundedBuffer(maxBytes int) *BoundedBuffer {
	if maxBytes <= 0 {
		maxBytes = DefaultMaxOutputBytes
	}
	return &BoundedBuffer{
		maxBytes: maxBytes,
	}
}

// Write writes data to the buffer up to maxBytes. Additional data is discarded and marked as truncated.
func (b *BoundedBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	b.total += int64(len(p))
	if b.buf.Len() >= b.maxBytes {
		b.truncated = true
		return len(p), nil
	}

	remaining := b.maxBytes - b.buf.Len()
	if len(p) > remaining {
		b.buf.Write(p[:remaining])
		b.truncated = true
		return len(p), nil
	}

	return b.buf.Write(p)
}

// String returns the buffered content, appending a truncation notice if the limit was exceeded.
func (b *BoundedBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()

	if !b.truncated {
		return b.buf.String()
	}
	return b.buf.String() + fmt.Sprintf("\n... [output truncated: exceeded %d bytes limit] ...", b.maxBytes)
}

// Truncated reports whether output was truncated due to exceeding the limit.
func (b *BoundedBuffer) Truncated() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.truncated
}

// TotalBytesWritten returns the total number of bytes written before truncation.
func (b *BoundedBuffer) TotalBytesWritten() int64 {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.total
}
