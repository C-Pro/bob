package safetensors

import "io"

// MmapFile represents a memory-mapped file resource.
type MmapFile interface {
	io.Closer
	Bytes() []byte
}
