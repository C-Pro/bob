//go:build !unix && !linux && !darwin && !freebsd && !openbsd && !netbsd && !windows

package safetensors

import (
	"fmt"
	"os"
)

type fallbackMmap struct {
	data []byte
	file *os.File
}

func openMmap(f *os.File) (MmapFile, error) {
	data, err := os.ReadFile(f.Name())
	if err != nil {
		return nil, fmt.Errorf("fallback read failed: %w", err)
	}
	return &fallbackMmap{data: data, file: f}, nil
}

func (m *fallbackMmap) Bytes() []byte {
	return m.data
}

func (m *fallbackMmap) Close() error {
	if m.file != nil {
		err := m.file.Close()
		m.file = nil
		return err
	}
	return nil
}
