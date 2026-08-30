//go:build unix || linux || darwin || freebsd || openbsd || netbsd

package safetensors

import (
	"fmt"
	"os"
	"syscall"
)

type unixMmap struct {
	data []byte
	file *os.File
}

func openMmap(f *os.File) (MmapFile, error) {
	fi, err := f.Stat()
	if err != nil {
		return nil, err
	}
	size := fi.Size()
	if size == 0 {
		return nil, fmt.Errorf("cannot mmap empty file")
	}

	data, err := syscall.Mmap(int(f.Fd()), 0, int(size), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, fmt.Errorf("mmap failed: %w", err)
	}

	return &unixMmap{data: data, file: f}, nil
}

func (m *unixMmap) Bytes() []byte {
	return m.data
}

func (m *unixMmap) Close() error {
	var err error
	if m.data != nil {
		err = syscall.Munmap(m.data)
		m.data = nil
	}
	if m.file != nil {
		if fErr := m.file.Close(); fErr != nil && err == nil {
			err = fErr
		}
		m.file = nil
	}
	return err
}
