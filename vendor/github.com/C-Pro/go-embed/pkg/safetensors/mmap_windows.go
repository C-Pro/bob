//go:build windows

package safetensors

import (
	"fmt"
	"os"
	"syscall"
	"unsafe"
)

type windowsMmap struct {
	data   []byte
	file   *os.File
	handle syscall.Handle
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

	hMap, err := syscall.CreateFileMapping(
		syscall.Handle(f.Fd()),
		nil,
		syscall.PAGE_READONLY,
		uint32(size>>32),
		uint32(size&0xffffffff),
		nil,
	)
	if err != nil && err != syscall.ERROR_SUCCESS {
		return nil, fmt.Errorf("CreateFileMapping failed: %w", err)
	}

	addr, err := syscall.MapViewOfFile(hMap, syscall.FILE_MAP_READ, 0, 0, uintptr(size))
	if err != nil && err != syscall.ERROR_SUCCESS {
		syscall.CloseHandle(hMap)
		return nil, fmt.Errorf("MapViewOfFile failed: %w", err)
	}

	data := unsafe.Slice((*byte)(unsafe.Pointer(addr)), size)
	return &windowsMmap{
		data:   data,
		file:   f,
		handle: hMap,
	}, nil
}

func (m *windowsMmap) Bytes() []byte {
	return m.data
}

func (m *windowsMmap) Close() error {
	var err error
	if len(m.data) > 0 {
		addr := uintptr(unsafe.Pointer(&m.data[0]))
		if uErr := syscall.UnmapViewOfFile(addr); uErr != nil && uErr != syscall.ERROR_SUCCESS {
			err = uErr
		}
		m.data = nil
	}
	if m.handle != 0 {
		if cErr := syscall.CloseHandle(m.handle); cErr != nil && cErr != syscall.ERROR_SUCCESS && err == nil {
			err = cErr
		}
		m.handle = 0
	}
	if m.file != nil {
		if fErr := m.file.Close(); fErr != nil && err == nil {
			err = fErr
		}
		m.file = nil
	}
	return err
}
