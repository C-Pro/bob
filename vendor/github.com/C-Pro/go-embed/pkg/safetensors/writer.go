package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"unsafe"
)

// TensorData holds raw bytes and metadata for writing a tensor into a Safetensors file.
type TensorData struct {
	DType string
	Shape []int
	Bytes []byte
}

// NewTensorF32 creates TensorData from a float32 slice.
func NewTensorF32(shape []int, data []float32) TensorData {
	raw := make([]byte, len(data)*4)
	for i, v := range data {
		binary.LittleEndian.PutUint32(raw[i*4:(i+1)*4], math.Float32bits(v))
	}
	return TensorData{
		DType: "F32",
		Shape: shape,
		Bytes: raw,
	}
}

// NewTensorBF16 creates TensorData from a uint16 BFloat16 slice.
func NewTensorBF16(shape []int, data []uint16) TensorData {
	raw := make([]byte, len(data)*2)
	for i, v := range data {
		binary.LittleEndian.PutUint16(raw[i*2:(i+1)*2], v)
	}
	return TensorData{
		DType: "BF16",
		Shape: shape,
		Bytes: raw,
	}
}

// NewTensorI8 creates TensorData from an int8 slice.
func NewTensorI8(shape []int, data []int8) TensorData {
	raw := make([]byte, len(data))
	if len(data) > 0 {
		src := unsafe.Slice((*byte)(unsafe.Pointer(&data[0])), len(data))
		copy(raw, src)
	}
	return TensorData{
		DType: "I8",
		Shape: shape,
		Bytes: raw,
	}
}

// WriteFile writes a map of tensors to a Safetensors file atomically with 64-byte aligned data offsets.
func WriteFile(targetPath string, tensors map[string]TensorData) error {
	dir := filepath.Dir(targetPath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create directory: %w", err)
	}

	tmpFile, err := os.CreateTemp(dir, "model_*.tmp")
	if err != nil {
		return fmt.Errorf("failed to create temporary file: %w", err)
	}
	tmpPath := tmpFile.Name()
	defer func() {
		tmpFile.Close()
		os.Remove(tmpPath)
	}()

	// Deterministic tensor ordering
	names := make([]string, 0, len(tensors))
	for name := range tensors {
		names = append(names, name)
	}
	sort.Strings(names)

	headerMap := make(map[string]TensorInfo, len(tensors))
	var currentOffset int64

	for _, name := range names {
		td := tensors[name]
		byteLen := int64(len(td.Bytes))
		headerMap[name] = TensorInfo{
			DType:       td.DType,
			Shape:       td.Shape,
			DataOffsets: [2]int64{currentOffset, currentOffset + byteLen},
		}
		currentOffset += byteLen
	}

	headerJSON, err := json.Marshal(headerMap)
	if err != nil {
		return fmt.Errorf("failed to marshal header JSON: %w", err)
	}

	// Pad header JSON with spaces so (8 + len(headerJSON)) is a multiple of 64
	totalHeaderSize := 8 + len(headerJSON)
	remainder := totalHeaderSize % 64
	if remainder != 0 {
		padding := 64 - remainder
		for i := 0; i < padding; i++ {
			headerJSON = append(headerJSON, ' ')
		}
	}

	// Write 8-byte little-endian header size
	headerLen := uint64(len(headerJSON))
	if err := binary.Write(tmpFile, binary.LittleEndian, headerLen); err != nil {
		return fmt.Errorf("failed to write header size: %w", err)
	}

	// Write header JSON
	if _, err := tmpFile.Write(headerJSON); err != nil {
		return fmt.Errorf("failed to write header JSON: %w", err)
	}

	// Write tensor byte payloads
	for _, name := range names {
		td := tensors[name]
		if _, err := tmpFile.Write(td.Bytes); err != nil {
			return fmt.Errorf("failed to write tensor bytes for %s: %w", name, err)
		}
	}

	if err := tmpFile.Sync(); err != nil {
		return fmt.Errorf("failed to sync file: %w", err)
	}
	if err := tmpFile.Close(); err != nil {
		return fmt.Errorf("failed to close file: %w", err)
	}

	// Atomic rename
	if err := os.Rename(tmpPath, targetPath); err != nil {
		return fmt.Errorf("failed to rename to target path: %w", err)
	}

	return nil
}
