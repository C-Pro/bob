package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"unsafe"
)

// TensorInfo contains metadata for a single tensor in a Safetensors file.
type TensorInfo struct {
	DType       string   `json:"dtype"`
	Shape       []int    `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"`
}

// Safetensors represents an opened, memory-mapped Safetensors file.
type Safetensors struct {
	mmap       MmapFile
	headerSize int64
	Tensors    map[string]TensorInfo
	dataOffset int64
}

// Open opens and memory-maps a Safetensors file.
func Open(path string) (*Safetensors, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open safetensors file: %w", err)
	}

	mmapFile, err := openMmap(f)
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("failed to memory-map safetensors file: %w", err)
	}

	allBytes := mmapFile.Bytes()
	if len(allBytes) < 8 {
		mmapFile.Close()
		return nil, fmt.Errorf("file too small for safetensors header")
	}

	headerLen := binary.LittleEndian.Uint64(allBytes[0:8])
	if headerLen == 0 || headerLen > uint64(len(allBytes)-8) || headerLen > 100*1024*1024 {
		mmapFile.Close()
		return nil, fmt.Errorf("invalid header length %d (file size %d)", headerLen, len(allBytes))
	}

	headerBytes := allBytes[8 : 8+headerLen]

	var rawMap map[string]json.RawMessage
	if err := json.Unmarshal(headerBytes, &rawMap); err != nil {
		mmapFile.Close()
		return nil, fmt.Errorf("failed to unmarshal header JSON: %w", err)
	}

	tensors := make(map[string]TensorInfo, len(rawMap))
	for k, v := range rawMap {
		if k == "__metadata__" {
			continue
		}
		var info TensorInfo
		if err := json.Unmarshal(v, &info); err != nil {
			mmapFile.Close()
			return nil, fmt.Errorf("failed to parse tensor info for %s: %w", k, err)
		}
		tensors[k] = info
	}

	dataOffset := int64(8 + headerLen)
	return &Safetensors{
		mmap:       mmapFile,
		headerSize: int64(headerLen),
		Tensors:    tensors,
		dataOffset: dataOffset,
	}, nil
}

// Close unmaps memory and closes the underlying file descriptor.
func (s *Safetensors) Close() error {
	if s != nil && s.mmap != nil {
		err := s.mmap.Close()
		s.mmap = nil
		return err
	}
	return nil
}

func validateTensorView(s *Safetensors, name string, expectedDTypes []string, bytesPerElement int) (int, []byte, error) {
	if s == nil || s.mmap == nil {
		return 0, nil, fmt.Errorf("safetensors file is nil or closed")
	}
	info, ok := s.Tensors[name]
	if !ok {
		return 0, nil, fmt.Errorf("tensor %q not found in safetensors", name)
	}
	matchedDType := false
	for _, dt := range expectedDTypes {
		if info.DType == dt {
			matchedDType = true
			break
		}
	}
	if !matchedDType {
		return 0, nil, fmt.Errorf("tensor %q has dtype %s, expected one of %v", name, info.DType, expectedDTypes)
	}

	if len(info.Shape) == 0 {
		return 0, nil, fmt.Errorf("tensor %q has empty shape", name)
	}
	numElements := 1
	for _, dim := range info.Shape {
		if dim < 0 {
			return 0, nil, fmt.Errorf("tensor %q has negative dimension %d", name, dim)
		}
		if dim > 0 && numElements > math.MaxInt/dim {
			return 0, nil, fmt.Errorf("tensor %q shape causes integer overflow", name)
		}
		numElements *= dim
	}

	if info.DataOffsets[0] < 0 || info.DataOffsets[1] < 0 || info.DataOffsets[0] > info.DataOffsets[1] {
		return 0, nil, fmt.Errorf("tensor %q has invalid offsets [%d, %d]", name, info.DataOffsets[0], info.DataOffsets[1])
	}
	if info.DataOffsets[1] > math.MaxInt64-s.dataOffset {
		return 0, nil, fmt.Errorf("tensor %q offsets overflow", name)
	}

	start := s.dataOffset + info.DataOffsets[0]
	end := s.dataOffset + info.DataOffsets[1]
	allBytes := s.mmap.Bytes()

	if start < 0 || end > int64(len(allBytes)) || start > end {
		return 0, nil, fmt.Errorf("tensor %q offsets [%d, %d] out of bounds (file size %d)", name, start, end, len(allBytes))
	}

	expectedBytes := int64(numElements) * int64(bytesPerElement)
	if end-start != expectedBytes {
		return 0, nil, fmt.Errorf("tensor %q byte length %d != expected %d", name, end-start, expectedBytes)
	}

	raw := allBytes[start:end]
	return numElements, raw, nil
}

// TensorF32View returns a float32 slice backed directly by mmap memory if 4-byte aligned,
// or a newly copied slice if unaligned.
func (s *Safetensors) TensorF32View(name string) ([]float32, error) {
	numElements, raw, err := validateTensorView(s, name, []string{"F32"}, 4)
	if err != nil {
		return nil, err
	}
	if numElements == 0 {
		return []float32{}, nil
	}

	if uintptr(unsafe.Pointer(&raw[0]))%4 == 0 {
		return unsafe.Slice((*float32)(unsafe.Pointer(&raw[0])), numElements), nil
	}

	// Unaligned fallback: copy
	out := make([]float32, numElements)
	for i := 0; i < numElements; i++ {
		bits := binary.LittleEndian.Uint32(raw[i*4 : (i+1)*4])
		out[i] = math.Float32frombits(bits)
	}
	return out, nil
}

// TensorBF16View returns a uint16 BFloat16 slice backed directly by mmap memory if 2-byte aligned,
// or a newly copied slice if unaligned.
func (s *Safetensors) TensorBF16View(name string) ([]uint16, error) {
	numElements, raw, err := validateTensorView(s, name, []string{"BF16"}, 2)
	if err != nil {
		return nil, err
	}
	if numElements == 0 {
		return []uint16{}, nil
	}

	if uintptr(unsafe.Pointer(&raw[0]))%2 == 0 {
		return unsafe.Slice((*uint16)(unsafe.Pointer(&raw[0])), numElements), nil
	}

	out := make([]uint16, numElements)
	for i := 0; i < numElements; i++ {
		out[i] = binary.LittleEndian.Uint16(raw[i*2 : (i+1)*2])
	}
	return out, nil
}

// TensorI8View returns an int8 slice backed directly by mmap memory.
func (s *Safetensors) TensorI8View(name string) ([]int8, error) {
	numElements, raw, err := validateTensorView(s, name, []string{"I8", "INT8"}, 1)
	if err != nil {
		return nil, err
	}
	if numElements == 0 {
		return []int8{}, nil
	}

	return unsafe.Slice((*int8)(unsafe.Pointer(&raw[0])), numElements), nil
}

// ReadTensorF32 reads a float32 tensor by name.
func (s *Safetensors) ReadTensorF32(name string) ([]float32, error) {
	return s.TensorF32View(name)
}

// ReadTensorF32Into reads a float32 tensor directly into an existing slice.
func (s *Safetensors) ReadTensorF32Into(name string, dest []float32) error {
	view, err := s.TensorF32View(name)
	if err != nil {
		return err
	}
	if len(dest) < len(view) {
		return fmt.Errorf("destination slice len (%d) < tensor elements (%d)", len(dest), len(view))
	}
	copy(dest, view)
	return nil
}
