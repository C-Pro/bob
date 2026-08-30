package engine

import (
	"math"
)

// PrecisionMode defines the numerical precision mode of the engine.
type PrecisionMode int

const (
	PrecisionFP32 PrecisionMode = iota
	PrecisionBF16
	PrecisionINT8
)

func (p PrecisionMode) String() string {
	switch p {
	case PrecisionBF16:
		return "BF16"
	case PrecisionINT8:
		return "INT8"
	default:
		return "FP32"
	}
}

// Float32ToBFloat16 converts an FP32 value to BFloat16 with round-to-nearest-even.
func Float32ToBFloat16(f float32) uint16 {
	bits := math.Float32bits(f)
	if math.IsNaN(float64(f)) {
		return uint16((bits >> 16) | 0x0040) // Return quiet NaN
	}
	// Round to nearest even
	lsb := (bits >> 16) & 1
	roundingBias := uint32(0x7FFF) + lsb
	bits += roundingBias
	return uint16(bits >> 16)
}

// BFloat16ToFloat32 converts a BFloat16 (uint16) back to FP32.
func BFloat16ToFloat32(bf uint16) float32 {
	return math.Float32frombits(uint32(bf) << 16)
}

// Float32sToBFloat16s converts a slice of FP32 floats into BFloat16 uint16s.
func Float32sToBFloat16s(src []float32) []uint16 {
	dst := make([]uint16, len(src))
	for i, v := range src {
		dst[i] = Float32ToBFloat16(v)
	}
	return dst
}

// BF16Linear represents a linear weight matrix stored in BFloat16 format.
type BF16Linear struct {
	Weight []uint16  // [rows * cols]
	Bias   []float32 // [rows]
	Rows   int
	Cols   int
}

// BF16Layer holds BFloat16 weights for a single transformer layer.
type BF16Layer struct {
	Query BF16Linear
	Key   BF16Linear
	Value BF16Linear
	Out   BF16Linear

	AttnNormW []float32
	AttnNormB []float32

	FFN1 BF16Linear
	FFN2 BF16Linear

	FFNNormW []float32
	FFNNormB []float32
}

// ConvertToBF16Model converts an FP32 Model into an in-memory BFloat16 Model,
// releasing FP32 weights to reduce memory footprint by 2x (449 MB -> 225 MB).
func ConvertToBF16Model(m *Model) *Model {
	if m == nil {
		return nil
	}
	if m.Precision == PrecisionBF16 {
		return m
	}

	bf16Model := &Model{
		Precision:           PrecisionBF16,
		IsQuantized:         true,
		PositionEmbeddings:  m.PositionEmbeddings,
		TokenTypeEmbeddings: m.TokenTypeEmbeddings,
		EmbeddingsNormW:     m.EmbeddingsNormW,
		EmbeddingsNormB:     m.EmbeddingsNormB,
		Tok:                 m.Tok,
	}

	// 1. Convert Word Embeddings to BF16
	bf16Model.BF16WordEmbeddings = Float32sToBFloat16s(m.WordEmbeddings)

	// 2. Convert 12 Transformer Layers to BF16
	for l := 0; l < NumLayers; l++ {
		layer := &m.Layers[l]
		bfl := &bf16Model.BF16Layers[l]

		bfl.Query = BF16Linear{
			Weight: Float32sToBFloat16s(layer.QueryWeight),
			Bias:   layer.QueryBias,
			Rows:   HiddenSize,
			Cols:   HiddenSize,
		}
		bfl.Key = BF16Linear{
			Weight: Float32sToBFloat16s(layer.KeyWeight),
			Bias:   layer.KeyBias,
			Rows:   HiddenSize,
			Cols:   HiddenSize,
		}
		bfl.Value = BF16Linear{
			Weight: Float32sToBFloat16s(layer.ValueWeight),
			Bias:   layer.ValueBias,
			Rows:   HiddenSize,
			Cols:   HiddenSize,
		}
		bfl.Out = BF16Linear{
			Weight: Float32sToBFloat16s(layer.OutWeight),
			Bias:   layer.OutBias,
			Rows:   HiddenSize,
			Cols:   HiddenSize,
		}
		bfl.AttnNormW = layer.AttnNormW
		bfl.AttnNormB = layer.AttnNormB

		bfl.FFN1 = BF16Linear{
			Weight: Float32sToBFloat16s(layer.FFN1Weight),
			Bias:   layer.FFN1Bias,
			Rows:   IntermediateSize,
			Cols:   HiddenSize,
		}
		bfl.FFN2 = BF16Linear{
			Weight: Float32sToBFloat16s(layer.FFN2Weight),
			Bias:   layer.FFN2Bias,
			Rows:   HiddenSize,
			Cols:   IntermediateSize,
		}
		bfl.FFNNormW = layer.FFNNormW
		bfl.FFNNormB = layer.FFNNormB
	}

	return bf16Model
}
