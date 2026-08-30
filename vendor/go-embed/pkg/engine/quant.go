package engine

import (
	"math"
)

// QuantizedLinear represents an INT8 weight-only quantized matrix with per-row FP32 scaling.
type QuantizedLinear struct {
	Weight []int8    // [rows * cols]
	Scale  []float32 // [rows]
	Bias   []float32 // [rows]
	Rows   int
	Cols   int
}

// QuantizedWordEmbeddings represents INT8 quantized token embeddings with per-token scale.
type QuantizedWordEmbeddings struct {
	Weight []int8    // [VocabSize * HiddenSize]
	Scale  []float32 // [VocabSize]
}

// QuantizedLayer holds INT8 quantized weights for a single transformer layer.
type QuantizedLayer struct {
	Query QuantizedLinear
	Key   QuantizedLinear
	Value QuantizedLinear
	Out   QuantizedLinear

	AttnNormW []float32
	AttnNormB []float32

	FFN1 QuantizedLinear
	FFN2 QuantizedLinear

	FFNNormW []float32
	FFNNormB []float32
}

// QuantizeMatrix converts an FP32 matrix of shape [rows, cols] into per-row INT8 and FP32 scale factors.
func QuantizeMatrix(weights []float32, rows, cols int) ([]int8, []float32) {
	if rows <= 0 || cols <= 0 || len(weights) < rows*cols {
		return nil, nil
	}
	qWeights := make([]int8, rows*cols)
	scales := make([]float32, rows)

	for r := 0; r < rows; r++ {
		rowOff := r * cols
		row := weights[rowOff : rowOff+cols]

		var maxAbs float32
		for _, v := range row {
			abs := float32(math.Abs(float64(v)))
			if math.IsNaN(float64(abs)) {
				continue
			}
			if abs > maxAbs {
				maxAbs = abs
			}
		}

		if maxAbs == 0 {
			scales[r] = 1.0
			continue
		}

		scale := maxAbs / 127.0
		invScale := float32(1.0) / scale
		scales[r] = scale

		for c := 0; c < cols; c++ {
			val := row[c] * invScale
			if math.IsNaN(float64(val)) {
				qWeights[rowOff+c] = 0
				continue
			}
			// Round to nearest integer and clamp to [-127, 127]
			rounded := int(math.Round(float64(val)))
			if rounded > 127 {
				rounded = 127
			} else if rounded < -127 {
				rounded = -127
			}
			qWeights[rowOff+c] = int8(rounded)
		}
	}

	return qWeights, scales
}

// QuantizeLinear creates a QuantizedLinear from FP32 weights and bias.
func QuantizeLinear(weight, bias []float32, rows, cols int) QuantizedLinear {
	if rows <= 0 || cols <= 0 {
		return QuantizedLinear{Rows: rows, Cols: cols}
	}
	qW, scale := QuantizeMatrix(weight, rows, cols)
	return QuantizedLinear{
		Weight: qW,
		Scale:  scale,
		Bias:   bias,
		Rows:   rows,
		Cols:   cols,
	}
}

// QuantizeModel converts an FP32 Model into an in-memory INT8 Quantized Model,
// releasing the heavy FP32 weight slices to reduce memory usage by ~3.6x.
func QuantizeModel(m *Model) *Model {
	if m == nil {
		return nil
	}
	if m.IsQuantized && m.Precision == PrecisionINT8 {
		return m
	}

	qModel := &Model{
		Precision:           PrecisionINT8,
		IsQuantized:         true,
		PositionEmbeddings:  m.PositionEmbeddings,
		TokenTypeEmbeddings: m.TokenTypeEmbeddings,
		EmbeddingsNormW:     m.EmbeddingsNormW,
		EmbeddingsNormB:     m.EmbeddingsNormB,
		Tok:                 m.Tok,
	}

	// 1. Quantize Word Embeddings
	vocabSize := len(m.WordEmbeddings) / HiddenSize
	if vocabSize > 0 {
		qW, qScale := QuantizeMatrix(m.WordEmbeddings, vocabSize, HiddenSize)
		qModel.QWordEmbeddings = QuantizedWordEmbeddings{
			Weight: qW,
			Scale:  qScale,
		}
	}

	// 2. Quantize 12 Transformer Layers
	for l := 0; l < NumLayers; l++ {
		layer := &m.Layers[l]
		ql := &qModel.QLayers[l]

		ql.Query = QuantizeLinear(layer.QueryWeight, layer.QueryBias, HiddenSize, HiddenSize)
		ql.Key = QuantizeLinear(layer.KeyWeight, layer.KeyBias, HiddenSize, HiddenSize)
		ql.Value = QuantizeLinear(layer.ValueWeight, layer.ValueBias, HiddenSize, HiddenSize)
		ql.Out = QuantizeLinear(layer.OutWeight, layer.OutBias, HiddenSize, HiddenSize)
		ql.AttnNormW = layer.AttnNormW
		ql.AttnNormB = layer.AttnNormB

		ql.FFN1 = QuantizeLinear(layer.FFN1Weight, layer.FFN1Bias, IntermediateSize, HiddenSize)
		ql.FFN2 = QuantizeLinear(layer.FFN2Weight, layer.FFN2Bias, HiddenSize, IntermediateSize)
		ql.FFNNormW = layer.FFNNormW
		ql.FFNNormB = layer.FFNNormB
	}

	return qModel
}
