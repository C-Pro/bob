package engine

import (
	"fmt"
	"os"
	"strings"

	"github.com/C-Pro/go-embed/pkg/safetensors"
	"github.com/C-Pro/go-embed/pkg/tokenizer"
)

const (
	HiddenSize       = 384
	IntermediateSize = 1536
	NumLayers        = 12
	NumHeads         = 12
	HeadDim          = 32
	MaxSeqLen        = 512
	VocabSize        = 250037
	LayerNormEps     = 1e-12
)

// Layer holds all parameter weights and biases for a single transformer layer.
// All matrices are stored in row-major order:
// - Query, Key, Value, Out: [384, 384] (each row has 384 elements)
// - FFN1: [1536, 384] (each row has 384 elements)
// - FFN2: [384, 1536] (each row has 1536 elements)
type Layer struct {
	QueryWeight []float32
	QueryBias   []float32
	KeyWeight   []float32
	KeyBias     []float32
	ValueWeight []float32
	ValueBias   []float32
	OutWeight   []float32
	OutBias     []float32
	AttnNormW   []float32
	AttnNormB   []float32

	FFN1Weight []float32
	FFN1Bias   []float32
	FFN2Weight []float32
	FFN2Bias   []float32
	FFNNormW   []float32
	FFNNormB   []float32
}

// Model represents the self-contained transformer embedding model.
// It supports FP32 full precision, BF16 16-bit float, and INT8 8-bit dynamic quantization.
type Model struct {
	Precision   PrecisionMode
	IsQuantized bool

	// FP32 Parameters
	WordEmbeddings      []float32 // [250037 * 384]
	PositionEmbeddings  []float32 // [512 * 384]
	TokenTypeEmbeddings []float32 // [2 * 384]
	EmbeddingsNormW     []float32 // [384]
	EmbeddingsNormB     []float32 // [384]
	Layers              [NumLayers]Layer

	// BF16 Parameters
	BF16WordEmbeddings []uint16 // [250037 * 384]
	BF16Layers         [NumLayers]BF16Layer

	// INT8 Quantized Parameters
	QWordEmbeddings QuantizedWordEmbeddings
	QLayers         [NumLayers]QuantizedLayer

	Tok *tokenizer.Tokenizer
	st  *safetensors.Safetensors
}

// Close releases any memory-mapped file resources held by the model.
func (m *Model) Close() error {
	if m.st != nil {
		err := m.st.Close()
		m.st = nil
		return err
	}
	return nil
}

// LoadModel loads the model directly from a safetensors file and tokenizer.json in FP32 precision.
func LoadModel(modelPath, tokenizerPath string) (*Model, error) {
	return LoadModelWithPrecision(modelPath, tokenizerPath, PrecisionFP32)
}

// LoadModelWithOptions loads model weights with optional INT8 quantization (backwards compatible).
func LoadModelWithOptions(modelPath, tokenizerPath string, quantize bool) (*Model, error) {
	if quantize {
		return LoadModelWithPrecision(modelPath, tokenizerPath, PrecisionINT8)
	}
	return LoadModelWithPrecision(modelPath, tokenizerPath, PrecisionFP32)
}

// VocabSize returns the vocabulary size of the loaded model.
func (m *Model) VocabSize() int {
	if len(m.WordEmbeddings) > 0 {
		return len(m.WordEmbeddings) / HiddenSize
	}
	if len(m.BF16WordEmbeddings) > 0 {
		return len(m.BF16WordEmbeddings) / HiddenSize
	}
	if len(m.QWordEmbeddings.Scale) > 0 {
		return len(m.QWordEmbeddings.Scale)
	}
	return VocabSize
}

func readTensorF32Flex(st *safetensors.Safetensors, name string) ([]float32, error) {
	if st == nil {
		return nil, fmt.Errorf("safetensors is nil")
	}
	if t, err := st.TensorF32View(name); err == nil {
		return t, nil
	}
	prefixes := []string{"roberta.", "bert.", "transformer.", "model."}
	for _, p := range prefixes {
		if t, err := st.TensorF32View(p + name); err == nil {
			return t, nil
		}
	}
	return nil, fmt.Errorf("tensor %q (and standard prefixes) not found in safetensors", name)
}

func readTensorBF16Flex(st *safetensors.Safetensors, name string) ([]uint16, error) {
	if st == nil {
		return nil, fmt.Errorf("safetensors is nil")
	}
	if t, err := st.TensorBF16View(name); err == nil {
		return t, nil
	}
	prefixes := []string{"roberta.", "bert.", "transformer.", "model."}
	for _, p := range prefixes {
		if t, err := st.TensorBF16View(p + name); err == nil {
			return t, nil
		}
	}
	return nil, fmt.Errorf("tensor %q (and standard prefixes) not found in safetensors", name)
}

func readTensorI8Flex(st *safetensors.Safetensors, name string) ([]int8, error) {
	if st == nil {
		return nil, fmt.Errorf("safetensors is nil")
	}
	if t, err := st.TensorI8View(name); err == nil {
		return t, nil
	}
	prefixes := []string{"roberta.", "bert.", "transformer.", "model."}
	for _, p := range prefixes {
		if t, err := st.TensorI8View(p + name); err == nil {
			return t, nil
		}
	}
	return nil, fmt.Errorf("tensor %q (and standard prefixes) not found in safetensors", name)
}

// Validate verifies that all tensor weights, biases, and parameters match the expected network architecture dimensions.
func (m *Model) Validate() error {
	if m == nil {
		return fmt.Errorf("model is nil")
	}
	if m.Tok == nil {
		return fmt.Errorf("model tokenizer is nil")
	}
	vocabSize := m.VocabSize()
	if vocabSize <= 0 {
		return fmt.Errorf("invalid vocabulary size %d", vocabSize)
	}

	if len(m.PositionEmbeddings) < MaxSeqLen*HiddenSize {
		return fmt.Errorf("position embeddings len %d < expected %d", len(m.PositionEmbeddings), MaxSeqLen*HiddenSize)
	}
	if len(m.EmbeddingsNormW) != HiddenSize || len(m.EmbeddingsNormB) != HiddenSize {
		return fmt.Errorf("embeddings norm weights/biases invalid length: W=%d, B=%d, expected %d", len(m.EmbeddingsNormW), len(m.EmbeddingsNormB), HiddenSize)
	}

	switch m.Precision {
	case PrecisionBF16:
		if len(m.BF16WordEmbeddings) < HiddenSize || len(m.BF16WordEmbeddings)%HiddenSize != 0 {
			return fmt.Errorf("bf16 word embeddings len %d invalid (must be multiple of %d)", len(m.BF16WordEmbeddings), HiddenSize)
		}
		for i := 0; i < NumLayers; i++ {
			l := &m.BF16Layers[i]
			if len(l.Query.Weight) != HiddenSize*HiddenSize || len(l.Query.Bias) != HiddenSize {
				return fmt.Errorf("layer %d query invalid dimensions", i)
			}
			if len(l.Key.Weight) != HiddenSize*HiddenSize || len(l.Key.Bias) != HiddenSize {
				return fmt.Errorf("layer %d key invalid dimensions", i)
			}
			if len(l.Value.Weight) != HiddenSize*HiddenSize || len(l.Value.Bias) != HiddenSize {
				return fmt.Errorf("layer %d value invalid dimensions", i)
			}
			if len(l.Out.Weight) != HiddenSize*HiddenSize || len(l.Out.Bias) != HiddenSize {
				return fmt.Errorf("layer %d out invalid dimensions", i)
			}
			if len(l.AttnNormW) != HiddenSize || len(l.AttnNormB) != HiddenSize {
				return fmt.Errorf("layer %d attn norm invalid dimensions", i)
			}
			if len(l.FFN1.Weight) != IntermediateSize*HiddenSize || len(l.FFN1.Bias) != IntermediateSize {
				return fmt.Errorf("layer %d ffn1 invalid dimensions", i)
			}
			if len(l.FFN2.Weight) != HiddenSize*IntermediateSize || len(l.FFN2.Bias) != HiddenSize {
				return fmt.Errorf("layer %d ffn2 invalid dimensions", i)
			}
			if len(l.FFNNormW) != HiddenSize || len(l.FFNNormB) != HiddenSize {
				return fmt.Errorf("layer %d ffn norm invalid dimensions", i)
			}
		}
	case PrecisionINT8:
		if len(m.QWordEmbeddings.Weight) < HiddenSize || len(m.QWordEmbeddings.Weight)%HiddenSize != 0 {
			return fmt.Errorf("int8 word embeddings len %d invalid", len(m.QWordEmbeddings.Weight))
		}
		if len(m.QWordEmbeddings.Scale) != len(m.QWordEmbeddings.Weight)/HiddenSize {
			return fmt.Errorf("int8 word embeddings scale len %d != tokens %d", len(m.QWordEmbeddings.Scale), len(m.QWordEmbeddings.Weight)/HiddenSize)
		}
		for i := 0; i < NumLayers; i++ {
			l := &m.QLayers[i]
			if len(l.Query.Weight) != HiddenSize*HiddenSize || len(l.Query.Scale) != HiddenSize || len(l.Query.Bias) != HiddenSize {
				return fmt.Errorf("layer %d query invalid dimensions", i)
			}
			if len(l.Key.Weight) != HiddenSize*HiddenSize || len(l.Key.Scale) != HiddenSize || len(l.Key.Bias) != HiddenSize {
				return fmt.Errorf("layer %d key invalid dimensions", i)
			}
			if len(l.Value.Weight) != HiddenSize*HiddenSize || len(l.Value.Scale) != HiddenSize || len(l.Value.Bias) != HiddenSize {
				return fmt.Errorf("layer %d value invalid dimensions", i)
			}
			if len(l.Out.Weight) != HiddenSize*HiddenSize || len(l.Out.Scale) != HiddenSize || len(l.Out.Bias) != HiddenSize {
				return fmt.Errorf("layer %d out invalid dimensions", i)
			}
			if len(l.AttnNormW) != HiddenSize || len(l.AttnNormB) != HiddenSize {
				return fmt.Errorf("layer %d attn norm invalid dimensions", i)
			}
			if len(l.FFN1.Weight) != IntermediateSize*HiddenSize || len(l.FFN1.Scale) != IntermediateSize || len(l.FFN1.Bias) != IntermediateSize {
				return fmt.Errorf("layer %d ffn1 invalid dimensions", i)
			}
			if len(l.FFN2.Weight) != HiddenSize*IntermediateSize || len(l.FFN2.Scale) != HiddenSize || len(l.FFN2.Bias) != HiddenSize {
				return fmt.Errorf("layer %d ffn2 invalid dimensions", i)
			}
			if len(l.FFNNormW) != HiddenSize || len(l.FFNNormB) != HiddenSize {
				return fmt.Errorf("layer %d ffn norm invalid dimensions", i)
			}
		}
	default:
		if len(m.WordEmbeddings) < HiddenSize || len(m.WordEmbeddings)%HiddenSize != 0 {
			return fmt.Errorf("word embeddings len %d invalid (must be multiple of %d)", len(m.WordEmbeddings), HiddenSize)
		}
		for i := 0; i < NumLayers; i++ {
			l := &m.Layers[i]
			if len(l.QueryWeight) != HiddenSize*HiddenSize || len(l.QueryBias) != HiddenSize {
				return fmt.Errorf("layer %d query invalid dimensions", i)
			}
			if len(l.KeyWeight) != HiddenSize*HiddenSize || len(l.KeyBias) != HiddenSize {
				return fmt.Errorf("layer %d key invalid dimensions", i)
			}
			if len(l.ValueWeight) != HiddenSize*HiddenSize || len(l.ValueBias) != HiddenSize {
				return fmt.Errorf("layer %d value invalid dimensions", i)
			}
			if len(l.OutWeight) != HiddenSize*HiddenSize || len(l.OutBias) != HiddenSize {
				return fmt.Errorf("layer %d out invalid dimensions", i)
			}
			if len(l.AttnNormW) != HiddenSize || len(l.AttnNormB) != HiddenSize {
				return fmt.Errorf("layer %d attn norm invalid dimensions", i)
			}
			if len(l.FFN1Weight) != IntermediateSize*HiddenSize || len(l.FFN1Bias) != IntermediateSize {
				return fmt.Errorf("layer %d ffn1 invalid dimensions", i)
			}
			if len(l.FFN2Weight) != HiddenSize*IntermediateSize || len(l.FFN2Bias) != HiddenSize {
				return fmt.Errorf("layer %d ffn2 invalid dimensions", i)
			}
			if len(l.FFNNormW) != HiddenSize || len(l.FFNNormB) != HiddenSize {
				return fmt.Errorf("layer %d ffn norm invalid dimensions", i)
			}
		}
	}
	return nil
}

// loadFP32Model directly memory-maps a full-precision model.safetensors file.
func loadFP32Model(modelPath, tokenizerPath string) (m *Model, err error) {
	tok, err := tokenizer.LoadFromFile(tokenizerPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load tokenizer: %w", err)
	}

	st, err := safetensors.Open(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open safetensors file: %w", err)
	}
	defer func() {
		if err != nil {
			st.Close()
		}
	}()

	m = &Model{
		Tok:       tok,
		Precision: PrecisionFP32,
		st:        st,
	}

	if m.WordEmbeddings, err = readTensorF32Flex(st, "embeddings.word_embeddings.weight"); err != nil {
		return nil, err
	}
	if m.PositionEmbeddings, err = readTensorF32Flex(st, "embeddings.position_embeddings.weight"); err != nil {
		return nil, err
	}
	if m.TokenTypeEmbeddings, err = readTensorF32Flex(st, "embeddings.token_type_embeddings.weight"); err != nil {
		m.TokenTypeEmbeddings = make([]float32, HiddenSize)
		err = nil
	}
	if m.EmbeddingsNormW, err = readTensorF32Flex(st, "embeddings.LayerNorm.weight"); err != nil {
		return nil, err
	}
	if m.EmbeddingsNormB, err = readTensorF32Flex(st, "embeddings.LayerNorm.bias"); err != nil {
		return nil, err
	}

	for i := 0; i < NumLayers; i++ {
		prefix := fmt.Sprintf("encoder.layer.%d.", i)
		l := &m.Layers[i]

		if l.QueryWeight, err = readTensorF32Flex(st, prefix+"attention.self.query.weight"); err != nil {
			return nil, err
		}
		if l.QueryBias, err = readTensorF32Flex(st, prefix+"attention.self.query.bias"); err != nil {
			return nil, err
		}
		if l.KeyWeight, err = readTensorF32Flex(st, prefix+"attention.self.key.weight"); err != nil {
			return nil, err
		}
		if l.KeyBias, err = readTensorF32Flex(st, prefix+"attention.self.key.bias"); err != nil {
			return nil, err
		}
		if l.ValueWeight, err = readTensorF32Flex(st, prefix+"attention.self.value.weight"); err != nil {
			return nil, err
		}
		if l.ValueBias, err = readTensorF32Flex(st, prefix+"attention.self.value.bias"); err != nil {
			return nil, err
		}
		if l.OutWeight, err = readTensorF32Flex(st, prefix+"attention.output.dense.weight"); err != nil {
			return nil, err
		}
		if l.OutBias, err = readTensorF32Flex(st, prefix+"attention.output.dense.bias"); err != nil {
			return nil, err
		}
		if l.AttnNormW, err = readTensorF32Flex(st, prefix+"attention.output.LayerNorm.weight"); err != nil {
			return nil, err
		}
		if l.AttnNormB, err = readTensorF32Flex(st, prefix+"attention.output.LayerNorm.bias"); err != nil {
			return nil, err
		}

		if l.FFN1Weight, err = readTensorF32Flex(st, prefix+"intermediate.dense.weight"); err != nil {
			return nil, err
		}
		if l.FFN1Bias, err = readTensorF32Flex(st, prefix+"intermediate.dense.bias"); err != nil {
			return nil, err
		}
		if l.FFN2Weight, err = readTensorF32Flex(st, prefix+"output.dense.weight"); err != nil {
			return nil, err
		}
		if l.FFN2Bias, err = readTensorF32Flex(st, prefix+"output.dense.bias"); err != nil {
			return nil, err
		}
		if l.FFNNormW, err = readTensorF32Flex(st, prefix+"output.LayerNorm.weight"); err != nil {
			return nil, err
		}
		if l.FFNNormB, err = readTensorF32Flex(st, prefix+"output.LayerNorm.bias"); err != nil {
			return nil, err
		}
	}

	if err := m.Validate(); err != nil {
		return nil, fmt.Errorf("model validation failed: %w", err)
	}

	return m, nil
}

// LoadBF16Model memory-maps a pre-quantized BFloat16 safetensors file.
func LoadBF16Model(modelPath, tokenizerPath string) (m *Model, err error) {
	tok, err := tokenizer.LoadFromFile(tokenizerPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load tokenizer: %w", err)
	}

	st, err := safetensors.Open(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open safetensors file: %w", err)
	}
	defer func() {
		if err != nil {
			st.Close()
		}
	}()

	m = &Model{
		Tok:       tok,
		Precision: PrecisionBF16,
		st:        st,
	}

	if m.BF16WordEmbeddings, err = readTensorBF16Flex(st, "embeddings.word_embeddings.weight"); err != nil {
		return nil, err
	}
	if m.PositionEmbeddings, err = readTensorF32Flex(st, "embeddings.position_embeddings.weight"); err != nil {
		return nil, err
	}
	if m.TokenTypeEmbeddings, err = readTensorF32Flex(st, "embeddings.token_type_embeddings.weight"); err != nil {
		m.TokenTypeEmbeddings = make([]float32, HiddenSize)
		err = nil
	}
	if m.EmbeddingsNormW, err = readTensorF32Flex(st, "embeddings.LayerNorm.weight"); err != nil {
		return nil, err
	}
	if m.EmbeddingsNormB, err = readTensorF32Flex(st, "embeddings.LayerNorm.bias"); err != nil {
		return nil, err
	}

	for i := 0; i < NumLayers; i++ {
		prefix := fmt.Sprintf("encoder.layer.%d.", i)
		l := &m.BF16Layers[i]
		f32L := &m.Layers[i]

		l.Query.Rows = HiddenSize
		l.Query.Cols = HiddenSize
		if l.Query.Weight, err = readTensorBF16Flex(st, prefix+"attention.self.query.weight"); err != nil {
			return nil, err
		}
		if f32L.QueryBias, err = readTensorF32Flex(st, prefix+"attention.self.query.bias"); err != nil {
			return nil, err
		}
		l.Query.Bias = f32L.QueryBias

		l.Key.Rows = HiddenSize
		l.Key.Cols = HiddenSize
		if l.Key.Weight, err = readTensorBF16Flex(st, prefix+"attention.self.key.weight"); err != nil {
			return nil, err
		}
		if f32L.KeyBias, err = readTensorF32Flex(st, prefix+"attention.self.key.bias"); err != nil {
			return nil, err
		}
		l.Key.Bias = f32L.KeyBias

		l.Value.Rows = HiddenSize
		l.Value.Cols = HiddenSize
		if l.Value.Weight, err = readTensorBF16Flex(st, prefix+"attention.self.value.weight"); err != nil {
			return nil, err
		}
		if f32L.ValueBias, err = readTensorF32Flex(st, prefix+"attention.self.value.bias"); err != nil {
			return nil, err
		}
		l.Value.Bias = f32L.ValueBias

		l.Out.Rows = HiddenSize
		l.Out.Cols = HiddenSize
		if l.Out.Weight, err = readTensorBF16Flex(st, prefix+"attention.output.dense.weight"); err != nil {
			return nil, err
		}
		if f32L.OutBias, err = readTensorF32Flex(st, prefix+"attention.output.dense.bias"); err != nil {
			return nil, err
		}
		l.Out.Bias = f32L.OutBias

		if l.AttnNormW, err = readTensorF32Flex(st, prefix+"attention.output.LayerNorm.weight"); err != nil {
			return nil, err
		}
		if l.AttnNormB, err = readTensorF32Flex(st, prefix+"attention.output.LayerNorm.bias"); err != nil {
			return nil, err
		}

		l.FFN1.Rows = IntermediateSize
		l.FFN1.Cols = HiddenSize
		if l.FFN1.Weight, err = readTensorBF16Flex(st, prefix+"intermediate.dense.weight"); err != nil {
			return nil, err
		}
		if f32L.FFN1Bias, err = readTensorF32Flex(st, prefix+"intermediate.dense.bias"); err != nil {
			return nil, err
		}
		l.FFN1.Bias = f32L.FFN1Bias

		l.FFN2.Rows = HiddenSize
		l.FFN2.Cols = IntermediateSize
		if l.FFN2.Weight, err = readTensorBF16Flex(st, prefix+"output.dense.weight"); err != nil {
			return nil, err
		}
		if f32L.FFN2Bias, err = readTensorF32Flex(st, prefix+"output.dense.bias"); err != nil {
			return nil, err
		}
		l.FFN2.Bias = f32L.FFN2Bias

		if l.FFNNormW, err = readTensorF32Flex(st, prefix+"output.LayerNorm.weight"); err != nil {
			return nil, err
		}
		if l.FFNNormB, err = readTensorF32Flex(st, prefix+"output.LayerNorm.bias"); err != nil {
			return nil, err
		}
	}

	if err := m.Validate(); err != nil {
		return nil, fmt.Errorf("model validation failed: %w", err)
	}

	return m, nil
}

// LoadQuantizedModel memory-maps a pre-quantized INT8 safetensors file.
func LoadQuantizedModel(modelPath, tokenizerPath string) (m *Model, err error) {
	tok, err := tokenizer.LoadFromFile(tokenizerPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load tokenizer: %w", err)
	}

	st, err := safetensors.Open(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open safetensors file: %w", err)
	}
	defer func() {
		if err != nil {
			st.Close()
		}
	}()

	m = &Model{
		Tok:         tok,
		Precision:   PrecisionINT8,
		IsQuantized: true,
		st:          st,
	}

	if m.QWordEmbeddings.Weight, err = readTensorI8Flex(st, "embeddings.word_embeddings.weight"); err != nil {
		return nil, err
	}
	if m.QWordEmbeddings.Scale, err = readTensorF32Flex(st, "embeddings.word_embeddings.scale"); err != nil {
		return nil, err
	}
	if m.PositionEmbeddings, err = readTensorF32Flex(st, "embeddings.position_embeddings.weight"); err != nil {
		return nil, err
	}
	if m.TokenTypeEmbeddings, err = readTensorF32Flex(st, "embeddings.token_type_embeddings.weight"); err != nil {
		m.TokenTypeEmbeddings = make([]float32, HiddenSize)
		err = nil
	}
	if m.EmbeddingsNormW, err = readTensorF32Flex(st, "embeddings.LayerNorm.weight"); err != nil {
		return nil, err
	}
	if m.EmbeddingsNormB, err = readTensorF32Flex(st, "embeddings.LayerNorm.bias"); err != nil {
		return nil, err
	}

	for i := 0; i < NumLayers; i++ {
		prefix := fmt.Sprintf("encoder.layer.%d.", i)
		l := &m.QLayers[i]

		l.Query.Rows = HiddenSize
		l.Query.Cols = HiddenSize
		if l.Query.Weight, err = readTensorI8Flex(st, prefix+"attention.self.query.weight"); err != nil {
			return nil, err
		}
		if l.Query.Scale, err = readTensorF32Flex(st, prefix+"attention.self.query.scale"); err != nil {
			return nil, err
		}
		if l.Query.Bias, err = readTensorF32Flex(st, prefix+"attention.self.query.bias"); err != nil {
			return nil, err
		}

		l.Key.Rows = HiddenSize
		l.Key.Cols = HiddenSize
		if l.Key.Weight, err = readTensorI8Flex(st, prefix+"attention.self.key.weight"); err != nil {
			return nil, err
		}
		if l.Key.Scale, err = readTensorF32Flex(st, prefix+"attention.self.key.scale"); err != nil {
			return nil, err
		}
		if l.Key.Bias, err = readTensorF32Flex(st, prefix+"attention.self.key.bias"); err != nil {
			return nil, err
		}

		l.Value.Rows = HiddenSize
		l.Value.Cols = HiddenSize
		if l.Value.Weight, err = readTensorI8Flex(st, prefix+"attention.self.value.weight"); err != nil {
			return nil, err
		}
		if l.Value.Scale, err = readTensorF32Flex(st, prefix+"attention.self.value.scale"); err != nil {
			return nil, err
		}
		if l.Value.Bias, err = readTensorF32Flex(st, prefix+"attention.self.value.bias"); err != nil {
			return nil, err
		}

		l.Out.Rows = HiddenSize
		l.Out.Cols = HiddenSize
		if l.Out.Weight, err = readTensorI8Flex(st, prefix+"attention.output.dense.weight"); err != nil {
			return nil, err
		}
		if l.Out.Scale, err = readTensorF32Flex(st, prefix+"attention.output.dense.scale"); err != nil {
			return nil, err
		}
		if l.Out.Bias, err = readTensorF32Flex(st, prefix+"attention.output.dense.bias"); err != nil {
			return nil, err
		}

		if l.AttnNormW, err = readTensorF32Flex(st, prefix+"attention.output.LayerNorm.weight"); err != nil {
			return nil, err
		}
		if l.AttnNormB, err = readTensorF32Flex(st, prefix+"attention.output.LayerNorm.bias"); err != nil {
			return nil, err
		}

		l.FFN1.Rows = IntermediateSize
		l.FFN1.Cols = HiddenSize
		if l.FFN1.Weight, err = readTensorI8Flex(st, prefix+"intermediate.dense.weight"); err != nil {
			return nil, err
		}
		if l.FFN1.Scale, err = readTensorF32Flex(st, prefix+"intermediate.dense.scale"); err != nil {
			return nil, err
		}
		if l.FFN1.Bias, err = readTensorF32Flex(st, prefix+"intermediate.dense.bias"); err != nil {
			return nil, err
		}

		l.FFN2.Rows = HiddenSize
		l.FFN2.Cols = IntermediateSize
		if l.FFN2.Weight, err = readTensorI8Flex(st, prefix+"output.dense.weight"); err != nil {
			return nil, err
		}
		if l.FFN2.Scale, err = readTensorF32Flex(st, prefix+"output.dense.scale"); err != nil {
			return nil, err
		}
		if l.FFN2.Bias, err = readTensorF32Flex(st, prefix+"output.dense.bias"); err != nil {
			return nil, err
		}

		if l.FFNNormW, err = readTensorF32Flex(st, prefix+"output.LayerNorm.weight"); err != nil {
			return nil, err
		}
		if l.FFNNormB, err = readTensorF32Flex(st, prefix+"output.LayerNorm.bias"); err != nil {
			return nil, err
		}
	}

	if err := m.Validate(); err != nil {
		return nil, fmt.Errorf("model validation failed: %w", err)
	}

	return m, nil
}

// SaveBF16Model saves a BFloat16 model to disk in Safetensors format.
func SaveBF16Model(m *Model, targetPath string) error {
	tensors := make(map[string]safetensors.TensorData)
	vocabSize := m.VocabSize()

	tensors["embeddings.word_embeddings.weight"] = safetensors.NewTensorBF16([]int{vocabSize, HiddenSize}, m.BF16WordEmbeddings)
	tensors["embeddings.position_embeddings.weight"] = safetensors.NewTensorF32([]int{len(m.PositionEmbeddings) / HiddenSize, HiddenSize}, m.PositionEmbeddings)
	if len(m.TokenTypeEmbeddings) > 0 {
		tensors["embeddings.token_type_embeddings.weight"] = safetensors.NewTensorF32([]int{len(m.TokenTypeEmbeddings) / HiddenSize, HiddenSize}, m.TokenTypeEmbeddings)
	}
	tensors["embeddings.LayerNorm.weight"] = safetensors.NewTensorF32([]int{HiddenSize}, m.EmbeddingsNormW)
	tensors["embeddings.LayerNorm.bias"] = safetensors.NewTensorF32([]int{HiddenSize}, m.EmbeddingsNormB)

	for i := 0; i < NumLayers; i++ {
		prefix := fmt.Sprintf("encoder.layer.%d.", i)
		l := &m.BF16Layers[i]

		tensors[prefix+"attention.self.query.weight"] = safetensors.NewTensorBF16([]int{HiddenSize, HiddenSize}, l.Query.Weight)
		tensors[prefix+"attention.self.query.bias"] = safetensors.NewTensorF32([]int{HiddenSize}, l.Query.Bias)

		tensors[prefix+"attention.self.key.weight"] = safetensors.NewTensorBF16([]int{HiddenSize, HiddenSize}, l.Key.Weight)
		tensors[prefix+"attention.self.key.bias"] = safetensors.NewTensorF32([]int{HiddenSize}, l.Key.Bias)

		tensors[prefix+"attention.self.value.weight"] = safetensors.NewTensorBF16([]int{HiddenSize, HiddenSize}, l.Value.Weight)
		tensors[prefix+"attention.self.value.bias"] = safetensors.NewTensorF32([]int{HiddenSize}, l.Value.Bias)

		tensors[prefix+"attention.output.dense.weight"] = safetensors.NewTensorBF16([]int{HiddenSize, HiddenSize}, l.Out.Weight)
		tensors[prefix+"attention.output.dense.bias"] = safetensors.NewTensorF32([]int{HiddenSize}, l.Out.Bias)

		tensors[prefix+"attention.output.LayerNorm.weight"] = safetensors.NewTensorF32([]int{HiddenSize}, l.AttnNormW)
		tensors[prefix+"attention.output.LayerNorm.bias"] = safetensors.NewTensorF32([]int{HiddenSize}, l.AttnNormB)

		tensors[prefix+"intermediate.dense.weight"] = safetensors.NewTensorBF16([]int{IntermediateSize, HiddenSize}, l.FFN1.Weight)
		tensors[prefix+"intermediate.dense.bias"] = safetensors.NewTensorF32([]int{IntermediateSize}, l.FFN1.Bias)

		tensors[prefix+"output.dense.weight"] = safetensors.NewTensorBF16([]int{HiddenSize, IntermediateSize}, l.FFN2.Weight)
		tensors[prefix+"output.dense.bias"] = safetensors.NewTensorF32([]int{HiddenSize}, l.FFN2.Bias)

		tensors[prefix+"output.LayerNorm.weight"] = safetensors.NewTensorF32([]int{HiddenSize}, l.FFNNormW)
		tensors[prefix+"output.LayerNorm.bias"] = safetensors.NewTensorF32([]int{HiddenSize}, l.FFNNormB)
	}

	return safetensors.WriteFile(targetPath, tensors)
}

// SaveQuantizedModel saves an INT8 quantized model to disk in Safetensors format.
func SaveQuantizedModel(m *Model, targetPath string) error {
	tensors := make(map[string]safetensors.TensorData)
	vocabSize := m.VocabSize()

	tensors["embeddings.word_embeddings.weight"] = safetensors.NewTensorI8([]int{vocabSize, HiddenSize}, m.QWordEmbeddings.Weight)
	tensors["embeddings.word_embeddings.scale"] = safetensors.NewTensorF32([]int{vocabSize}, m.QWordEmbeddings.Scale)
	tensors["embeddings.position_embeddings.weight"] = safetensors.NewTensorF32([]int{len(m.PositionEmbeddings) / HiddenSize, HiddenSize}, m.PositionEmbeddings)
	if len(m.TokenTypeEmbeddings) > 0 {
		tensors["embeddings.token_type_embeddings.weight"] = safetensors.NewTensorF32([]int{len(m.TokenTypeEmbeddings) / HiddenSize, HiddenSize}, m.TokenTypeEmbeddings)
	}
	tensors["embeddings.LayerNorm.weight"] = safetensors.NewTensorF32([]int{HiddenSize}, m.EmbeddingsNormW)
	tensors["embeddings.LayerNorm.bias"] = safetensors.NewTensorF32([]int{HiddenSize}, m.EmbeddingsNormB)

	for i := 0; i < NumLayers; i++ {
		prefix := fmt.Sprintf("encoder.layer.%d.", i)
		l := &m.QLayers[i]

		tensors[prefix+"attention.self.query.weight"] = safetensors.NewTensorI8([]int{HiddenSize, HiddenSize}, l.Query.Weight)
		tensors[prefix+"attention.self.query.scale"] = safetensors.NewTensorF32([]int{HiddenSize}, l.Query.Scale)
		tensors[prefix+"attention.self.query.bias"] = safetensors.NewTensorF32([]int{HiddenSize}, l.Query.Bias)

		tensors[prefix+"attention.self.key.weight"] = safetensors.NewTensorI8([]int{HiddenSize, HiddenSize}, l.Key.Weight)
		tensors[prefix+"attention.self.key.scale"] = safetensors.NewTensorF32([]int{HiddenSize}, l.Key.Scale)
		tensors[prefix+"attention.self.key.bias"] = safetensors.NewTensorF32([]int{HiddenSize}, l.Key.Bias)

		tensors[prefix+"attention.self.value.weight"] = safetensors.NewTensorI8([]int{HiddenSize, HiddenSize}, l.Value.Weight)
		tensors[prefix+"attention.self.value.scale"] = safetensors.NewTensorF32([]int{HiddenSize}, l.Value.Scale)
		tensors[prefix+"attention.self.value.bias"] = safetensors.NewTensorF32([]int{HiddenSize}, l.Value.Bias)

		tensors[prefix+"attention.output.dense.weight"] = safetensors.NewTensorI8([]int{HiddenSize, HiddenSize}, l.Out.Weight)
		tensors[prefix+"attention.output.dense.scale"] = safetensors.NewTensorF32([]int{HiddenSize}, l.Out.Scale)
		tensors[prefix+"attention.output.dense.bias"] = safetensors.NewTensorF32([]int{HiddenSize}, l.Out.Bias)

		tensors[prefix+"attention.output.LayerNorm.weight"] = safetensors.NewTensorF32([]int{HiddenSize}, l.AttnNormW)
		tensors[prefix+"attention.output.LayerNorm.bias"] = safetensors.NewTensorF32([]int{HiddenSize}, l.AttnNormB)

		tensors[prefix+"intermediate.dense.weight"] = safetensors.NewTensorI8([]int{IntermediateSize, HiddenSize}, l.FFN1.Weight)
		tensors[prefix+"intermediate.dense.scale"] = safetensors.NewTensorF32([]int{IntermediateSize}, l.FFN1.Scale)
		tensors[prefix+"intermediate.dense.bias"] = safetensors.NewTensorF32([]int{IntermediateSize}, l.FFN1.Bias)

		tensors[prefix+"output.dense.weight"] = safetensors.NewTensorI8([]int{HiddenSize, IntermediateSize}, l.FFN2.Weight)
		tensors[prefix+"output.dense.scale"] = safetensors.NewTensorF32([]int{HiddenSize}, l.FFN2.Scale)
		tensors[prefix+"output.dense.bias"] = safetensors.NewTensorF32([]int{HiddenSize}, l.FFN2.Bias)

		tensors[prefix+"output.LayerNorm.weight"] = safetensors.NewTensorF32([]int{HiddenSize}, l.FFNNormW)
		tensors[prefix+"output.LayerNorm.bias"] = safetensors.NewTensorF32([]int{HiddenSize}, l.FFNNormB)
	}

	return safetensors.WriteFile(targetPath, tensors)
}

// LoadModelWithPrecision loads model weights with memory mapping and caching of quantized weights on disk.
func LoadModelWithPrecision(modelPath, tokenizerPath string, prec PrecisionMode) (*Model, error) {
	switch prec {
	case PrecisionBF16:
		if strings.HasSuffix(modelPath, "_bf16.safetensors") {
			return LoadBF16Model(modelPath, tokenizerPath)
		}
		derivedPath := strings.TrimSuffix(modelPath, ".safetensors") + "_bf16.safetensors"
		if fi, err := os.Stat(derivedPath); err == nil && fi.Size() > 0 {
			return LoadBF16Model(derivedPath, tokenizerPath)
		}

		// Load FP32, convert, save to disk, and mmap
		fp32M, err := loadFP32Model(modelPath, tokenizerPath)
		if err != nil {
			return nil, err
		}
		bf16M := ConvertToBF16Model(fp32M)
		saveErr := SaveBF16Model(bf16M, derivedPath)
		fp32M.Close()

		if saveErr == nil {
			return LoadBF16Model(derivedPath, tokenizerPath)
		}
		return bf16M, nil

	case PrecisionINT8:
		if strings.HasSuffix(modelPath, "_int8.safetensors") {
			return LoadQuantizedModel(modelPath, tokenizerPath)
		}
		derivedPath := strings.TrimSuffix(modelPath, ".safetensors") + "_int8.safetensors"
		if fi, err := os.Stat(derivedPath); err == nil && fi.Size() > 0 {
			return LoadQuantizedModel(derivedPath, tokenizerPath)
		}

		// Load FP32, quantize, save to disk, and mmap
		fp32M, err := loadFP32Model(modelPath, tokenizerPath)
		if err != nil {
			return nil, err
		}
		qM := QuantizeModel(fp32M)
		saveErr := SaveQuantizedModel(qM, derivedPath)
		fp32M.Close()

		if saveErr == nil {
			return LoadQuantizedModel(derivedPath, tokenizerPath)
		}
		return qM, nil

	default:
		return loadFP32Model(modelPath, tokenizerPath)
	}
}
