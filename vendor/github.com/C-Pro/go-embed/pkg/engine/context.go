package engine

import (
	"fmt"
	"strings"
	"sync"

	"github.com/C-Pro/go-embed/pkg/tokenizer"
)

const (
	scaleFactor = float32(0.17677669529663687) // 1.0 / sqrt(32)

	DefaultWindowSize = 512
	DefaultOverlap    = 256
)

// InferenceContext encapsulates all pre-allocated scratchpad buffers
// needed to execute forward inference passes.
type InferenceContext struct {
	Model         *Model
	UseSIMD       bool
	WindowSize    int
	Overlap       int
	QueryPrefix   string
	PassagePrefix string

	RuneBuf       []rune
	AllTokens     []int
	InputIDs      []int
	AttentionMask []int8
	DPBuf         []tokenizer.DPState

	HiddenStates []float32 // [512 * 384]
	Residual     []float32 // [512 * 384]
	Q            []float32 // [512 * 384]
	K            []float32 // [512 * 384]
	V            []float32 // [512 * 384]
	AttnContext  []float32 // [512 * 384]
	AttnOut      []float32 // [512 * 384]
	Scores       []float32 // [512]
	FFNMid       []float32 // [512 * 1536]
	FFNOut       []float32 // [512 * 384]
	Pooled       []float32 // [384]
	Output       []float32 // [384]
}

// NewContext creates a new InferenceContext with default window size (512) and overlap (256).
func NewContext(m *Model) *InferenceContext {
	return NewContextWithOptions(m, DefaultWindowSize, DefaultOverlap, "", "")
}

// NewContextWithOptions creates a new InferenceContext with custom window size, overlap, and prefixes.
func NewContextWithOptions(m *Model, windowSize, overlap int, queryPrefix, passagePrefix string) *InferenceContext {
	if windowSize <= 0 || windowSize > MaxSeqLen {
		windowSize = MaxSeqLen
	}
	if overlap < 0 {
		overlap = 0
	} else if overlap >= windowSize-2 {
		overlap = (windowSize - 2) / 2
	}

	return &InferenceContext{
		Model:         m,
		UseSIMD:       HasSIMD,
		WindowSize:    windowSize,
		Overlap:       overlap,
		QueryPrefix:   queryPrefix,
		PassagePrefix: passagePrefix,
		RuneBuf:       make([]rune, 0, 1024),
		AllTokens:     make([]int, 0, 1024),
		InputIDs:      make([]int, 0, MaxSeqLen),
		AttentionMask: make([]int8, 0, MaxSeqLen),
		DPBuf:         make([]tokenizer.DPState, 0, 2048),
		HiddenStates:  make([]float32, MaxSeqLen*HiddenSize),
		Residual:      make([]float32, MaxSeqLen*HiddenSize),
		Q:             make([]float32, MaxSeqLen*HiddenSize),
		K:             make([]float32, MaxSeqLen*HiddenSize),
		V:             make([]float32, MaxSeqLen*HiddenSize),
		AttnContext:   make([]float32, MaxSeqLen*HiddenSize),
		AttnOut:       make([]float32, MaxSeqLen*HiddenSize),
		Scores:        make([]float32, MaxSeqLen),
		FFNMid:        make([]float32, MaxSeqLen*IntermediateSize),
		FFNOut:        make([]float32, MaxSeqLen*HiddenSize),
		Pooled:        make([]float32, HiddenSize),
		Output:        make([]float32, HiddenSize),
	}
}

// ContextPool manages reusable InferenceContext instances for high concurrency.
type ContextPool struct {
	pool sync.Pool
}

// NewContextPool creates a new pool of inference contexts for the given model with configured options.
func NewContextPool(m *Model, windowSize, overlap int, queryPrefix, passagePrefix string) *ContextPool {
	return &ContextPool{
		pool: sync.Pool{
			New: func() any {
				return NewContextWithOptions(m, windowSize, overlap, queryPrefix, passagePrefix)
			},
		},
	}
}

// Get borrows an InferenceContext from the pool.
func (cp *ContextPool) Get() *InferenceContext {
	return cp.pool.Get().(*InferenceContext)
}

// Put returns an InferenceContext back to the pool.
func (cp *ContextPool) Put(ctx *InferenceContext) {
	cp.pool.Put(ctx)
}

// Embed generates L2-normalized 384-dimensional embeddings for raw text across sliding windows.
// It returns a slice of vectors, with one vector for each window.
func (ctx *InferenceContext) Embed(text string) ([][]float32, error) {
	if ctx == nil || ctx.Model == nil || ctx.Model.Tok == nil {
		return nil, fmt.Errorf("inference context or model is not initialized")
	}
	ctx.RuneBuf, ctx.AllTokens = ctx.Model.Tok.EncodeRawIntoZeroAlloc(
		text,
		ctx.RuneBuf,
		ctx.AllTokens,
		ctx.DPBuf,
	)

	return ctx.EmbedTokens(ctx.AllTokens), nil
}

// EmbedQuery embeds text with the configured query prefix.
func (ctx *InferenceContext) EmbedQuery(text string) ([][]float32, error) {
	if ctx == nil || ctx.Model == nil {
		return nil, fmt.Errorf("inference context or model is not initialized")
	}
	if ctx.QueryPrefix != "" && !strings.HasPrefix(text, ctx.QueryPrefix) {
		text = ctx.QueryPrefix + text
	}
	return ctx.Embed(text)
}

// EmbedPassage embeds text with the configured passage prefix.
func (ctx *InferenceContext) EmbedPassage(text string) ([][]float32, error) {
	if ctx == nil || ctx.Model == nil {
		return nil, fmt.Errorf("inference context or model is not initialized")
	}
	if ctx.PassagePrefix != "" && !strings.HasPrefix(text, ctx.PassagePrefix) {
		text = ctx.PassagePrefix + text
	}
	return ctx.Embed(text)
}

// EmbedTokens generates embeddings for a slice of content token IDs using sliding window chunking with overlap.
func (ctx *InferenceContext) EmbedTokens(tokens []int) [][]float32 {
	if ctx == nil || ctx.Model == nil {
		return nil
	}
	windowSize := ctx.WindowSize
	if windowSize <= 0 || windowSize > MaxSeqLen {
		windowSize = MaxSeqLen
	}
	maxContent := windowSize - 2
	if maxContent <= 0 {
		maxContent = 1
	}

	if len(tokens) <= maxContent {
		seqLen := len(tokens) + 2
		if seqLen > MaxSeqLen {
			seqLen = MaxSeqLen
		}
		if cap(ctx.InputIDs) < seqLen {
			ctx.InputIDs = make([]int, seqLen)
		} else {
			ctx.InputIDs = ctx.InputIDs[:seqLen]
		}
		ctx.InputIDs[0] = tokenizer.BOS_ID
		if seqLen > 2 {
			copy(ctx.InputIDs[1:seqLen-1], tokens[:seqLen-2])
		}
		ctx.InputIDs[seqLen-1] = tokenizer.EOS_ID

		if cap(ctx.AttentionMask) < seqLen {
			ctx.AttentionMask = make([]int8, seqLen)
		} else {
			ctx.AttentionMask = ctx.AttentionMask[:seqLen]
		}
		for i := 0; i < seqLen; i++ {
			ctx.AttentionMask[i] = 1
		}

		vec := make([]float32, HiddenSize)
		ctx.Forward(seqLen, vec)
		return [][]float32{vec}
	}

	overlap := ctx.Overlap
	if overlap < 0 {
		overlap = 0
	} else if overlap >= maxContent {
		overlap = maxContent / 2
	}
	stride := maxContent - overlap
	if stride <= 0 {
		stride = 1
	}

	numChunks := (len(tokens) - overlap + stride - 1) / stride
	if numChunks < 1 {
		numChunks = 1
	}
	results := make([][]float32, 0, numChunks)

	for start := 0; start < len(tokens); start += stride {
		end := start + maxContent
		if end > len(tokens) {
			end = len(tokens)
		}
		chunk := tokens[start:end]
		seqLen := len(chunk) + 2
		if seqLen > MaxSeqLen {
			seqLen = MaxSeqLen
		}

		if cap(ctx.InputIDs) < seqLen {
			ctx.InputIDs = make([]int, seqLen)
		} else {
			ctx.InputIDs = ctx.InputIDs[:seqLen]
		}
		ctx.InputIDs[0] = tokenizer.BOS_ID
		if seqLen > 2 {
			copy(ctx.InputIDs[1:seqLen-1], chunk[:seqLen-2])
		}
		ctx.InputIDs[seqLen-1] = tokenizer.EOS_ID

		if cap(ctx.AttentionMask) < seqLen {
			ctx.AttentionMask = make([]int8, seqLen)
		} else {
			ctx.AttentionMask = ctx.AttentionMask[:seqLen]
		}
		for i := 0; i < seqLen; i++ {
			ctx.AttentionMask[i] = 1
		}

		vec := make([]float32, HiddenSize)
		ctx.Forward(seqLen, vec)
		results = append(results, vec)

		if end == len(tokens) {
			break
		}
	}

	return results
}

// EmbedTokenIDs runs the forward pass given pre-computed token IDs and optional attention mask.
// If len(inputIDs) <= MaxSeqLen, it executes directly as a single window.
// If len(inputIDs) > MaxSeqLen, it windows across the content tokens.
func (ctx *InferenceContext) EmbedTokenIDs(inputIDs []int, attnMask []int8) ([][]float32, error) {
	if ctx == nil || ctx.Model == nil {
		return nil, fmt.Errorf("inference context or model is not initialized")
	}
	if len(inputIDs) == 0 {
		return nil, fmt.Errorf("empty inputIDs")
	}
	if len(inputIDs) <= MaxSeqLen {
		seqLen := len(inputIDs)
		ctx.InputIDs = append(ctx.InputIDs[:0], inputIDs...)
		ctx.AttentionMask = ctx.AttentionMask[:0]
		for i := 0; i < seqLen; i++ {
			if attnMask != nil && i < len(attnMask) {
				ctx.AttentionMask = append(ctx.AttentionMask, attnMask[i])
			} else {
				ctx.AttentionMask = append(ctx.AttentionMask, 1)
			}
		}

		vec := make([]float32, HiddenSize)
		ctx.Forward(seqLen, vec)
		return [][]float32{vec}, nil
	}

	// Strip outer BOS/EOS if already present before chunking
	raw := inputIDs
	if len(raw) > 0 && raw[0] == tokenizer.BOS_ID {
		raw = raw[1:]
	}
	if len(raw) > 0 && raw[len(raw)-1] == tokenizer.EOS_ID {
		raw = raw[:len(raw)-1]
	}

	return ctx.EmbedTokens(raw), nil
}

// Forward executes the transformer encoder, mean pooling, and L2 normalization.
func (ctx *InferenceContext) Forward(seqLen int, out []float32) []float32 {
	if ctx == nil || ctx.Model == nil || seqLen <= 0 {
		return out
	}
	if seqLen > MaxSeqLen {
		seqLen = MaxSeqLen
	}
	if len(out) < HiddenSize {
		out = make([]float32, HiddenSize)
	}
	m := ctx.Model

	matMul := MatVecMulAddScalar
	matMulQ := MatVecMulAddINT8Scalar
	matMulBF16 := MatVecMulAddBF16Scalar
	layerNorm := LayerNormScalar
	gelu := GELUScalar
	l2Norm := L2NormalizeScalar

	if ctx.UseSIMD && HasSIMD {
		matMul = MatVecMulAddSIMD
		matMulQ = MatVecMulAddINT8SIMD
		matMulBF16 = MatVecMulAddBF16SIMD
		layerNorm = LayerNormSIMD
		gelu = GELUSIMD
		l2Norm = L2NormalizeSIMD
	}

	// 1. Embeddings lookup & LayerNorm
	vocabLimit := m.VocabSize()
	if vocabLimit <= 0 {
		return out
	}

	for t := 0; t < seqLen; t++ {
		id := tokenizer.UNK_ID
		if t < len(ctx.InputIDs) {
			id = ctx.InputIDs[t]
		}
		if id < 0 || id >= vocabLimit {
			if tokenizer.UNK_ID >= 0 && tokenizer.UNK_ID < vocabLimit {
				id = tokenizer.UNK_ID
			} else {
				id = 0
			}
		}

		pOffset := t * HiddenSize
		hOffset := t * HiddenSize

		var pEmb []float32
		if pOffset+HiddenSize <= len(m.PositionEmbeddings) {
			pEmb = m.PositionEmbeddings[pOffset : pOffset+HiddenSize]
		}
		var tEmb []float32
		if len(m.TokenTypeEmbeddings) >= HiddenSize {
			tEmb = m.TokenTypeEmbeddings[:HiddenSize]
		}
		sumSlice := ctx.Residual[hOffset : hOffset+HiddenSize]

		switch m.Precision {
		case PrecisionBF16:
			wOffset := id * HiddenSize
			var wWords []uint16
			if wOffset+HiddenSize <= len(m.BF16WordEmbeddings) {
				wWords = m.BF16WordEmbeddings[wOffset : wOffset+HiddenSize]
			}
			for d := 0; d < HiddenSize; d++ {
				var val float32
				if d < len(wWords) {
					val = BFloat16ToFloat32(wWords[d])
				}
				if d < len(pEmb) {
					val += pEmb[d]
				}
				if d < len(tEmb) {
					val += tEmb[d]
				}
				sumSlice[d] = val
			}
		case PrecisionINT8:
			wOffset := id * HiddenSize
			var qScale float32 = 1.0
			if id < len(m.QWordEmbeddings.Scale) {
				qScale = m.QWordEmbeddings.Scale[id]
			}
			var qWeights []int8
			if wOffset+HiddenSize <= len(m.QWordEmbeddings.Weight) {
				qWeights = m.QWordEmbeddings.Weight[wOffset : wOffset+HiddenSize]
			}
			for d := 0; d < HiddenSize; d++ {
				var val float32
				if d < len(qWeights) {
					val = float32(qWeights[d]) * qScale
				}
				if d < len(pEmb) {
					val += pEmb[d]
				}
				if d < len(tEmb) {
					val += tEmb[d]
				}
				sumSlice[d] = val
			}
		default:
			wOffset := id * HiddenSize
			var wEmb []float32
			if wOffset+HiddenSize <= len(m.WordEmbeddings) {
				wEmb = m.WordEmbeddings[wOffset : wOffset+HiddenSize]
			}
			for d := 0; d < HiddenSize; d++ {
				var val float32
				if d < len(wEmb) {
					val = wEmb[d]
				}
				if d < len(pEmb) {
					val += pEmb[d]
				}
				if d < len(tEmb) {
					val += tEmb[d]
				}
				sumSlice[d] = val
			}
		}

		layerNorm(
			sumSlice,
			m.EmbeddingsNormW,
			m.EmbeddingsNormB,
			ctx.HiddenStates[hOffset:hOffset+HiddenSize],
			HiddenSize,
			LayerNormEps,
		)
	}

	// 2. 12 Transformer Layers
	for l := 0; l < NumLayers; l++ {
		switch m.Precision {
		case PrecisionBF16:
			layer := &m.BF16Layers[l]

			// Multi-head self attention projections Q, K, V
			for t := 0; t < seqLen; t++ {
				tOffset := t * HiddenSize
				xt := ctx.HiddenStates[tOffset : tOffset+HiddenSize]

				matMulBF16(xt, layer.Query.Weight, layer.Query.Bias, ctx.Q[tOffset:tOffset+HiddenSize], HiddenSize, HiddenSize)
				matMulBF16(xt, layer.Key.Weight, layer.Key.Bias, ctx.K[tOffset:tOffset+HiddenSize], HiddenSize, HiddenSize)
				matMulBF16(xt, layer.Value.Weight, layer.Value.Bias, ctx.V[tOffset:tOffset+HiddenSize], HiddenSize, HiddenSize)
			}

			// Scaled dot-product attention per head
			for h := 0; h < NumHeads; h++ {
				hOffset := h * HeadDim

				for t := 0; t < seqLen; t++ {
					qt := ctx.Q[t*HiddenSize+hOffset : t*HiddenSize+hOffset+HeadDim]

					// Compute scores against all positions s
					for s := 0; s < seqLen; s++ {
						ks := ctx.K[s*HiddenSize+hOffset : s*HiddenSize+hOffset+HeadDim]
						var dot float32
						for d := 0; d < HeadDim; d++ {
							dot += qt[d] * ks[d]
						}
						ctx.Scores[s] = dot * scaleFactor
					}

					// Softmax over sequence scores
					SoftmaxScalar(ctx.Scores[:seqLen], ctx.Scores[:seqLen], seqLen)

					// Weighted sum of values
					for d := 0; d < HeadDim; d++ {
						var sumVal float32
						for s := 0; s < seqLen; s++ {
							sumVal += ctx.Scores[s] * ctx.V[s*HiddenSize+hOffset+d]
						}
						ctx.AttnContext[t*HiddenSize+hOffset+d] = sumVal
					}
				}
			}

			// Attention output projection, residual connection, and LayerNorm
			for t := 0; t < seqLen; t++ {
				tOffset := t * HiddenSize
				cSlice := ctx.AttnContext[tOffset : tOffset+HiddenSize]
				matMulBF16(cSlice, layer.Out.Weight, layer.Out.Bias, ctx.AttnOut[tOffset:tOffset+HiddenSize], HiddenSize, HiddenSize)

				resSlice := ctx.Residual[tOffset : tOffset+HiddenSize]
				hSlice := ctx.HiddenStates[tOffset : tOffset+HiddenSize]
				attnOutSlice := ctx.AttnOut[tOffset : tOffset+HiddenSize]

				for d := 0; d < HiddenSize; d++ {
					resSlice[d] = hSlice[d] + attnOutSlice[d]
				}

				layerNorm(resSlice, layer.AttnNormW, layer.AttnNormB, hSlice, HiddenSize, LayerNormEps)
			}

			// Feed-Forward Network: Linear (384 -> 1536) -> GELU -> Linear (1536 -> 384) -> Residual -> LayerNorm
			for t := 0; t < seqLen; t++ {
				tOffsetH := t * HiddenSize
				tOffsetFFN := t * IntermediateSize

				xt := ctx.HiddenStates[tOffsetH : tOffsetH+HiddenSize]
				ffnMid := ctx.FFNMid[tOffsetFFN : tOffsetFFN+IntermediateSize]
				ffnOut := ctx.FFNOut[tOffsetH : tOffsetH+HiddenSize]

				matMulBF16(xt, layer.FFN1.Weight, layer.FFN1.Bias, ffnMid, HiddenSize, IntermediateSize)
				gelu(ffnMid, ffnMid, IntermediateSize)
				matMulBF16(ffnMid, layer.FFN2.Weight, layer.FFN2.Bias, ffnOut, IntermediateSize, HiddenSize)

				resSlice := ctx.Residual[tOffsetH : tOffsetH+HiddenSize]
				for d := 0; d < HiddenSize; d++ {
					resSlice[d] = xt[d] + ffnOut[d]
				}

				layerNorm(resSlice, layer.FFNNormW, layer.FFNNormB, xt, HiddenSize, LayerNormEps)
			}

		case PrecisionINT8:
			layer := &m.QLayers[l]

			// Multi-head self attention projections Q, K, V
			for t := 0; t < seqLen; t++ {
				tOffset := t * HiddenSize
				xt := ctx.HiddenStates[tOffset : tOffset+HiddenSize]

				matMulQ(xt, layer.Query.Weight, layer.Query.Scale, layer.Query.Bias, ctx.Q[tOffset:tOffset+HiddenSize], HiddenSize, HiddenSize)
				matMulQ(xt, layer.Key.Weight, layer.Key.Scale, layer.Key.Bias, ctx.K[tOffset:tOffset+HiddenSize], HiddenSize, HiddenSize)
				matMulQ(xt, layer.Value.Weight, layer.Value.Scale, layer.Value.Bias, ctx.V[tOffset:tOffset+HiddenSize], HiddenSize, HiddenSize)
			}

			// Scaled dot-product attention per head
			for h := 0; h < NumHeads; h++ {
				hOffset := h * HeadDim

				for t := 0; t < seqLen; t++ {
					qt := ctx.Q[t*HiddenSize+hOffset : t*HiddenSize+hOffset+HeadDim]

					// Compute scores against all positions s
					for s := 0; s < seqLen; s++ {
						ks := ctx.K[s*HiddenSize+hOffset : s*HiddenSize+hOffset+HeadDim]
						var dot float32
						for d := 0; d < HeadDim; d++ {
							dot += qt[d] * ks[d]
						}
						ctx.Scores[s] = dot * scaleFactor
					}

					// Softmax over sequence scores
					SoftmaxScalar(ctx.Scores[:seqLen], ctx.Scores[:seqLen], seqLen)

					// Weighted sum of values
					for d := 0; d < HeadDim; d++ {
						var sumVal float32
						for s := 0; s < seqLen; s++ {
							sumVal += ctx.Scores[s] * ctx.V[s*HiddenSize+hOffset+d]
						}
						ctx.AttnContext[t*HiddenSize+hOffset+d] = sumVal
					}
				}
			}

			// Attention output projection, residual connection, and LayerNorm
			for t := 0; t < seqLen; t++ {
				tOffset := t * HiddenSize
				cSlice := ctx.AttnContext[tOffset : tOffset+HiddenSize]
				matMulQ(cSlice, layer.Out.Weight, layer.Out.Scale, layer.Out.Bias, ctx.AttnOut[tOffset:tOffset+HiddenSize], HiddenSize, HiddenSize)

				resSlice := ctx.Residual[tOffset : tOffset+HiddenSize]
				hSlice := ctx.HiddenStates[tOffset : tOffset+HiddenSize]
				attnOutSlice := ctx.AttnOut[tOffset : tOffset+HiddenSize]

				for d := 0; d < HiddenSize; d++ {
					resSlice[d] = hSlice[d] + attnOutSlice[d]
				}

				layerNorm(resSlice, layer.AttnNormW, layer.AttnNormB, hSlice, HiddenSize, LayerNormEps)
			}

			// Feed-Forward Network: Linear (384 -> 1536) -> GELU -> Linear (1536 -> 384) -> Residual -> LayerNorm
			for t := 0; t < seqLen; t++ {
				tOffsetH := t * HiddenSize
				tOffsetFFN := t * IntermediateSize

				xt := ctx.HiddenStates[tOffsetH : tOffsetH+HiddenSize]
				ffnMid := ctx.FFNMid[tOffsetFFN : tOffsetFFN+IntermediateSize]
				ffnOut := ctx.FFNOut[tOffsetH : tOffsetH+HiddenSize]

				matMulQ(xt, layer.FFN1.Weight, layer.FFN1.Scale, layer.FFN1.Bias, ffnMid, HiddenSize, IntermediateSize)
				gelu(ffnMid, ffnMid, IntermediateSize)
				matMulQ(ffnMid, layer.FFN2.Weight, layer.FFN2.Scale, layer.FFN2.Bias, ffnOut, IntermediateSize, HiddenSize)

				resSlice := ctx.Residual[tOffsetH : tOffsetH+HiddenSize]
				for d := 0; d < HiddenSize; d++ {
					resSlice[d] = xt[d] + ffnOut[d]
				}

				layerNorm(resSlice, layer.FFNNormW, layer.FFNNormB, xt, HiddenSize, LayerNormEps)
			}

		default:
			layer := &m.Layers[l]

			// Multi-head self attention projections Q, K, V
			for t := 0; t < seqLen; t++ {
				tOffset := t * HiddenSize
				xt := ctx.HiddenStates[tOffset : tOffset+HiddenSize]

				matMul(xt, layer.QueryWeight, layer.QueryBias, ctx.Q[tOffset:tOffset+HiddenSize], HiddenSize, HiddenSize)
				matMul(xt, layer.KeyWeight, layer.KeyBias, ctx.K[tOffset:tOffset+HiddenSize], HiddenSize, HiddenSize)
				matMul(xt, layer.ValueWeight, layer.ValueBias, ctx.V[tOffset:tOffset+HiddenSize], HiddenSize, HiddenSize)
			}

			// Scaled dot-product attention per head
			for h := 0; h < NumHeads; h++ {
				hOffset := h * HeadDim

				for t := 0; t < seqLen; t++ {
					qt := ctx.Q[t*HiddenSize+hOffset : t*HiddenSize+hOffset+HeadDim]

					// Compute scores against all positions s
					for s := 0; s < seqLen; s++ {
						ks := ctx.K[s*HiddenSize+hOffset : s*HiddenSize+hOffset+HeadDim]
						var dot float32
						for d := 0; d < HeadDim; d++ {
							dot += qt[d] * ks[d]
						}
						ctx.Scores[s] = dot * scaleFactor
					}

					// Softmax over sequence scores
					SoftmaxScalar(ctx.Scores[:seqLen], ctx.Scores[:seqLen], seqLen)

					// Weighted sum of values
					for d := 0; d < HeadDim; d++ {
						var sumVal float32
						for s := 0; s < seqLen; s++ {
							sumVal += ctx.Scores[s] * ctx.V[s*HiddenSize+hOffset+d]
						}
						ctx.AttnContext[t*HiddenSize+hOffset+d] = sumVal
					}
				}
			}

			// Attention output projection, residual connection, and LayerNorm
			for t := 0; t < seqLen; t++ {
				tOffset := t * HiddenSize
				cSlice := ctx.AttnContext[tOffset : tOffset+HiddenSize]
				matMul(cSlice, layer.OutWeight, layer.OutBias, ctx.AttnOut[tOffset:tOffset+HiddenSize], HiddenSize, HiddenSize)

				resSlice := ctx.Residual[tOffset : tOffset+HiddenSize]
				hSlice := ctx.HiddenStates[tOffset : tOffset+HiddenSize]
				attnOutSlice := ctx.AttnOut[tOffset : tOffset+HiddenSize]

				for d := 0; d < HiddenSize; d++ {
					resSlice[d] = hSlice[d] + attnOutSlice[d]
				}

				layerNorm(resSlice, layer.AttnNormW, layer.AttnNormB, hSlice, HiddenSize, LayerNormEps)
			}

			// Feed-Forward Network: Linear (384 -> 1536) -> GELU -> Linear (1536 -> 384) -> Residual -> LayerNorm
			for t := 0; t < seqLen; t++ {
				tOffsetH := t * HiddenSize
				tOffsetFFN := t * IntermediateSize

				xt := ctx.HiddenStates[tOffsetH : tOffsetH+HiddenSize]
				ffnMid := ctx.FFNMid[tOffsetFFN : tOffsetFFN+IntermediateSize]
				ffnOut := ctx.FFNOut[tOffsetH : tOffsetH+HiddenSize]

				matMul(xt, layer.FFN1Weight, layer.FFN1Bias, ffnMid, HiddenSize, IntermediateSize)
				gelu(ffnMid, ffnMid, IntermediateSize)
				matMul(ffnMid, layer.FFN2Weight, layer.FFN2Bias, ffnOut, IntermediateSize, HiddenSize)

				resSlice := ctx.Residual[tOffsetH : tOffsetH+HiddenSize]
				for d := 0; d < HiddenSize; d++ {
					resSlice[d] = xt[d] + ffnOut[d]
				}

				layerNorm(resSlice, layer.FFNNormW, layer.FFNNormB, xt, HiddenSize, LayerNormEps)
			}
		}
	}

	// 3. Mean Pooling
	MeanPoolScalar(ctx.HiddenStates, ctx.AttentionMask, ctx.Pooled, seqLen, HiddenSize)

	// 4. L2 Normalization
	l2Norm(ctx.Pooled, out, HiddenSize)

	return out
}

// CosineSimilarity computes cosine similarity between two 384-dimensional embeddings.
func CosineSimilarity(a, b []float32) float32 {
	if HasSIMD {
		return CosineSimilaritySIMD(a, b, HiddenSize)
	}
	return CosineSimilarityScalar(a, b, HiddenSize)
}
