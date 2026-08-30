// Package embed provides pure Go transformer-based text embeddings with SIMD acceleration,
// dynamic quantization (BF16, INT8), sliding window chunking, and Hugging Face model downloading.
package embed

import (
	"github.com/C-Pro/go-embed/pkg/engine"
)

// Re-export core types.
type (
	// Engine is the thread-safe, production-ready inference engine for text embedding generation.
	Engine = engine.Engine

	// Option configures engine loading, precision, model storage parameters, and prefix formatting.
	Option = engine.Option

	// Model represents the self-contained transformer embedding model.
	Model = engine.Model

	// Layer holds parameter weights and biases for a single transformer layer.
	Layer = engine.Layer

	// BF16Layer holds BFloat16 quantized weights for a single transformer layer.
	BF16Layer = engine.BF16Layer

	// BF16Linear represents a linear weight matrix stored in BFloat16 format.
	BF16Linear = engine.BF16Linear

	// QuantizedLayer holds INT8 quantized weights for a single transformer layer.
	QuantizedLayer = engine.QuantizedLayer

	// QuantizedLinear represents an INT8 weight-only quantized matrix with per-row FP32 scaling.
	QuantizedLinear = engine.QuantizedLinear

	// QuantizedWordEmbeddings represents INT8 quantized token embeddings with per-token scale.
	QuantizedWordEmbeddings = engine.QuantizedWordEmbeddings

	// PrecisionMode defines the numerical precision mode of the engine.
	PrecisionMode = engine.PrecisionMode

	// InferenceContext encapsulates all pre-allocated scratchpad buffers
	// needed to execute forward inference passes.
	InferenceContext = engine.InferenceContext

	// ContextPool manages reusable InferenceContext instances for high concurrency.
	ContextPool = engine.ContextPool
)

// Re-export precision constants.
const (
	PrecisionFP32 = engine.PrecisionFP32
	PrecisionBF16 = engine.PrecisionBF16
	PrecisionINT8 = engine.PrecisionINT8
)

// Re-export architectural and operational constants.
const (
	DefaultModelName  = engine.DefaultModelName
	DefaultWindowSize = engine.DefaultWindowSize
	DefaultOverlap    = engine.DefaultOverlap

	HiddenSize       = engine.HiddenSize
	IntermediateSize = engine.IntermediateSize
	NumLayers        = engine.NumLayers
	NumHeads         = engine.NumHeads
	HeadDim          = engine.HeadDim
	MaxSeqLen        = engine.MaxSeqLen
	VocabSize        = engine.VocabSize
	LayerNormEps     = engine.LayerNormEps
	HuggingFaceBase  = engine.HuggingFaceBase
	HasSIMD          = engine.HasSIMD
)

// NewEngine initializes a text embedding engine using ergonomic functional options.
// By default, it checks the local data directory for "intfloat/multilingual-e5-small" weights,
// automatically downloads them from Hugging Face if missing, and loads the model in FP32 precision.
//
// Examples:
//
//	eng, err := embed.NewEngine() // Default: downloads if missing, uses FP32
//	eng, err := embed.NewEngine(embed.WithDataDir("/var/models"), embed.WithBF16())
//	eng, err := embed.NewEngine(embed.WithModelName("intfloat/multilingual-e5-small"), embed.WithINT8())
func NewEngine(opts ...Option) (*Engine, error) {
	return engine.NewEngine(opts...)
}

// New creates a new Engine instance with explicit model paths in FP32 precision.
func New(modelPath, tokenizerPath string) (*Engine, error) {
	return engine.New(modelPath, tokenizerPath)
}

// NewBF16 creates a new Engine instance with explicit model paths in BFloat16 precision.
func NewBF16(modelPath, tokenizerPath string) (*Engine, error) {
	return engine.NewBF16(modelPath, tokenizerPath)
}

// NewQuantized creates a new Engine instance with explicit model paths in INT8 precision.
func NewQuantized(modelPath, tokenizerPath string) (*Engine, error) {
	return engine.NewQuantized(modelPath, tokenizerPath)
}

// NewWithModel creates an Engine directly from an existing pre-loaded Model instance.
func NewWithModel(m *Model) *Engine {
	return engine.NewWithModel(m)
}

// NewWithOptions creates a new Engine with custom configuration options.
func NewWithOptions(modelPath, tokenizerPath string, opts ...Option) (*Engine, error) {
	return engine.NewWithOptions(modelPath, tokenizerPath, opts...)
}

// WithDataDir overrides the directory where model files are stored and looked up.
// If the model files are not found in this directory, they are automatically downloaded here.
func WithDataDir(dir string) Option {
	return engine.WithDataDir(dir)
}

// WithModelName overrides the Hugging Face repository name (default: "intfloat/multilingual-e5-small").
func WithModelName(name string) Option {
	return engine.WithModelName(name)
}

// WithModelPath provides explicit paths to local model.safetensors and tokenizer.json files,
// bypassing automatic directory resolution and downloading.
func WithModelPath(modelPath, tokenizerPath string) Option {
	return engine.WithModelPath(modelPath, tokenizerPath)
}

// WithPrecision sets the numerical precision mode (PrecisionFP32, PrecisionBF16, or PrecisionINT8).
func WithPrecision(prec PrecisionMode) Option {
	return engine.WithPrecision(prec)
}

// WithBF16 configures the engine to use BFloat16 16-bit float weights.
func WithBF16() Option {
	return engine.WithBF16()
}

// WithINT8 configures the engine to use INT8 weight-only dynamic quantization.
func WithINT8() Option {
	return engine.WithINT8()
}

// WithQuantization enables or disables INT8 weight quantization at model load time.
func WithQuantization(enabled bool) Option {
	return engine.WithQuantization(enabled)
}

// WithChunking configures the sliding window size and overlap token count for inputs exceeding max tokens.
func WithChunking(windowSize, overlap int) Option {
	return engine.WithChunking(windowSize, overlap)
}

// WithOverlap configures the number of overlapping tokens between successive sliding windows.
func WithOverlap(overlap int) Option {
	return engine.WithOverlap(overlap)
}

// WithPrefixes sets custom query and passage prefixes, overriding automatic detection.
func WithPrefixes(queryPrefix, passagePrefix string) Option {
	return engine.WithPrefixes(queryPrefix, passagePrefix)
}

// WithQueryPrefix sets an explicit query prefix.
func WithQueryPrefix(prefix string) Option {
	return engine.WithQueryPrefix(prefix)
}

// WithPassagePrefix sets an explicit passage prefix.
func WithPassagePrefix(prefix string) Option {
	return engine.WithPassagePrefix(prefix)
}

// WithNoPrefixes disables query and passage prefix prepending completely.
func WithNoPrefixes() Option {
	return engine.WithNoPrefixes()
}

// WithSilentDownload silences stdout progress reporting during automatic Hugging Face file downloads.
func WithSilentDownload(silent bool) Option {
	return engine.WithSilentDownload(silent)
}

// LoadModel loads the model directly from a safetensors file and tokenizer.json in FP32 precision.
func LoadModel(modelPath, tokenizerPath string) (*Model, error) {
	return engine.LoadModel(modelPath, tokenizerPath)
}

// LoadModelWithOptions loads model weights with optional INT8 quantization (backwards compatible).
func LoadModelWithOptions(modelPath, tokenizerPath string, quantize bool) (*Model, error) {
	return engine.LoadModelWithOptions(modelPath, tokenizerPath, quantize)
}

// LoadModelWithPrecision loads model weights directly into the requested precision mode.
func LoadModelWithPrecision(modelPath, tokenizerPath string, prec PrecisionMode) (*Model, error) {
	return engine.LoadModelWithPrecision(modelPath, tokenizerPath, prec)
}

// LoadBF16Model loads model weights with BFloat16 precision.
func LoadBF16Model(modelPath, tokenizerPath string) (*Model, error) {
	return engine.LoadBF16Model(modelPath, tokenizerPath)
}

// LoadQuantizedModel loads model weights with INT8 quantization.
func LoadQuantizedModel(modelPath, tokenizerPath string) (*Model, error) {
	return engine.LoadQuantizedModel(modelPath, tokenizerPath)
}

// SaveBF16Model saves a model in BFloat16 safetensors format to targetPath.
func SaveBF16Model(m *Model, targetPath string) error {
	return engine.SaveBF16Model(m, targetPath)
}

// SaveQuantizedModel saves a quantized model to targetPath.
func SaveQuantizedModel(m *Model, targetPath string) error {
	return engine.SaveQuantizedModel(m, targetPath)
}

// EnsureModelFiles checks if the required model weights (model.safetensors and tokenizer.json)
// exist in targetDir. If they do not exist, it automatically downloads them from Hugging Face.
func EnsureModelFiles(dataDir, modelName string, silent bool) (safetensorsPath, tokenizerPath string, err error) {
	return engine.EnsureModelFiles(dataDir, modelName, silent)
}

// DetectModelPrefixes detects the query and passage prefixes for a model.
func DetectModelPrefixes(dataDir, modelName string, silent bool) (queryPrefix, passagePrefix string) {
	return engine.DetectModelPrefixes(dataDir, modelName, silent)
}

// CosineSimilarity computes cosine similarity between two 384-dimensional embeddings.
func CosineSimilarity(a, b []float32) float32 {
	return engine.CosineSimilarity(a, b)
}

// QuantizeMatrix converts an FP32 matrix of shape [rows, cols] into per-row INT8 and FP32 scale factors.
func QuantizeMatrix(weights []float32, rows, cols int) ([]int8, []float32) {
	return engine.QuantizeMatrix(weights, rows, cols)
}

// Float32ToBFloat16 converts an FP32 value to BFloat16 with round-to-nearest-even.
func Float32ToBFloat16(f float32) uint16 {
	return engine.Float32ToBFloat16(f)
}

// BFloat16ToFloat32 converts a BFloat16 (uint16) back to FP32.
func BFloat16ToFloat32(bf uint16) float32 {
	return engine.BFloat16ToFloat32(bf)
}

// Float32sToBFloat16s converts a slice of FP32 floats into BFloat16 uint16s.
func Float32sToBFloat16s(src []float32) []uint16 {
	return engine.Float32sToBFloat16s(src)
}

// NewContext creates a new InferenceContext with default window size (512) and overlap (256).
func NewContext(m *Model) *InferenceContext {
	return engine.NewContext(m)
}

// NewContextWithOptions creates a new InferenceContext with custom window size, overlap, and prefixes.
func NewContextWithOptions(m *Model, windowSize, overlap int, queryPrefix, passagePrefix string) *InferenceContext {
	return engine.NewContextWithOptions(m, windowSize, overlap, queryPrefix, passagePrefix)
}

// NewContextPool creates a new pool of inference contexts for the given model with configured options.
func NewContextPool(m *Model, windowSize, overlap int, queryPrefix, passagePrefix string) *ContextPool {
	return engine.NewContextPool(m, windowSize, overlap, queryPrefix, passagePrefix)
}
