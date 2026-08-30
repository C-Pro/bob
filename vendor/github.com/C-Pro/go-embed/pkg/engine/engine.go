package engine

import (
	"fmt"
	"sync"
)

// Engine is the thread-safe, production-ready inference engine for text embedding generation.
type Engine struct {
	model         *Model
	pool          *ContextPool
	queryPrefix   string
	passagePrefix string
}

// Option configures engine loading, precision, model storage parameters, and prefix formatting.
type Option func(*engineConfig)

type engineConfig struct {
	modelName           string
	dataDir             string
	modelPath           string
	tokenizerPath       string
	precision           PrecisionMode
	windowSize          int
	overlap             int
	queryPrefix         string
	passagePrefix       string
	hasExplicitPrefixes bool
	silent              bool
}

// WithDataDir overrides the directory where model files are stored and looked up.
// If the model files are not found in this directory, they are automatically downloaded here.
func WithDataDir(dir string) Option {
	return func(c *engineConfig) {
		c.dataDir = dir
	}
}

// WithModelName overrides the Hugging Face repository name (default: "intfloat/multilingual-e5-small").
func WithModelName(name string) Option {
	return func(c *engineConfig) {
		c.modelName = name
	}
}

// WithModelPath provides explicit paths to local model.safetensors and tokenizer.json files,
// bypassing automatic directory resolution and downloading.
func WithModelPath(modelPath, tokenizerPath string) Option {
	return func(c *engineConfig) {
		c.modelPath = modelPath
		c.tokenizerPath = tokenizerPath
	}
}

// WithPrecision sets the numerical precision mode (PrecisionFP32, PrecisionBF16, or PrecisionINT8).
func WithPrecision(prec PrecisionMode) Option {
	return func(c *engineConfig) {
		c.precision = prec
	}
}

// WithBF16 configures the engine to use BFloat16 16-bit float weights.
// Note: BF16 trades off a small amount of compute instruction speed for a 50% reduction in RAM footprint (225 MB vs 449 MB).
func WithBF16() Option {
	return func(c *engineConfig) {
		c.precision = PrecisionBF16
	}
}

// WithINT8 configures the engine to use INT8 weight-only dynamic quantization (125 MB RAM).
func WithINT8() Option {
	return func(c *engineConfig) {
		c.precision = PrecisionINT8
	}
}

// WithQuantization enables or disables INT8 weight quantization at model load time.
func WithQuantization(enabled bool) Option {
	return func(c *engineConfig) {
		if enabled {
			c.precision = PrecisionINT8
		} else {
			c.precision = PrecisionFP32
		}
	}
}

// WithChunking configures the sliding window size and overlap token count for inputs exceeding max tokens.
func WithChunking(windowSize, overlap int) Option {
	return func(c *engineConfig) {
		c.windowSize = windowSize
		c.overlap = overlap
	}
}

// WithOverlap configures the number of overlapping tokens between successive sliding windows.
func WithOverlap(overlap int) Option {
	return func(c *engineConfig) {
		c.overlap = overlap
	}
}

// WithPrefixes sets custom query and passage prefixes, overriding automatic detection.
func WithPrefixes(queryPrefix, passagePrefix string) Option {
	return func(c *engineConfig) {
		c.queryPrefix = queryPrefix
		c.passagePrefix = passagePrefix
		c.hasExplicitPrefixes = true
	}
}

// WithQueryPrefix sets an explicit query prefix.
func WithQueryPrefix(prefix string) Option {
	return func(c *engineConfig) {
		c.queryPrefix = prefix
		c.hasExplicitPrefixes = true
	}
}

// WithPassagePrefix sets an explicit passage prefix.
func WithPassagePrefix(prefix string) Option {
	return func(c *engineConfig) {
		c.passagePrefix = prefix
		c.hasExplicitPrefixes = true
	}
}

// WithNoPrefixes disables query and passage prefix prepending completely.
func WithNoPrefixes() Option {
	return func(c *engineConfig) {
		c.queryPrefix = ""
		c.passagePrefix = ""
		c.hasExplicitPrefixes = true
	}
}

// WithSilentDownload silences stdout progress reporting during automatic Hugging Face file downloads.
func WithSilentDownload(silent bool) Option {
	return func(c *engineConfig) {
		c.silent = silent
	}
}

// NewEngine initializes a text embedding engine using ergonomic functional options.
// By default, it checks the local data directory for "intfloat/multilingual-e5-small" weights,
// automatically downloads them from Hugging Face if missing, and loads the model in FP32 precision.
//
// Examples:
//
//	eng, err := engine.NewEngine() // Default: downloads if missing, uses FP32
//	eng, err := engine.NewEngine(engine.WithDataDir("/var/models"), engine.WithBF16())
//	eng, err := engine.NewEngine(engine.WithModelName("intfloat/multilingual-e5-small"), engine.WithINT8())
func NewEngine(opts ...Option) (*Engine, error) {
	cfg := engineConfig{
		modelName:  DefaultModelName,
		precision:  PrecisionFP32,
		windowSize: DefaultWindowSize,
		overlap:    DefaultOverlap,
		silent:     false,
	}
	for _, opt := range opts {
		opt(&cfg)
	}

	modelPath := cfg.modelPath
	tokenizerPath := cfg.tokenizerPath

	if modelPath == "" || tokenizerPath == "" {
		var err error
		modelPath, tokenizerPath, err = EnsureModelFiles(cfg.dataDir, cfg.modelName, cfg.silent)
		if err != nil {
			return nil, fmt.Errorf("failed to prepare model files: %w", err)
		}
	}

	queryPrefix := cfg.queryPrefix
	passagePrefix := cfg.passagePrefix
	if !cfg.hasExplicitPrefixes {
		queryPrefix, passagePrefix = DetectModelPrefixes(cfg.dataDir, cfg.modelName, cfg.silent)
	}

	m, err := LoadModelWithPrecision(modelPath, tokenizerPath, cfg.precision)
	if err != nil {
		return nil, fmt.Errorf("failed to load model: %w", err)
	}

	return &Engine{
		model:         m,
		pool:          NewContextPool(m, cfg.windowSize, cfg.overlap, queryPrefix, passagePrefix),
		queryPrefix:   queryPrefix,
		passagePrefix: passagePrefix,
	}, nil
}

// New creates a new Engine instance with explicit model paths in FP32 precision.
func New(modelPath, tokenizerPath string) (*Engine, error) {
	return NewEngine(WithModelPath(modelPath, tokenizerPath), WithPrecision(PrecisionFP32))
}

// NewBF16 creates a new Engine instance with explicit model paths in BFloat16 precision.
func NewBF16(modelPath, tokenizerPath string) (*Engine, error) {
	return NewEngine(WithModelPath(modelPath, tokenizerPath), WithBF16())
}

// NewQuantized creates a new Engine instance with explicit model paths in INT8 precision.
func NewQuantized(modelPath, tokenizerPath string) (*Engine, error) {
	return NewEngine(WithModelPath(modelPath, tokenizerPath), WithINT8())
}

// NewWithModel creates an Engine directly from an existing pre-loaded Model instance.
func NewWithModel(m *Model) *Engine {
	return &Engine{
		model: m,
		pool:  NewContextPool(m, DefaultWindowSize, DefaultOverlap, "", ""),
	}
}

// NewWithOptions creates a new Engine with custom configuration options.
func NewWithOptions(modelPath, tokenizerPath string, opts ...Option) (*Engine, error) {
	allOpts := append([]Option{WithModelPath(modelPath, tokenizerPath)}, opts...)
	return NewEngine(allOpts...)
}

// Precision returns the numerical precision mode of the engine.
func (e *Engine) Precision() PrecisionMode {
	return e.model.Precision
}

// IsQuantized returns true if the engine is running in INT8 quantized mode.
func (e *Engine) IsQuantized() bool {
	return e.model.Precision == PrecisionINT8
}

// Model returns the underlying Model.
func (e *Engine) Model() *Model {
	return e.model
}

// QueryPrefix returns the configured prefix prepended to queries by EmbedQuery.
func (e *Engine) QueryPrefix() string {
	return e.queryPrefix
}

// PassagePrefix returns the configured prefix prepended to passages by EmbedPassage.
func (e *Engine) PassagePrefix() string {
	return e.passagePrefix
}

// Close releases any memory-mapped file resources held by the engine's model.
func (e *Engine) Close() error {
	if e != nil && e.model != nil {
		return e.model.Close()
	}
	return nil
}

// Embed generates L2-normalized 384-dimensional embeddings for the input text across sliding windows.
// If the input text exceeds max tokens, it returns a slice of vectors, one for each window chunk.
func (e *Engine) Embed(text string) ([][]float32, error) {
	if e == nil || e.pool == nil {
		return nil, fmt.Errorf("engine is not initialized")
	}
	ctx := e.pool.Get()
	defer e.pool.Put(ctx)

	return ctx.Embed(text)
}

// EmbedQuery generates embeddings for a query with standard 'query: ' prefix.
func (e *Engine) EmbedQuery(text string) ([][]float32, error) {
	if e == nil || e.pool == nil {
		return nil, fmt.Errorf("engine is not initialized")
	}
	ctx := e.pool.Get()
	defer e.pool.Put(ctx)

	return ctx.EmbedQuery(text)
}

// EmbedPassage generates embeddings for a passage with standard 'passage: ' prefix.
func (e *Engine) EmbedPassage(text string) ([][]float32, error) {
	if e == nil || e.pool == nil {
		return nil, fmt.Errorf("engine is not initialized")
	}
	ctx := e.pool.Get()
	defer e.pool.Put(ctx)

	return ctx.EmbedPassage(text)
}

// EmbedBatch generates embeddings concurrently across multiple texts.
func (e *Engine) EmbedBatch(texts []string) ([][][]float32, error) {
	if e == nil || e.pool == nil {
		return nil, fmt.Errorf("engine is not initialized")
	}
	if len(texts) == 0 {
		return [][][]float32{}, nil
	}

	results := make([][][]float32, len(texts))
	var wg sync.WaitGroup
	errCh := make(chan error, len(texts))

	for i, text := range texts {
		wg.Add(1)
		go func(idx int, txt string) {
			defer wg.Done()
			ctx := e.pool.Get()
			defer e.pool.Put(ctx)

			embs, err := ctx.Embed(txt)
			if err != nil {
				errCh <- fmt.Errorf("batch item %d failed: %w", idx, err)
				return
			}
			results[idx] = embs
		}(i, text)
	}

	wg.Wait()
	close(errCh)

	if len(errCh) > 0 {
		return nil, <-errCh
	}
	return results, nil
}

// Similarity calculates the maximum cosine similarity across all chunk pairs between two text strings.
func (e *Engine) Similarity(textA, textB string) (float32, error) {
	if e == nil || e.pool == nil {
		return 0, fmt.Errorf("engine is not initialized")
	}
	embA, err := e.Embed(textA)
	if err != nil {
		return 0, err
	}
	embB, err := e.Embed(textB)
	if err != nil {
		return 0, err
	}
	if len(embA) == 0 || len(embB) == 0 {
		return 0, nil
	}

	var maxSim float32 = -1.0
	for _, va := range embA {
		for _, vb := range embB {
			sim := CosineSimilarity(va, vb)
			if sim > maxSim {
				maxSim = sim
			}
		}
	}
	return maxSim, nil
}
