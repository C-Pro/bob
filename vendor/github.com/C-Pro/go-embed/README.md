# go-embed

[![CI](https://github.com/C-Pro/go-embed/actions/workflows/ci.yml/badge.svg)](https://github.com/C-Pro/go-embed/actions/workflows/ci.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/C-Pro/go-embed.svg)](https://pkg.go.dev/github.com/C-Pro/go-embed)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A high-performance, pure Go (CGO-free), CPU-only inference library for tokenization and text embedding generation with **zero external dependencies**.

`go-embed` natively executes [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small) (XLM-RoBERTa architecture: 12 layers, hidden dimension 384, 12 attention heads, 1536 intermediate size, ~250k vocabulary) with exact numerical parity to PyTorch/Hugging Face Transformers.

---

## Key Features

- **Zero Dependencies / Pure Go:** Zero third-party dependencies — built entirely on the Go standard library.
- **CGO-Free:** Zero external C/C++ libraries, BLAS/LAPACK binaries, or Python runtimes required. Cross-compiles to any Go-supported platform out of the box.
- **Multiple Precision Modes:**
  - **FP32 Full Precision (Default):** Maximum compute performance (`~54 ms` short query latency, 449 MB RAM).
  - **BFloat16 (BF16):** Native 16-bit float weights reduce RAM to **225 MB (2× reduction)** with $>99.999\%$ fidelity and zero scaling overhead.
  - **INT8 Dynamic Quantization (W8A32):** Dynamic 8-bit integer weights reduce RAM to **~125 MB (3.6× reduction)** with $>99.97\%$ fidelity.
- **Sliding Window Chunking with Overlap:** Full support for input texts of arbitrary length. Inputs exceeding the model sequence length are tokenized across the full text and chunked with configurable sliding windows and overlaps (`WithChunking(windowSize, overlap)`), returning a slice of 384-dimensional vectors (`[][]float32`).
- **Zero-Copy Memory Mapping (`mmap`) & Instant Startup:** Model weights across all precision modes (FP32, BF16, and INT8) are memory-mapped directly into virtual address space via cross-platform OS syscalls (Windows `CreateFileMapping` / POSIX `syscall.Mmap`). Word embeddings (~85% of model size) are demand-paged on-the-fly by the OS kernel, enabling instant `0.00s` startup time and automatic page cache eviction under memory pressure.
- **Automatic Quantized Disk Caching:** When BF16 or INT8 precision is selected, `go-embed` automatically generates and saves 64-byte SIMD-aligned `model_bf16.safetensors` or `model_int8.safetensors` on disk during first load for instant subsequent `mmap` executions.
- **Zero Heap Allocations in Steady-State:** Pre-allocated scratchpad buffers (`InferenceContext`) achieve **`0 B/op` and `0 allocs/op`** during inference across all precision modes.
- **Hardware Acceleration via Go 1.27 SIMD:** Leverages Go 1.27's standard `simd.Float32s` with 8-way unrolled fused multiply-add (`MulAdd`), vectorized LayerNorm, fast GELU, and cosine similarity. Includes portable scalar fallback when compiled without SIMD flags.
- **Exact Numerical Parity:** Validated against PyTorch reference outputs with Cosine Similarity $\ge 0.999999$ across short queries, full sentences, code snippets, empty strings, and 512-token max-length sequences.
- **Cross-Lingual Support:** Supports 100+ languages (English, Russian, Indonesian, German, Chinese, etc.) with semantic consistency $\ge 0.80$ across language pairs.
- **Built-in Tokenizer & Safetensors Loader/Writer:** Includes a pure Go SentencePiece/Unigram tokenizer (with NFKC normalization and Viterbi DP search) and `.safetensors` reader/writer.

---

## Precision Modes & Benchmarks

*Benchmarked on AMD Ryzen 7 PRO 8840U (16 threads), Go 1.27, Linux amd64.*

| Sequence Length | Precision | RAM Footprint | Single-Query Latency | Allocations | Fidelity vs PyTorch |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Short Query ($L=32$)** | **FP32** | 449 MB | **`54 ms`** | **`0 B / 0 allocs`** | 100% (Baseline) |
| **Short Query ($L=32$)** | **BF16** | **225 MB** *(2×)* | **`128 ms`** | **`0 B / 0 allocs`** | **`0.999996`** |
| **Short Query ($L=32$)** | **INT8** | **125 MB** *(3.6×)* | `326 ms` | **`0 B / 0 allocs`** | `0.999787` |
| **Long Sequence ($L=512$)** | **FP32** | 449 MB | **`5.57 s`** | **`0 B / 0 allocs`** | 100% (Baseline) |
| **Long Sequence ($L=512$)** | **BF16** | **225 MB** *(2×)* | **`5.89 s`** | **`0 B / 0 allocs`** | **`0.999996`** |
| **Long Sequence ($L=512$)** | **INT8** | **125 MB** *(3.6×)* | `15.50 s` | **`0 B / 0 allocs`** | `0.999787` |

---

## Installation & Requirements

- **Go 1.27+** (recommended for hardware SIMD acceleration) or **Go 1.21+** (for scalar fallback).

To install the module:

```bash
go get github.com/C-Pro/go-embed
```

---

## Quick Start

### 1. Minimal Setup (Zero Manual Downloads)

`NewEngine()` automatically checks the local models directory and downloads `multilingual-e5-small` weights from Hugging Face if not already present:

```go
package main

import (
	"fmt"
	"log"

	"github.com/C-Pro/go-embed"
)

func main() {
	// Initialize engine: automatically downloads model if missing, uses FP32 by default
	eng, err := embed.NewEngine()
	if err != nil {
		log.Fatalf("Failed to initialize engine: %v", err)
	}
	defer eng.Close()

	// E5 models use "query: " and "passage: " prefixes for asymmetric retrieval
	query := "how to implement consensus in distributed systems?"
	relPassage := "Consensus algorithms like Raft and Paxos ensure consistency across nodes."
	irrelPassage := "Authentic Italian tiramisu recipe with mascarpone and espresso."

	// Generate 384-dimensional L2-normalized embeddings (returns [][]float32 for sliding window chunks)
	qEmbs, _ := eng.EmbedQuery(query)
	pRelEmbs, _ := eng.EmbedPassage(relPassage)
	pIrrelEmbs, _ := eng.EmbedPassage(irrelPassage)

	// Compute Cosine Similarity
	simRel := embed.CosineSimilarity(qEmbs[0], pRelEmbs[0])
	simIrrel := embed.CosineSimilarity(qEmbs[0], pIrrelEmbs[0])

	fmt.Printf("Query: %s\n", query)
	fmt.Printf("Similarity to relevant passage:   %.4f\n", simRel)   // ~0.87
	fmt.Printf("Similarity to irrelevant passage: %.4f\n", simIrrel) // ~0.73
}
```

---

### 2. Ergonomic Functional Options & Dynamic Prefix Detection

`go-embed` automatically detects task prefix requirements:
1. From **`config_sentence_transformers.json`** if present in the Hugging Face repository metadata (`prompts` dictionary).
2. From **known model families** (e.g. `e5` uses `"query: "` and `"passage: "`, `bge` uses retrieval instruction, `MiniLM` uses no prefix).
3. Via **explicit user overrides**:

```go
// Custom task prefixes overriding auto-detection
eng, err := embed.NewEngine(
    embed.WithPrefixes("search_query: ", "search_document: "),
)

// Disable automatic prefixes completely (treat as symmetric model)
eng, err := embed.NewEngine(
    embed.WithNoPrefixes(),
)

// Configure sliding window size and overlap for documents with >512 tokens
eng, err := embed.NewEngine(
    embed.WithChunking(512, 256), // Window 512, Overlap 256
)

// BFloat16 mode (cuts RAM in half to 225 MB; trades off slight compute speed for memory)
eng, err := embed.NewEngine(
    embed.WithDataDir("./my_models"),
    embed.WithBF16(),
)

// INT8 dynamic quantization mode (125 MB RAM for edge/memory-constrained environments)
eng, err := embed.NewEngine(
    embed.WithDataDir("./my_models"),
    embed.WithINT8(),
)

// Override Hugging Face repository name
eng, err := embed.NewEngine(
    embed.WithModelName("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
)
```

---

## Supported Models & Architectures

`go-embed` natively executes any transformer model based on the **XLM-RoBERTa / RoBERTa (12-layer, 384-dim)** architecture with SentencePiece tokenization and `.safetensors` weights.

### Architecture Specifications

| Parameter | Supported Specification |
| :--- | :--- |
| **Architecture Family** | XLM-RoBERTa / RoBERTa / BERT-compatible Encoder |
| **Hidden Dimension ($D$)** | `384` |
| **Transformer Layers ($L$)** | `12` |
| **Attention Heads ($H$)** | `12` ($d_{\text{head}} = 32$) |
| **Intermediate FFN Size ($D_{\text{ffn}}$)** | `1536` |
| **Layer Normalization** | Post-LN ($\epsilon = 10^{-5}$) |
| **Activation Function** | GELU |
| **Tokenizer Algorithm** | SentencePiece (Unigram) with NFKC normalization |
| **Max Sequence Length** | 512 tokens (unlimited via sliding windows) |
| **File Formats** | `model.safetensors` + `tokenizer.json` |

---

### Verified Models (Out-of-the-Box)

The following models are verified and automatically downloaded from Hugging Face when specified via `WithModelName(...)`:

#### 1. [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small) (Default)
- **Best for:** Semantic search, document retrieval, and question answering across 100+ languages.
- **Prefix convention:** Requires `query: ` and `passage: ` prefixes for asymmetric retrieval.
- **Example:**
  ```go
  eng, err := embed.NewEngine(
      embed.WithModelName("intfloat/multilingual-e5-small"),
  )
  qEmbs, _ := eng.EmbedQuery("how to implement consensus in distributed systems?")
  pEmbs, _ := eng.EmbedPassage("Consensus algorithms like Raft and Paxos ensure consistency.")
  ```

#### 2. [`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- **Best for:** Semantic similarity, sentence paraphrasing, deduplication, and cross-lingual clustering across 50+ languages.
- **Prefix convention:** Direct text input (no prefix required).
- **Example:**
  ```go
  eng, err := embed.NewEngine(
      embed.WithModelName("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
      embed.WithBF16(), // Cuts RAM in half (225 MB)
  )
  embs1, _ := eng.Embed("The cat is sleeping peacefully on the sofa.")
  embs2, _ := eng.Embed("Die Katze schläft friedlich auf dem Sofa.")
  similarity, _ := eng.Similarity("The cat is sleeping peacefully on the sofa.", "Die Katze schläft friedlich auf dem Sofa.") // ~0.9846
  ```

---

### 3. Reusable Context Pool

For high-throughput web servers or pipeline processing, reuse pre-allocated buffers with `ContextPool`:

```go
package main

import (
	"fmt"
	"github.com/C-Pro/go-embed"
)

func main() {
	// Initialize engine with default options (auto-downloads if missing)
	eng, err := embed.NewEngine()
	if err != nil {
		panic(err)
	}
	defer eng.Close()

	// ContextPool manages reusable memory scratchpads for goroutines
	pool := embed.NewContextPool(eng.Model(), embed.DefaultWindowSize, embed.DefaultOverlap, "", "")
	ctx := pool.Get()
	defer pool.Put(ctx)

	// Generates embeddings across sliding windows
	embs, err := ctx.Embed("query: low latency Go inference")
	if err != nil {
		panic(err)
	}

	fmt.Printf("Generated %d vector(s), first 3 dimensions: [%.4f, %.4f, %.4f]\n", len(embs), embs[0][0], embs[0][1], embs[0][2])
}
```

---

### 4. Parallel Batch Embedding

Process multiple queries concurrently across CPU cores:

```go
texts := []string{
	"query: Kubernetes cluster architecture",
	"query: Distributed key-value store in Go",
	"query: Lock-free ring buffer implementation",
}

// Concurrently computes embeddings across available CPU threads
embeddings, err := eng.EmbedBatch(texts)
if err != nil {
	log.Fatal(err)
}
fmt.Printf("Computed %d embeddings\n", len(embeddings))
```

---

### 5. Precision Modes & Trade-offs (FP32 vs BF16 vs INT8)

`go-embed` provides three precision modes to balance memory footprint and latency:

* **FP32 Full Precision (`449 MB RAM`, default):** Fastest compute speed for short queries (`54 ms`) with 100% baseline numerical precision.
* **BFloat16 (`225 MB RAM`, `WithBF16()`):** **Trades off slight CPU instruction speed for a 50% memory reduction.** Because weights are stored as `uint16` and unpacked into 32-bit SIMD registers during computation, short query latency is `128 ms`, but for long sequences ($L=512$) the latency difference is only ~5% (`5.89 s` vs `5.57 s`) due to better CPU cache locality.
* **INT8 Dynamic Quantization (`125 MB RAM`, `WithINT8()`):** Maximum memory reduction (3.6× smaller) for resource-constrained edge environments.

```go
// BFloat16 mode (2x smaller RAM, >99.999% fidelity)
bf16Eng, err := embed.NewEngine(embed.WithBF16())

// INT8 mode (3.6x smaller RAM)
int8Eng, err := embed.NewEngine(embed.WithINT8())
```

---

## Interactive CLI Tool

`go-embed` includes an interactive CLI utility for generating embeddings and computing semantic similarities:

```bash
# Interactive REPL
GOEXPERIMENT=simd go run cmd/embed/main.go -mode=cli

# Direct Similarity Comparison
GOEXPERIMENT=simd go run cmd/embed/main.go -mode=sim \
  -t1="how to implement consensus in distributed systems?" \
  -t2="Consensus algorithms like Raft and Paxos ensure state machine consistency."

# Single Query Embedding
GOEXPERIMENT=simd go run cmd/embed/main.go -mode=query -t1="machine learning in Go"
```

---

## Running Tests & Verification

```bash
# Run all unit tests, golden parity verification, and cross-lingual sanity tests
GOEXPERIMENT=simd go test -v ./...

# Run zero-allocation assertion (verifies 0.0 allocs/op)
GOEXPERIMENT=simd go test -v -run TestZeroAllocations ./pkg/engine

# Run comparative benchmarks
GOEXPERIMENT=simd go test -run=^$ -bench=. -benchmem ./pkg/engine
```

---

## Architecture & Project Structure

```
.
├── .github/
│   └── workflows/ci.yml    # GitHub Actions CI (Semgrep, OSV scanner, lint, tests)
├── cmd/
│   └── embed/              # Interactive CLI application
├── pkg/
│   ├── engine/             # Core inference engine, transformer layers & scratchpad pool
│   │   ├── downloader.go   # Automatic Hugging Face model downloader and caching
│   │   ├── ops_simd.go     # Go 1.27 hardware SIMD vector kernels (FMA, LayerNorm, GELU)
│   │   ├── ops_scalar.go   # Unrolled portable scalar math kernels
│   │   ├── quant.go        # INT8 dynamic quantization structures and kernels
│   │   ├── quant_bf16.go   # BFloat16 16-bit float packaging and math
│   │   ├── context.go      # Pre-allocated zero-allocation scratchpad and sync.Pool
│   │   ├── model.go        # Contiguous model parameter layouts & safetensors loader
│   │   └── engine.go       # High-level ergonomic API (NewEngine with functional options)
│   ├── safetensors/        # Pure Go .safetensors binary reader, writer & OS mmap
│   │   ├── mmap.go         # MmapFile interface
│   │   ├── mmap_unix.go    # POSIX / Linux / Darwin syscall.Mmap
│   │   ├── mmap_windows.go # Windows CreateFileMapping / MapViewOfFile
│   │   ├── mmap_fallback.go# Portable fallback
│   │   ├── writer.go       # Atomic 64-byte aligned safetensors writer
│   │   └── safetensors.go  # Safetensors memory-mapped tensor views
│   └── tokenizer/          # Pure Go SentencePiece/Unigram tokenizer & Viterbi search
└── testdata/
    └── golden.json         # 26 PyTorch/Transformers ground truth vectors
```

---

## Acknowledgments

- [`github.com/nlpodyssey/spago`](https://github.com/nlpodyssey/spago) for the original pure-Go machine learning framework and foundation.
- [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small) for the multilingual embedding model weights.
- [`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) for the paraphrase embedding model weights.

## License

MIT License.

