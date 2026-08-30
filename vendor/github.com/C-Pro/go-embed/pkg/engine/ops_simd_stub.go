//go:build !goexperiment.simd

package engine

// HasSIMD indicates SIMD hardware acceleration is inactive in fallback mode.
const HasSIMD = false

func MatVecMulAddSIMD(x []float32, weight []float32, bias []float32, out []float32, inDim, outDim int) {
	MatVecMulAddScalar(x, weight, bias, out, inDim, outDim)
}

func MatVecMulAddINT8SIMD(x []float32, weight []int8, scale []float32, bias []float32, out []float32, inDim, outDim int) {
	MatVecMulAddINT8Scalar(x, weight, scale, bias, out, inDim, outDim)
}

func MatVecMulAddBF16SIMD(x []float32, weight []uint16, bias []float32, out []float32, inDim, outDim int) {
	MatVecMulAddBF16Scalar(x, weight, bias, out, inDim, outDim)
}

func LayerNormSIMD(x []float32, weight, bias, out []float32, dim int, eps float32) {
	LayerNormScalar(x, weight, bias, out, dim, eps)
}

func GELUSIMD(x, out []float32, n int) {
	GELUScalar(x, out, n)
}

func GELUApproxSIMD(x, out []float32, n int) {
	GELUApproxScalar(x, out, n)
}

func L2NormalizeSIMD(v, out []float32, dim int) {
	L2NormalizeScalar(v, out, dim)
}

func CosineSimilaritySIMD(a, b []float32, dim int) float32 {
	return CosineSimilarityScalar(a, b, dim)
}
