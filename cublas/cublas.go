//go:build !cuda && !cuda11 && !cuda12 && !cuda13

package cublas

// SrotmParams contains Givens transformation parameters returned
// by the Float32 Srotm method.
type SrotmParams struct {
	Flag float32
	H    [4]float32
}

// DrotmParams contains Givens transformation parameters returned
// by the Float64 Drotm method.
type DrotmParams struct {
	Flag float64
	H    [4]float64
}

// Order is used to specify the matrix storage format. We still interact with
// an API that allows client calls to specify order, so this is here to document that fact.
type Order byte

const (
	RowMajor Order = iota // Row Major
	ColMajor              // Column Major (cublas assumes all matrices be stored in this order)
)

// Transpose specifies the transposition operation for CUBLAS operations.
type Transpose int

const (
	NoTrans   Transpose = 0 // No transpose operation
	Trans     Transpose = 1 // Transpose operation
	ConjTrans Transpose = 2 // Conjugate transpose operation
)

// Side specifies which side of the matrix operation is performed.
type Side int

const (
	Left  Side = 0 // Operation performed from the left
	Right Side = 1 // Operation performed from the right
)

// Diag specifies whether a matrix is unit triangular or not.
type Diag int

const (
	NonUnit Diag = 0 // Matrix is not unit triangular
	Unit    Diag = 1 // Matrix is unit triangular
)

// Uplo specifies whether the matrix is upper or lower triangular.
type Uplo int

const (
	Upper Uplo = 0 // Upper triangular matrix
	Lower Uplo = 1 // Lower triangular matrix
)
