//go:build cuda || cuda11 || cuda12 || cuda13

package cublas

/*
#include <cublas_v2.h>
*/
import "C"

type SrotmParams struct {
	flag float32
	h    [4]float32
}

type DrotmParams struct {
	flag float64
	h    [4]float64
}

// Order is used to specify the matrix storage format. We still interact with
// an API that allows client calls to specify order, so this is here to document that fact.
type Order byte

const (
	RowMajor Order = iota // Row Major
	ColMajor              // Column Major (cublas assumes all matrices be stored in this order)
)

// Transpose specifies the transposition operation for CUBLAS operations.
// This directly maps to cublasOperation_t values for optimal performance.
type Transpose C.cublasOperation_t

const (
	NoTrans   Transpose = Transpose(C.CUBLAS_OP_N) // No transpose operation
	Trans     Transpose = Transpose(C.CUBLAS_OP_T) // Transpose operation
	ConjTrans Transpose = Transpose(C.CUBLAS_OP_C) // Conjugate transpose operation
)

// Side specifies which side of the matrix operation is performed.
// This directly maps to cublasSideMode_t values for optimal performance.
type Side C.cublasSideMode_t

const (
	Left  Side = Side(C.CUBLAS_SIDE_LEFT)  // Operation performed from the left
	Right Side = Side(C.CUBLAS_SIDE_RIGHT) // Operation performed from the right
)

// Diag specifies whether a matrix is unit triangular or not.
// This directly maps to cublasDiagType_t values for optimal performance.
type Diag C.cublasDiagType_t

const (
	NonUnit Diag = Diag(C.CUBLAS_DIAG_NON_UNIT) // Matrix is not unit triangular
	Unit    Diag = Diag(C.CUBLAS_DIAG_UNIT)     // Matrix is unit triangular
)

// Uplo specifies whether the matrix is upper or lower triangular.
// This directly maps to cublasFillMode_t values for optimal performance.
type Uplo C.cublasFillMode_t

const (
	Upper Uplo = Uplo(C.CUBLAS_FILL_MODE_UPPER) // Upper triangular matrix
	Lower Uplo = Uplo(C.CUBLAS_FILL_MODE_LOWER) // Lower triangular matrix
)
