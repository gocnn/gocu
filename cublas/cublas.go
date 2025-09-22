package cublas

/*
#include <cublas_v2.h>
*/
import "C"
import (
	"sync"
)

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

// PointerMode
type PointerMode byte

const (
	Host PointerMode = iota
	Device
)

// Transpose specifies the transposition operation for CUBLAS operations.
// This directly maps to cublasOperation_t values for optimal performance.
type Transpose C.cublasOperation_t

const (
	NoTranspose   Transpose = Transpose(C.CUBLAS_OP_N) // No transpose operation
	Transpose_    Transpose = Transpose(C.CUBLAS_OP_T) // Transpose operation
	ConjTranspose Transpose = Transpose(C.CUBLAS_OP_C) // Conjugate transpose operation
)

// Side specifies which side of the matrix operation is performed.
// This directly maps to cublasSideMode_t values for optimal performance.
type Side C.cublasSideMode_t

const (
	LeftSide  Side = Side(C.CUBLAS_SIDE_LEFT)  // Operation performed from the left
	RightSide Side = Side(C.CUBLAS_SIDE_RIGHT) // Operation performed from the right
)

// Diag specifies whether a matrix is unit triangular or not.
// This directly maps to cublasDiagType_t values for optimal performance.
type Diag C.cublasDiagType_t

const (
	NonUnitDiag Diag = Diag(C.CUBLAS_DIAG_NON_UNIT) // Matrix is not unit triangular
	UnitDiag    Diag = Diag(C.CUBLAS_DIAG_UNIT)     // Matrix is unit triangular
)

// Uplo specifies whether the matrix is upper or lower triangular.
// This directly maps to cublasFillMode_t values for optimal performance.
type Uplo C.cublasFillMode_t

const (
	UpperTriangular Uplo = Uplo(C.CUBLAS_FILL_MODE_UPPER) // Upper triangular matrix
	LowerTriangular Uplo = Uplo(C.CUBLAS_FILL_MODE_LOWER) // Lower triangular matrix
)

// Handler is the standard cuBLAS handler.
// By default it assumes that the data is in  RowMajor, DESPITE the fact that cuBLAS
// takes ColMajor only. This is done for the ease of use of developers writing in Go.
//
// Use New to create a new BLAS handler.
// Use the various ConsOpts to set the options
type Handler struct {
	Handle

	e error

	sync.RWMutex
}

func New() *Handler {
	handle, err := Create()
	if err != nil {
		panic(err)
	}
	handler := &Handler{Handle: handle}
	return handler
}

func (h *Handler) Err() error {
	h.RLock()
	defer h.RUnlock()
	return h.e
}

func (h *Handler) Close() error {
	h.Lock()
	defer h.Unlock()
	if err := h.Handle.Destroy(); err != nil {
		return err
	}
	return nil
}
