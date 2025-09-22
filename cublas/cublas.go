package cublas

/*
#include <cublas_v2.h>
*/
import "C"
import (
	"sync"

	"github.com/gocnn/gocu"
	"github.com/gocnn/gomat/blas"
)

type srotmParams struct {
	flag float32
	h    [4]float32
}

type drotmParams struct {
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

const (
	NoTrans   = C.CUBLAS_OP_N // NoTrans represents the no-transpose operation
	Trans     = C.CUBLAS_OP_T // Trans represents the transpose operation
	ConjTrans = C.CUBLAS_OP_C // ConjTrans represents the conjugate transpose operation

	Upper = C.CUBLAS_FILL_MODE_UPPER // Upper is used to specify that the matrix is an upper triangular matrix
	Lower = C.CUBLAS_FILL_MODE_LOWER // Lower is used to specify that the matrix is an lower triangular matrix

	NonUnit = C.CUBLAS_DIAG_NON_UNIT // NonUnit is used to specify that the matrix is not a unit triangular matrix
	Unit    = C.CUBLAS_DIAG_UNIT     // Unit is used to specify that the matrix is a unit triangular matrix

	Left  = C.CUBLAS_SIDE_LEFT  // Left is used to specify a multiplication op is performed from the left
	Right = C.CUBLAS_SIDE_RIGHT // Right is used to specify a multiplication op is performed from the right
)

func trans2cublasTrans(t blas.Transpose) C.cublasOperation_t {
	switch t {
	case blas.NoTrans:
		return NoTrans
	case blas.Trans:
		return Trans
	case blas.ConjTrans:
		return ConjTrans
	}
	panic("Unreachable")
}

func side2cublasSide(s blas.Side) C.cublasSideMode_t {
	switch s {
	case blas.Left:
		return Left
	case blas.Right:
		return Right
	}
	panic("Unreachable")
}

func diag2cublasDiag(d blas.Diag) C.cublasDiagType_t {
	switch d {
	case blas.Unit:
		return Unit
	case blas.NonUnit:
		return NonUnit
	}
	panic("Unreachable")
}

func uplo2cublasUplo(u blas.Uplo) C.cublasFillMode_t {
	switch u {
	case blas.Upper:
		return Upper
	case blas.Lower:
		return Lower
	}
	panic("Unreachable")
}

// Standard is the standard cuBLAS handler.
// By default it assumes that the data is in  RowMajor, DESPITE the fact that cuBLAS
// takes ColMajor only. This is done for the ease of use of developers writing in Go.
//
// Use New to create a new BLAS handler.
// Use the various ConsOpts to set the options
type Standard struct {
	h C.cublasHandle_t
	o Order
	m PointerMode
	e error

	ctx       gocu.Context
	dataOnDev bool

	sync.Mutex
}

func New() *Standard {
	var handle C.cublasHandle_t
	if err := Check(C.cublasCreate(&handle)); err != nil {
		panic(err)
	}

	impl := &Standard{
		h: handle,
	}

	return impl
}

func (impl *Standard) SetContext(ctx gocu.Context) {
	impl.Lock()
	defer impl.Unlock()
	impl.ctx = ctx
}

func (impl *Standard) WithNativeData() {
	impl.Lock()
	defer impl.Unlock()
	impl.dataOnDev = false
}

func (impl *Standard) Err() error { return impl.e }

func (impl *Standard) Close() error {
	impl.Lock()
	defer impl.Unlock()

	var empty C.cublasHandle_t
	if impl.h == empty {
		return nil
	}
	if err := Check(C.cublasDestroy(impl.h)); err != nil {
		return err
	}
	impl.h = empty
	impl.ctx = gocu.Context{}
	return nil
}
