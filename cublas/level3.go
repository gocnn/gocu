package cublas

// #include <cublas_v2.h>
import "C"
import (
	"github.com/gocnn/gocu/cudart"
	"github.com/gocnn/gomat/blas" // Only for error constants
)

func (impl *Standard) Sgemm(tA, tB Transpose, m, n, k int, alpha float32, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta float32, c cudart.DevicePtr, ldc int) {
	if impl.e != nil {
		return
	}

	if tA != NoTranspose && tA != Transpose_ && tA != ConjTranspose {
		panic(blas.ErrBadTranspose)
	}
	if tB != NoTranspose && tB != Transpose_ && tB != ConjTranspose {
		panic(blas.ErrBadTranspose)
	}
	if m < 0 {
		panic(blas.ErrMLT0)
	}
	if n < 0 {
		panic(blas.ErrMLT0)
	}
	if k < 0 {
		panic(blas.ErrKLT0)
	}
	impl.e = Check(C.cublasSgemm(C.cublasHandle_t(impl.h), C.cublasOperation_t(tA), C.cublasOperation_t(tB), C.int(m), C.int(n), C.int(k), (*C.float)(&alpha), (*C.float)(a), C.int(lda), (*C.float)(b), C.int(ldb), (*C.float)(&beta), (*C.float)(c), C.int(ldc)))
}
