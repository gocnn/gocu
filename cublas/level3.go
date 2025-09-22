package cublas

// #include <cublas_v2.h>
import "C"
import (
	"github.com/gocnn/gocu/cudart"
	"github.com/gocnn/gomat/blas"
)

func (impl *Standard) Sgemm(tA, tB blas.Transpose, m, n, k int, alpha float32, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta float32, c cudart.DevicePtr, ldc int) {
	if impl.e != nil {
		return
	}

	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic(blas.ErrBadTranspose)
	}
	if tB != blas.NoTrans && tB != blas.Trans && tB != blas.ConjTrans {
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
	impl.e = Check(C.cublasSgemm(C.cublasHandle_t(impl.h), trans2cublasTrans(tA), trans2cublasTrans(tB), C.int(m), C.int(n), C.int(k), (*C.float)(&alpha), (*C.float)(a), C.int(lda), (*C.float)(b), C.int(ldb), (*C.float)(&beta), (*C.float)(c), C.int(ldc)))
}
