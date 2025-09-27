package cublas

// #include <cublas_v2.h>
import "C"
import (
	"unsafe"

	"github.com/gocnn/gocu/cudart"
	"github.com/gocnn/gomat/blas"
)

func (h *Handler) Sgemm(tA, tB Transpose, m, n, k int, alpha float32, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta float32, c cudart.DevicePtr, ldc int) {
	if h.e != nil {
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
	h.e = Check(C.cublasSgemm(C.cublasHandle_t(h.h), C.cublasOperation_t(tA), C.cublasOperation_t(tB), C.int(m), C.int(n), C.int(k), (*C.float)(&alpha), (*C.float)(a), C.int(lda), (*C.float)(b), C.int(ldb), (*C.float)(&beta), (*C.float)(c), C.int(ldc)))
}

// Dgemm performs the matrix-matrix operation C = alpha*op(A)*op(B) + beta*C for double precision real matrices.
// op(X) is either X, X^T depending on the transpose parameter.
func (h *Handler) Dgemm(tA, tB Transpose, m, n, k int, alpha float64, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta float64, c cudart.DevicePtr, ldc int) {
	if h.e != nil {
		return
	}
	if m < 0 {
		panic(blas.ErrMLT0)
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if k < 0 {
		panic(blas.ErrKLT0)
	}
	if lda < 1 {
		panic(blas.ErrShortA)
	}
	if ldb < 1 {
		panic(blas.ErrShortB)
	}
	if ldc < 1 {
		panic(blas.ErrShortC)
	}

	h.e = Check(C.cublasDgemm(C.cublasHandle_t(h.h), C.cublasOperation_t(tA), C.cublasOperation_t(tB), C.int(m), C.int(n), C.int(k), (*C.double)(&alpha), (*C.double)(a), C.int(lda), (*C.double)(b), C.int(ldb), (*C.double)(&beta), (*C.double)(c), C.int(ldc)))
}

// Cgemm performs the matrix-matrix operation C = alpha*op(A)*op(B) + beta*C for single precision complex matrices.
// op(X) is either X, X^T, or X^H depending on the transpose parameter.
func (h *Handler) Cgemm(tA, tB Transpose, m, n, k int, alpha complex64, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta complex64, c cudart.DevicePtr, ldc int) {
	if h.e != nil {
		return
	}
	if m < 0 {
		panic(blas.ErrMLT0)
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if k < 0 {
		panic(blas.ErrKLT0)
	}
	if lda < 1 {
		panic(blas.ErrShortA)
	}
	if ldb < 1 {
		panic(blas.ErrShortB)
	}
	if ldc < 1 {
		panic(blas.ErrShortC)
	}

	h.e = Check(C.cublasCgemm(C.cublasHandle_t(h.h), C.cublasOperation_t(tA), C.cublasOperation_t(tB), C.int(m), C.int(n), C.int(k), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(a), C.int(lda), (*C.cuComplex)(b), C.int(ldb), (*C.cuComplex)(unsafe.Pointer(&beta)), (*C.cuComplex)(c), C.int(ldc)))
}

// Zgemm performs the matrix-matrix operation C = alpha*op(A)*op(B) + beta*C for double precision complex matrices.
// op(X) is either X, X^T, or X^H depending on the transpose parameter.
func (h *Handler) Zgemm(tA, tB Transpose, m, n, k int, alpha complex128, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta complex128, c cudart.DevicePtr, ldc int) {
	if h.e != nil {
		return
	}
	if m < 0 {
		panic(blas.ErrMLT0)
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if k < 0 {
		panic(blas.ErrKLT0)
	}
	if lda < 1 {
		panic(blas.ErrShortA)
	}
	if ldb < 1 {
		panic(blas.ErrShortB)
	}
	if ldc < 1 {
		panic(blas.ErrShortC)
	}

	h.e = Check(C.cublasZgemm(C.cublasHandle_t(h.h), C.cublasOperation_t(tA), C.cublasOperation_t(tB), C.int(m), C.int(n), C.int(k), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(a), C.int(lda), (*C.cuDoubleComplex)(b), C.int(ldb), (*C.cuDoubleComplex)(unsafe.Pointer(&beta)), (*C.cuDoubleComplex)(c), C.int(ldc)))
}

// Cgemm3m performs the matrix-matrix operation C = alpha*op(A)*op(B) + beta*C for single precision complex matrices
// using the 3M algorithm which reduces the number of real multiplications from 4 to 3 per complex multiplication.
func (h *Handler) Cgemm3m(tA, tB Transpose, m, n, k int, alpha complex64, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta complex64, c cudart.DevicePtr, ldc int) {
	if h.e != nil {
		return
	}
	if m < 0 {
		panic(blas.ErrMLT0)
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if k < 0 {
		panic(blas.ErrKLT0)
	}
	if lda < 1 {
		panic(blas.ErrShortA)
	}
	if ldb < 1 {
		panic(blas.ErrShortB)
	}
	if ldc < 1 {
		panic(blas.ErrShortC)
	}

	h.e = Check(C.cublasCgemm3m(C.cublasHandle_t(h.h), C.cublasOperation_t(tA), C.cublasOperation_t(tB), C.int(m), C.int(n), C.int(k), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(a), C.int(lda), (*C.cuComplex)(b), C.int(ldb), (*C.cuComplex)(unsafe.Pointer(&beta)), (*C.cuComplex)(c), C.int(ldc)))
}

// Zgemm3m performs the matrix-matrix operation C = alpha*op(A)*op(B) + beta*C for double precision complex matrices
// using the 3M algorithm which reduces the number of real multiplications from 4 to 3 per complex multiplication.
func (h *Handler) Zgemm3m(tA, tB Transpose, m, n, k int, alpha complex128, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta complex128, c cudart.DevicePtr, ldc int) {
	if h.e != nil {
		return
	}
	if m < 0 {
		panic(blas.ErrMLT0)
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if k < 0 {
		panic(blas.ErrKLT0)
	}
	if lda < 1 {
		panic(blas.ErrShortA)
	}
	if ldb < 1 {
		panic(blas.ErrShortB)
	}
	if ldc < 1 {
		panic(blas.ErrShortC)
	}

	h.e = Check(C.cublasZgemm3m(C.cublasHandle_t(h.h), C.cublasOperation_t(tA), C.cublasOperation_t(tB), C.int(m), C.int(n), C.int(k), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(a), C.int(lda), (*C.cuDoubleComplex)(b), C.int(ldb), (*C.cuDoubleComplex)(unsafe.Pointer(&beta)), (*C.cuDoubleComplex)(c), C.int(ldc)))
}

// Ssymm performs the symmetric matrix-matrix operation C = alpha*A*B + beta*C or C = alpha*B*A + beta*C for single precision real matrices.
// A is a symmetric matrix.
func (h *Handler) Ssymm(side Side, uplo Uplo, m, n int, alpha float32, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta float32, c cudart.DevicePtr, ldc int) {
	if h.e != nil {
		return
	}
	if m < 0 {
		panic(blas.ErrMLT0)
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if lda < 1 {
		panic(blas.ErrShortA)
	}
	if ldb < 1 {
		panic(blas.ErrShortB)
	}
	if ldc < 1 {
		panic(blas.ErrShortC)
	}

	h.e = Check(C.cublasSsymm(C.cublasHandle_t(h.h), C.cublasSideMode_t(side), C.cublasFillMode_t(uplo), C.int(m), C.int(n), (*C.float)(&alpha), (*C.float)(a), C.int(lda), (*C.float)(b), C.int(ldb), (*C.float)(&beta), (*C.float)(c), C.int(ldc)))
}

// Dsymm performs the symmetric matrix-matrix operation C = alpha*A*B + beta*C or C = alpha*B*A + beta*C for double precision real matrices.
// A is a symmetric matrix.
func (h *Handler) Dsymm(side Side, uplo Uplo, m, n int, alpha float64, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta float64, c cudart.DevicePtr, ldc int) {
	if h.e != nil {
		return
	}
	if m < 0 {
		panic(blas.ErrMLT0)
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if lda < 1 {
		panic(blas.ErrShortA)
	}
	if ldb < 1 {
		panic(blas.ErrShortB)
	}
	if ldc < 1 {
		panic(blas.ErrShortC)
	}

	h.e = Check(C.cublasDsymm(C.cublasHandle_t(h.h), C.cublasSideMode_t(side), C.cublasFillMode_t(uplo), C.int(m), C.int(n), (*C.double)(&alpha), (*C.double)(a), C.int(lda), (*C.double)(b), C.int(ldb), (*C.double)(&beta), (*C.double)(c), C.int(ldc)))
}

// Csymm performs the symmetric matrix-matrix operation C = alpha*A*B + beta*C or C = alpha*B*A + beta*C for single precision complex matrices.
// A is a symmetric matrix (not Hermitian).
func (h *Handler) Csymm(side Side, uplo Uplo, m, n int, alpha complex64, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta complex64, c cudart.DevicePtr, ldc int) {
	if h.e != nil {
		return
	}
	if m < 0 {
		panic(blas.ErrMLT0)
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if lda < 1 {
		panic(blas.ErrShortA)
	}
	if ldb < 1 {
		panic(blas.ErrShortB)
	}
	if ldc < 1 {
		panic(blas.ErrShortC)
	}

	h.e = Check(C.cublasCsymm(C.cublasHandle_t(h.h), C.cublasSideMode_t(side), C.cublasFillMode_t(uplo), C.int(m), C.int(n), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(a), C.int(lda), (*C.cuComplex)(b), C.int(ldb), (*C.cuComplex)(unsafe.Pointer(&beta)), (*C.cuComplex)(c), C.int(ldc)))
}

// Zsymm performs the symmetric matrix-matrix operation C = alpha*A*B + beta*C or C = alpha*B*A + beta*C for double precision complex matrices.
// A is a symmetric matrix (not Hermitian).
func (h *Handler) Zsymm(side Side, uplo Uplo, m, n int, alpha complex128, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta complex128, c cudart.DevicePtr, ldc int) {
	if h.e != nil {
		return
	}
	if m < 0 {
		panic(blas.ErrMLT0)
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if lda < 1 {
		panic(blas.ErrShortA)
	}
	if ldb < 1 {
		panic(blas.ErrShortB)
	}
	if ldc < 1 {
		panic(blas.ErrShortC)
	}

	h.e = Check(C.cublasZsymm(C.cublasHandle_t(h.h), C.cublasSideMode_t(side), C.cublasFillMode_t(uplo), C.int(m), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(a), C.int(lda), (*C.cuDoubleComplex)(b), C.int(ldb), (*C.cuDoubleComplex)(unsafe.Pointer(&beta)), (*C.cuDoubleComplex)(c), C.int(ldc)))
}

// Ssyrk performs the symmetric rank-k update C = alpha*A*A^T + beta*C or C = alpha*A^T*A + beta*C for single precision real matrices.
// C is a symmetric matrix.
func (h *Handler) Ssyrk(uplo Uplo, trans Transpose, n, k int, alpha float32, a cudart.DevicePtr, lda int, beta float32, c cudart.DevicePtr, ldc int) {
	if h.e != nil {
		return
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if k < 0 {
		panic(blas.ErrKLT0)
	}
	if lda < 1 {
		panic(blas.ErrShortA)
	}
	if ldc < 1 {
		panic(blas.ErrShortC)
	}

	h.e = Check(C.cublasSsyrk(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.int(n), C.int(k), (*C.float)(&alpha), (*C.float)(a), C.int(lda), (*C.float)(&beta), (*C.float)(c), C.int(ldc)))
}

// Dsyrk performs the symmetric rank-k update C = alpha*A*A^T + beta*C or C = alpha*A^T*A + beta*C for double precision real matrices.
// C is a symmetric matrix.
func (h *Handler) Dsyrk(uplo Uplo, trans Transpose, n, k int, alpha float64, a cudart.DevicePtr, lda int, beta float64, c cudart.DevicePtr, ldc int) {
	if h.e != nil {
		return
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if k < 0 {
		panic(blas.ErrKLT0)
	}
	if lda < 1 {
		panic(blas.ErrShortA)
	}
	if ldc < 1 {
		panic(blas.ErrShortC)
	}

	h.e = Check(C.cublasDsyrk(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.int(n), C.int(k), (*C.double)(&alpha), (*C.double)(a), C.int(lda), (*C.double)(&beta), (*C.double)(c), C.int(ldc)))
}

// Csyrk performs the symmetric rank-k update C = alpha*A*A^T + beta*C or C = alpha*A^T*A + beta*C for single precision complex matrices.
// C is a symmetric matrix (not Hermitian).
func (h *Handler) Csyrk(uplo Uplo, trans Transpose, n, k int, alpha complex64, a cudart.DevicePtr, lda int, beta complex64, c cudart.DevicePtr, ldc int) {
	if h.e != nil {
		return
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if k < 0 {
		panic(blas.ErrKLT0)
	}
	if lda < 1 {
		panic(blas.ErrShortA)
	}
	if ldc < 1 {
		panic(blas.ErrShortC)
	}

	h.e = Check(C.cublasCsyrk(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.int(n), C.int(k), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(a), C.int(lda), (*C.cuComplex)(unsafe.Pointer(&beta)), (*C.cuComplex)(c), C.int(ldc)))
}

// Zsyrk performs the symmetric rank-k update C = alpha*A*A^T + beta*C or C = alpha*A^T*A + beta*C for double precision complex matrices.
// C is a symmetric matrix (not Hermitian).
func (h *Handler) Zsyrk(uplo Uplo, trans Transpose, n, k int, alpha complex128, a cudart.DevicePtr, lda int, beta complex128, c cudart.DevicePtr, ldc int) {
	if h.e != nil {
		return
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if k < 0 {
		panic(blas.ErrKLT0)
	}
	if lda < 1 {
		panic(blas.ErrShortA)
	}
	if ldc < 1 {
		panic(blas.ErrShortC)
	}

	h.e = Check(C.cublasZsyrk(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.int(n), C.int(k), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(a), C.int(lda), (*C.cuDoubleComplex)(unsafe.Pointer(&beta)), (*C.cuDoubleComplex)(c), C.int(ldc)))
}

// Ssyr2k performs the symmetric rank-2k update C = alpha*(A*B^T + B*A^T) + beta*C or C = alpha*(A^T*B + B^T*A) + beta*C for single precision real matrices.
// C is a symmetric matrix.
func (h *Handler) Ssyr2k(uplo Uplo, trans Transpose, n, k int, alpha float32, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta float32, c cudart.DevicePtr, ldc int) {
	if h.e != nil {
		return
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if k < 0 {
		panic(blas.ErrKLT0)
	}
	if lda < 1 {
		panic(blas.ErrShortA)
	}
	if ldb < 1 {
		panic(blas.ErrShortB)
	}
	if ldc < 1 {
		panic(blas.ErrShortC)
	}

	h.e = Check(C.cublasSsyr2k(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.int(n), C.int(k), (*C.float)(&alpha), (*C.float)(a), C.int(lda), (*C.float)(b), C.int(ldb), (*C.float)(&beta), (*C.float)(c), C.int(ldc)))
}

// Dsyr2k performs the symmetric rank-2k update C = alpha*(A*B^T + B*A^T) + beta*C or C = alpha*(A^T*B + B^T*A) + beta*C for double precision real matrices.
// C is a symmetric matrix.
func (h *Handler) Dsyr2k(uplo Uplo, trans Transpose, n, k int, alpha float64, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta float64, c cudart.DevicePtr, ldc int) {
	if h.e != nil {
		return
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if k < 0 {
		panic(blas.ErrKLT0)
	}
	if lda < 1 {
		panic(blas.ErrShortA)
	}
	if ldb < 1 {
		panic(blas.ErrShortB)
	}
	if ldc < 1 {
		panic(blas.ErrShortC)
	}

	h.e = Check(C.cublasDsyr2k(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.int(n), C.int(k), (*C.double)(&alpha), (*C.double)(a), C.int(lda), (*C.double)(b), C.int(ldb), (*C.double)(&beta), (*C.double)(c), C.int(ldc)))
}

// Csyr2k performs the symmetric rank-2k update C = alpha*(A*B^T + B*A^T) + beta*C or C = alpha*(A^T*B + B^T*A) + beta*C for single precision complex matrices.
// C is a symmetric matrix (not Hermitian).
func (h *Handler) Csyr2k(uplo Uplo, trans Transpose, n, k int, alpha complex64, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta complex64, c cudart.DevicePtr, ldc int) {
	if h.e != nil {
		return
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if k < 0 {
		panic(blas.ErrKLT0)
	}
	if lda < 1 {
		panic(blas.ErrShortA)
	}
	if ldb < 1 {
		panic(blas.ErrShortB)
	}
	if ldc < 1 {
		panic(blas.ErrShortC)
	}

	h.e = Check(C.cublasCsyr2k(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.int(n), C.int(k), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(a), C.int(lda), (*C.cuComplex)(b), C.int(ldb), (*C.cuComplex)(unsafe.Pointer(&beta)), (*C.cuComplex)(c), C.int(ldc)))
}

// Zsyr2k performs the symmetric rank-2k update C = alpha*(A*B^T + B*A^T) + beta*C or C = alpha*(A^T*B + B^T*A) + beta*C for double precision complex matrices.
// C is a symmetric matrix (not Hermitian).
func (h *Handler) Zsyr2k(uplo Uplo, trans Transpose, n, k int, alpha complex128, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta complex128, c cudart.DevicePtr, ldc int) {
	if h.e != nil {
		return
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if k < 0 {
		panic(blas.ErrKLT0)
	}
	if lda < 1 {
		panic(blas.ErrShortA)
	}
	if ldb < 1 {
		panic(blas.ErrShortB)
	}
	if ldc < 1 {
		panic(blas.ErrShortC)
	}

	h.e = Check(C.cublasZsyr2k(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.int(n), C.int(k), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(a), C.int(lda), (*C.cuDoubleComplex)(b), C.int(ldb), (*C.cuDoubleComplex)(unsafe.Pointer(&beta)), (*C.cuDoubleComplex)(c), C.int(ldc)))
}

// Ssyrkx performs the symmetric rank-kx update C = alpha*A*B^T + beta*C or C = alpha*A^T*B + beta*C for single precision real matrices.
// This is a generalization of SYRK where A and B can be different matrices.
func (h *Handler) Ssyrkx(uplo Uplo, trans Transpose, n, k int, alpha float32, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta float32, c cudart.DevicePtr, ldc int) {
	if h.e != nil {
		return
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if k < 0 {
		panic(blas.ErrKLT0)
	}
	if lda < 1 {
		panic(blas.ErrShortA)
	}
	if ldb < 1 {
		panic(blas.ErrShortB)
	}
	if ldc < 1 {
		panic(blas.ErrShortC)
	}

	h.e = Check(C.cublasSsyrkx(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.int(n), C.int(k), (*C.float)(&alpha), (*C.float)(a), C.int(lda), (*C.float)(b), C.int(ldb), (*C.float)(&beta), (*C.float)(c), C.int(ldc)))
}

// Dsyrkx performs the symmetric rank-kx update C = alpha*A*B^T + beta*C or C = alpha*A^T*B + beta*C for double precision real matrices.
// This is a generalization of SYRK where A and B can be different matrices.
func (h *Handler) Dsyrkx(uplo Uplo, trans Transpose, n, k int, alpha float64, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta float64, c cudart.DevicePtr, ldc int) {
	if h.e != nil {
		return
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if k < 0 {
		panic(blas.ErrKLT0)
	}
	if lda < 1 {
		panic(blas.ErrShortA)
	}
	if ldb < 1 {
		panic(blas.ErrShortB)
	}
	if ldc < 1 {
		panic(blas.ErrShortC)
	}

	h.e = Check(C.cublasDsyrkx(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.int(n), C.int(k), (*C.double)(&alpha), (*C.double)(a), C.int(lda), (*C.double)(b), C.int(ldb), (*C.double)(&beta), (*C.double)(c), C.int(ldc)))
}

// Csyrkx performs the symmetric rank-kx update C = alpha*A*B^T + beta*C or C = alpha*A^T*B + beta*C for single precision complex matrices.
// This is a generalization of SYRK where A and B can be different matrices. C is a symmetric matrix (not Hermitian).
func (h *Handler) Csyrkx(uplo Uplo, trans Transpose, n, k int, alpha complex64, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta complex64, c cudart.DevicePtr, ldc int) {
	if h.e != nil {
		return
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if k < 0 {
		panic(blas.ErrKLT0)
	}
	if lda < 1 {
		panic(blas.ErrShortA)
	}
	if ldb < 1 {
		panic(blas.ErrShortB)
	}
	if ldc < 1 {
		panic(blas.ErrShortC)
	}

	h.e = Check(C.cublasCsyrkx(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.int(n), C.int(k), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(a), C.int(lda), (*C.cuComplex)(b), C.int(ldb), (*C.cuComplex)(unsafe.Pointer(&beta)), (*C.cuComplex)(c), C.int(ldc)))
}

// Zsyrkx performs the symmetric rank-kx update C = alpha*A*B^T + beta*C or C = alpha*A^T*B + beta*C for double precision complex matrices.
// This is a generalization of SYRK where A and B can be different matrices. C is a symmetric matrix (not Hermitian).
func (h *Handler) Zsyrkx(uplo Uplo, trans Transpose, n, k int, alpha complex128, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta complex128, c cudart.DevicePtr, ldc int) {
	if h.e != nil {
		return
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if k < 0 {
		panic(blas.ErrKLT0)
	}
	if lda < 1 {
		panic(blas.ErrShortA)
	}
	if ldb < 1 {
		panic(blas.ErrShortB)
	}
	if ldc < 1 {
		panic(blas.ErrShortC)
	}

	h.e = Check(C.cublasZsyrkx(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.int(n), C.int(k), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(a), C.int(lda), (*C.cuDoubleComplex)(b), C.int(ldb), (*C.cuDoubleComplex)(unsafe.Pointer(&beta)), (*C.cuDoubleComplex)(c), C.int(ldc)))
}

// Strmm performs the triangular matrix-matrix operation C = alpha*op(A)*B or C = alpha*B*op(A) for single precision real matrices.
// A is a triangular matrix.
func (h *Handler) Strmm(side Side, uplo Uplo, trans Transpose, diag Diag, m, n int, alpha float32, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, c cudart.DevicePtr, ldc int) {
	if h.e != nil {
		return
	}
	if m < 0 {
		panic(blas.ErrMLT0)
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if lda < 1 {
		panic(blas.ErrShortA)
	}
	if ldb < 1 {
		panic(blas.ErrShortB)
	}
	if ldc < 1 {
		panic(blas.ErrShortC)
	}

	h.e = Check(C.cublasStrmm(C.cublasHandle_t(h.h), C.cublasSideMode_t(side), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.cublasDiagType_t(diag), C.int(m), C.int(n), (*C.float)(&alpha), (*C.float)(a), C.int(lda), (*C.float)(b), C.int(ldb), (*C.float)(c), C.int(ldc)))
}

// Dtrmm performs the triangular matrix-matrix operation C = alpha*op(A)*B or C = alpha*B*op(A) for double precision real matrices.
// A is a triangular matrix.
func (h *Handler) Dtrmm(side Side, uplo Uplo, trans Transpose, diag Diag, m, n int, alpha float64, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, c cudart.DevicePtr, ldc int) {
	if h.e != nil {
		return
	}
	if m < 0 {
		panic(blas.ErrMLT0)
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if lda < 1 {
		panic(blas.ErrShortA)
	}
	if ldb < 1 {
		panic(blas.ErrShortB)
	}
	if ldc < 1 {
		panic(blas.ErrShortC)
	}

	h.e = Check(C.cublasDtrmm(C.cublasHandle_t(h.h), C.cublasSideMode_t(side), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.cublasDiagType_t(diag), C.int(m), C.int(n), (*C.double)(&alpha), (*C.double)(a), C.int(lda), (*C.double)(b), C.int(ldb), (*C.double)(c), C.int(ldc)))
}

// Ctrmm performs the triangular matrix-matrix operation C = alpha*op(A)*B or C = alpha*B*op(A) for single precision complex matrices.
// A is a triangular matrix.
func (h *Handler) Ctrmm(side Side, uplo Uplo, trans Transpose, diag Diag, m, n int, alpha complex64, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, c cudart.DevicePtr, ldc int) {
	if h.e != nil {
		return
	}
	if m < 0 {
		panic(blas.ErrMLT0)
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if lda < 1 {
		panic(blas.ErrShortA)
	}
	if ldb < 1 {
		panic(blas.ErrShortB)
	}
	if ldc < 1 {
		panic(blas.ErrShortC)
	}

	h.e = Check(C.cublasCtrmm(C.cublasHandle_t(h.h), C.cublasSideMode_t(side), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.cublasDiagType_t(diag), C.int(m), C.int(n), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(a), C.int(lda), (*C.cuComplex)(b), C.int(ldb), (*C.cuComplex)(c), C.int(ldc)))
}

// Ztrmm performs the triangular matrix-matrix operation C = alpha*op(A)*B or C = alpha*B*op(A) for double precision complex matrices.
// A is a triangular matrix.
func (h *Handler) Ztrmm(side Side, uplo Uplo, trans Transpose, diag Diag, m, n int, alpha complex128, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, c cudart.DevicePtr, ldc int) {
	if h.e != nil {
		return
	}
	if m < 0 {
		panic(blas.ErrMLT0)
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if lda < 1 {
		panic(blas.ErrShortA)
	}
	if ldb < 1 {
		panic(blas.ErrShortB)
	}
	if ldc < 1 {
		panic(blas.ErrShortC)
	}

	h.e = Check(C.cublasZtrmm(C.cublasHandle_t(h.h), C.cublasSideMode_t(side), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.cublasDiagType_t(diag), C.int(m), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(a), C.int(lda), (*C.cuDoubleComplex)(b), C.int(ldb), (*C.cuDoubleComplex)(c), C.int(ldc)))
}

// Strsm solves the triangular matrix equation op(A)*X = alpha*B or X*op(A) = alpha*B for single precision real matrices.
// A is a triangular matrix, and the solution X overwrites B.
func (h *Handler) Strsm(side Side, uplo Uplo, trans Transpose, diag Diag, m, n int, alpha float32, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int) {
	if h.e != nil {
		return
	}
	if m < 0 {
		panic(blas.ErrMLT0)
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if lda < 1 {
		panic(blas.ErrShortA)
	}
	if ldb < 1 {
		panic(blas.ErrShortB)
	}

	h.e = Check(C.cublasStrsm(C.cublasHandle_t(h.h), C.cublasSideMode_t(side), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.cublasDiagType_t(diag), C.int(m), C.int(n), (*C.float)(&alpha), (*C.float)(a), C.int(lda), (*C.float)(b), C.int(ldb)))
}

// Dtrsm solves the triangular matrix equation op(A)*X = alpha*B or X*op(A) = alpha*B for double precision real matrices.
// A is a triangular matrix, and the solution X overwrites B.
func (h *Handler) Dtrsm(side Side, uplo Uplo, trans Transpose, diag Diag, m, n int, alpha float64, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int) {
	if h.e != nil {
		return
	}
	if m < 0 {
		panic(blas.ErrMLT0)
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if lda < 1 {
		panic(blas.ErrShortA)
	}
	if ldb < 1 {
		panic(blas.ErrShortB)
	}

	h.e = Check(C.cublasDtrsm(C.cublasHandle_t(h.h), C.cublasSideMode_t(side), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.cublasDiagType_t(diag), C.int(m), C.int(n), (*C.double)(&alpha), (*C.double)(a), C.int(lda), (*C.double)(b), C.int(ldb)))
}

// Ctrsm solves the triangular matrix equation op(A)*X = alpha*B or X*op(A) = alpha*B for single precision complex matrices.
// A is a triangular matrix, and the solution X overwrites B.
func (h *Handler) Ctrsm(side Side, uplo Uplo, trans Transpose, diag Diag, m, n int, alpha complex64, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int) {
	if h.e != nil {
		return
	}
	if m < 0 {
		panic(blas.ErrMLT0)
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if lda < 1 {
		panic(blas.ErrShortA)
	}
	if ldb < 1 {
		panic(blas.ErrShortB)
	}

	h.e = Check(C.cublasCtrsm(C.cublasHandle_t(h.h), C.cublasSideMode_t(side), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.cublasDiagType_t(diag), C.int(m), C.int(n), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(a), C.int(lda), (*C.cuComplex)(b), C.int(ldb)))
}

// Ztrsm solves the triangular matrix equation op(A)*X = alpha*B or X*op(A) = alpha*B for double precision complex matrices.
// A is a triangular matrix, and the solution X overwrites B.
func (h *Handler) Ztrsm(side Side, uplo Uplo, trans Transpose, diag Diag, m, n int, alpha complex128, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int) {
	if h.e != nil {
		return
	}
	if m < 0 {
		panic(blas.ErrMLT0)
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if lda < 1 {
		panic(blas.ErrShortA)
	}
	if ldb < 1 {
		panic(blas.ErrShortB)
	}

	h.e = Check(C.cublasZtrsm(C.cublasHandle_t(h.h), C.cublasSideMode_t(side), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.cublasDiagType_t(diag), C.int(m), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(a), C.int(lda), (*C.cuDoubleComplex)(b), C.int(ldb)))
}

// Chemm performs the Hermitian matrix-matrix operation C = alpha*A*B + beta*C or C = alpha*B*A + beta*C for single precision complex matrices.
// A is a Hermitian matrix.
func (h *Handler) Chemm(side Side, uplo Uplo, m, n int, alpha complex64, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta complex64, c cudart.DevicePtr, ldc int) {
	if h.e != nil {
		return
	}
	if m < 0 {
		panic(blas.ErrMLT0)
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if lda < 1 {
		panic(blas.ErrShortA)
	}
	if ldb < 1 {
		panic(blas.ErrShortB)
	}
	if ldc < 1 {
		panic(blas.ErrShortC)
	}

	h.e = Check(C.cublasChemm(C.cublasHandle_t(h.h), C.cublasSideMode_t(side), C.cublasFillMode_t(uplo), C.int(m), C.int(n), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(a), C.int(lda), (*C.cuComplex)(b), C.int(ldb), (*C.cuComplex)(unsafe.Pointer(&beta)), (*C.cuComplex)(c), C.int(ldc)))
}

// Zhemm performs the Hermitian matrix-matrix operation C = alpha*A*B + beta*C or C = alpha*B*A + beta*C for double precision complex matrices.
// A is a Hermitian matrix.
func (h *Handler) Zhemm(side Side, uplo Uplo, m, n int, alpha complex128, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta complex128, c cudart.DevicePtr, ldc int) {
	if h.e != nil {
		return
	}
	if m < 0 {
		panic(blas.ErrMLT0)
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if lda < 1 {
		panic(blas.ErrShortA)
	}
	if ldb < 1 {
		panic(blas.ErrShortB)
	}
	if ldc < 1 {
		panic(blas.ErrShortC)
	}

	h.e = Check(C.cublasZhemm(C.cublasHandle_t(h.h), C.cublasSideMode_t(side), C.cublasFillMode_t(uplo), C.int(m), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(a), C.int(lda), (*C.cuDoubleComplex)(b), C.int(ldb), (*C.cuDoubleComplex)(unsafe.Pointer(&beta)), (*C.cuDoubleComplex)(c), C.int(ldc)))
}

// Cherk performs the Hermitian rank-k update C = alpha*A*A^H + beta*C or C = alpha*A^H*A + beta*C for single precision complex matrices.
// C is a Hermitian matrix. Note that alpha and beta are real scalars.
func (h *Handler) Cherk(uplo Uplo, trans Transpose, n, k int, alpha float32, a cudart.DevicePtr, lda int, beta float32, c cudart.DevicePtr, ldc int) {
	if h.e != nil {
		return
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if k < 0 {
		panic(blas.ErrKLT0)
	}
	if lda < 1 {
		panic(blas.ErrShortA)
	}
	if ldc < 1 {
		panic(blas.ErrShortC)
	}

	h.e = Check(C.cublasCherk(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.int(n), C.int(k), (*C.float)(&alpha), (*C.cuComplex)(a), C.int(lda), (*C.float)(&beta), (*C.cuComplex)(c), C.int(ldc)))
}

// Zherk performs the Hermitian rank-k update C = alpha*A*A^H + beta*C or C = alpha*A^H*A + beta*C for double precision complex matrices.
// C is a Hermitian matrix. Note that alpha and beta are real scalars.
func (h *Handler) Zherk(uplo Uplo, trans Transpose, n, k int, alpha float64, a cudart.DevicePtr, lda int, beta float64, c cudart.DevicePtr, ldc int) {
	if h.e != nil {
		return
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if k < 0 {
		panic(blas.ErrKLT0)
	}
	if lda < 1 {
		panic(blas.ErrShortA)
	}
	if ldc < 1 {
		panic(blas.ErrShortC)
	}

	h.e = Check(C.cublasZherk(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.int(n), C.int(k), (*C.double)(&alpha), (*C.cuDoubleComplex)(a), C.int(lda), (*C.double)(&beta), (*C.cuDoubleComplex)(c), C.int(ldc)))
}

// Cher2k performs the Hermitian rank-2k update C = alpha*(A*B^H + B*A^H) + beta*C or C = alpha*(A^H*B + B^H*A) + beta*C for single precision complex matrices.
// C is a Hermitian matrix. Note that beta is a real scalar.
func (h *Handler) Cher2k(uplo Uplo, trans Transpose, n, k int, alpha complex64, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta float32, c cudart.DevicePtr, ldc int) {
	if h.e != nil {
		return
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if k < 0 {
		panic(blas.ErrKLT0)
	}
	if lda < 1 {
		panic(blas.ErrShortA)
	}
	if ldb < 1 {
		panic(blas.ErrShortB)
	}
	if ldc < 1 {
		panic(blas.ErrShortC)
	}

	h.e = Check(C.cublasCher2k(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.int(n), C.int(k), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(a), C.int(lda), (*C.cuComplex)(b), C.int(ldb), (*C.float)(&beta), (*C.cuComplex)(c), C.int(ldc)))
}

// Zher2k performs the Hermitian rank-2k update C = alpha*(A*B^H + B*A^H) + beta*C or C = alpha*(A^H*B + B^H*A) + beta*C for double precision complex matrices.
// C is a Hermitian matrix. Note that beta is a real scalar.
func (h *Handler) Zher2k(uplo Uplo, trans Transpose, n, k int, alpha complex128, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta float64, c cudart.DevicePtr, ldc int) {
	if h.e != nil {
		return
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if k < 0 {
		panic(blas.ErrKLT0)
	}
	if lda < 1 {
		panic(blas.ErrShortA)
	}
	if ldb < 1 {
		panic(blas.ErrShortB)
	}
	if ldc < 1 {
		panic(blas.ErrShortC)
	}

	h.e = Check(C.cublasZher2k(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.int(n), C.int(k), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(a), C.int(lda), (*C.cuDoubleComplex)(b), C.int(ldb), (*C.double)(&beta), (*C.cuDoubleComplex)(c), C.int(ldc)))
}

// Cherkx performs the Hermitian rank-kx update C = alpha*A*B^H + beta*C or C = alpha*A^H*B + beta*C for single precision complex matrices.
// This is a generalization of HERK where A and B can be different matrices. Note that beta is a real scalar.
func (h *Handler) Cherkx(uplo Uplo, trans Transpose, n, k int, alpha complex64, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta float32, c cudart.DevicePtr, ldc int) {
	if h.e != nil {
		return
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if k < 0 {
		panic(blas.ErrKLT0)
	}
	if lda < 1 {
		panic(blas.ErrShortA)
	}
	if ldb < 1 {
		panic(blas.ErrShortB)
	}
	if ldc < 1 {
		panic(blas.ErrShortC)
	}

	h.e = Check(C.cublasCherkx(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.int(n), C.int(k), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(a), C.int(lda), (*C.cuComplex)(b), C.int(ldb), (*C.float)(&beta), (*C.cuComplex)(c), C.int(ldc)))
}

// Zherkx performs the Hermitian rank-kx update C = alpha*A*B^H + beta*C or C = alpha*A^H*B + beta*C for double precision complex matrices.
// This is a generalization of HERK where A and B can be different matrices. Note that beta is a real scalar.
func (h *Handler) Zherkx(uplo Uplo, trans Transpose, n, k int, alpha complex128, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta float64, c cudart.DevicePtr, ldc int) {
	if h.e != nil {
		return
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if k < 0 {
		panic(blas.ErrKLT0)
	}
	if lda < 1 {
		panic(blas.ErrShortA)
	}
	if ldb < 1 {
		panic(blas.ErrShortB)
	}
	if ldc < 1 {
		panic(blas.ErrShortC)
	}

	h.e = Check(C.cublasZherkx(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.int(n), C.int(k), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(a), C.int(lda), (*C.cuDoubleComplex)(b), C.int(ldb), (*C.double)(&beta), (*C.cuDoubleComplex)(c), C.int(ldc)))
}
