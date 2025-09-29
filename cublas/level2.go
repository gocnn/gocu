package cublas

// #include <cublas_v2.h>
import "C"
import (
	"unsafe"

	"github.com/gocnn/gocu/cudart"
	"github.com/gocnn/gomat/blas"
)

// Sgbmv performs the matrix-vector operation y = alpha*A*x + beta*y for single precision real band matrix.
// A is an m-by-n band matrix with kl sub-diagonals and ku super-diagonals.
// The matrix A is stored in band storage format.
func Sgbmv(h *Handle, trans Transpose, m, n, kl, ku int, alpha float32, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta float32, y cudart.DevicePtr, incY int) error {

	if m < 0 {
		panic(blas.ErrMLT0)
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if kl < 0 {
		panic(blas.ErrKLT0)
	}
	if ku < 0 {
		panic(blas.ErrKLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}
	if lda < kl+ku+1 {
		panic(blas.ErrShortA)
	}

	return Check(C.cublasSgbmv(C.cublasHandle_t(h.h), C.cublasOperation_t(trans), C.int(m), C.int(n), C.int(kl), C.int(ku), (*C.float)(&alpha), (*C.float)(A), C.int(lda), (*C.float)(x), C.int(incX), (*C.float)(&beta), (*C.float)(y), C.int(incY)))
}

// Dgbmv performs the matrix-vector operation y = alpha*A*x + beta*y for double precision real band matrix.
// A is an m-by-n band matrix with kl sub-diagonals and ku super-diagonals.
// The matrix A is stored in band storage format.
func Dgbmv(h *Handle, trans Transpose, m, n, kl, ku int, alpha float64, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta float64, y cudart.DevicePtr, incY int) error {

	if m < 0 {
		panic(blas.ErrMLT0)
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if kl < 0 {
		panic(blas.ErrKLT0)
	}
	if ku < 0 {
		panic(blas.ErrKLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}
	if lda < kl+ku+1 {
		panic(blas.ErrShortA)
	}

	return Check(C.cublasDgbmv(C.cublasHandle_t(h.h), C.cublasOperation_t(trans), C.int(m), C.int(n), C.int(kl), C.int(ku), (*C.double)(&alpha), (*C.double)(A), C.int(lda), (*C.double)(x), C.int(incX), (*C.double)(&beta), (*C.double)(y), C.int(incY)))
}

// Cgbmv performs the matrix-vector operation y = alpha*A*x + beta*y for single precision complex band matrix.
// A is an m-by-n band matrix with kl sub-diagonals and ku super-diagonals.
// The matrix A is stored in band storage format.
func Cgbmv(h *Handle, trans Transpose, m, n, kl, ku int, alpha complex64, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta complex64, y cudart.DevicePtr, incY int) error {

	if m < 0 {
		panic(blas.ErrMLT0)
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if kl < 0 {
		panic(blas.ErrKLT0)
	}
	if ku < 0 {
		panic(blas.ErrKLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}
	if lda < kl+ku+1 {
		panic(blas.ErrShortA)
	}

	return Check(C.cublasCgbmv(C.cublasHandle_t(h.h), C.cublasOperation_t(trans), C.int(m), C.int(n), C.int(kl), C.int(ku), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(A), C.int(lda), (*C.cuComplex)(x), C.int(incX), (*C.cuComplex)(unsafe.Pointer(&beta)), (*C.cuComplex)(y), C.int(incY)))
}

// Zgbmv performs the matrix-vector operation y = alpha*A*x + beta*y for double precision complex band matrix.
// A is an m-by-n band matrix with kl sub-diagonals and ku super-diagonals.
// The matrix A is stored in band storage format.
func Zgbmv(h *Handle, trans Transpose, m, n, kl, ku int, alpha complex128, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta complex128, y cudart.DevicePtr, incY int) error {

	if m < 0 {
		panic(blas.ErrMLT0)
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if kl < 0 {
		panic(blas.ErrKLT0)
	}
	if ku < 0 {
		panic(blas.ErrKLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}
	if lda < kl+ku+1 {
		panic(blas.ErrShortA)
	}

	return Check(C.cublasZgbmv(C.cublasHandle_t(h.h), C.cublasOperation_t(trans), C.int(m), C.int(n), C.int(kl), C.int(ku), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(A), C.int(lda), (*C.cuDoubleComplex)(x), C.int(incX), (*C.cuDoubleComplex)(unsafe.Pointer(&beta)), (*C.cuDoubleComplex)(y), C.int(incY)))
}

// Sgemv performs the matrix-vector operation y = alpha*A*x + beta*y for single precision real general matrix.
// A is an m-by-n general matrix.
func Sgemv(h *Handle, trans Transpose, m, n int, alpha float32, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta float32, y cudart.DevicePtr, incY int) error {

	if m < 0 {
		panic(blas.ErrMLT0)
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}
	if lda < m {
		panic(blas.ErrShortA)
	}

	return Check(C.cublasSgemv(C.cublasHandle_t(h.h), C.cublasOperation_t(trans), C.int(m), C.int(n), (*C.float)(&alpha), (*C.float)(A), C.int(lda), (*C.float)(x), C.int(incX), (*C.float)(&beta), (*C.float)(y), C.int(incY)))
}

// Dgemv performs the matrix-vector operation y = alpha*A*x + beta*y for double precision real general matrix.
// A is an m-by-n general matrix.
func Dgemv(h *Handle, trans Transpose, m, n int, alpha float64, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta float64, y cudart.DevicePtr, incY int) error {

	if m < 0 {
		panic(blas.ErrMLT0)
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}
	if lda < m {
		panic(blas.ErrShortA)
	}

	return Check(C.cublasDgemv(C.cublasHandle_t(h.h), C.cublasOperation_t(trans), C.int(m), C.int(n), (*C.double)(&alpha), (*C.double)(A), C.int(lda), (*C.double)(x), C.int(incX), (*C.double)(&beta), (*C.double)(y), C.int(incY)))
}

// Cgemv performs the matrix-vector operation y = alpha*A*x + beta*y for single precision complex general matrix.
// A is an m-by-n general matrix.
func Cgemv(h *Handle, trans Transpose, m, n int, alpha complex64, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta complex64, y cudart.DevicePtr, incY int) error {

	if m < 0 {
		panic(blas.ErrMLT0)
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}
	if lda < m {
		panic(blas.ErrShortA)
	}

	return Check(C.cublasCgemv(C.cublasHandle_t(h.h), C.cublasOperation_t(trans), C.int(m), C.int(n), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(A), C.int(lda), (*C.cuComplex)(x), C.int(incX), (*C.cuComplex)(unsafe.Pointer(&beta)), (*C.cuComplex)(y), C.int(incY)))
}

// Zgemv performs the matrix-vector operation y = alpha*A*x + beta*y for double precision complex general matrix.
// A is an m-by-n general matrix.
func Zgemv(h *Handle, trans Transpose, m, n int, alpha complex128, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta complex128, y cudart.DevicePtr, incY int) error {

	if m < 0 {
		panic(blas.ErrMLT0)
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}
	if lda < m {
		panic(blas.ErrShortA)
	}

	return Check(C.cublasZgemv(C.cublasHandle_t(h.h), C.cublasOperation_t(trans), C.int(m), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(A), C.int(lda), (*C.cuDoubleComplex)(x), C.int(incX), (*C.cuDoubleComplex)(unsafe.Pointer(&beta)), (*C.cuDoubleComplex)(y), C.int(incY)))
}

// Sger performs the rank-1 update A = alpha*x*y^T + A for single precision real matrix.
// A is an m-by-n general matrix.
func Sger(h *Handle, m, n int, alpha float32, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, A cudart.DevicePtr, lda int) error {

	if m < 0 {
		panic(blas.ErrMLT0)
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}
	if lda < m {
		panic(blas.ErrShortA)
	}

	return Check(C.cublasSger(C.cublasHandle_t(h.h), C.int(m), C.int(n), (*C.float)(&alpha), (*C.float)(x), C.int(incX), (*C.float)(y), C.int(incY), (*C.float)(A), C.int(lda)))
}

// Dger performs the rank-1 update A = alpha*x*y^T + A for double precision real matrix.
// A is an m-by-n general matrix.
func Dger(h *Handle, m, n int, alpha float64, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, A cudart.DevicePtr, lda int) error {

	if m < 0 {
		panic(blas.ErrMLT0)
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}
	if lda < m {
		panic(blas.ErrShortA)
	}

	return Check(C.cublasDger(C.cublasHandle_t(h.h), C.int(m), C.int(n), (*C.double)(&alpha), (*C.double)(x), C.int(incX), (*C.double)(y), C.int(incY), (*C.double)(A), C.int(lda)))
}

// Cgeru performs the rank-1 update A = alpha*x*y^T + A for single precision complex matrix (unconjugated).
// A is an m-by-n general matrix.
func Cgeru(h *Handle, m, n int, alpha complex64, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, A cudart.DevicePtr, lda int) error {

	if m < 0 {
		panic(blas.ErrMLT0)
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}
	if lda < m {
		panic(blas.ErrShortA)
	}

	return Check(C.cublasCgeru(C.cublasHandle_t(h.h), C.int(m), C.int(n), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(x), C.int(incX), (*C.cuComplex)(y), C.int(incY), (*C.cuComplex)(A), C.int(lda)))
}

// Cgerc performs the rank-1 update A = alpha*x*y^H + A for single precision complex matrix (conjugated).
// A is an m-by-n general matrix.
func Cgerc(h *Handle, m, n int, alpha complex64, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, A cudart.DevicePtr, lda int) error {

	if m < 0 {
		panic(blas.ErrMLT0)
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}
	if lda < m {
		panic(blas.ErrShortA)
	}

	return Check(C.cublasCgerc(C.cublasHandle_t(h.h), C.int(m), C.int(n), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(x), C.int(incX), (*C.cuComplex)(y), C.int(incY), (*C.cuComplex)(A), C.int(lda)))
}

// Zgeru performs the rank-1 update A = alpha*x*y^T + A for double precision complex matrix (unconjugated).
// A is an m-by-n general matrix.
func Zgeru(h *Handle, m, n int, alpha complex128, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, A cudart.DevicePtr, lda int) error {

	if m < 0 {
		panic(blas.ErrMLT0)
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}
	if lda < m {
		panic(blas.ErrShortA)
	}

	return Check(C.cublasZgeru(C.cublasHandle_t(h.h), C.int(m), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(x), C.int(incX), (*C.cuDoubleComplex)(y), C.int(incY), (*C.cuDoubleComplex)(A), C.int(lda)))
}

// Zgerc performs the rank-1 update A = alpha*x*y^H + A for double precision complex matrix (conjugated).
// A is an m-by-n general matrix.
func Zgerc(h *Handle, m, n int, alpha complex128, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, A cudart.DevicePtr, lda int) error {

	if m < 0 {
		panic(blas.ErrMLT0)
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}
	if lda < m {
		panic(blas.ErrShortA)
	}

	return Check(C.cublasZgerc(C.cublasHandle_t(h.h), C.int(m), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(x), C.int(incX), (*C.cuDoubleComplex)(y), C.int(incY), (*C.cuDoubleComplex)(A), C.int(lda)))
}

// Ssbmv performs the matrix-vector operation y = alpha*A*x + beta*y for single precision real symmetric band matrix.
// A is an n-by-n symmetric band matrix with k super-diagonals.
func Ssbmv(h *Handle, uplo Uplo, n, k int, alpha float32, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta float32, y cudart.DevicePtr, incY int) error {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if k < 0 {
		panic(blas.ErrKLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}
	if lda < k+1 {
		panic(blas.ErrShortA)
	}

	return Check(C.cublasSsbmv(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.int(n), C.int(k), (*C.float)(&alpha), (*C.float)(A), C.int(lda), (*C.float)(x), C.int(incX), (*C.float)(&beta), (*C.float)(y), C.int(incY)))
}

// Dsbmv performs the matrix-vector operation y = alpha*A*x + beta*y for double precision real symmetric band matrix.
// A is an n-by-n symmetric band matrix with k super-diagonals.
func Dsbmv(h *Handle, uplo Uplo, n, k int, alpha float64, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta float64, y cudart.DevicePtr, incY int) error {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if k < 0 {
		panic(blas.ErrKLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}
	if lda < k+1 {
		panic(blas.ErrShortA)
	}

	return Check(C.cublasDsbmv(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.int(n), C.int(k), (*C.double)(&alpha), (*C.double)(A), C.int(lda), (*C.double)(x), C.int(incX), (*C.double)(&beta), (*C.double)(y), C.int(incY)))
}

// Sspmv performs the matrix-vector operation y = alpha*A*x + beta*y for single precision real symmetric packed matrix.
// A is an n-by-n symmetric matrix stored in packed format.
func Sspmv(h *Handle, uplo Uplo, n int, alpha float32, AP cudart.DevicePtr, x cudart.DevicePtr, incX int, beta float32, y cudart.DevicePtr, incY int) error {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}

	return Check(C.cublasSspmv(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.int(n), (*C.float)(&alpha), (*C.float)(AP), (*C.float)(x), C.int(incX), (*C.float)(&beta), (*C.float)(y), C.int(incY)))
}

// Dspmv performs the matrix-vector operation y = alpha*A*x + beta*y for double precision real symmetric packed matrix.
// A is an n-by-n symmetric matrix stored in packed format.
func Dspmv(h *Handle, uplo Uplo, n int, alpha float64, AP cudart.DevicePtr, x cudart.DevicePtr, incX int, beta float64, y cudart.DevicePtr, incY int) error {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}

	return Check(C.cublasDspmv(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.int(n), (*C.double)(&alpha), (*C.double)(AP), (*C.double)(x), C.int(incX), (*C.double)(&beta), (*C.double)(y), C.int(incY)))
}

// Sspr performs the symmetric rank-1 update AP = alpha*x*x^T + AP for single precision real symmetric packed matrix.
// AP is an n-by-n symmetric matrix stored in packed format.
func Sspr(h *Handle, uplo Uplo, n int, alpha float32, x cudart.DevicePtr, incX int, AP cudart.DevicePtr) error {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	return Check(C.cublasSspr(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.int(n), (*C.float)(&alpha), (*C.float)(x), C.int(incX), (*C.float)(AP)))
}

// Dspr performs the symmetric rank-1 update AP = alpha*x*x^T + AP for double precision real symmetric packed matrix.
// AP is an n-by-n symmetric matrix stored in packed format.
func Dspr(h *Handle, uplo Uplo, n int, alpha float64, x cudart.DevicePtr, incX int, AP cudart.DevicePtr) error {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	return Check(C.cublasDspr(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.int(n), (*C.double)(&alpha), (*C.double)(x), C.int(incX), (*C.double)(AP)))
}

// Sspr2 performs the symmetric rank-2 update AP = alpha*x*y^T + alpha*y*x^T + AP for single precision real symmetric packed matrix.
// AP is an n-by-n symmetric matrix stored in packed format.
func Sspr2(h *Handle, uplo Uplo, n int, alpha float32, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, AP cudart.DevicePtr) error {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}

	return Check(C.cublasSspr2(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.int(n), (*C.float)(&alpha), (*C.float)(x), C.int(incX), (*C.float)(y), C.int(incY), (*C.float)(AP)))
}

// Dspr2 performs the symmetric rank-2 update AP = alpha*x*y^T + alpha*y*x^T + AP for double precision real symmetric packed matrix.
// AP is an n-by-n symmetric matrix stored in packed format.
func Dspr2(h *Handle, uplo Uplo, n int, alpha float64, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, AP cudart.DevicePtr) error {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}

	return Check(C.cublasDspr2(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.int(n), (*C.double)(&alpha), (*C.double)(x), C.int(incX), (*C.double)(y), C.int(incY), (*C.double)(AP)))
}

// Ssymv performs the matrix-vector operation y = alpha*A*x + beta*y for single precision real symmetric matrix.
// A is an n-by-n symmetric matrix.
func Ssymv(h *Handle, uplo Uplo, n int, alpha float32, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta float32, y cudart.DevicePtr, incY int) error {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}
	if lda < n {
		panic(blas.ErrShortA)
	}

	return Check(C.cublasSsymv(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.int(n), (*C.float)(&alpha), (*C.float)(A), C.int(lda), (*C.float)(x), C.int(incX), (*C.float)(&beta), (*C.float)(y), C.int(incY)))
}

// Dsymv performs the matrix-vector operation y = alpha*A*x + beta*y for double precision real symmetric matrix.
// A is an n-by-n symmetric matrix.
func Dsymv(h *Handle, uplo Uplo, n int, alpha float64, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta float64, y cudart.DevicePtr, incY int) error {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}
	if lda < n {
		panic(blas.ErrShortA)
	}

	return Check(C.cublasDsymv(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.int(n), (*C.double)(&alpha), (*C.double)(A), C.int(lda), (*C.double)(x), C.int(incX), (*C.double)(&beta), (*C.double)(y), C.int(incY)))
}

// Csymv performs the matrix-vector operation y = alpha*A*x + beta*y for single precision complex symmetric matrix.
// A is an n-by-n symmetric matrix (not Hermitian).
func Csymv(h *Handle, uplo Uplo, n int, alpha complex64, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta complex64, y cudart.DevicePtr, incY int) error {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}
	if lda < n {
		panic(blas.ErrShortA)
	}

	return Check(C.cublasCsymv(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.int(n), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(A), C.int(lda), (*C.cuComplex)(x), C.int(incX), (*C.cuComplex)(unsafe.Pointer(&beta)), (*C.cuComplex)(y), C.int(incY)))
}

// Zsymv performs the matrix-vector operation y = alpha*A*x + beta*y for double precision complex symmetric matrix.
// A is an n-by-n symmetric matrix (not Hermitian).
func Zsymv(h *Handle, uplo Uplo, n int, alpha complex128, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta complex128, y cudart.DevicePtr, incY int) error {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}
	if lda < n {
		panic(blas.ErrShortA)
	}

	return Check(C.cublasZsymv(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(A), C.int(lda), (*C.cuDoubleComplex)(x), C.int(incX), (*C.cuDoubleComplex)(unsafe.Pointer(&beta)), (*C.cuDoubleComplex)(y), C.int(incY)))
}

// Ssyr performs the symmetric rank-1 update A = alpha*x*x^T + A for single precision real symmetric matrix.
// A is an n-by-n symmetric matrix.
func Ssyr(h *Handle, uplo Uplo, n int, alpha float32, x cudart.DevicePtr, incX int, A cudart.DevicePtr, lda int) error {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if lda < n {
		panic(blas.ErrShortA)
	}

	return Check(C.cublasSsyr(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.int(n), (*C.float)(&alpha), (*C.float)(x), C.int(incX), (*C.float)(A), C.int(lda)))
}

// Dsyr performs the symmetric rank-1 update A = alpha*x*x^T + A for double precision real symmetric matrix.
// A is an n-by-n symmetric matrix.
func Dsyr(h *Handle, uplo Uplo, n int, alpha float64, x cudart.DevicePtr, incX int, A cudart.DevicePtr, lda int) error {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if lda < n {
		panic(blas.ErrShortA)
	}

	return Check(C.cublasDsyr(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.int(n), (*C.double)(&alpha), (*C.double)(x), C.int(incX), (*C.double)(A), C.int(lda)))
}

// Csyr performs the symmetric rank-1 update A = alpha*x*x^T + A for single precision complex symmetric matrix.
// A is an n-by-n symmetric matrix (not Hermitian).
func Csyr(h *Handle, uplo Uplo, n int, alpha complex64, x cudart.DevicePtr, incX int, A cudart.DevicePtr, lda int) error {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if lda < n {
		panic(blas.ErrShortA)
	}

	return Check(C.cublasCsyr(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.int(n), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(x), C.int(incX), (*C.cuComplex)(A), C.int(lda)))
}

// Zsyr performs the symmetric rank-1 update A = alpha*x*x^T + A for double precision complex symmetric matrix.
// A is an n-by-n symmetric matrix (not Hermitian).
func Zsyr(h *Handle, uplo Uplo, n int, alpha complex128, x cudart.DevicePtr, incX int, A cudart.DevicePtr, lda int) error {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if lda < n {
		panic(blas.ErrShortA)
	}

	return Check(C.cublasZsyr(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(x), C.int(incX), (*C.cuDoubleComplex)(A), C.int(lda)))
}

// Ssyr2 performs the symmetric rank-2 update A = alpha*x*y^T + alpha*y*x^T + A for single precision real symmetric matrix.
// A is an n-by-n symmetric matrix.
func Ssyr2(h *Handle, uplo Uplo, n int, alpha float32, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, A cudart.DevicePtr, lda int) error {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}
	if lda < n {
		panic(blas.ErrShortA)
	}

	return Check(C.cublasSsyr2(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.int(n), (*C.float)(&alpha), (*C.float)(x), C.int(incX), (*C.float)(y), C.int(incY), (*C.float)(A), C.int(lda)))
}

// Dsyr2 performs the symmetric rank-2 update A = alpha*x*y^T + alpha*y*x^T + A for double precision real symmetric matrix.
// A is an n-by-n symmetric matrix.
func Dsyr2(h *Handle, uplo Uplo, n int, alpha float64, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, A cudart.DevicePtr, lda int) error {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}
	if lda < n {
		panic(blas.ErrShortA)
	}

	return Check(C.cublasDsyr2(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.int(n), (*C.double)(&alpha), (*C.double)(x), C.int(incX), (*C.double)(y), C.int(incY), (*C.double)(A), C.int(lda)))
}

// Csyr2 performs the symmetric rank-2 update A = alpha*x*y^T + alpha*y*x^T + A for single precision complex symmetric matrix.
// A is an n-by-n symmetric matrix (not Hermitian).
func Csyr2(h *Handle, uplo Uplo, n int, alpha complex64, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, A cudart.DevicePtr, lda int) error {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}
	if lda < n {
		panic(blas.ErrShortA)
	}

	return Check(C.cublasCsyr2(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.int(n), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(x), C.int(incX), (*C.cuComplex)(y), C.int(incY), (*C.cuComplex)(A), C.int(lda)))
}

// Zsyr2 performs the symmetric rank-2 update A = alpha*x*y^T + alpha*y*x^T + A for double precision complex symmetric matrix.
// A is an n-by-n symmetric matrix (not Hermitian).
func Zsyr2(h *Handle, uplo Uplo, n int, alpha complex128, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, A cudart.DevicePtr, lda int) error {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}
	if lda < n {
		panic(blas.ErrShortA)
	}

	return Check(C.cublasZsyr2(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(x), C.int(incX), (*C.cuDoubleComplex)(y), C.int(incY), (*C.cuDoubleComplex)(A), C.int(lda)))
}

// Stbmv performs the matrix-vector operation x = A*x, x = A^T*x for single precision real triangular band matrix.
// A is an n-by-n triangular band matrix with k super-diagonals or sub-diagonals.
func Stbmv(h *Handle, uplo Uplo, trans Transpose, diag Diag, n, k int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) error {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if k < 0 {
		panic(blas.ErrKLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if lda < k+1 {
		panic(blas.ErrShortA)
	}

	return Check(C.cublasStbmv(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.cublasDiagType_t(diag), C.int(n), C.int(k), (*C.float)(A), C.int(lda), (*C.float)(x), C.int(incX)))
}

// Dtbmv performs the matrix-vector operation x = A*x, x = A^T*x for double precision real triangular band matrix.
// A is an n-by-n triangular band matrix with k super-diagonals or sub-diagonals.
func Dtbmv(h *Handle, uplo Uplo, trans Transpose, diag Diag, n, k int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) error {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if k < 0 {
		panic(blas.ErrKLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if lda < k+1 {
		panic(blas.ErrShortA)
	}

	return Check(C.cublasDtbmv(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.cublasDiagType_t(diag), C.int(n), C.int(k), (*C.double)(A), C.int(lda), (*C.double)(x), C.int(incX)))
}

// Ctbmv performs the matrix-vector operation x = A*x, x = A^T*x, x = A^H*x for single precision complex triangular band matrix.
// A is an n-by-n triangular band matrix with k super-diagonals or sub-diagonals.
func Ctbmv(h *Handle, uplo Uplo, trans Transpose, diag Diag, n, k int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) error {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if k < 0 {
		panic(blas.ErrKLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if lda < k+1 {
		panic(blas.ErrShortA)
	}

	return Check(C.cublasCtbmv(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.cublasDiagType_t(diag), C.int(n), C.int(k), (*C.cuComplex)(A), C.int(lda), (*C.cuComplex)(x), C.int(incX)))
}

// Ztbmv performs the matrix-vector operation x = A*x, x = A^T*x, x = A^H*x for double precision complex triangular band matrix.
// A is an n-by-n triangular band matrix with k super-diagonals or sub-diagonals.
func Ztbmv(h *Handle, uplo Uplo, trans Transpose, diag Diag, n, k int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) error {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if k < 0 {
		panic(blas.ErrKLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if lda < k+1 {
		panic(blas.ErrShortA)
	}

	return Check(C.cublasZtbmv(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.cublasDiagType_t(diag), C.int(n), C.int(k), (*C.cuDoubleComplex)(A), C.int(lda), (*C.cuDoubleComplex)(x), C.int(incX)))
}

// Stbsv solves the triangular band system A*x = b, A^T*x = b for single precision real triangular band matrix.
// A is an n-by-n triangular band matrix with k super-diagonals or sub-diagonals.
// The vector x contains the right-hand side b on input and the solution on output.
func Stbsv(h *Handle, uplo Uplo, trans Transpose, diag Diag, n, k int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) error {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if k < 0 {
		panic(blas.ErrKLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if lda < k+1 {
		panic(blas.ErrShortA)
	}

	return Check(C.cublasStbsv(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.cublasDiagType_t(diag), C.int(n), C.int(k), (*C.float)(A), C.int(lda), (*C.float)(x), C.int(incX)))
}

// Dtbsv solves the triangular band system A*x = b, A^T*x = b for double precision real triangular band matrix.
// A is an n-by-n triangular band matrix with k super-diagonals or sub-diagonals.
// The vector x contains the right-hand side b on input and the solution on output.
func Dtbsv(h *Handle, uplo Uplo, trans Transpose, diag Diag, n, k int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) error {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if k < 0 {
		panic(blas.ErrKLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if lda < k+1 {
		panic(blas.ErrShortA)
	}

	return Check(C.cublasDtbsv(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.cublasDiagType_t(diag), C.int(n), C.int(k), (*C.double)(A), C.int(lda), (*C.double)(x), C.int(incX)))
}

// Ctbsv solves the triangular band system A*x = b, A^T*x = b, A^H*x = b for single precision complex triangular band matrix.
// A is an n-by-n triangular band matrix with k super-diagonals or sub-diagonals.
// The vector x contains the right-hand side b on input and the solution on output.
func Ctbsv(h *Handle, uplo Uplo, trans Transpose, diag Diag, n, k int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) error {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if k < 0 {
		panic(blas.ErrKLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if lda < k+1 {
		panic(blas.ErrShortA)
	}

	return Check(C.cublasCtbsv(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.cublasDiagType_t(diag), C.int(n), C.int(k), (*C.cuComplex)(A), C.int(lda), (*C.cuComplex)(x), C.int(incX)))
}

// Ztbsv solves the triangular band system A*x = b, A^T*x = b, A^H*x = b for double precision complex triangular band matrix.
// A is an n-by-n triangular band matrix with k super-diagonals or sub-diagonals.
// The vector x contains the right-hand side b on input and the solution on output.
func Ztbsv(h *Handle, uplo Uplo, trans Transpose, diag Diag, n, k int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) error {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if k < 0 {
		panic(blas.ErrKLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if lda < k+1 {
		panic(blas.ErrShortA)
	}

	return Check(C.cublasZtbsv(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.cublasDiagType_t(diag), C.int(n), C.int(k), (*C.cuDoubleComplex)(A), C.int(lda), (*C.cuDoubleComplex)(x), C.int(incX)))
}

// Stpmv performs the matrix-vector operation x = A*x, x = A^T*x for single precision real triangular packed matrix.
// A is an n-by-n triangular matrix stored in packed format.
func Stpmv(h *Handle, uplo Uplo, trans Transpose, diag Diag, n int, AP cudart.DevicePtr, x cudart.DevicePtr, incX int) error {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	return Check(C.cublasStpmv(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.cublasDiagType_t(diag), C.int(n), (*C.float)(AP), (*C.float)(x), C.int(incX)))
}

// Dtpmv performs the matrix-vector operation x = A*x, x = A^T*x for double precision real triangular packed matrix.
// A is an n-by-n triangular matrix stored in packed format.
func Dtpmv(h *Handle, uplo Uplo, trans Transpose, diag Diag, n int, AP cudart.DevicePtr, x cudart.DevicePtr, incX int) error {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	return Check(C.cublasDtpmv(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.cublasDiagType_t(diag), C.int(n), (*C.double)(AP), (*C.double)(x), C.int(incX)))
}

// Ctpmv performs the matrix-vector operation x = A*x, x = A^T*x, x = A^H*x for single precision complex triangular packed matrix.
// A is an n-by-n triangular matrix stored in packed format.
func Ctpmv(h *Handle, uplo Uplo, trans Transpose, diag Diag, n int, AP cudart.DevicePtr, x cudart.DevicePtr, incX int) error {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	return Check(C.cublasCtpmv(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.cublasDiagType_t(diag), C.int(n), (*C.cuComplex)(AP), (*C.cuComplex)(x), C.int(incX)))
}

// Ztpmv performs the matrix-vector operation x = A*x, x = A^T*x, x = A^H*x for double precision complex triangular packed matrix.
// A is an n-by-n triangular matrix stored in packed format.
func Ztpmv(h *Handle, uplo Uplo, trans Transpose, diag Diag, n int, AP cudart.DevicePtr, x cudart.DevicePtr, incX int) error {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	return Check(C.cublasZtpmv(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.cublasDiagType_t(diag), C.int(n), (*C.cuDoubleComplex)(AP), (*C.cuDoubleComplex)(x), C.int(incX)))
}

// Stpsv solves the triangular packed system A*x = b, A^T*x = b for single precision real triangular packed matrix.
// A is an n-by-n triangular matrix stored in packed format.
// The vector x contains the right-hand side b on input and the solution on output.
func Stpsv(h *Handle, uplo Uplo, trans Transpose, diag Diag, n int, AP cudart.DevicePtr, x cudart.DevicePtr, incX int) error {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	return Check(C.cublasStpsv(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.cublasDiagType_t(diag), C.int(n), (*C.float)(AP), (*C.float)(x), C.int(incX)))
}

// Dtpsv solves the triangular packed system A*x = b, A^T*x = b for double precision real triangular packed matrix.
// A is an n-by-n triangular matrix stored in packed format.
// The vector x contains the right-hand side b on input and the solution on output.
func Dtpsv(h *Handle, uplo Uplo, trans Transpose, diag Diag, n int, AP cudart.DevicePtr, x cudart.DevicePtr, incX int) error {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	return Check(C.cublasDtpsv(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.cublasDiagType_t(diag), C.int(n), (*C.double)(AP), (*C.double)(x), C.int(incX)))
}

// Ctpsv solves the triangular packed system A*x = b, A^T*x = b, A^H*x = b for single precision complex triangular packed matrix.
// A is an n-by-n triangular matrix stored in packed format.
// The vector x contains the right-hand side b on input and the solution on output.
func Ctpsv(h *Handle, uplo Uplo, trans Transpose, diag Diag, n int, AP cudart.DevicePtr, x cudart.DevicePtr, incX int) error {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	return Check(C.cublasCtpsv(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.cublasDiagType_t(diag), C.int(n), (*C.cuComplex)(AP), (*C.cuComplex)(x), C.int(incX)))
}

// Ztpsv solves the triangular packed system A*x = b, A^T*x = b, A^H*x = b for double precision complex triangular packed matrix.
// A is an n-by-n triangular matrix stored in packed format.
// The vector x contains the right-hand side b on input and the solution on output.
func Ztpsv(h *Handle, uplo Uplo, trans Transpose, diag Diag, n int, AP cudart.DevicePtr, x cudart.DevicePtr, incX int) error {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	return Check(C.cublasZtpsv(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.cublasDiagType_t(diag), C.int(n), (*C.cuDoubleComplex)(AP), (*C.cuDoubleComplex)(x), C.int(incX)))
}

// Strmv performs the matrix-vector operation x = A*x, x = A^T*x for single precision real triangular matrix.
// A is an n-by-n triangular matrix.
func Strmv(h *Handle, uplo Uplo, trans Transpose, diag Diag, n int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) error {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if lda < n {
		panic(blas.ErrShortA)
	}

	return Check(C.cublasStrmv(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.cublasDiagType_t(diag), C.int(n), (*C.float)(A), C.int(lda), (*C.float)(x), C.int(incX)))
}

// Dtrmv performs the matrix-vector operation x = A*x, x = A^T*x for double precision real triangular matrix.
// A is an n-by-n triangular matrix.
func Dtrmv(h *Handle, uplo Uplo, trans Transpose, diag Diag, n int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) error {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if lda < n {
		panic(blas.ErrShortA)
	}

	return Check(C.cublasDtrmv(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.cublasDiagType_t(diag), C.int(n), (*C.double)(A), C.int(lda), (*C.double)(x), C.int(incX)))
}

// Ctrmv performs the matrix-vector operation x = A*x, x = A^T*x, x = A^H*x for single precision complex triangular matrix.
// A is an n-by-n triangular matrix.
func Ctrmv(h *Handle, uplo Uplo, trans Transpose, diag Diag, n int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) error {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if lda < n {
		panic(blas.ErrShortA)
	}

	return Check(C.cublasCtrmv(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.cublasDiagType_t(diag), C.int(n), (*C.cuComplex)(A), C.int(lda), (*C.cuComplex)(x), C.int(incX)))
}

// Ztrmv performs the matrix-vector operation x = A*x, x = A^T*x, x = A^H*x for double precision complex triangular matrix.
// A is an n-by-n triangular matrix.
func Ztrmv(h *Handle, uplo Uplo, trans Transpose, diag Diag, n int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) error {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if lda < n {
		panic(blas.ErrShortA)
	}

	return Check(C.cublasZtrmv(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.cublasDiagType_t(diag), C.int(n), (*C.cuDoubleComplex)(A), C.int(lda), (*C.cuDoubleComplex)(x), C.int(incX)))
}

// Strsv solves the triangular system A*x = b, A^T*x = b for single precision real triangular matrix.
// A is an n-by-n triangular matrix.
// The vector x contains the right-hand side b on input and the solution on output.
func Strsv(h *Handle, uplo Uplo, trans Transpose, diag Diag, n int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) error {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if lda < n {
		panic(blas.ErrShortA)
	}

	return Check(C.cublasStrsv(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.cublasDiagType_t(diag), C.int(n), (*C.float)(A), C.int(lda), (*C.float)(x), C.int(incX)))
}

// Dtrsv solves the triangular system A*x = b, A^T*x = b for double precision real triangular matrix.
// A is an n-by-n triangular matrix.
// The vector x contains the right-hand side b on input and the solution on output.
func Dtrsv(h *Handle, uplo Uplo, trans Transpose, diag Diag, n int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) error {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if lda < n {
		panic(blas.ErrShortA)
	}

	return Check(C.cublasDtrsv(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.cublasDiagType_t(diag), C.int(n), (*C.double)(A), C.int(lda), (*C.double)(x), C.int(incX)))
}

// Ctrsv solves the triangular system A*x = b, A^T*x = b, A^H*x = b for single precision complex triangular matrix.
// A is an n-by-n triangular matrix.
// The vector x contains the right-hand side b on input and the solution on output.
func Ctrsv(h *Handle, uplo Uplo, trans Transpose, diag Diag, n int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) error {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if lda < n {
		panic(blas.ErrShortA)
	}

	return Check(C.cublasCtrsv(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.cublasDiagType_t(diag), C.int(n), (*C.cuComplex)(A), C.int(lda), (*C.cuComplex)(x), C.int(incX)))
}

// Ztrsv solves the triangular system A*x = b, A^T*x = b, A^H*x = b for double precision complex triangular matrix.
// A is an n-by-n triangular matrix.
// The vector x contains the right-hand side b on input and the solution on output.
func Ztrsv(h *Handle, uplo Uplo, trans Transpose, diag Diag, n int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) error {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if lda < n {
		panic(blas.ErrShortA)
	}

	return Check(C.cublasZtrsv(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.cublasDiagType_t(diag), C.int(n), (*C.cuDoubleComplex)(A), C.int(lda), (*C.cuDoubleComplex)(x), C.int(incX)))
}

// Chemv performs the matrix-vector operation y = alpha*A*x + beta*y for single precision complex Hermitian matrix.
// A is an n-by-n Hermitian matrix (A = A^H).
func Chemv(h *Handle, uplo Uplo, n int, alpha complex64, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta complex64, y cudart.DevicePtr, incY int) error {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}
	if lda < n {
		panic(blas.ErrShortA)
	}

	return Check(C.cublasChemv(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.int(n), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(A), C.int(lda), (*C.cuComplex)(x), C.int(incX), (*C.cuComplex)(unsafe.Pointer(&beta)), (*C.cuComplex)(y), C.int(incY)))
}

// Zhemv performs the matrix-vector operation y = alpha*A*x + beta*y for double precision complex Hermitian matrix.
// A is an n-by-n Hermitian matrix (A = A^H).
func Zhemv(h *Handle, uplo Uplo, n int, alpha complex128, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta complex128, y cudart.DevicePtr, incY int) error {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}
	if lda < n {
		panic(blas.ErrShortA)
	}

	return Check(C.cublasZhemv(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(A), C.int(lda), (*C.cuDoubleComplex)(x), C.int(incX), (*C.cuDoubleComplex)(unsafe.Pointer(&beta)), (*C.cuDoubleComplex)(y), C.int(incY)))
}

// Chbmv performs the matrix-vector operation y = alpha*A*x + beta*y for single precision complex Hermitian band matrix.
// A is an n-by-n Hermitian band matrix with k super-diagonals.
func Chbmv(h *Handle, uplo Uplo, n, k int, alpha complex64, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta complex64, y cudart.DevicePtr, incY int) error {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if k < 0 {
		panic(blas.ErrKLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}
	if lda < k+1 {
		panic(blas.ErrShortA)
	}

	return Check(C.cublasChbmv(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.int(n), C.int(k), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(A), C.int(lda), (*C.cuComplex)(x), C.int(incX), (*C.cuComplex)(unsafe.Pointer(&beta)), (*C.cuComplex)(y), C.int(incY)))
}

// Zhbmv performs the matrix-vector operation y = alpha*A*x + beta*y for double precision complex Hermitian band matrix.
// A is an n-by-n Hermitian band matrix with k super-diagonals.
func Zhbmv(h *Handle, uplo Uplo, n, k int, alpha complex128, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta complex128, y cudart.DevicePtr, incY int) error {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if k < 0 {
		panic(blas.ErrKLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}
	if lda < k+1 {
		panic(blas.ErrShortA)
	}

	return Check(C.cublasZhbmv(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.int(n), C.int(k), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(A), C.int(lda), (*C.cuDoubleComplex)(x), C.int(incX), (*C.cuDoubleComplex)(unsafe.Pointer(&beta)), (*C.cuDoubleComplex)(y), C.int(incY)))
}

// Chpmv performs the matrix-vector operation y = alpha*A*x + beta*y for single precision complex Hermitian packed matrix.
// A is an n-by-n Hermitian matrix stored in packed format.
func Chpmv(h *Handle, uplo Uplo, n int, alpha complex64, AP cudart.DevicePtr, x cudart.DevicePtr, incX int, beta complex64, y cudart.DevicePtr, incY int) error {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}

	return Check(C.cublasChpmv(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.int(n), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(AP), (*C.cuComplex)(x), C.int(incX), (*C.cuComplex)(unsafe.Pointer(&beta)), (*C.cuComplex)(y), C.int(incY)))
}

// Zhpmv performs the matrix-vector operation y = alpha*A*x + beta*y for double precision complex Hermitian packed matrix.
// A is an n-by-n Hermitian matrix stored in packed format.
func Zhpmv(h *Handle, uplo Uplo, n int, alpha complex128, AP cudart.DevicePtr, x cudart.DevicePtr, incX int, beta complex128, y cudart.DevicePtr, incY int) error {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}

	return Check(C.cublasZhpmv(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(AP), (*C.cuDoubleComplex)(x), C.int(incX), (*C.cuDoubleComplex)(unsafe.Pointer(&beta)), (*C.cuDoubleComplex)(y), C.int(incY)))
}

// Cher performs the Hermitian rank-1 update A = alpha*x*x^H + A for single precision complex Hermitian matrix.
// A is an n-by-n Hermitian matrix. Note: alpha must be real for Hermitian matrices.
func Cher(h *Handle, uplo Uplo, n int, alpha float32, x cudart.DevicePtr, incX int, A cudart.DevicePtr, lda int) error {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if lda < n {
		panic(blas.ErrShortA)
	}

	return Check(C.cublasCher(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.int(n), (*C.float)(&alpha), (*C.cuComplex)(x), C.int(incX), (*C.cuComplex)(A), C.int(lda)))
}

// Zher performs the Hermitian rank-1 update A = alpha*x*x^H + A for double precision complex Hermitian matrix.
// A is an n-by-n Hermitian matrix. Note: alpha must be real for Hermitian matrices.
func Zher(h *Handle, uplo Uplo, n int, alpha float64, x cudart.DevicePtr, incX int, A cudart.DevicePtr, lda int) error {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if lda < n {
		panic(blas.ErrShortA)
	}

	return Check(C.cublasZher(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.int(n), (*C.double)(&alpha), (*C.cuDoubleComplex)(x), C.int(incX), (*C.cuDoubleComplex)(A), C.int(lda)))
}

// Cher2 performs the Hermitian rank-2 update A = alpha*x*y^H + conj(alpha)*y*x^H + A for single precision complex Hermitian matrix.
// A is an n-by-n Hermitian matrix.
func Cher2(h *Handle, uplo Uplo, n int, alpha complex64, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, A cudart.DevicePtr, lda int) error {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}
	if lda < n {
		panic(blas.ErrShortA)
	}

	return Check(C.cublasCher2(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.int(n), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(x), C.int(incX), (*C.cuComplex)(y), C.int(incY), (*C.cuComplex)(A), C.int(lda)))
}

// Zher2 performs the Hermitian rank-2 update A = alpha*x*y^H + conj(alpha)*y*x^H + A for double precision complex Hermitian matrix.
// A is an n-by-n Hermitian matrix.
func Zher2(h *Handle, uplo Uplo, n int, alpha complex128, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, A cudart.DevicePtr, lda int) error {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}
	if lda < n {
		panic(blas.ErrShortA)
	}

	return Check(C.cublasZher2(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(x), C.int(incX), (*C.cuDoubleComplex)(y), C.int(incY), (*C.cuDoubleComplex)(A), C.int(lda)))
}

// Chpr performs the Hermitian rank-1 update AP = alpha*x*x^H + AP for single precision complex Hermitian packed matrix.
// AP is an n-by-n Hermitian matrix stored in packed format. Note: alpha must be real for Hermitian matrices.
func Chpr(h *Handle, uplo Uplo, n int, alpha float32, x cudart.DevicePtr, incX int, AP cudart.DevicePtr) error {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	return Check(C.cublasChpr(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.int(n), (*C.float)(&alpha), (*C.cuComplex)(x), C.int(incX), (*C.cuComplex)(AP)))
}

// Zhpr performs the Hermitian rank-1 update AP = alpha*x*x^H + AP for double precision complex Hermitian packed matrix.
// AP is an n-by-n Hermitian matrix stored in packed format. Note: alpha must be real for Hermitian matrices.
func Zhpr(h *Handle, uplo Uplo, n int, alpha float64, x cudart.DevicePtr, incX int, AP cudart.DevicePtr) error {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	return Check(C.cublasZhpr(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.int(n), (*C.double)(&alpha), (*C.cuDoubleComplex)(x), C.int(incX), (*C.cuDoubleComplex)(AP)))
}

// Chpr2 performs the Hermitian rank-2 update AP = alpha*x*y^H + conj(alpha)*y*x^H + AP for single precision complex Hermitian packed matrix.
// AP is an n-by-n Hermitian matrix stored in packed format.
func Chpr2(h *Handle, uplo Uplo, n int, alpha complex64, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, AP cudart.DevicePtr) error {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}

	return Check(C.cublasChpr2(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.int(n), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(x), C.int(incX), (*C.cuComplex)(y), C.int(incY), (*C.cuComplex)(AP)))
}

// Zhpr2 performs the Hermitian rank-2 update AP = alpha*x*y^H + conj(alpha)*y*x^H + AP for double precision complex Hermitian packed matrix.
// AP is an n-by-n Hermitian matrix stored in packed format.
func Zhpr2(h *Handle, uplo Uplo, n int, alpha complex128, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, AP cudart.DevicePtr) error {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}

	return Check(C.cublasZhpr2(C.cublasHandle_t(h.h), C.cublasFillMode_t(uplo), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(x), C.int(incX), (*C.cuDoubleComplex)(y), C.int(incY), (*C.cuDoubleComplex)(AP)))
}

// SgemvBatched performs multiple matrix-vector operations y[i] = alpha*A[i]*x[i] + beta*y[i] for single precision real matrices.
// This function performs batchCount independent GEMV operations in parallel.
func SgemvBatched(h *Handle, trans Transpose, m, n int, alpha float32, Aarray []cudart.DevicePtr, lda int, xarray []cudart.DevicePtr, incX int, beta float32, yarray []cudart.DevicePtr, incY int, batchCount int) error {

	if m < 0 {
		panic(blas.ErrMLT0)
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}
	if lda < m {
		panic(blas.ErrShortA)
	}
	if batchCount < 0 {
		panic("batchCount must be non-negative")
	}
	if len(Aarray) < batchCount || len(xarray) < batchCount || len(yarray) < batchCount {
		panic("array lengths must be at least batchCount")
	}

	return Check(C.cublasSgemvBatched(C.cublasHandle_t(h.h), C.cublasOperation_t(trans), C.int(m), C.int(n), (*C.float)(&alpha), (**C.float)(unsafe.Pointer(&Aarray[0])), C.int(lda), (**C.float)(unsafe.Pointer(&xarray[0])), C.int(incX), (*C.float)(&beta), (**C.float)(unsafe.Pointer(&yarray[0])), C.int(incY), C.int(batchCount)))
}

// DgemvBatched performs multiple matrix-vector operations y[i] = alpha*A[i]*x[i] + beta*y[i] for double precision real matrices.
// This function performs batchCount independent GEMV operations in parallel.
func DgemvBatched(h *Handle, trans Transpose, m, n int, alpha float64, Aarray []cudart.DevicePtr, lda int, xarray []cudart.DevicePtr, incX int, beta float64, yarray []cudart.DevicePtr, incY int, batchCount int) error {

	if m < 0 {
		panic(blas.ErrMLT0)
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}
	if lda < m {
		panic(blas.ErrShortA)
	}
	if batchCount < 0 {
		panic("batchCount must be non-negative")
	}
	if len(Aarray) < batchCount || len(xarray) < batchCount || len(yarray) < batchCount {
		panic("array lengths must be at least batchCount")
	}

	return Check(C.cublasDgemvBatched(C.cublasHandle_t(h.h), C.cublasOperation_t(trans), C.int(m), C.int(n), (*C.double)(&alpha), (**C.double)(unsafe.Pointer(&Aarray[0])), C.int(lda), (**C.double)(unsafe.Pointer(&xarray[0])), C.int(incX), (*C.double)(&beta), (**C.double)(unsafe.Pointer(&yarray[0])), C.int(incY), C.int(batchCount)))
}

// CgemvBatched performs multiple matrix-vector operations y[i] = alpha*A[i]*x[i] + beta*y[i] for single precision complex matrices.
// This function performs batchCount independent GEMV operations in parallel.
func CgemvBatched(h *Handle, trans Transpose, m, n int, alpha complex64, Aarray []cudart.DevicePtr, lda int, xarray []cudart.DevicePtr, incX int, beta complex64, yarray []cudart.DevicePtr, incY int, batchCount int) error {

	if m < 0 {
		panic(blas.ErrMLT0)
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}
	if lda < m {
		panic(blas.ErrShortA)
	}
	if batchCount < 0 {
		panic("batchCount must be non-negative")
	}
	if len(Aarray) < batchCount || len(xarray) < batchCount || len(yarray) < batchCount {
		panic("array lengths must be at least batchCount")
	}

	return Check(C.cublasCgemvBatched(C.cublasHandle_t(h.h), C.cublasOperation_t(trans), C.int(m), C.int(n), (*C.cuComplex)(unsafe.Pointer(&alpha)), (**C.cuComplex)(unsafe.Pointer(&Aarray[0])), C.int(lda), (**C.cuComplex)(unsafe.Pointer(&xarray[0])), C.int(incX), (*C.cuComplex)(unsafe.Pointer(&beta)), (**C.cuComplex)(unsafe.Pointer(&yarray[0])), C.int(incY), C.int(batchCount)))
}

// ZgemvBatched performs multiple matrix-vector operations y[i] = alpha*A[i]*x[i] + beta*y[i] for double precision complex matrices.
// This function performs batchCount independent GEMV operations in parallel.
func ZgemvBatched(h *Handle, trans Transpose, m, n int, alpha complex128, Aarray []cudart.DevicePtr, lda int, xarray []cudart.DevicePtr, incX int, beta complex128, yarray []cudart.DevicePtr, incY int, batchCount int) error {

	if m < 0 {
		panic(blas.ErrMLT0)
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}
	if lda < m {
		panic(blas.ErrShortA)
	}
	if batchCount < 0 {
		panic("batchCount must be non-negative")
	}
	if len(Aarray) < batchCount || len(xarray) < batchCount || len(yarray) < batchCount {
		panic("array lengths must be at least batchCount")
	}

	return Check(C.cublasZgemvBatched(C.cublasHandle_t(h.h), C.cublasOperation_t(trans), C.int(m), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (**C.cuDoubleComplex)(unsafe.Pointer(&Aarray[0])), C.int(lda), (**C.cuDoubleComplex)(unsafe.Pointer(&xarray[0])), C.int(incX), (*C.cuDoubleComplex)(unsafe.Pointer(&beta)), (**C.cuDoubleComplex)(unsafe.Pointer(&yarray[0])), C.int(incY), C.int(batchCount)))
}

// SgemvStridedBatched performs multiple matrix-vector operations y[i] = alpha*A[i]*x[i] + beta*y[i] for single precision real matrices.
// This function uses stride-based memory layout for better performance with contiguous data.
func SgemvStridedBatched(h *Handle, trans Transpose, m, n int, alpha float32, A cudart.DevicePtr, lda int, strideA int64, x cudart.DevicePtr, incX int, stridex int64, beta float32, y cudart.DevicePtr, incY int, stridey int64, batchCount int) error {

	if m < 0 {
		panic(blas.ErrMLT0)
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}
	if lda < m {
		panic(blas.ErrShortA)
	}
	if batchCount < 0 {
		panic("batchCount must be non-negative")
	}

	return Check(C.cublasSgemvStridedBatched(C.cublasHandle_t(h.h), C.cublasOperation_t(trans), C.int(m), C.int(n), (*C.float)(&alpha), (*C.float)(A), C.int(lda), C.longlong(strideA), (*C.float)(x), C.int(incX), C.longlong(stridex), (*C.float)(&beta), (*C.float)(y), C.int(incY), C.longlong(stridey), C.int(batchCount)))
}

// DgemvStridedBatched performs multiple matrix-vector operations y[i] = alpha*A[i]*x[i] + beta*y[i] for double precision real matrices.
// This function uses stride-based memory layout for better performance with contiguous data.
func DgemvStridedBatched(h *Handle, trans Transpose, m, n int, alpha float64, A cudart.DevicePtr, lda int, strideA int64, x cudart.DevicePtr, incX int, stridex int64, beta float64, y cudart.DevicePtr, incY int, stridey int64, batchCount int) error {

	if m < 0 {
		panic(blas.ErrMLT0)
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}
	if lda < m {
		panic(blas.ErrShortA)
	}
	if batchCount < 0 {
		panic("batchCount must be non-negative")
	}

	return Check(C.cublasDgemvStridedBatched(C.cublasHandle_t(h.h), C.cublasOperation_t(trans), C.int(m), C.int(n), (*C.double)(&alpha), (*C.double)(A), C.int(lda), C.longlong(strideA), (*C.double)(x), C.int(incX), C.longlong(stridex), (*C.double)(&beta), (*C.double)(y), C.int(incY), C.longlong(stridey), C.int(batchCount)))
}

// CgemvStridedBatched performs multiple matrix-vector operations y[i] = alpha*A[i]*x[i] + beta*y[i] for single precision complex matrices.
// This function uses stride-based memory layout for better performance with contiguous data.
func CgemvStridedBatched(h *Handle, trans Transpose, m, n int, alpha complex64, A cudart.DevicePtr, lda int, strideA int64, x cudart.DevicePtr, incX int, stridex int64, beta complex64, y cudart.DevicePtr, incY int, stridey int64, batchCount int) error {

	if m < 0 {
		panic(blas.ErrMLT0)
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}
	if lda < m {
		panic(blas.ErrShortA)
	}
	if batchCount < 0 {
		panic("batchCount must be non-negative")
	}

	return Check(C.cublasCgemvStridedBatched(C.cublasHandle_t(h.h), C.cublasOperation_t(trans), C.int(m), C.int(n), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(A), C.int(lda), C.longlong(strideA), (*C.cuComplex)(x), C.int(incX), C.longlong(stridex), (*C.cuComplex)(unsafe.Pointer(&beta)), (*C.cuComplex)(y), C.int(incY), C.longlong(stridey), C.int(batchCount)))
}

// ZgemvStridedBatched performs multiple matrix-vector operations y[i] = alpha*A[i]*x[i] + beta*y[i] for double precision complex matrices.
// This function uses stride-based memory layout for better performance with contiguous data.
func ZgemvStridedBatched(h *Handle, trans Transpose, m, n int, alpha complex128, A cudart.DevicePtr, lda int, strideA int64, x cudart.DevicePtr, incX int, stridex int64, beta complex128, y cudart.DevicePtr, incY int, stridey int64, batchCount int) error {

	if m < 0 {
		panic(blas.ErrMLT0)
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}
	if lda < m {
		panic(blas.ErrShortA)
	}
	if batchCount < 0 {
		panic("batchCount must be non-negative")
	}

	return Check(C.cublasZgemvStridedBatched(C.cublasHandle_t(h.h), C.cublasOperation_t(trans), C.int(m), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(A), C.int(lda), C.longlong(strideA), (*C.cuDoubleComplex)(x), C.int(incX), C.longlong(stridex), (*C.cuDoubleComplex)(unsafe.Pointer(&beta)), (*C.cuDoubleComplex)(y), C.int(incY), C.longlong(stridey), C.int(batchCount)))
}
