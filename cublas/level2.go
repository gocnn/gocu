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
func (impl *Standard) Sgbmv(trans Transpose, m, n, kl, ku int, alpha float32, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta float32, y cudart.DevicePtr, incY int) {
	if impl.e != nil {
		return
	}
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

	impl.e = Check(C.cublasSgbmv(C.cublasHandle_t(impl.h), C.cublasOperation_t(trans), C.int(m), C.int(n), C.int(kl), C.int(ku), (*C.float)(&alpha), (*C.float)(A), C.int(lda), (*C.float)(x), C.int(incX), (*C.float)(&beta), (*C.float)(y), C.int(incY)))
}

// Dgbmv performs the matrix-vector operation y = alpha*A*x + beta*y for double precision real band matrix.
// A is an m-by-n band matrix with kl sub-diagonals and ku super-diagonals.
// The matrix A is stored in band storage format.
func (impl *Standard) Dgbmv(trans Transpose, m, n, kl, ku int, alpha float64, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta float64, y cudart.DevicePtr, incY int) {
	if impl.e != nil {
		return
	}
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

	impl.e = Check(C.cublasDgbmv(C.cublasHandle_t(impl.h), C.cublasOperation_t(trans), C.int(m), C.int(n), C.int(kl), C.int(ku), (*C.double)(&alpha), (*C.double)(A), C.int(lda), (*C.double)(x), C.int(incX), (*C.double)(&beta), (*C.double)(y), C.int(incY)))
}

// Cgbmv performs the matrix-vector operation y = alpha*A*x + beta*y for single precision complex band matrix.
// A is an m-by-n band matrix with kl sub-diagonals and ku super-diagonals.
// The matrix A is stored in band storage format.
func (impl *Standard) Cgbmv(trans Transpose, m, n, kl, ku int, alpha complex64, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta complex64, y cudart.DevicePtr, incY int) {
	if impl.e != nil {
		return
	}
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

	impl.e = Check(C.cublasCgbmv(C.cublasHandle_t(impl.h), C.cublasOperation_t(trans), C.int(m), C.int(n), C.int(kl), C.int(ku), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(A), C.int(lda), (*C.cuComplex)(x), C.int(incX), (*C.cuComplex)(unsafe.Pointer(&beta)), (*C.cuComplex)(y), C.int(incY)))
}

// Zgbmv performs the matrix-vector operation y = alpha*A*x + beta*y for double precision complex band matrix.
// A is an m-by-n band matrix with kl sub-diagonals and ku super-diagonals.
// The matrix A is stored in band storage format.
func (impl *Standard) Zgbmv(trans Transpose, m, n, kl, ku int, alpha complex128, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta complex128, y cudart.DevicePtr, incY int) {
	if impl.e != nil {
		return
	}
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

	impl.e = Check(C.cublasZgbmv(C.cublasHandle_t(impl.h), C.cublasOperation_t(trans), C.int(m), C.int(n), C.int(kl), C.int(ku), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(A), C.int(lda), (*C.cuDoubleComplex)(x), C.int(incX), (*C.cuDoubleComplex)(unsafe.Pointer(&beta)), (*C.cuDoubleComplex)(y), C.int(incY)))
}

// Sgemv performs the matrix-vector operation y = alpha*A*x + beta*y for single precision real general matrix.
// A is an m-by-n general matrix.
func (impl *Standard) Sgemv(trans Transpose, m, n int, alpha float32, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta float32, y cudart.DevicePtr, incY int) {
	if impl.e != nil {
		return
	}
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

	impl.e = Check(C.cublasSgemv(C.cublasHandle_t(impl.h), C.cublasOperation_t(trans), C.int(m), C.int(n), (*C.float)(&alpha), (*C.float)(A), C.int(lda), (*C.float)(x), C.int(incX), (*C.float)(&beta), (*C.float)(y), C.int(incY)))
}

// Dgemv performs the matrix-vector operation y = alpha*A*x + beta*y for double precision real general matrix.
// A is an m-by-n general matrix.
func (impl *Standard) Dgemv(trans Transpose, m, n int, alpha float64, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta float64, y cudart.DevicePtr, incY int) {
	if impl.e != nil {
		return
	}
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

	impl.e = Check(C.cublasDgemv(C.cublasHandle_t(impl.h), C.cublasOperation_t(trans), C.int(m), C.int(n), (*C.double)(&alpha), (*C.double)(A), C.int(lda), (*C.double)(x), C.int(incX), (*C.double)(&beta), (*C.double)(y), C.int(incY)))
}

// Cgemv performs the matrix-vector operation y = alpha*A*x + beta*y for single precision complex general matrix.
// A is an m-by-n general matrix.
func (impl *Standard) Cgemv(trans Transpose, m, n int, alpha complex64, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta complex64, y cudart.DevicePtr, incY int) {
	if impl.e != nil {
		return
	}
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

	impl.e = Check(C.cublasCgemv(C.cublasHandle_t(impl.h), C.cublasOperation_t(trans), C.int(m), C.int(n), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(A), C.int(lda), (*C.cuComplex)(x), C.int(incX), (*C.cuComplex)(unsafe.Pointer(&beta)), (*C.cuComplex)(y), C.int(incY)))
}

// Zgemv performs the matrix-vector operation y = alpha*A*x + beta*y for double precision complex general matrix.
// A is an m-by-n general matrix.
func (impl *Standard) Zgemv(trans Transpose, m, n int, alpha complex128, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta complex128, y cudart.DevicePtr, incY int) {
	if impl.e != nil {
		return
	}
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

	impl.e = Check(C.cublasZgemv(C.cublasHandle_t(impl.h), C.cublasOperation_t(trans), C.int(m), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(A), C.int(lda), (*C.cuDoubleComplex)(x), C.int(incX), (*C.cuDoubleComplex)(unsafe.Pointer(&beta)), (*C.cuDoubleComplex)(y), C.int(incY)))
}

// Sger performs the rank-1 update A = alpha*x*y^T + A for single precision real matrix.
// A is an m-by-n general matrix.
func (impl *Standard) Sger(m, n int, alpha float32, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, A cudart.DevicePtr, lda int) {
	if impl.e != nil {
		return
	}
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

	impl.e = Check(C.cublasSger(C.cublasHandle_t(impl.h), C.int(m), C.int(n), (*C.float)(&alpha), (*C.float)(x), C.int(incX), (*C.float)(y), C.int(incY), (*C.float)(A), C.int(lda)))
}

// Dger performs the rank-1 update A = alpha*x*y^T + A for double precision real matrix.
// A is an m-by-n general matrix.
func (impl *Standard) Dger(m, n int, alpha float64, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, A cudart.DevicePtr, lda int) {
	if impl.e != nil {
		return
	}
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

	impl.e = Check(C.cublasDger(C.cublasHandle_t(impl.h), C.int(m), C.int(n), (*C.double)(&alpha), (*C.double)(x), C.int(incX), (*C.double)(y), C.int(incY), (*C.double)(A), C.int(lda)))
}

// Cgeru performs the rank-1 update A = alpha*x*y^T + A for single precision complex matrix (unconjugated).
// A is an m-by-n general matrix.
func (impl *Standard) Cgeru(m, n int, alpha complex64, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, A cudart.DevicePtr, lda int) {
	if impl.e != nil {
		return
	}
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

	impl.e = Check(C.cublasCgeru(C.cublasHandle_t(impl.h), C.int(m), C.int(n), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(x), C.int(incX), (*C.cuComplex)(y), C.int(incY), (*C.cuComplex)(A), C.int(lda)))
}

// Cgerc performs the rank-1 update A = alpha*x*y^H + A for single precision complex matrix (conjugated).
// A is an m-by-n general matrix.
func (impl *Standard) Cgerc(m, n int, alpha complex64, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, A cudart.DevicePtr, lda int) {
	if impl.e != nil {
		return
	}
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

	impl.e = Check(C.cublasCgerc(C.cublasHandle_t(impl.h), C.int(m), C.int(n), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(x), C.int(incX), (*C.cuComplex)(y), C.int(incY), (*C.cuComplex)(A), C.int(lda)))
}

// Zgeru performs the rank-1 update A = alpha*x*y^T + A for double precision complex matrix (unconjugated).
// A is an m-by-n general matrix.
func (impl *Standard) Zgeru(m, n int, alpha complex128, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, A cudart.DevicePtr, lda int) {
	if impl.e != nil {
		return
	}
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

	impl.e = Check(C.cublasZgeru(C.cublasHandle_t(impl.h), C.int(m), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(x), C.int(incX), (*C.cuDoubleComplex)(y), C.int(incY), (*C.cuDoubleComplex)(A), C.int(lda)))
}

// Zgerc performs the rank-1 update A = alpha*x*y^H + A for double precision complex matrix (conjugated).
// A is an m-by-n general matrix.
func (impl *Standard) Zgerc(m, n int, alpha complex128, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, A cudart.DevicePtr, lda int) {
	if impl.e != nil {
		return
	}
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

	impl.e = Check(C.cublasZgerc(C.cublasHandle_t(impl.h), C.int(m), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(x), C.int(incX), (*C.cuDoubleComplex)(y), C.int(incY), (*C.cuDoubleComplex)(A), C.int(lda)))
}

// Ssbmv performs the matrix-vector operation y = alpha*A*x + beta*y for single precision real symmetric band matrix.
// A is an n-by-n symmetric band matrix with k super-diagonals.
func (impl *Standard) Ssbmv(uplo Uplo, n, k int, alpha float32, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta float32, y cudart.DevicePtr, incY int) {
	if impl.e != nil {
		return
	}
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

	impl.e = Check(C.cublasSsbmv(C.cublasHandle_t(impl.h), C.cublasFillMode_t(uplo), C.int(n), C.int(k), (*C.float)(&alpha), (*C.float)(A), C.int(lda), (*C.float)(x), C.int(incX), (*C.float)(&beta), (*C.float)(y), C.int(incY)))
}

// Dsbmv performs the matrix-vector operation y = alpha*A*x + beta*y for double precision real symmetric band matrix.
// A is an n-by-n symmetric band matrix with k super-diagonals.
func (impl *Standard) Dsbmv(uplo Uplo, n, k int, alpha float64, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta float64, y cudart.DevicePtr, incY int) {
	if impl.e != nil {
		return
	}
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

	impl.e = Check(C.cublasDsbmv(C.cublasHandle_t(impl.h), C.cublasFillMode_t(uplo), C.int(n), C.int(k), (*C.double)(&alpha), (*C.double)(A), C.int(lda), (*C.double)(x), C.int(incX), (*C.double)(&beta), (*C.double)(y), C.int(incY)))
}

// Sspmv performs the matrix-vector operation y = alpha*A*x + beta*y for single precision real symmetric packed matrix.
// A is an n-by-n symmetric matrix stored in packed format.
func (impl *Standard) Sspmv(uplo Uplo, n int, alpha float32, AP cudart.DevicePtr, x cudart.DevicePtr, incX int, beta float32, y cudart.DevicePtr, incY int) {
	if impl.e != nil {
		return
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

	impl.e = Check(C.cublasSspmv(C.cublasHandle_t(impl.h), C.cublasFillMode_t(uplo), C.int(n), (*C.float)(&alpha), (*C.float)(AP), (*C.float)(x), C.int(incX), (*C.float)(&beta), (*C.float)(y), C.int(incY)))
}

// Dspmv performs the matrix-vector operation y = alpha*A*x + beta*y for double precision real symmetric packed matrix.
// A is an n-by-n symmetric matrix stored in packed format.
func (impl *Standard) Dspmv(uplo Uplo, n int, alpha float64, AP cudart.DevicePtr, x cudart.DevicePtr, incX int, beta float64, y cudart.DevicePtr, incY int) {
	if impl.e != nil {
		return
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

	impl.e = Check(C.cublasDspmv(C.cublasHandle_t(impl.h), C.cublasFillMode_t(uplo), C.int(n), (*C.double)(&alpha), (*C.double)(AP), (*C.double)(x), C.int(incX), (*C.double)(&beta), (*C.double)(y), C.int(incY)))
}

// Sspr performs the symmetric rank-1 update AP = alpha*x*x^T + AP for single precision real symmetric packed matrix.
// AP is an n-by-n symmetric matrix stored in packed format.
func (impl *Standard) Sspr(uplo Uplo, n int, alpha float32, x cudart.DevicePtr, incX int, AP cudart.DevicePtr) {
	if impl.e != nil {
		return
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	impl.e = Check(C.cublasSspr(C.cublasHandle_t(impl.h), C.cublasFillMode_t(uplo), C.int(n), (*C.float)(&alpha), (*C.float)(x), C.int(incX), (*C.float)(AP)))
}

// Dspr performs the symmetric rank-1 update AP = alpha*x*x^T + AP for double precision real symmetric packed matrix.
// AP is an n-by-n symmetric matrix stored in packed format.
func (impl *Standard) Dspr(uplo Uplo, n int, alpha float64, x cudart.DevicePtr, incX int, AP cudart.DevicePtr) {
	if impl.e != nil {
		return
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	impl.e = Check(C.cublasDspr(C.cublasHandle_t(impl.h), C.cublasFillMode_t(uplo), C.int(n), (*C.double)(&alpha), (*C.double)(x), C.int(incX), (*C.double)(AP)))
}

// Sspr2 performs the symmetric rank-2 update AP = alpha*x*y^T + alpha*y*x^T + AP for single precision real symmetric packed matrix.
// AP is an n-by-n symmetric matrix stored in packed format.
func (impl *Standard) Sspr2(uplo Uplo, n int, alpha float32, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, AP cudart.DevicePtr) {
	if impl.e != nil {
		return
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

	impl.e = Check(C.cublasSspr2(C.cublasHandle_t(impl.h), C.cublasFillMode_t(uplo), C.int(n), (*C.float)(&alpha), (*C.float)(x), C.int(incX), (*C.float)(y), C.int(incY), (*C.float)(AP)))
}

// Dspr2 performs the symmetric rank-2 update AP = alpha*x*y^T + alpha*y*x^T + AP for double precision real symmetric packed matrix.
// AP is an n-by-n symmetric matrix stored in packed format.
func (impl *Standard) Dspr2(uplo Uplo, n int, alpha float64, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, AP cudart.DevicePtr) {
	if impl.e != nil {
		return
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

	impl.e = Check(C.cublasDspr2(C.cublasHandle_t(impl.h), C.cublasFillMode_t(uplo), C.int(n), (*C.double)(&alpha), (*C.double)(x), C.int(incX), (*C.double)(y), C.int(incY), (*C.double)(AP)))
}

// Ssymv performs the matrix-vector operation y = alpha*A*x + beta*y for single precision real symmetric matrix.
// A is an n-by-n symmetric matrix.
func (impl *Standard) Ssymv(uplo Uplo, n int, alpha float32, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta float32, y cudart.DevicePtr, incY int) {
	if impl.e != nil {
		return
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
	if lda < n {
		panic(blas.ErrShortA)
	}

	impl.e = Check(C.cublasSsymv(C.cublasHandle_t(impl.h), C.cublasFillMode_t(uplo), C.int(n), (*C.float)(&alpha), (*C.float)(A), C.int(lda), (*C.float)(x), C.int(incX), (*C.float)(&beta), (*C.float)(y), C.int(incY)))
}

// Dsymv performs the matrix-vector operation y = alpha*A*x + beta*y for double precision real symmetric matrix.
// A is an n-by-n symmetric matrix.
func (impl *Standard) Dsymv(uplo Uplo, n int, alpha float64, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta float64, y cudart.DevicePtr, incY int) {
	if impl.e != nil {
		return
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
	if lda < n {
		panic(blas.ErrShortA)
	}

	impl.e = Check(C.cublasDsymv(C.cublasHandle_t(impl.h), C.cublasFillMode_t(uplo), C.int(n), (*C.double)(&alpha), (*C.double)(A), C.int(lda), (*C.double)(x), C.int(incX), (*C.double)(&beta), (*C.double)(y), C.int(incY)))
}

// Csymv performs the matrix-vector operation y = alpha*A*x + beta*y for single precision complex symmetric matrix.
// A is an n-by-n symmetric matrix (not Hermitian).
func (impl *Standard) Csymv(uplo Uplo, n int, alpha complex64, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta complex64, y cudart.DevicePtr, incY int) {
	if impl.e != nil {
		return
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
	if lda < n {
		panic(blas.ErrShortA)
	}

	impl.e = Check(C.cublasCsymv(C.cublasHandle_t(impl.h), C.cublasFillMode_t(uplo), C.int(n), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(A), C.int(lda), (*C.cuComplex)(x), C.int(incX), (*C.cuComplex)(unsafe.Pointer(&beta)), (*C.cuComplex)(y), C.int(incY)))
}

// Zsymv performs the matrix-vector operation y = alpha*A*x + beta*y for double precision complex symmetric matrix.
// A is an n-by-n symmetric matrix (not Hermitian).
func (impl *Standard) Zsymv(uplo Uplo, n int, alpha complex128, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta complex128, y cudart.DevicePtr, incY int) {
	if impl.e != nil {
		return
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
	if lda < n {
		panic(blas.ErrShortA)
	}

	impl.e = Check(C.cublasZsymv(C.cublasHandle_t(impl.h), C.cublasFillMode_t(uplo), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(A), C.int(lda), (*C.cuDoubleComplex)(x), C.int(incX), (*C.cuDoubleComplex)(unsafe.Pointer(&beta)), (*C.cuDoubleComplex)(y), C.int(incY)))
}

// Ssyr performs the symmetric rank-1 update A = alpha*x*x^T + A for single precision real symmetric matrix.
// A is an n-by-n symmetric matrix.
func (impl *Standard) Ssyr(uplo Uplo, n int, alpha float32, x cudart.DevicePtr, incX int, A cudart.DevicePtr, lda int) {
	if impl.e != nil {
		return
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if lda < n {
		panic(blas.ErrShortA)
	}

	impl.e = Check(C.cublasSsyr(C.cublasHandle_t(impl.h), C.cublasFillMode_t(uplo), C.int(n), (*C.float)(&alpha), (*C.float)(x), C.int(incX), (*C.float)(A), C.int(lda)))
}

// Dsyr performs the symmetric rank-1 update A = alpha*x*x^T + A for double precision real symmetric matrix.
// A is an n-by-n symmetric matrix.
func (impl *Standard) Dsyr(uplo Uplo, n int, alpha float64, x cudart.DevicePtr, incX int, A cudart.DevicePtr, lda int) {
	if impl.e != nil {
		return
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if lda < n {
		panic(blas.ErrShortA)
	}

	impl.e = Check(C.cublasDsyr(C.cublasHandle_t(impl.h), C.cublasFillMode_t(uplo), C.int(n), (*C.double)(&alpha), (*C.double)(x), C.int(incX), (*C.double)(A), C.int(lda)))
}

// Csyr performs the symmetric rank-1 update A = alpha*x*x^T + A for single precision complex symmetric matrix.
// A is an n-by-n symmetric matrix (not Hermitian).
func (impl *Standard) Csyr(uplo Uplo, n int, alpha complex64, x cudart.DevicePtr, incX int, A cudart.DevicePtr, lda int) {
	if impl.e != nil {
		return
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if lda < n {
		panic(blas.ErrShortA)
	}

	impl.e = Check(C.cublasCsyr(C.cublasHandle_t(impl.h), C.cublasFillMode_t(uplo), C.int(n), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(x), C.int(incX), (*C.cuComplex)(A), C.int(lda)))
}

// Zsyr performs the symmetric rank-1 update A = alpha*x*x^T + A for double precision complex symmetric matrix.
// A is an n-by-n symmetric matrix (not Hermitian).
func (impl *Standard) Zsyr(uplo Uplo, n int, alpha complex128, x cudart.DevicePtr, incX int, A cudart.DevicePtr, lda int) {
	if impl.e != nil {
		return
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if lda < n {
		panic(blas.ErrShortA)
	}

	impl.e = Check(C.cublasZsyr(C.cublasHandle_t(impl.h), C.cublasFillMode_t(uplo), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(x), C.int(incX), (*C.cuDoubleComplex)(A), C.int(lda)))
}

// Ssyr2 performs the symmetric rank-2 update A = alpha*x*y^T + alpha*y*x^T + A for single precision real symmetric matrix.
// A is an n-by-n symmetric matrix.
func (impl *Standard) Ssyr2(uplo Uplo, n int, alpha float32, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, A cudart.DevicePtr, lda int) {
	if impl.e != nil {
		return
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
	if lda < n {
		panic(blas.ErrShortA)
	}

	impl.e = Check(C.cublasSsyr2(C.cublasHandle_t(impl.h), C.cublasFillMode_t(uplo), C.int(n), (*C.float)(&alpha), (*C.float)(x), C.int(incX), (*C.float)(y), C.int(incY), (*C.float)(A), C.int(lda)))
}

// Dsyr2 performs the symmetric rank-2 update A = alpha*x*y^T + alpha*y*x^T + A for double precision real symmetric matrix.
// A is an n-by-n symmetric matrix.
func (impl *Standard) Dsyr2(uplo Uplo, n int, alpha float64, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, A cudart.DevicePtr, lda int) {
	if impl.e != nil {
		return
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
	if lda < n {
		panic(blas.ErrShortA)
	}

	impl.e = Check(C.cublasDsyr2(C.cublasHandle_t(impl.h), C.cublasFillMode_t(uplo), C.int(n), (*C.double)(&alpha), (*C.double)(x), C.int(incX), (*C.double)(y), C.int(incY), (*C.double)(A), C.int(lda)))
}

// Csyr2 performs the symmetric rank-2 update A = alpha*x*y^T + alpha*y*x^T + A for single precision complex symmetric matrix.
// A is an n-by-n symmetric matrix (not Hermitian).
func (impl *Standard) Csyr2(uplo Uplo, n int, alpha complex64, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, A cudart.DevicePtr, lda int) {
	if impl.e != nil {
		return
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
	if lda < n {
		panic(blas.ErrShortA)
	}

	impl.e = Check(C.cublasCsyr2(C.cublasHandle_t(impl.h), C.cublasFillMode_t(uplo), C.int(n), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(x), C.int(incX), (*C.cuComplex)(y), C.int(incY), (*C.cuComplex)(A), C.int(lda)))
}

// Zsyr2 performs the symmetric rank-2 update A = alpha*x*y^T + alpha*y*x^T + A for double precision complex symmetric matrix.
// A is an n-by-n symmetric matrix (not Hermitian).
func (impl *Standard) Zsyr2(uplo Uplo, n int, alpha complex128, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, A cudart.DevicePtr, lda int) {
	if impl.e != nil {
		return
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
	if lda < n {
		panic(blas.ErrShortA)
	}

	impl.e = Check(C.cublasZsyr2(C.cublasHandle_t(impl.h), C.cublasFillMode_t(uplo), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(x), C.int(incX), (*C.cuDoubleComplex)(y), C.int(incY), (*C.cuDoubleComplex)(A), C.int(lda)))
}

// Stbmv performs the matrix-vector operation x = A*x, x = A^T*x for single precision real triangular band matrix.
// A is an n-by-n triangular band matrix with k super-diagonals or sub-diagonals.
func (impl *Standard) Stbmv(uplo Uplo, trans Transpose, diag Diag, n, k int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) {
	if impl.e != nil {
		return
	}
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

	impl.e = Check(C.cublasStbmv(C.cublasHandle_t(impl.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.cublasDiagType_t(diag), C.int(n), C.int(k), (*C.float)(A), C.int(lda), (*C.float)(x), C.int(incX)))
}

// Dtbmv performs the matrix-vector operation x = A*x, x = A^T*x for double precision real triangular band matrix.
// A is an n-by-n triangular band matrix with k super-diagonals or sub-diagonals.
func (impl *Standard) Dtbmv(uplo Uplo, trans Transpose, diag Diag, n, k int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) {
	if impl.e != nil {
		return
	}
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

	impl.e = Check(C.cublasDtbmv(C.cublasHandle_t(impl.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.cublasDiagType_t(diag), C.int(n), C.int(k), (*C.double)(A), C.int(lda), (*C.double)(x), C.int(incX)))
}

// Ctbmv performs the matrix-vector operation x = A*x, x = A^T*x, x = A^H*x for single precision complex triangular band matrix.
// A is an n-by-n triangular band matrix with k super-diagonals or sub-diagonals.
func (impl *Standard) Ctbmv(uplo Uplo, trans Transpose, diag Diag, n, k int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) {
	if impl.e != nil {
		return
	}
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

	impl.e = Check(C.cublasCtbmv(C.cublasHandle_t(impl.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.cublasDiagType_t(diag), C.int(n), C.int(k), (*C.cuComplex)(A), C.int(lda), (*C.cuComplex)(x), C.int(incX)))
}

// Ztbmv performs the matrix-vector operation x = A*x, x = A^T*x, x = A^H*x for double precision complex triangular band matrix.
// A is an n-by-n triangular band matrix with k super-diagonals or sub-diagonals.
func (impl *Standard) Ztbmv(uplo Uplo, trans Transpose, diag Diag, n, k int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) {
	if impl.e != nil {
		return
	}
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

	impl.e = Check(C.cublasZtbmv(C.cublasHandle_t(impl.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.cublasDiagType_t(diag), C.int(n), C.int(k), (*C.cuDoubleComplex)(A), C.int(lda), (*C.cuDoubleComplex)(x), C.int(incX)))
}

// Stbsv solves the triangular band system A*x = b, A^T*x = b for single precision real triangular band matrix.
// A is an n-by-n triangular band matrix with k super-diagonals or sub-diagonals.
// The vector x contains the right-hand side b on input and the solution on output.
func (impl *Standard) Stbsv(uplo Uplo, trans Transpose, diag Diag, n, k int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) {
	if impl.e != nil {
		return
	}
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

	impl.e = Check(C.cublasStbsv(C.cublasHandle_t(impl.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.cublasDiagType_t(diag), C.int(n), C.int(k), (*C.float)(A), C.int(lda), (*C.float)(x), C.int(incX)))
}

// Dtbsv solves the triangular band system A*x = b, A^T*x = b for double precision real triangular band matrix.
// A is an n-by-n triangular band matrix with k super-diagonals or sub-diagonals.
// The vector x contains the right-hand side b on input and the solution on output.
func (impl *Standard) Dtbsv(uplo Uplo, trans Transpose, diag Diag, n, k int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) {
	if impl.e != nil {
		return
	}
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

	impl.e = Check(C.cublasDtbsv(C.cublasHandle_t(impl.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.cublasDiagType_t(diag), C.int(n), C.int(k), (*C.double)(A), C.int(lda), (*C.double)(x), C.int(incX)))
}

// Ctbsv solves the triangular band system A*x = b, A^T*x = b, A^H*x = b for single precision complex triangular band matrix.
// A is an n-by-n triangular band matrix with k super-diagonals or sub-diagonals.
// The vector x contains the right-hand side b on input and the solution on output.
func (impl *Standard) Ctbsv(uplo Uplo, trans Transpose, diag Diag, n, k int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) {
	if impl.e != nil {
		return
	}
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

	impl.e = Check(C.cublasCtbsv(C.cublasHandle_t(impl.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.cublasDiagType_t(diag), C.int(n), C.int(k), (*C.cuComplex)(A), C.int(lda), (*C.cuComplex)(x), C.int(incX)))
}

// Ztbsv solves the triangular band system A*x = b, A^T*x = b, A^H*x = b for double precision complex triangular band matrix.
// A is an n-by-n triangular band matrix with k super-diagonals or sub-diagonals.
// The vector x contains the right-hand side b on input and the solution on output.
func (impl *Standard) Ztbsv(uplo Uplo, trans Transpose, diag Diag, n, k int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) {
	if impl.e != nil {
		return
	}
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

	impl.e = Check(C.cublasZtbsv(C.cublasHandle_t(impl.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.cublasDiagType_t(diag), C.int(n), C.int(k), (*C.cuDoubleComplex)(A), C.int(lda), (*C.cuDoubleComplex)(x), C.int(incX)))
}

// Stpmv performs the matrix-vector operation x = A*x, x = A^T*x for single precision real triangular packed matrix.
// A is an n-by-n triangular matrix stored in packed format.
func (impl *Standard) Stpmv(uplo Uplo, trans Transpose, diag Diag, n int, AP cudart.DevicePtr, x cudart.DevicePtr, incX int) {
	if impl.e != nil {
		return
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	impl.e = Check(C.cublasStpmv(C.cublasHandle_t(impl.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.cublasDiagType_t(diag), C.int(n), (*C.float)(AP), (*C.float)(x), C.int(incX)))
}

// Dtpmv performs the matrix-vector operation x = A*x, x = A^T*x for double precision real triangular packed matrix.
// A is an n-by-n triangular matrix stored in packed format.
func (impl *Standard) Dtpmv(uplo Uplo, trans Transpose, diag Diag, n int, AP cudart.DevicePtr, x cudart.DevicePtr, incX int) {
	if impl.e != nil {
		return
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	impl.e = Check(C.cublasDtpmv(C.cublasHandle_t(impl.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.cublasDiagType_t(diag), C.int(n), (*C.double)(AP), (*C.double)(x), C.int(incX)))
}

// Ctpmv performs the matrix-vector operation x = A*x, x = A^T*x, x = A^H*x for single precision complex triangular packed matrix.
// A is an n-by-n triangular matrix stored in packed format.
func (impl *Standard) Ctpmv(uplo Uplo, trans Transpose, diag Diag, n int, AP cudart.DevicePtr, x cudart.DevicePtr, incX int) {
	if impl.e != nil {
		return
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	impl.e = Check(C.cublasCtpmv(C.cublasHandle_t(impl.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.cublasDiagType_t(diag), C.int(n), (*C.cuComplex)(AP), (*C.cuComplex)(x), C.int(incX)))
}

// Ztpmv performs the matrix-vector operation x = A*x, x = A^T*x, x = A^H*x for double precision complex triangular packed matrix.
// A is an n-by-n triangular matrix stored in packed format.
func (impl *Standard) Ztpmv(uplo Uplo, trans Transpose, diag Diag, n int, AP cudart.DevicePtr, x cudart.DevicePtr, incX int) {
	if impl.e != nil {
		return
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	impl.e = Check(C.cublasZtpmv(C.cublasHandle_t(impl.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.cublasDiagType_t(diag), C.int(n), (*C.cuDoubleComplex)(AP), (*C.cuDoubleComplex)(x), C.int(incX)))
}

// Stpsv solves the triangular packed system A*x = b, A^T*x = b for single precision real triangular packed matrix.
// A is an n-by-n triangular matrix stored in packed format.
// The vector x contains the right-hand side b on input and the solution on output.
func (impl *Standard) Stpsv(uplo Uplo, trans Transpose, diag Diag, n int, AP cudart.DevicePtr, x cudart.DevicePtr, incX int) {
	if impl.e != nil {
		return
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	impl.e = Check(C.cublasStpsv(C.cublasHandle_t(impl.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.cublasDiagType_t(diag), C.int(n), (*C.float)(AP), (*C.float)(x), C.int(incX)))
}

// Dtpsv solves the triangular packed system A*x = b, A^T*x = b for double precision real triangular packed matrix.
// A is an n-by-n triangular matrix stored in packed format.
// The vector x contains the right-hand side b on input and the solution on output.
func (impl *Standard) Dtpsv(uplo Uplo, trans Transpose, diag Diag, n int, AP cudart.DevicePtr, x cudart.DevicePtr, incX int) {
	if impl.e != nil {
		return
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	impl.e = Check(C.cublasDtpsv(C.cublasHandle_t(impl.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.cublasDiagType_t(diag), C.int(n), (*C.double)(AP), (*C.double)(x), C.int(incX)))
}

// Ctpsv solves the triangular packed system A*x = b, A^T*x = b, A^H*x = b for single precision complex triangular packed matrix.
// A is an n-by-n triangular matrix stored in packed format.
// The vector x contains the right-hand side b on input and the solution on output.
func (impl *Standard) Ctpsv(uplo Uplo, trans Transpose, diag Diag, n int, AP cudart.DevicePtr, x cudart.DevicePtr, incX int) {
	if impl.e != nil {
		return
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	impl.e = Check(C.cublasCtpsv(C.cublasHandle_t(impl.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.cublasDiagType_t(diag), C.int(n), (*C.cuComplex)(AP), (*C.cuComplex)(x), C.int(incX)))
}

// Ztpsv solves the triangular packed system A*x = b, A^T*x = b, A^H*x = b for double precision complex triangular packed matrix.
// A is an n-by-n triangular matrix stored in packed format.
// The vector x contains the right-hand side b on input and the solution on output.
func (impl *Standard) Ztpsv(uplo Uplo, trans Transpose, diag Diag, n int, AP cudart.DevicePtr, x cudart.DevicePtr, incX int) {
	if impl.e != nil {
		return
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	impl.e = Check(C.cublasZtpsv(C.cublasHandle_t(impl.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.cublasDiagType_t(diag), C.int(n), (*C.cuDoubleComplex)(AP), (*C.cuDoubleComplex)(x), C.int(incX)))
}

// Strmv performs the matrix-vector operation x = A*x, x = A^T*x for single precision real triangular matrix.
// A is an n-by-n triangular matrix.
func (impl *Standard) Strmv(uplo Uplo, trans Transpose, diag Diag, n int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) {
	if impl.e != nil {
		return
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if lda < n {
		panic(blas.ErrShortA)
	}

	impl.e = Check(C.cublasStrmv(C.cublasHandle_t(impl.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.cublasDiagType_t(diag), C.int(n), (*C.float)(A), C.int(lda), (*C.float)(x), C.int(incX)))
}

// Dtrmv performs the matrix-vector operation x = A*x, x = A^T*x for double precision real triangular matrix.
// A is an n-by-n triangular matrix.
func (impl *Standard) Dtrmv(uplo Uplo, trans Transpose, diag Diag, n int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) {
	if impl.e != nil {
		return
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if lda < n {
		panic(blas.ErrShortA)
	}

	impl.e = Check(C.cublasDtrmv(C.cublasHandle_t(impl.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.cublasDiagType_t(diag), C.int(n), (*C.double)(A), C.int(lda), (*C.double)(x), C.int(incX)))
}

// Ctrmv performs the matrix-vector operation x = A*x, x = A^T*x, x = A^H*x for single precision complex triangular matrix.
// A is an n-by-n triangular matrix.
func (impl *Standard) Ctrmv(uplo Uplo, trans Transpose, diag Diag, n int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) {
	if impl.e != nil {
		return
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if lda < n {
		panic(blas.ErrShortA)
	}

	impl.e = Check(C.cublasCtrmv(C.cublasHandle_t(impl.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.cublasDiagType_t(diag), C.int(n), (*C.cuComplex)(A), C.int(lda), (*C.cuComplex)(x), C.int(incX)))
}

// Ztrmv performs the matrix-vector operation x = A*x, x = A^T*x, x = A^H*x for double precision complex triangular matrix.
// A is an n-by-n triangular matrix.
func (impl *Standard) Ztrmv(uplo Uplo, trans Transpose, diag Diag, n int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) {
	if impl.e != nil {
		return
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if lda < n {
		panic(blas.ErrShortA)
	}

	impl.e = Check(C.cublasZtrmv(C.cublasHandle_t(impl.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.cublasDiagType_t(diag), C.int(n), (*C.cuDoubleComplex)(A), C.int(lda), (*C.cuDoubleComplex)(x), C.int(incX)))
}

// Strsv solves the triangular system A*x = b, A^T*x = b for single precision real triangular matrix.
// A is an n-by-n triangular matrix.
// The vector x contains the right-hand side b on input and the solution on output.
func (impl *Standard) Strsv(uplo Uplo, trans Transpose, diag Diag, n int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) {
	if impl.e != nil {
		return
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if lda < n {
		panic(blas.ErrShortA)
	}

	impl.e = Check(C.cublasStrsv(C.cublasHandle_t(impl.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.cublasDiagType_t(diag), C.int(n), (*C.float)(A), C.int(lda), (*C.float)(x), C.int(incX)))
}

// Dtrsv solves the triangular system A*x = b, A^T*x = b for double precision real triangular matrix.
// A is an n-by-n triangular matrix.
// The vector x contains the right-hand side b on input and the solution on output.
func (impl *Standard) Dtrsv(uplo Uplo, trans Transpose, diag Diag, n int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) {
	if impl.e != nil {
		return
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if lda < n {
		panic(blas.ErrShortA)
	}

	impl.e = Check(C.cublasDtrsv(C.cublasHandle_t(impl.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.cublasDiagType_t(diag), C.int(n), (*C.double)(A), C.int(lda), (*C.double)(x), C.int(incX)))
}

// Ctrsv solves the triangular system A*x = b, A^T*x = b, A^H*x = b for single precision complex triangular matrix.
// A is an n-by-n triangular matrix.
// The vector x contains the right-hand side b on input and the solution on output.
func (impl *Standard) Ctrsv(uplo Uplo, trans Transpose, diag Diag, n int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) {
	if impl.e != nil {
		return
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if lda < n {
		panic(blas.ErrShortA)
	}

	impl.e = Check(C.cublasCtrsv(C.cublasHandle_t(impl.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.cublasDiagType_t(diag), C.int(n), (*C.cuComplex)(A), C.int(lda), (*C.cuComplex)(x), C.int(incX)))
}

// Ztrsv solves the triangular system A*x = b, A^T*x = b, A^H*x = b for double precision complex triangular matrix.
// A is an n-by-n triangular matrix.
// The vector x contains the right-hand side b on input and the solution on output.
func (impl *Standard) Ztrsv(uplo Uplo, trans Transpose, diag Diag, n int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) {
	if impl.e != nil {
		return
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if lda < n {
		panic(blas.ErrShortA)
	}

	impl.e = Check(C.cublasZtrsv(C.cublasHandle_t(impl.h), C.cublasFillMode_t(uplo), C.cublasOperation_t(trans), C.cublasDiagType_t(diag), C.int(n), (*C.cuDoubleComplex)(A), C.int(lda), (*C.cuDoubleComplex)(x), C.int(incX)))
}

// Chemv performs the matrix-vector operation y = alpha*A*x + beta*y for single precision complex Hermitian matrix.
// A is an n-by-n Hermitian matrix (A = A^H).
func (impl *Standard) Chemv(uplo Uplo, n int, alpha complex64, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta complex64, y cudart.DevicePtr, incY int) {
	if impl.e != nil {
		return
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
	if lda < n {
		panic(blas.ErrShortA)
	}

	impl.e = Check(C.cublasChemv(C.cublasHandle_t(impl.h), C.cublasFillMode_t(uplo), C.int(n), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(A), C.int(lda), (*C.cuComplex)(x), C.int(incX), (*C.cuComplex)(unsafe.Pointer(&beta)), (*C.cuComplex)(y), C.int(incY)))
}

// Zhemv performs the matrix-vector operation y = alpha*A*x + beta*y for double precision complex Hermitian matrix.
// A is an n-by-n Hermitian matrix (A = A^H).
func (impl *Standard) Zhemv(uplo Uplo, n int, alpha complex128, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta complex128, y cudart.DevicePtr, incY int) {
	if impl.e != nil {
		return
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
	if lda < n {
		panic(blas.ErrShortA)
	}

	impl.e = Check(C.cublasZhemv(C.cublasHandle_t(impl.h), C.cublasFillMode_t(uplo), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(A), C.int(lda), (*C.cuDoubleComplex)(x), C.int(incX), (*C.cuDoubleComplex)(unsafe.Pointer(&beta)), (*C.cuDoubleComplex)(y), C.int(incY)))
}

// Chbmv performs the matrix-vector operation y = alpha*A*x + beta*y for single precision complex Hermitian band matrix.
// A is an n-by-n Hermitian band matrix with k super-diagonals.
func (impl *Standard) Chbmv(uplo Uplo, n, k int, alpha complex64, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta complex64, y cudart.DevicePtr, incY int) {
	if impl.e != nil {
		return
	}
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

	impl.e = Check(C.cublasChbmv(C.cublasHandle_t(impl.h), C.cublasFillMode_t(uplo), C.int(n), C.int(k), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(A), C.int(lda), (*C.cuComplex)(x), C.int(incX), (*C.cuComplex)(unsafe.Pointer(&beta)), (*C.cuComplex)(y), C.int(incY)))
}

// Zhbmv performs the matrix-vector operation y = alpha*A*x + beta*y for double precision complex Hermitian band matrix.
// A is an n-by-n Hermitian band matrix with k super-diagonals.
func (impl *Standard) Zhbmv(uplo Uplo, n, k int, alpha complex128, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta complex128, y cudart.DevicePtr, incY int) {
	if impl.e != nil {
		return
	}
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

	impl.e = Check(C.cublasZhbmv(C.cublasHandle_t(impl.h), C.cublasFillMode_t(uplo), C.int(n), C.int(k), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(A), C.int(lda), (*C.cuDoubleComplex)(x), C.int(incX), (*C.cuDoubleComplex)(unsafe.Pointer(&beta)), (*C.cuDoubleComplex)(y), C.int(incY)))
}

// Chpmv performs the matrix-vector operation y = alpha*A*x + beta*y for single precision complex Hermitian packed matrix.
// A is an n-by-n Hermitian matrix stored in packed format.
func (impl *Standard) Chpmv(uplo Uplo, n int, alpha complex64, AP cudart.DevicePtr, x cudart.DevicePtr, incX int, beta complex64, y cudart.DevicePtr, incY int) {
	if impl.e != nil {
		return
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

	impl.e = Check(C.cublasChpmv(C.cublasHandle_t(impl.h), C.cublasFillMode_t(uplo), C.int(n), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(AP), (*C.cuComplex)(x), C.int(incX), (*C.cuComplex)(unsafe.Pointer(&beta)), (*C.cuComplex)(y), C.int(incY)))
}

// Zhpmv performs the matrix-vector operation y = alpha*A*x + beta*y for double precision complex Hermitian packed matrix.
// A is an n-by-n Hermitian matrix stored in packed format.
func (impl *Standard) Zhpmv(uplo Uplo, n int, alpha complex128, AP cudart.DevicePtr, x cudart.DevicePtr, incX int, beta complex128, y cudart.DevicePtr, incY int) {
	if impl.e != nil {
		return
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

	impl.e = Check(C.cublasZhpmv(C.cublasHandle_t(impl.h), C.cublasFillMode_t(uplo), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(AP), (*C.cuDoubleComplex)(x), C.int(incX), (*C.cuDoubleComplex)(unsafe.Pointer(&beta)), (*C.cuDoubleComplex)(y), C.int(incY)))
}

// Cher performs the Hermitian rank-1 update A = alpha*x*x^H + A for single precision complex Hermitian matrix.
// A is an n-by-n Hermitian matrix. Note: alpha must be real for Hermitian matrices.
func (impl *Standard) Cher(uplo Uplo, n int, alpha float32, x cudart.DevicePtr, incX int, A cudart.DevicePtr, lda int) {
	if impl.e != nil {
		return
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if lda < n {
		panic(blas.ErrShortA)
	}

	impl.e = Check(C.cublasCher(C.cublasHandle_t(impl.h), C.cublasFillMode_t(uplo), C.int(n), (*C.float)(&alpha), (*C.cuComplex)(x), C.int(incX), (*C.cuComplex)(A), C.int(lda)))
}

// Zher performs the Hermitian rank-1 update A = alpha*x*x^H + A for double precision complex Hermitian matrix.
// A is an n-by-n Hermitian matrix. Note: alpha must be real for Hermitian matrices.
func (impl *Standard) Zher(uplo Uplo, n int, alpha float64, x cudart.DevicePtr, incX int, A cudart.DevicePtr, lda int) {
	if impl.e != nil {
		return
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if lda < n {
		panic(blas.ErrShortA)
	}

	impl.e = Check(C.cublasZher(C.cublasHandle_t(impl.h), C.cublasFillMode_t(uplo), C.int(n), (*C.double)(&alpha), (*C.cuDoubleComplex)(x), C.int(incX), (*C.cuDoubleComplex)(A), C.int(lda)))
}

// Cher2 performs the Hermitian rank-2 update A = alpha*x*y^H + conj(alpha)*y*x^H + A for single precision complex Hermitian matrix.
// A is an n-by-n Hermitian matrix.
func (impl *Standard) Cher2(uplo Uplo, n int, alpha complex64, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, A cudart.DevicePtr, lda int) {
	if impl.e != nil {
		return
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
	if lda < n {
		panic(blas.ErrShortA)
	}

	impl.e = Check(C.cublasCher2(C.cublasHandle_t(impl.h), C.cublasFillMode_t(uplo), C.int(n), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(x), C.int(incX), (*C.cuComplex)(y), C.int(incY), (*C.cuComplex)(A), C.int(lda)))
}

// Zher2 performs the Hermitian rank-2 update A = alpha*x*y^H + conj(alpha)*y*x^H + A for double precision complex Hermitian matrix.
// A is an n-by-n Hermitian matrix.
func (impl *Standard) Zher2(uplo Uplo, n int, alpha complex128, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, A cudart.DevicePtr, lda int) {
	if impl.e != nil {
		return
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
	if lda < n {
		panic(blas.ErrShortA)
	}

	impl.e = Check(C.cublasZher2(C.cublasHandle_t(impl.h), C.cublasFillMode_t(uplo), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(x), C.int(incX), (*C.cuDoubleComplex)(y), C.int(incY), (*C.cuDoubleComplex)(A), C.int(lda)))
}

// Chpr performs the Hermitian rank-1 update AP = alpha*x*x^H + AP for single precision complex Hermitian packed matrix.
// AP is an n-by-n Hermitian matrix stored in packed format. Note: alpha must be real for Hermitian matrices.
func (impl *Standard) Chpr(uplo Uplo, n int, alpha float32, x cudart.DevicePtr, incX int, AP cudart.DevicePtr) {
	if impl.e != nil {
		return
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	impl.e = Check(C.cublasChpr(C.cublasHandle_t(impl.h), C.cublasFillMode_t(uplo), C.int(n), (*C.float)(&alpha), (*C.cuComplex)(x), C.int(incX), (*C.cuComplex)(AP)))
}

// Zhpr performs the Hermitian rank-1 update AP = alpha*x*x^H + AP for double precision complex Hermitian packed matrix.
// AP is an n-by-n Hermitian matrix stored in packed format. Note: alpha must be real for Hermitian matrices.
func (impl *Standard) Zhpr(uplo Uplo, n int, alpha float64, x cudart.DevicePtr, incX int, AP cudart.DevicePtr) {
	if impl.e != nil {
		return
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	impl.e = Check(C.cublasZhpr(C.cublasHandle_t(impl.h), C.cublasFillMode_t(uplo), C.int(n), (*C.double)(&alpha), (*C.cuDoubleComplex)(x), C.int(incX), (*C.cuDoubleComplex)(AP)))
}

// Chpr2 performs the Hermitian rank-2 update AP = alpha*x*y^H + conj(alpha)*y*x^H + AP for single precision complex Hermitian packed matrix.
// AP is an n-by-n Hermitian matrix stored in packed format.
func (impl *Standard) Chpr2(uplo Uplo, n int, alpha complex64, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, AP cudart.DevicePtr) {
	if impl.e != nil {
		return
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

	impl.e = Check(C.cublasChpr2(C.cublasHandle_t(impl.h), C.cublasFillMode_t(uplo), C.int(n), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(x), C.int(incX), (*C.cuComplex)(y), C.int(incY), (*C.cuComplex)(AP)))
}

// Zhpr2 performs the Hermitian rank-2 update AP = alpha*x*y^H + conj(alpha)*y*x^H + AP for double precision complex Hermitian packed matrix.
// AP is an n-by-n Hermitian matrix stored in packed format.
func (impl *Standard) Zhpr2(uplo Uplo, n int, alpha complex128, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, AP cudart.DevicePtr) {
	if impl.e != nil {
		return
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

	impl.e = Check(C.cublasZhpr2(C.cublasHandle_t(impl.h), C.cublasFillMode_t(uplo), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(x), C.int(incX), (*C.cuDoubleComplex)(y), C.int(incY), (*C.cuDoubleComplex)(AP)))
}
