package cublas

// #include <cublas_v2.h>
import "C"
import (
	"unsafe"

	"github.com/gocnn/gomat/blas"
)

// Sgbmv performs the matrix-vector operation y = alpha*A*x + beta*y for single precision real band matrix.
// A is an m-by-n band matrix with kl sub-diagonals and ku super-diagonals.
// The matrix A is stored in band storage format.
func (impl *Standard) Sgbmv(trans blas.Transpose, m, n, kl, ku int, alpha float32, A []float32, lda int, x []float32, incX int, beta float32, y []float32, incY int) {
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
		panic("kl must be non-negative")
	}
	if ku < 0 {
		panic("ku must be non-negative")
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

	// Check vector lengths based on transpose
	var xLen, yLen int
	if trans == blas.NoTrans {
		xLen, yLen = n, m
	} else {
		xLen, yLen = m, n
	}

	if len(x) == 0 || len(y) == 0 || len(A) == 0 {
		return
	}
	if (incX > 0 && (xLen-1)*incX >= len(x)) || (incX < 0 && (1-xLen)*incX >= len(x)) {
		panic(blas.ErrShortX)
	}
	if (incY > 0 && (yLen-1)*incY >= len(y)) || (incY < 0 && (1-yLen)*incY >= len(y)) {
		panic(blas.ErrShortY)
	}
	if lda*(n-1)+kl+ku+1 > len(A) {
		panic(blas.ErrShortA)
	}

	impl.e = Check(C.cublasSgbmv(C.cublasHandle_t(impl.h), C.cublasOperation_t(trans), C.int(m), C.int(n), C.int(kl), C.int(ku), (*C.float)(&alpha), (*C.float)(&A[0]), C.int(lda), (*C.float)(&x[0]), C.int(incX), (*C.float)(&beta), (*C.float)(&y[0]), C.int(incY)))
}

// Dgbmv performs the matrix-vector operation y = alpha*A*x + beta*y for double precision real band matrix.
// A is an m-by-n band matrix with kl sub-diagonals and ku super-diagonals.
// The matrix A is stored in band storage format.
func (impl *Standard) Dgbmv(trans blas.Transpose, m, n, kl, ku int, alpha float64, A []float64, lda int, x []float64, incX int, beta float64, y []float64, incY int) {
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
		panic("kl must be non-negative")
	}
	if ku < 0 {
		panic("ku must be non-negative")
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

	// Check vector lengths based on transpose
	var xLen, yLen int
	if trans == blas.NoTrans {
		xLen, yLen = n, m
	} else {
		xLen, yLen = m, n
	}

	if len(x) == 0 || len(y) == 0 || len(A) == 0 {
		return
	}
	if (incX > 0 && (xLen-1)*incX >= len(x)) || (incX < 0 && (1-xLen)*incX >= len(x)) {
		panic(blas.ErrShortX)
	}
	if (incY > 0 && (yLen-1)*incY >= len(y)) || (incY < 0 && (1-yLen)*incY >= len(y)) {
		panic(blas.ErrShortY)
	}
	if lda*(n-1)+kl+ku+1 > len(A) {
		panic(blas.ErrShortA)
	}

	impl.e = Check(C.cublasDgbmv(C.cublasHandle_t(impl.h), C.cublasOperation_t(trans), C.int(m), C.int(n), C.int(kl), C.int(ku), (*C.double)(&alpha), (*C.double)(&A[0]), C.int(lda), (*C.double)(&x[0]), C.int(incX), (*C.double)(&beta), (*C.double)(&y[0]), C.int(incY)))
}

// Cgbmv performs the matrix-vector operation y = alpha*A*x + beta*y for single precision complex band matrix.
// A is an m-by-n band matrix with kl sub-diagonals and ku super-diagonals.
// The matrix A is stored in band storage format.
func (impl *Standard) Cgbmv(trans blas.Transpose, m, n, kl, ku int, alpha complex64, A []complex64, lda int, x []complex64, incX int, beta complex64, y []complex64, incY int) {
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
		panic("kl must be non-negative")
	}
	if ku < 0 {
		panic("ku must be non-negative")
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

	// Check vector lengths based on transpose
	var xLen, yLen int
	if trans == blas.NoTrans {
		xLen, yLen = n, m
	} else {
		xLen, yLen = m, n
	}

	if len(x) == 0 || len(y) == 0 || len(A) == 0 {
		return
	}
	if (incX > 0 && (xLen-1)*incX >= len(x)) || (incX < 0 && (1-xLen)*incX >= len(x)) {
		panic(blas.ErrShortX)
	}
	if (incY > 0 && (yLen-1)*incY >= len(y)) || (incY < 0 && (1-yLen)*incY >= len(y)) {
		panic(blas.ErrShortY)
	}
	if lda*(n-1)+kl+ku+1 > len(A) {
		panic(blas.ErrShortA)
	}

	impl.e = Check(C.cublasCgbmv(C.cublasHandle_t(impl.h), C.cublasOperation_t(trans), C.int(m), C.int(n), C.int(kl), C.int(ku), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(unsafe.Pointer(&A[0])), C.int(lda), (*C.cuComplex)(unsafe.Pointer(&x[0])), C.int(incX), (*C.cuComplex)(unsafe.Pointer(&beta)), (*C.cuComplex)(unsafe.Pointer(&y[0])), C.int(incY)))
}

// Zgbmv performs the matrix-vector operation y = alpha*A*x + beta*y for double precision complex band matrix.
// A is an m-by-n band matrix with kl sub-diagonals and ku super-diagonals.
// The matrix A is stored in band storage format.
func (impl *Standard) Zgbmv(trans blas.Transpose, m, n, kl, ku int, alpha complex128, A []complex128, lda int, x []complex128, incX int, beta complex128, y []complex128, incY int) {
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
		panic("kl must be non-negative")
	}
	if ku < 0 {
		panic("ku must be non-negative")
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

	// Check vector lengths based on transpose
	var xLen, yLen int
	if trans == blas.NoTrans {
		xLen, yLen = n, m
	} else {
		xLen, yLen = m, n
	}

	if len(x) == 0 || len(y) == 0 || len(A) == 0 {
		return
	}
	if (incX > 0 && (xLen-1)*incX >= len(x)) || (incX < 0 && (1-xLen)*incX >= len(x)) {
		panic(blas.ErrShortX)
	}
	if (incY > 0 && (yLen-1)*incY >= len(y)) || (incY < 0 && (1-yLen)*incY >= len(y)) {
		panic(blas.ErrShortY)
	}
	if lda*(n-1)+kl+ku+1 > len(A) {
		panic(blas.ErrShortA)
	}

	impl.e = Check(C.cublasZgbmv(C.cublasHandle_t(impl.h), C.cublasOperation_t(trans), C.int(m), C.int(n), C.int(kl), C.int(ku), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(unsafe.Pointer(&A[0])), C.int(lda), (*C.cuDoubleComplex)(unsafe.Pointer(&x[0])), C.int(incX), (*C.cuDoubleComplex)(unsafe.Pointer(&beta)), (*C.cuDoubleComplex)(unsafe.Pointer(&y[0])), C.int(incY)))
}
