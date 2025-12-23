//go:build !cuda && !cuda11 && !cuda12 && !cuda13

package cublas

import (
	"github.com/gocnn/gocu/cudart"
)

// Level 3 BLAS - General matrix-matrix operations
func Sgemm(h *Handle, tA, tB Transpose, m, n, k int, alpha float32, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta float32, c cudart.DevicePtr, ldc int) error {
	return ErrNotAvailable
}
func Dgemm(h *Handle, tA, tB Transpose, m, n, k int, alpha float64, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta float64, c cudart.DevicePtr, ldc int) error {
	return ErrNotAvailable
}
func Cgemm(h *Handle, tA, tB Transpose, m, n, k int, alpha complex64, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta complex64, c cudart.DevicePtr, ldc int) error {
	return ErrNotAvailable
}
func Zgemm(h *Handle, tA, tB Transpose, m, n, k int, alpha complex128, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta complex128, c cudart.DevicePtr, ldc int) error {
	return ErrNotAvailable
}

// Level 3 BLAS - 3M algorithm variants
func Cgemm3m(h *Handle, tA, tB Transpose, m, n, k int, alpha complex64, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta complex64, c cudart.DevicePtr, ldc int) error {
	return ErrNotAvailable
}
func Zgemm3m(h *Handle, tA, tB Transpose, m, n, k int, alpha complex128, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta complex128, c cudart.DevicePtr, ldc int) error {
	return ErrNotAvailable
}

// Level 3 BLAS - Symmetric matrix-matrix operations
func Ssymm(h *Handle, side Side, uplo Uplo, m, n int, alpha float32, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta float32, c cudart.DevicePtr, ldc int) error {
	return ErrNotAvailable
}
func Dsymm(h *Handle, side Side, uplo Uplo, m, n int, alpha float64, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta float64, c cudart.DevicePtr, ldc int) error {
	return ErrNotAvailable
}
func Csymm(h *Handle, side Side, uplo Uplo, m, n int, alpha complex64, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta complex64, c cudart.DevicePtr, ldc int) error {
	return ErrNotAvailable
}
func Zsymm(h *Handle, side Side, uplo Uplo, m, n int, alpha complex128, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta complex128, c cudart.DevicePtr, ldc int) error {
	return ErrNotAvailable
}

// Level 3 BLAS - Symmetric rank-k update operations
func Ssyrk(h *Handle, uplo Uplo, trans Transpose, n, k int, alpha float32, a cudart.DevicePtr, lda int, beta float32, c cudart.DevicePtr, ldc int) error {
	return ErrNotAvailable
}
func Dsyrk(h *Handle, uplo Uplo, trans Transpose, n, k int, alpha float64, a cudart.DevicePtr, lda int, beta float64, c cudart.DevicePtr, ldc int) error {
	return ErrNotAvailable
}
func Csyrk(h *Handle, uplo Uplo, trans Transpose, n, k int, alpha complex64, a cudart.DevicePtr, lda int, beta complex64, c cudart.DevicePtr, ldc int) error {
	return ErrNotAvailable
}
func Zsyrk(h *Handle, uplo Uplo, trans Transpose, n, k int, alpha complex128, a cudart.DevicePtr, lda int, beta complex128, c cudart.DevicePtr, ldc int) error {
	return ErrNotAvailable
}

// Level 3 BLAS - Symmetric rank-2k update operations
func Ssyr2k(h *Handle, uplo Uplo, trans Transpose, n, k int, alpha float32, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta float32, c cudart.DevicePtr, ldc int) error {
	return ErrNotAvailable
}
func Dsyr2k(h *Handle, uplo Uplo, trans Transpose, n, k int, alpha float64, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta float64, c cudart.DevicePtr, ldc int) error {
	return ErrNotAvailable
}
func Csyr2k(h *Handle, uplo Uplo, trans Transpose, n, k int, alpha complex64, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta complex64, c cudart.DevicePtr, ldc int) error {
	return ErrNotAvailable
}
func Zsyr2k(h *Handle, uplo Uplo, trans Transpose, n, k int, alpha complex128, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta complex128, c cudart.DevicePtr, ldc int) error {
	return ErrNotAvailable
}

// Level 3 BLAS - Symmetric rank-kx update operations
func Ssyrkx(h *Handle, uplo Uplo, trans Transpose, n, k int, alpha float32, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta float32, c cudart.DevicePtr, ldc int) error {
	return ErrNotAvailable
}
func Dsyrkx(h *Handle, uplo Uplo, trans Transpose, n, k int, alpha float64, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta float64, c cudart.DevicePtr, ldc int) error {
	return ErrNotAvailable
}
func Csyrkx(h *Handle, uplo Uplo, trans Transpose, n, k int, alpha complex64, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta complex64, c cudart.DevicePtr, ldc int) error {
	return ErrNotAvailable
}
func Zsyrkx(h *Handle, uplo Uplo, trans Transpose, n, k int, alpha complex128, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta complex128, c cudart.DevicePtr, ldc int) error {
	return ErrNotAvailable
}

// Level 3 BLAS - Triangular matrix-matrix operations
func Strmm(h *Handle, side Side, uplo Uplo, trans Transpose, diag Diag, m, n int, alpha float32, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, c cudart.DevicePtr, ldc int) error {
	return ErrNotAvailable
}
func Dtrmm(h *Handle, side Side, uplo Uplo, trans Transpose, diag Diag, m, n int, alpha float64, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, c cudart.DevicePtr, ldc int) error {
	return ErrNotAvailable
}
func Ctrmm(h *Handle, side Side, uplo Uplo, trans Transpose, diag Diag, m, n int, alpha complex64, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, c cudart.DevicePtr, ldc int) error {
	return ErrNotAvailable
}
func Ztrmm(h *Handle, side Side, uplo Uplo, trans Transpose, diag Diag, m, n int, alpha complex128, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, c cudart.DevicePtr, ldc int) error {
	return ErrNotAvailable
}

// Level 3 BLAS - Triangular solve operations
func Strsm(h *Handle, side Side, uplo Uplo, trans Transpose, diag Diag, m, n int, alpha float32, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int) error {
	return ErrNotAvailable
}
func Dtrsm(h *Handle, side Side, uplo Uplo, trans Transpose, diag Diag, m, n int, alpha float64, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int) error {
	return ErrNotAvailable
}
func Ctrsm(h *Handle, side Side, uplo Uplo, trans Transpose, diag Diag, m, n int, alpha complex64, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int) error {
	return ErrNotAvailable
}
func Ztrsm(h *Handle, side Side, uplo Uplo, trans Transpose, diag Diag, m, n int, alpha complex128, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int) error {
	return ErrNotAvailable
}

// Level 3 BLAS - Hermitian matrix-matrix operations
func Chemm(h *Handle, side Side, uplo Uplo, m, n int, alpha complex64, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta complex64, c cudart.DevicePtr, ldc int) error {
	return ErrNotAvailable
}
func Zhemm(h *Handle, side Side, uplo Uplo, m, n int, alpha complex128, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta complex128, c cudart.DevicePtr, ldc int) error {
	return ErrNotAvailable
}

// Level 3 BLAS - Hermitian rank-k update operations
func Cherk(h *Handle, uplo Uplo, trans Transpose, n, k int, alpha float32, a cudart.DevicePtr, lda int, beta float32, c cudart.DevicePtr, ldc int) error {
	return ErrNotAvailable
}
func Zherk(h *Handle, uplo Uplo, trans Transpose, n, k int, alpha float64, a cudart.DevicePtr, lda int, beta float64, c cudart.DevicePtr, ldc int) error {
	return ErrNotAvailable
}

// Level 3 BLAS - Hermitian rank-2k update operations
func Cher2k(h *Handle, uplo Uplo, trans Transpose, n, k int, alpha complex64, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta float32, c cudart.DevicePtr, ldc int) error {
	return ErrNotAvailable
}
func Zher2k(h *Handle, uplo Uplo, trans Transpose, n, k int, alpha complex128, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta float64, c cudart.DevicePtr, ldc int) error {
	return ErrNotAvailable
}

// Level 3 BLAS - Hermitian rank-kx update operations
func Cherkx(h *Handle, uplo Uplo, trans Transpose, n, k int, alpha complex64, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta float32, c cudart.DevicePtr, ldc int) error {
	return ErrNotAvailable
}
func Zherkx(h *Handle, uplo Uplo, trans Transpose, n, k int, alpha complex128, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta float64, c cudart.DevicePtr, ldc int) error {
	return ErrNotAvailable
}
