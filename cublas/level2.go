//go:build !cuda && !cuda11 && !cuda12 && !cuda13

package cublas

import (
	"github.com/gocnn/gocu/cudart"
)

// Level 2 BLAS - Band matrix-vector operations
func Sgbmv(h *Handle, trans Transpose, m, n, kl, ku int, alpha float32, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta float32, y cudart.DevicePtr, incY int) error {
	return ErrNotAvailable
}
func Dgbmv(h *Handle, trans Transpose, m, n, kl, ku int, alpha float64, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta float64, y cudart.DevicePtr, incY int) error {
	return ErrNotAvailable
}
func Cgbmv(h *Handle, trans Transpose, m, n, kl, ku int, alpha complex64, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta complex64, y cudart.DevicePtr, incY int) error {
	return ErrNotAvailable
}
func Zgbmv(h *Handle, trans Transpose, m, n, kl, ku int, alpha complex128, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta complex128, y cudart.DevicePtr, incY int) error {
	return ErrNotAvailable
}

// Level 2 BLAS - General matrix-vector operations
func Sgemv(h *Handle, trans Transpose, m, n int, alpha float32, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta float32, y cudart.DevicePtr, incY int) error {
	return ErrNotAvailable
}
func Dgemv(h *Handle, trans Transpose, m, n int, alpha float64, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta float64, y cudart.DevicePtr, incY int) error {
	return ErrNotAvailable
}
func Cgemv(h *Handle, trans Transpose, m, n int, alpha complex64, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta complex64, y cudart.DevicePtr, incY int) error {
	return ErrNotAvailable
}
func Zgemv(h *Handle, trans Transpose, m, n int, alpha complex128, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta complex128, y cudart.DevicePtr, incY int) error {
	return ErrNotAvailable
}

// Level 2 BLAS - Rank-1 update operations
func Sger(h *Handle, m, n int, alpha float32, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, A cudart.DevicePtr, lda int) error {
	return ErrNotAvailable
}
func Dger(h *Handle, m, n int, alpha float64, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, A cudart.DevicePtr, lda int) error {
	return ErrNotAvailable
}
func Cgeru(h *Handle, m, n int, alpha complex64, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, A cudart.DevicePtr, lda int) error {
	return ErrNotAvailable
}
func Cgerc(h *Handle, m, n int, alpha complex64, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, A cudart.DevicePtr, lda int) error {
	return ErrNotAvailable
}
func Zgeru(h *Handle, m, n int, alpha complex128, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, A cudart.DevicePtr, lda int) error {
	return ErrNotAvailable
}
func Zgerc(h *Handle, m, n int, alpha complex128, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, A cudart.DevicePtr, lda int) error {
	return ErrNotAvailable
}

// Level 2 BLAS - Symmetric band matrix-vector operations
func Ssbmv(h *Handle, uplo Uplo, n, k int, alpha float32, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta float32, y cudart.DevicePtr, incY int) error {
	return ErrNotAvailable
}
func Dsbmv(h *Handle, uplo Uplo, n, k int, alpha float64, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta float64, y cudart.DevicePtr, incY int) error {
	return ErrNotAvailable
}

// Level 2 BLAS - Symmetric packed matrix-vector operations
func Sspmv(h *Handle, uplo Uplo, n int, alpha float32, AP cudart.DevicePtr, x cudart.DevicePtr, incX int, beta float32, y cudart.DevicePtr, incY int) error {
	return ErrNotAvailable
}
func Dspmv(h *Handle, uplo Uplo, n int, alpha float64, AP cudart.DevicePtr, x cudart.DevicePtr, incX int, beta float64, y cudart.DevicePtr, incY int) error {
	return ErrNotAvailable
}

// Level 2 BLAS - Symmetric rank-1 update operations
func Sspr(h *Handle, uplo Uplo, n int, alpha float32, x cudart.DevicePtr, incX int, AP cudart.DevicePtr) error {
	return ErrNotAvailable
}
func Dspr(h *Handle, uplo Uplo, n int, alpha float64, x cudart.DevicePtr, incX int, AP cudart.DevicePtr) error {
	return ErrNotAvailable
}

// Level 2 BLAS - Symmetric rank-2 update operations
func Sspr2(h *Handle, uplo Uplo, n int, alpha float32, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, AP cudart.DevicePtr) error {
	return ErrNotAvailable
}
func Dspr2(h *Handle, uplo Uplo, n int, alpha float64, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, AP cudart.DevicePtr) error {
	return ErrNotAvailable
}

// Level 2 BLAS - Symmetric matrix-vector operations
func Ssymv(h *Handle, uplo Uplo, n int, alpha float32, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta float32, y cudart.DevicePtr, incY int) error {
	return ErrNotAvailable
}
func Dsymv(h *Handle, uplo Uplo, n int, alpha float64, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta float64, y cudart.DevicePtr, incY int) error {
	return ErrNotAvailable
}
func Csymv(h *Handle, uplo Uplo, n int, alpha complex64, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta complex64, y cudart.DevicePtr, incY int) error {
	return ErrNotAvailable
}
func Zsymv(h *Handle, uplo Uplo, n int, alpha complex128, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta complex128, y cudart.DevicePtr, incY int) error {
	return ErrNotAvailable
}

// Level 2 BLAS - Symmetric rank-1 update operations
func Ssyr(h *Handle, uplo Uplo, n int, alpha float32, x cudart.DevicePtr, incX int, A cudart.DevicePtr, lda int) error {
	return ErrNotAvailable
}
func Dsyr(h *Handle, uplo Uplo, n int, alpha float64, x cudart.DevicePtr, incX int, A cudart.DevicePtr, lda int) error {
	return ErrNotAvailable
}
func Csyr(h *Handle, uplo Uplo, n int, alpha complex64, x cudart.DevicePtr, incX int, A cudart.DevicePtr, lda int) error {
	return ErrNotAvailable
}
func Zsyr(h *Handle, uplo Uplo, n int, alpha complex128, x cudart.DevicePtr, incX int, A cudart.DevicePtr, lda int) error {
	return ErrNotAvailable
}

// Level 2 BLAS - Symmetric rank-2 update operations
func Ssyr2(h *Handle, uplo Uplo, n int, alpha float32, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, A cudart.DevicePtr, lda int) error {
	return ErrNotAvailable
}
func Dsyr2(h *Handle, uplo Uplo, n int, alpha float64, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, A cudart.DevicePtr, lda int) error {
	return ErrNotAvailable
}
func Csyr2(h *Handle, uplo Uplo, n int, alpha complex64, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, A cudart.DevicePtr, lda int) error {
	return ErrNotAvailable
}
func Zsyr2(h *Handle, uplo Uplo, n int, alpha complex128, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, A cudart.DevicePtr, lda int) error {
	return ErrNotAvailable
}

// Level 2 BLAS - Triangular band matrix-vector operations
func Stbmv(h *Handle, uplo Uplo, trans Transpose, diag Diag, n, k int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) error {
	return ErrNotAvailable
}
func Dtbmv(h *Handle, uplo Uplo, trans Transpose, diag Diag, n, k int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) error {
	return ErrNotAvailable
}
func Ctbmv(h *Handle, uplo Uplo, trans Transpose, diag Diag, n, k int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) error {
	return ErrNotAvailable
}
func Ztbmv(h *Handle, uplo Uplo, trans Transpose, diag Diag, n, k int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) error {
	return ErrNotAvailable
}

// Level 2 BLAS - Triangular band solve operations
func Stbsv(h *Handle, uplo Uplo, trans Transpose, diag Diag, n, k int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) error {
	return ErrNotAvailable
}
func Dtbsv(h *Handle, uplo Uplo, trans Transpose, diag Diag, n, k int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) error {
	return ErrNotAvailable
}
func Ctbsv(h *Handle, uplo Uplo, trans Transpose, diag Diag, n, k int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) error {
	return ErrNotAvailable
}
func Ztbsv(h *Handle, uplo Uplo, trans Transpose, diag Diag, n, k int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) error {
	return ErrNotAvailable
}

// Level 2 BLAS - Triangular packed matrix-vector operations
func Stpmv(h *Handle, uplo Uplo, trans Transpose, diag Diag, n int, AP cudart.DevicePtr, x cudart.DevicePtr, incX int) error {
	return ErrNotAvailable
}
func Dtpmv(h *Handle, uplo Uplo, trans Transpose, diag Diag, n int, AP cudart.DevicePtr, x cudart.DevicePtr, incX int) error {
	return ErrNotAvailable
}
func Ctpmv(h *Handle, uplo Uplo, trans Transpose, diag Diag, n int, AP cudart.DevicePtr, x cudart.DevicePtr, incX int) error {
	return ErrNotAvailable
}
func Ztpmv(h *Handle, uplo Uplo, trans Transpose, diag Diag, n int, AP cudart.DevicePtr, x cudart.DevicePtr, incX int) error {
	return ErrNotAvailable
}

// Level 2 BLAS - Triangular packed solve operations
func Stpsv(h *Handle, uplo Uplo, trans Transpose, diag Diag, n int, AP cudart.DevicePtr, x cudart.DevicePtr, incX int) error {
	return ErrNotAvailable
}
func Dtpsv(h *Handle, uplo Uplo, trans Transpose, diag Diag, n int, AP cudart.DevicePtr, x cudart.DevicePtr, incX int) error {
	return ErrNotAvailable
}
func Ctpsv(h *Handle, uplo Uplo, trans Transpose, diag Diag, n int, AP cudart.DevicePtr, x cudart.DevicePtr, incX int) error {
	return ErrNotAvailable
}
func Ztpsv(h *Handle, uplo Uplo, trans Transpose, diag Diag, n int, AP cudart.DevicePtr, x cudart.DevicePtr, incX int) error {
	return ErrNotAvailable
}

// Level 2 BLAS - Triangular matrix-vector operations
func Strmv(h *Handle, uplo Uplo, trans Transpose, diag Diag, n int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) error {
	return ErrNotAvailable
}
func Dtrmv(h *Handle, uplo Uplo, trans Transpose, diag Diag, n int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) error {
	return ErrNotAvailable
}
func Ctrmv(h *Handle, uplo Uplo, trans Transpose, diag Diag, n int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) error {
	return ErrNotAvailable
}
func Ztrmv(h *Handle, uplo Uplo, trans Transpose, diag Diag, n int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) error {
	return ErrNotAvailable
}

// Level 2 BLAS - Triangular solve operations
func Strsv(h *Handle, uplo Uplo, trans Transpose, diag Diag, n int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) error {
	return ErrNotAvailable
}
func Dtrsv(h *Handle, uplo Uplo, trans Transpose, diag Diag, n int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) error {
	return ErrNotAvailable
}
func Ctrsv(h *Handle, uplo Uplo, trans Transpose, diag Diag, n int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) error {
	return ErrNotAvailable
}
func Ztrsv(h *Handle, uplo Uplo, trans Transpose, diag Diag, n int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) error {
	return ErrNotAvailable
}

// Level 2 BLAS - Hermitian matrix-vector operations
func Chemv(h *Handle, uplo Uplo, n int, alpha complex64, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta complex64, y cudart.DevicePtr, incY int) error {
	return ErrNotAvailable
}
func Zhemv(h *Handle, uplo Uplo, n int, alpha complex128, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta complex128, y cudart.DevicePtr, incY int) error {
	return ErrNotAvailable
}

// Level 2 BLAS - Hermitian band matrix-vector operations
func Chbmv(h *Handle, uplo Uplo, n, k int, alpha complex64, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta complex64, y cudart.DevicePtr, incY int) error {
	return ErrNotAvailable
}
func Zhbmv(h *Handle, uplo Uplo, n, k int, alpha complex128, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta complex128, y cudart.DevicePtr, incY int) error {
	return ErrNotAvailable
}

// Level 2 BLAS - Hermitian packed matrix-vector operations
func Chpmv(h *Handle, uplo Uplo, n int, alpha complex64, AP cudart.DevicePtr, x cudart.DevicePtr, incX int, beta complex64, y cudart.DevicePtr, incY int) error {
	return ErrNotAvailable
}
func Zhpmv(h *Handle, uplo Uplo, n int, alpha complex128, AP cudart.DevicePtr, x cudart.DevicePtr, incX int, beta complex128, y cudart.DevicePtr, incY int) error {
	return ErrNotAvailable
}

// Level 2 BLAS - Hermitian rank-1 update operations
func Cher(h *Handle, uplo Uplo, n int, alpha float32, x cudart.DevicePtr, incX int, A cudart.DevicePtr, lda int) error {
	return ErrNotAvailable
}
func Zher(h *Handle, uplo Uplo, n int, alpha float64, x cudart.DevicePtr, incX int, A cudart.DevicePtr, lda int) error {
	return ErrNotAvailable
}

// Level 2 BLAS - Hermitian rank-2 update operations
func Cher2(h *Handle, uplo Uplo, n int, alpha complex64, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, A cudart.DevicePtr, lda int) error {
	return ErrNotAvailable
}
func Zher2(h *Handle, uplo Uplo, n int, alpha complex128, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, A cudart.DevicePtr, lda int) error {
	return ErrNotAvailable
}

// Level 2 BLAS - Hermitian packed rank-1 update operations
func Chpr(h *Handle, uplo Uplo, n int, alpha float32, x cudart.DevicePtr, incX int, AP cudart.DevicePtr) error {
	return ErrNotAvailable
}
func Zhpr(h *Handle, uplo Uplo, n int, alpha float64, x cudart.DevicePtr, incX int, AP cudart.DevicePtr) error {
	return ErrNotAvailable
}

// Level 2 BLAS - Hermitian packed rank-2 update operations
func Chpr2(h *Handle, uplo Uplo, n int, alpha complex64, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, AP cudart.DevicePtr) error {
	return ErrNotAvailable
}
func Zhpr2(h *Handle, uplo Uplo, n int, alpha complex128, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, AP cudart.DevicePtr) error {
	return ErrNotAvailable
}

// Level 2 BLAS - Batched operations
func SgemvBatched(h *Handle, trans Transpose, m, n int, alpha float32, Aarray []cudart.DevicePtr, lda int, xarray []cudart.DevicePtr, incX int, beta float32, yarray []cudart.DevicePtr, incY int, batchCount int) error {
	return ErrNotAvailable
}
func DgemvBatched(h *Handle, trans Transpose, m, n int, alpha float64, Aarray []cudart.DevicePtr, lda int, xarray []cudart.DevicePtr, incX int, beta float64, yarray []cudart.DevicePtr, incY int, batchCount int) error {
	return ErrNotAvailable
}
func CgemvBatched(h *Handle, trans Transpose, m, n int, alpha complex64, Aarray []cudart.DevicePtr, lda int, xarray []cudart.DevicePtr, incX int, beta complex64, yarray []cudart.DevicePtr, incY int, batchCount int) error {
	return ErrNotAvailable
}
func ZgemvBatched(h *Handle, trans Transpose, m, n int, alpha complex128, Aarray []cudart.DevicePtr, lda int, xarray []cudart.DevicePtr, incX int, beta complex128, yarray []cudart.DevicePtr, incY int, batchCount int) error {
	return ErrNotAvailable
}

// Level 2 BLAS - Strided batched operations
func SgemvStridedBatched(h *Handle, trans Transpose, m, n int, alpha float32, A cudart.DevicePtr, lda int, strideA int64, x cudart.DevicePtr, incX int, stridex int64, beta float32, y cudart.DevicePtr, incY int, stridey int64, batchCount int) error {
	return ErrNotAvailable
}
func DgemvStridedBatched(h *Handle, trans Transpose, m, n int, alpha float64, A cudart.DevicePtr, lda int, strideA int64, x cudart.DevicePtr, incX int, stridex int64, beta float64, y cudart.DevicePtr, incY int, stridey int64, batchCount int) error {
	return ErrNotAvailable
}
func CgemvStridedBatched(h *Handle, trans Transpose, m, n int, alpha complex64, A cudart.DevicePtr, lda int, strideA int64, x cudart.DevicePtr, incX int, stridex int64, beta complex64, y cudart.DevicePtr, incY int, stridey int64, batchCount int) error {
	return ErrNotAvailable
}
func ZgemvStridedBatched(h *Handle, trans Transpose, m, n int, alpha complex128, A cudart.DevicePtr, lda int, strideA int64, x cudart.DevicePtr, incX int, stridex int64, beta complex128, y cudart.DevicePtr, incY int, stridey int64, batchCount int) error {
	return ErrNotAvailable
}
