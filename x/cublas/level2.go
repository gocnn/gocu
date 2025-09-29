package cublas

import (
	"github.com/gocnn/gocu/cublas"
	"github.com/gocnn/gocu/cudart"
)

// Level 2 BLAS Functions - Band Matrix-Vector Operations

// Sgbmv performs the matrix-vector operation y = alpha*A*x + beta*y for single precision real band matrix.
func (h *CudaBlas) Sgbmv(trans cublas.Transpose, m, n, kl, ku int, alpha float32, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta float32, y cudart.DevicePtr, incY int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Sgbmv(h.handle, trans, m, n, kl, ku, alpha, A, lda, x, incX, beta, y, incY)
}

// Dgbmv performs the matrix-vector operation y = alpha*A*x + beta*y for double precision real band matrix.
func (h *CudaBlas) Dgbmv(trans cublas.Transpose, m, n, kl, ku int, alpha float64, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta float64, y cudart.DevicePtr, incY int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Dgbmv(h.handle, trans, m, n, kl, ku, alpha, A, lda, x, incX, beta, y, incY)
}

// Cgbmv performs the matrix-vector operation y = alpha*A*x + beta*y for single precision complex band matrix.
func (h *CudaBlas) Cgbmv(trans cublas.Transpose, m, n, kl, ku int, alpha complex64, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta complex64, y cudart.DevicePtr, incY int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Cgbmv(h.handle, trans, m, n, kl, ku, alpha, A, lda, x, incX, beta, y, incY)
}

// Zgbmv performs the matrix-vector operation y = alpha*A*x + beta*y for double precision complex band matrix.
func (h *CudaBlas) Zgbmv(trans cublas.Transpose, m, n, kl, ku int, alpha complex128, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta complex128, y cudart.DevicePtr, incY int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Zgbmv(h.handle, trans, m, n, kl, ku, alpha, A, lda, x, incX, beta, y, incY)
}

// Level 2 BLAS Functions - General Matrix-Vector Operations

// Sgemv performs the matrix-vector operation y = alpha*A*x + beta*y for single precision real matrix.
func (h *CudaBlas) Sgemv(trans cublas.Transpose, m, n int, alpha float32, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta float32, y cudart.DevicePtr, incY int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Sgemv(h.handle, trans, m, n, alpha, A, lda, x, incX, beta, y, incY)
}

// Dgemv performs the matrix-vector operation y = alpha*A*x + beta*y for double precision real matrix.
func (h *CudaBlas) Dgemv(trans cublas.Transpose, m, n int, alpha float64, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta float64, y cudart.DevicePtr, incY int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Dgemv(h.handle, trans, m, n, alpha, A, lda, x, incX, beta, y, incY)
}

// Cgemv performs the matrix-vector operation y = alpha*A*x + beta*y for single precision complex matrix.
func (h *CudaBlas) Cgemv(trans cublas.Transpose, m, n int, alpha complex64, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta complex64, y cudart.DevicePtr, incY int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Cgemv(h.handle, trans, m, n, alpha, A, lda, x, incX, beta, y, incY)
}

// Zgemv performs the matrix-vector operation y = alpha*A*x + beta*y for double precision complex matrix.
func (h *CudaBlas) Zgemv(trans cublas.Transpose, m, n int, alpha complex128, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta complex128, y cudart.DevicePtr, incY int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Zgemv(h.handle, trans, m, n, alpha, A, lda, x, incX, beta, y, incY)
}

// Level 2 BLAS Functions - General Rank-1 Update Operations

// Sger performs the rank-1 update A = alpha*x*y^T + A for single precision real matrix.
func (h *CudaBlas) Sger(m, n int, alpha float32, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, A cudart.DevicePtr, lda int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Sger(h.handle, m, n, alpha, x, incX, y, incY, A, lda)
}

// Dger performs the rank-1 update A = alpha*x*y^T + A for double precision real matrix.
func (h *CudaBlas) Dger(m, n int, alpha float64, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, A cudart.DevicePtr, lda int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Dger(h.handle, m, n, alpha, x, incX, y, incY, A, lda)
}

// Cgeru performs the rank-1 update A = alpha*x*y^T + A for single precision complex matrix (unconjugated).
func (h *CudaBlas) Cgeru(m, n int, alpha complex64, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, A cudart.DevicePtr, lda int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Cgeru(h.handle, m, n, alpha, x, incX, y, incY, A, lda)
}

// Cgerc performs the rank-1 update A = alpha*x*y^H + A for single precision complex matrix (conjugated).
func (h *CudaBlas) Cgerc(m, n int, alpha complex64, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, A cudart.DevicePtr, lda int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Cgerc(h.handle, m, n, alpha, x, incX, y, incY, A, lda)
}

// Zgeru performs the rank-1 update A = alpha*x*y^T + A for double precision complex matrix (unconjugated).
func (h *CudaBlas) Zgeru(m, n int, alpha complex128, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, A cudart.DevicePtr, lda int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Zgeru(h.handle, m, n, alpha, x, incX, y, incY, A, lda)
}

// Zgerc performs the rank-1 update A = alpha*x*y^H + A for double precision complex matrix (conjugated).
func (h *CudaBlas) Zgerc(m, n int, alpha complex128, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, A cudart.DevicePtr, lda int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Zgerc(h.handle, m, n, alpha, x, incX, y, incY, A, lda)
}

// Level 2 BLAS Functions - Symmetric Band Matrix-Vector Operations

// Ssbmv performs the matrix-vector operation y = alpha*A*x + beta*y for single precision real symmetric band matrix.
func (h *CudaBlas) Ssbmv(uplo cublas.Uplo, n, k int, alpha float32, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta float32, y cudart.DevicePtr, incY int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Ssbmv(h.handle, uplo, n, k, alpha, A, lda, x, incX, beta, y, incY)
}

// Dsbmv performs the matrix-vector operation y = alpha*A*x + beta*y for double precision real symmetric band matrix.
func (h *CudaBlas) Dsbmv(uplo cublas.Uplo, n, k int, alpha float64, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta float64, y cudart.DevicePtr, incY int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Dsbmv(h.handle, uplo, n, k, alpha, A, lda, x, incX, beta, y, incY)
}

// Level 2 BLAS Functions - Symmetric Packed Matrix-Vector Operations

// Sspmv performs the matrix-vector operation y = alpha*A*x + beta*y for single precision real symmetric packed matrix.
func (h *CudaBlas) Sspmv(uplo cublas.Uplo, n int, alpha float32, AP cudart.DevicePtr, x cudart.DevicePtr, incX int, beta float32, y cudart.DevicePtr, incY int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Sspmv(h.handle, uplo, n, alpha, AP, x, incX, beta, y, incY)
}

// Dspmv performs the matrix-vector operation y = alpha*A*x + beta*y for double precision real symmetric packed matrix.
func (h *CudaBlas) Dspmv(uplo cublas.Uplo, n int, alpha float64, AP cudart.DevicePtr, x cudart.DevicePtr, incX int, beta float64, y cudart.DevicePtr, incY int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Dspmv(h.handle, uplo, n, alpha, AP, x, incX, beta, y, incY)
}

// Level 2 BLAS Functions - Symmetric Packed Rank-1 Update Operations

// Sspr performs the symmetric rank-1 update AP = alpha*x*x^T + AP for single precision real packed matrix.
func (h *CudaBlas) Sspr(uplo cublas.Uplo, n int, alpha float32, x cudart.DevicePtr, incX int, AP cudart.DevicePtr) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Sspr(h.handle, uplo, n, alpha, x, incX, AP)
}

// Dspr performs the symmetric rank-1 update AP = alpha*x*x^T + AP for double precision real packed matrix.
func (h *CudaBlas) Dspr(uplo cublas.Uplo, n int, alpha float64, x cudart.DevicePtr, incX int, AP cudart.DevicePtr) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Dspr(h.handle, uplo, n, alpha, x, incX, AP)
}

// Level 2 BLAS Functions - Symmetric Packed Rank-2 Update Operations

// Sspr2 performs the symmetric rank-2 update AP = alpha*x*y^T + alpha*y*x^T + AP for single precision real packed matrix.
func (h *CudaBlas) Sspr2(uplo cublas.Uplo, n int, alpha float32, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, AP cudart.DevicePtr) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Sspr2(h.handle, uplo, n, alpha, x, incX, y, incY, AP)
}

// Dspr2 performs the symmetric rank-2 update AP = alpha*x*y^T + alpha*y*x^T + AP for double precision real packed matrix.
func (h *CudaBlas) Dspr2(uplo cublas.Uplo, n int, alpha float64, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, AP cudart.DevicePtr) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Dspr2(h.handle, uplo, n, alpha, x, incX, y, incY, AP)
}

// Level 2 BLAS Functions - Symmetric Matrix-Vector Operations

// Ssymv performs the matrix-vector operation y = alpha*A*x + beta*y for single precision real symmetric matrix.
func (h *CudaBlas) Ssymv(uplo cublas.Uplo, n int, alpha float32, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta float32, y cudart.DevicePtr, incY int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Ssymv(h.handle, uplo, n, alpha, A, lda, x, incX, beta, y, incY)
}

// Dsymv performs the matrix-vector operation y = alpha*A*x + beta*y for double precision real symmetric matrix.
func (h *CudaBlas) Dsymv(uplo cublas.Uplo, n int, alpha float64, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta float64, y cudart.DevicePtr, incY int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Dsymv(h.handle, uplo, n, alpha, A, lda, x, incX, beta, y, incY)
}

// Csymv performs the matrix-vector operation y = alpha*A*x + beta*y for single precision complex symmetric matrix.
func (h *CudaBlas) Csymv(uplo cublas.Uplo, n int, alpha complex64, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta complex64, y cudart.DevicePtr, incY int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Csymv(h.handle, uplo, n, alpha, A, lda, x, incX, beta, y, incY)
}

// Zsymv performs the matrix-vector operation y = alpha*A*x + beta*y for double precision complex symmetric matrix.
func (h *CudaBlas) Zsymv(uplo cublas.Uplo, n int, alpha complex128, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta complex128, y cudart.DevicePtr, incY int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Zsymv(h.handle, uplo, n, alpha, A, lda, x, incX, beta, y, incY)
}

// Level 2 BLAS Functions - Symmetric Rank-1 Update Operations

// Ssyr performs the symmetric rank-1 update A = alpha*x*x^T + A for single precision real matrix.
func (h *CudaBlas) Ssyr(uplo cublas.Uplo, n int, alpha float32, x cudart.DevicePtr, incX int, A cudart.DevicePtr, lda int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Ssyr(h.handle, uplo, n, alpha, x, incX, A, lda)
}

// Dsyr performs the symmetric rank-1 update A = alpha*x*x^T + A for double precision real matrix.
func (h *CudaBlas) Dsyr(uplo cublas.Uplo, n int, alpha float64, x cudart.DevicePtr, incX int, A cudart.DevicePtr, lda int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Dsyr(h.handle, uplo, n, alpha, x, incX, A, lda)
}

// Csyr performs the symmetric rank-1 update A = alpha*x*x^T + A for single precision complex matrix.
func (h *CudaBlas) Csyr(uplo cublas.Uplo, n int, alpha complex64, x cudart.DevicePtr, incX int, A cudart.DevicePtr, lda int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Csyr(h.handle, uplo, n, alpha, x, incX, A, lda)
}

// Zsyr performs the symmetric rank-1 update A = alpha*x*x^T + A for double precision complex matrix.
func (h *CudaBlas) Zsyr(uplo cublas.Uplo, n int, alpha complex128, x cudart.DevicePtr, incX int, A cudart.DevicePtr, lda int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Zsyr(h.handle, uplo, n, alpha, x, incX, A, lda)
}

// Level 2 BLAS Functions - Symmetric Rank-2 Update Operations

// Ssyr2 performs the symmetric rank-2 update A = alpha*x*y^T + alpha*y*x^T + A for single precision real matrix.
func (h *CudaBlas) Ssyr2(uplo cublas.Uplo, n int, alpha float32, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, A cudart.DevicePtr, lda int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Ssyr2(h.handle, uplo, n, alpha, x, incX, y, incY, A, lda)
}

// Dsyr2 performs the symmetric rank-2 update A = alpha*x*y^T + alpha*y*x^T + A for double precision real matrix.
func (h *CudaBlas) Dsyr2(uplo cublas.Uplo, n int, alpha float64, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, A cudart.DevicePtr, lda int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Dsyr2(h.handle, uplo, n, alpha, x, incX, y, incY, A, lda)
}

// Csyr2 performs the symmetric rank-2 update A = alpha*x*y^T + alpha*y*x^T + A for single precision complex matrix.
func (h *CudaBlas) Csyr2(uplo cublas.Uplo, n int, alpha complex64, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, A cudart.DevicePtr, lda int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Csyr2(h.handle, uplo, n, alpha, x, incX, y, incY, A, lda)
}

// Zsyr2 performs the symmetric rank-2 update A = alpha*x*y^T + alpha*y*x^T + A for double precision complex matrix.
func (h *CudaBlas) Zsyr2(uplo cublas.Uplo, n int, alpha complex128, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, A cudart.DevicePtr, lda int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Zsyr2(h.handle, uplo, n, alpha, x, incX, y, incY, A, lda)
}

// Level 2 BLAS Functions - Triangular Band Matrix-Vector Operations

// Stbmv performs the matrix-vector operation x = A*x or x = A^T*x for single precision real triangular band matrix.
func (h *CudaBlas) Stbmv(uplo cublas.Uplo, trans cublas.Transpose, diag cublas.Diag, n, k int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Stbmv(h.handle, uplo, trans, diag, n, k, A, lda, x, incX)
}

// Dtbmv performs the matrix-vector operation x = A*x or x = A^T*x for double precision real triangular band matrix.
func (h *CudaBlas) Dtbmv(uplo cublas.Uplo, trans cublas.Transpose, diag cublas.Diag, n, k int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Dtbmv(h.handle, uplo, trans, diag, n, k, A, lda, x, incX)
}

// Ctbmv performs the matrix-vector operation x = A*x, x = A^T*x, or x = A^H*x for single precision complex triangular band matrix.
func (h *CudaBlas) Ctbmv(uplo cublas.Uplo, trans cublas.Transpose, diag cublas.Diag, n, k int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Ctbmv(h.handle, uplo, trans, diag, n, k, A, lda, x, incX)
}

// Ztbmv performs the matrix-vector operation x = A*x, x = A^T*x, or x = A^H*x for double precision complex triangular band matrix.
func (h *CudaBlas) Ztbmv(uplo cublas.Uplo, trans cublas.Transpose, diag cublas.Diag, n, k int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Ztbmv(h.handle, uplo, trans, diag, n, k, A, lda, x, incX)
}

// Level 2 BLAS Functions - Triangular Band System Solve Operations

// Stbsv solves the triangular band system A*x = b or A^T*x = b for single precision real matrix.
func (h *CudaBlas) Stbsv(uplo cublas.Uplo, trans cublas.Transpose, diag cublas.Diag, n, k int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Stbsv(h.handle, uplo, trans, diag, n, k, A, lda, x, incX)
}

// Dtbsv solves the triangular band system A*x = b or A^T*x = b for double precision real matrix.
func (h *CudaBlas) Dtbsv(uplo cublas.Uplo, trans cublas.Transpose, diag cublas.Diag, n, k int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Dtbsv(h.handle, uplo, trans, diag, n, k, A, lda, x, incX)
}

// Ctbsv solves the triangular band system A*x = b, A^T*x = b, or A^H*x = b for single precision complex matrix.
func (h *CudaBlas) Ctbsv(uplo cublas.Uplo, trans cublas.Transpose, diag cublas.Diag, n, k int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Ctbsv(h.handle, uplo, trans, diag, n, k, A, lda, x, incX)
}

// Ztbsv solves the triangular band system A*x = b, A^T*x = b, or A^H*x = b for double precision complex matrix.
func (h *CudaBlas) Ztbsv(uplo cublas.Uplo, trans cublas.Transpose, diag cublas.Diag, n, k int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Ztbsv(h.handle, uplo, trans, diag, n, k, A, lda, x, incX)
}

// Level 2 BLAS Functions - Triangular Packed Matrix-Vector Operations

// Stpmv performs the matrix-vector operation x = A*x or x = A^T*x for single precision real triangular packed matrix.
func (h *CudaBlas) Stpmv(uplo cublas.Uplo, trans cublas.Transpose, diag cublas.Diag, n int, AP cudart.DevicePtr, x cudart.DevicePtr, incX int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Stpmv(h.handle, uplo, trans, diag, n, AP, x, incX)
}

// Dtpmv performs the matrix-vector operation x = A*x or x = A^T*x for double precision real triangular packed matrix.
func (h *CudaBlas) Dtpmv(uplo cublas.Uplo, trans cublas.Transpose, diag cublas.Diag, n int, AP cudart.DevicePtr, x cudart.DevicePtr, incX int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Dtpmv(h.handle, uplo, trans, diag, n, AP, x, incX)
}

// Ctpmv performs the matrix-vector operation x = A*x, x = A^T*x, or x = A^H*x for single precision complex triangular packed matrix.
func (h *CudaBlas) Ctpmv(uplo cublas.Uplo, trans cublas.Transpose, diag cublas.Diag, n int, AP cudart.DevicePtr, x cudart.DevicePtr, incX int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Ctpmv(h.handle, uplo, trans, diag, n, AP, x, incX)
}

// Ztpmv performs the matrix-vector operation x = A*x, x = A^T*x, or x = A^H*x for double precision complex triangular packed matrix.
func (h *CudaBlas) Ztpmv(uplo cublas.Uplo, trans cublas.Transpose, diag cublas.Diag, n int, AP cudart.DevicePtr, x cudart.DevicePtr, incX int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Ztpmv(h.handle, uplo, trans, diag, n, AP, x, incX)
}

// Level 2 BLAS Functions - Triangular Packed System Solve Operations

// Stpsv solves the triangular packed system A*x = b or A^T*x = b for single precision real matrix.
func (h *CudaBlas) Stpsv(uplo cublas.Uplo, trans cublas.Transpose, diag cublas.Diag, n int, AP cudart.DevicePtr, x cudart.DevicePtr, incX int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Stpsv(h.handle, uplo, trans, diag, n, AP, x, incX)
}

// Dtpsv solves the triangular packed system A*x = b or A^T*x = b for double precision real matrix.
func (h *CudaBlas) Dtpsv(uplo cublas.Uplo, trans cublas.Transpose, diag cublas.Diag, n int, AP cudart.DevicePtr, x cudart.DevicePtr, incX int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Dtpsv(h.handle, uplo, trans, diag, n, AP, x, incX)
}

// Ctpsv solves the triangular packed system A*x = b, A^T*x = b, or A^H*x = b for single precision complex matrix.
func (h *CudaBlas) Ctpsv(uplo cublas.Uplo, trans cublas.Transpose, diag cublas.Diag, n int, AP cudart.DevicePtr, x cudart.DevicePtr, incX int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Ctpsv(h.handle, uplo, trans, diag, n, AP, x, incX)
}

// Ztpsv solves the triangular packed system A*x = b, A^T*x = b, or A^H*x = b for double precision complex matrix.
func (h *CudaBlas) Ztpsv(uplo cublas.Uplo, trans cublas.Transpose, diag cublas.Diag, n int, AP cudart.DevicePtr, x cudart.DevicePtr, incX int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Ztpsv(h.handle, uplo, trans, diag, n, AP, x, incX)
}

// Level 2 BLAS Functions - Triangular Matrix-Vector Operations

// Strmv performs the matrix-vector operation x = A*x or x = A^T*x for single precision real triangular matrix.
func (h *CudaBlas) Strmv(uplo cublas.Uplo, trans cublas.Transpose, diag cublas.Diag, n int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Strmv(h.handle, uplo, trans, diag, n, A, lda, x, incX)
}

// Dtrmv performs the matrix-vector operation x = A*x or x = A^T*x for double precision real triangular matrix.
func (h *CudaBlas) Dtrmv(uplo cublas.Uplo, trans cublas.Transpose, diag cublas.Diag, n int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Dtrmv(h.handle, uplo, trans, diag, n, A, lda, x, incX)
}

// Ctrmv performs the matrix-vector operation x = A*x, x = A^T*x, or x = A^H*x for single precision complex triangular matrix.
func (h *CudaBlas) Ctrmv(uplo cublas.Uplo, trans cublas.Transpose, diag cublas.Diag, n int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Ctrmv(h.handle, uplo, trans, diag, n, A, lda, x, incX)
}

// Ztrmv performs the matrix-vector operation x = A*x, x = A^T*x, or x = A^H*x for double precision complex triangular matrix.
func (h *CudaBlas) Ztrmv(uplo cublas.Uplo, trans cublas.Transpose, diag cublas.Diag, n int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Ztrmv(h.handle, uplo, trans, diag, n, A, lda, x, incX)
}

// Level 2 BLAS Functions - Triangular System Solve Operations

// Strsv solves the triangular system A*x = b or A^T*x = b for single precision real matrix.
func (h *CudaBlas) Strsv(uplo cublas.Uplo, trans cublas.Transpose, diag cublas.Diag, n int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Strsv(h.handle, uplo, trans, diag, n, A, lda, x, incX)
}

// Dtrsv solves the triangular system A*x = b or A^T*x = b for double precision real matrix.
func (h *CudaBlas) Dtrsv(uplo cublas.Uplo, trans cublas.Transpose, diag cublas.Diag, n int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Dtrsv(h.handle, uplo, trans, diag, n, A, lda, x, incX)
}

// Ctrsv solves the triangular system A*x = b, A^T*x = b, or A^H*x = b for single precision complex matrix.
func (h *CudaBlas) Ctrsv(uplo cublas.Uplo, trans cublas.Transpose, diag cublas.Diag, n int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Ctrsv(h.handle, uplo, trans, diag, n, A, lda, x, incX)
}

// Ztrsv solves the triangular system A*x = b, A^T*x = b, or A^H*x = b for double precision complex matrix.
func (h *CudaBlas) Ztrsv(uplo cublas.Uplo, trans cublas.Transpose, diag cublas.Diag, n int, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Ztrsv(h.handle, uplo, trans, diag, n, A, lda, x, incX)
}

// Level 2 BLAS Functions - Hermitian Matrix-Vector Operations

// Chemv performs the matrix-vector operation y = alpha*A*x + beta*y for single precision complex Hermitian matrix.
func (h *CudaBlas) Chemv(uplo cublas.Uplo, n int, alpha complex64, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta complex64, y cudart.DevicePtr, incY int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Chemv(h.handle, uplo, n, alpha, A, lda, x, incX, beta, y, incY)
}

// Zhemv performs the matrix-vector operation y = alpha*A*x + beta*y for double precision complex Hermitian matrix.
func (h *CudaBlas) Zhemv(uplo cublas.Uplo, n int, alpha complex128, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta complex128, y cudart.DevicePtr, incY int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Zhemv(h.handle, uplo, n, alpha, A, lda, x, incX, beta, y, incY)
}

// Level 2 BLAS Functions - Hermitian Band Matrix-Vector Operations

// Chbmv performs the matrix-vector operation y = alpha*A*x + beta*y for single precision complex Hermitian band matrix.
func (h *CudaBlas) Chbmv(uplo cublas.Uplo, n, k int, alpha complex64, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta complex64, y cudart.DevicePtr, incY int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Chbmv(h.handle, uplo, n, k, alpha, A, lda, x, incX, beta, y, incY)
}

// Zhbmv performs the matrix-vector operation y = alpha*A*x + beta*y for double precision complex Hermitian band matrix.
func (h *CudaBlas) Zhbmv(uplo cublas.Uplo, n, k int, alpha complex128, A cudart.DevicePtr, lda int, x cudart.DevicePtr, incX int, beta complex128, y cudart.DevicePtr, incY int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Zhbmv(h.handle, uplo, n, k, alpha, A, lda, x, incX, beta, y, incY)
}

// Level 2 BLAS Functions - Hermitian Packed Matrix-Vector Operations

// Chpmv performs the matrix-vector operation y = alpha*A*x + beta*y for single precision complex Hermitian packed matrix.
func (h *CudaBlas) Chpmv(uplo cublas.Uplo, n int, alpha complex64, AP cudart.DevicePtr, x cudart.DevicePtr, incX int, beta complex64, y cudart.DevicePtr, incY int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Chpmv(h.handle, uplo, n, alpha, AP, x, incX, beta, y, incY)
}

// Zhpmv performs the matrix-vector operation y = alpha*A*x + beta*y for double precision complex Hermitian packed matrix.
func (h *CudaBlas) Zhpmv(uplo cublas.Uplo, n int, alpha complex128, AP cudart.DevicePtr, x cudart.DevicePtr, incX int, beta complex128, y cudart.DevicePtr, incY int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Zhpmv(h.handle, uplo, n, alpha, AP, x, incX, beta, y, incY)
}

// Level 2 BLAS Functions - Hermitian Rank-1 Update Operations

// Cher performs the Hermitian rank-1 update A = alpha*x*x^H + A for single precision complex matrix.
func (h *CudaBlas) Cher(uplo cublas.Uplo, n int, alpha float32, x cudart.DevicePtr, incX int, A cudart.DevicePtr, lda int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Cher(h.handle, uplo, n, alpha, x, incX, A, lda)
}

// Zher performs the Hermitian rank-1 update A = alpha*x*x^H + A for double precision complex matrix.
func (h *CudaBlas) Zher(uplo cublas.Uplo, n int, alpha float64, x cudart.DevicePtr, incX int, A cudart.DevicePtr, lda int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Zher(h.handle, uplo, n, alpha, x, incX, A, lda)
}

// Level 2 BLAS Functions - Hermitian Rank-2 Update Operations

// Cher2 performs the Hermitian rank-2 update A = alpha*x*y^H + conj(alpha)*y*x^H + A for single precision complex matrix.
func (h *CudaBlas) Cher2(uplo cublas.Uplo, n int, alpha complex64, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, A cudart.DevicePtr, lda int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Cher2(h.handle, uplo, n, alpha, x, incX, y, incY, A, lda)
}

// Zher2 performs the Hermitian rank-2 update A = alpha*x*y^H + conj(alpha)*y*x^H + A for double precision complex matrix.
func (h *CudaBlas) Zher2(uplo cublas.Uplo, n int, alpha complex128, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, A cudart.DevicePtr, lda int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Zher2(h.handle, uplo, n, alpha, x, incX, y, incY, A, lda)
}

// Level 2 BLAS Functions - Hermitian Packed Rank-1 Update Operations

// Chpr performs the Hermitian packed rank-1 update AP = alpha*x*x^H + AP for single precision complex matrix.
func (h *CudaBlas) Chpr(uplo cublas.Uplo, n int, alpha float32, x cudart.DevicePtr, incX int, AP cudart.DevicePtr) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Chpr(h.handle, uplo, n, alpha, x, incX, AP)
}

// Zhpr performs the Hermitian packed rank-1 update AP = alpha*x*x^H + AP for double precision complex matrix.
func (h *CudaBlas) Zhpr(uplo cublas.Uplo, n int, alpha float64, x cudart.DevicePtr, incX int, AP cudart.DevicePtr) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Zhpr(h.handle, uplo, n, alpha, x, incX, AP)
}

// Level 2 BLAS Functions - Hermitian Packed Rank-2 Update Operations

// Chpr2 performs the Hermitian packed rank-2 update AP = alpha*x*y^H + conj(alpha)*y*x^H + AP for single precision complex matrix.
func (h *CudaBlas) Chpr2(uplo cublas.Uplo, n int, alpha complex64, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, AP cudart.DevicePtr) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Chpr2(h.handle, uplo, n, alpha, x, incX, y, incY, AP)
}

// Zhpr2 performs the Hermitian packed rank-2 update AP = alpha*x*y^H + conj(alpha)*y*x^H + AP for double precision complex matrix.
func (h *CudaBlas) Zhpr2(uplo cublas.Uplo, n int, alpha complex128, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, AP cudart.DevicePtr) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Zhpr2(h.handle, uplo, n, alpha, x, incX, y, incY, AP)
}
