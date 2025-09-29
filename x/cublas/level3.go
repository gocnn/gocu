package cublas

import (
	"github.com/gocnn/gocu/cublas"
	"github.com/gocnn/gocu/cudart"
)

// Level 3 BLAS Functions - General Matrix-Matrix Operations

// Sgemm performs the matrix-matrix operation C = alpha*op(A)*op(B) + beta*C for single precision real matrices.
func (h *CudaBlas) Sgemm(tA, tB cublas.Transpose, m, n, k int, alpha float32, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta float32, c cudart.DevicePtr, ldc int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Sgemm(h.handle, tA, tB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}

// Dgemm performs the matrix-matrix operation C = alpha*op(A)*op(B) + beta*C for double precision real matrices.
func (h *CudaBlas) Dgemm(tA, tB cublas.Transpose, m, n, k int, alpha float64, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta float64, c cudart.DevicePtr, ldc int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Dgemm(h.handle, tA, tB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}

// Cgemm performs the matrix-matrix operation C = alpha*op(A)*op(B) + beta*C for single precision complex matrices.
func (h *CudaBlas) Cgemm(tA, tB cublas.Transpose, m, n, k int, alpha complex64, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta complex64, c cudart.DevicePtr, ldc int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Cgemm(h.handle, tA, tB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}

// Zgemm performs the matrix-matrix operation C = alpha*op(A)*op(B) + beta*C for double precision complex matrices.
func (h *CudaBlas) Zgemm(tA, tB cublas.Transpose, m, n, k int, alpha complex128, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta complex128, c cudart.DevicePtr, ldc int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Zgemm(h.handle, tA, tB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}

// Cgemm3m performs the matrix-matrix operation C = alpha*op(A)*op(B) + beta*C for single precision complex matrices using 3M algorithm.
func (h *CudaBlas) Cgemm3m(tA, tB cublas.Transpose, m, n, k int, alpha complex64, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta complex64, c cudart.DevicePtr, ldc int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Cgemm3m(h.handle, tA, tB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}

// Zgemm3m performs the matrix-matrix operation C = alpha*op(A)*op(B) + beta*C for double precision complex matrices using 3M algorithm.
func (h *CudaBlas) Zgemm3m(tA, tB cublas.Transpose, m, n, k int, alpha complex128, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta complex128, c cudart.DevicePtr, ldc int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Zgemm3m(h.handle, tA, tB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}

// Level 3 BLAS Functions - Symmetric Matrix-Matrix Operations

// Ssymm performs the symmetric matrix-matrix operation C = alpha*A*B + beta*C or C = alpha*B*A + beta*C for single precision real matrices.
func (h *CudaBlas) Ssymm(side cublas.Side, uplo cublas.Uplo, m, n int, alpha float32, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta float32, c cudart.DevicePtr, ldc int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Ssymm(h.handle, side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc)
}

// Dsymm performs the symmetric matrix-matrix operation C = alpha*A*B + beta*C or C = alpha*B*A + beta*C for double precision real matrices.
func (h *CudaBlas) Dsymm(side cublas.Side, uplo cublas.Uplo, m, n int, alpha float64, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta float64, c cudart.DevicePtr, ldc int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Dsymm(h.handle, side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc)
}

// Csymm performs the symmetric matrix-matrix operation C = alpha*A*B + beta*C or C = alpha*B*A + beta*C for single precision complex matrices.
func (h *CudaBlas) Csymm(side cublas.Side, uplo cublas.Uplo, m, n int, alpha complex64, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta complex64, c cudart.DevicePtr, ldc int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Csymm(h.handle, side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc)
}

// Zsymm performs the symmetric matrix-matrix operation C = alpha*A*B + beta*C or C = alpha*B*A + beta*C for double precision complex matrices.
func (h *CudaBlas) Zsymm(side cublas.Side, uplo cublas.Uplo, m, n int, alpha complex128, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta complex128, c cudart.DevicePtr, ldc int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Zsymm(h.handle, side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc)
}

// Level 3 BLAS Functions - Symmetric Rank-K Update Operations

// Ssyrk performs the symmetric rank-k update C = alpha*A*A^T + beta*C or C = alpha*A^T*A + beta*C for single precision real matrices.
func (h *CudaBlas) Ssyrk(uplo cublas.Uplo, trans cublas.Transpose, n, k int, alpha float32, a cudart.DevicePtr, lda int, beta float32, c cudart.DevicePtr, ldc int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Ssyrk(h.handle, uplo, trans, n, k, alpha, a, lda, beta, c, ldc)
}

// Dsyrk performs the symmetric rank-k update C = alpha*A*A^T + beta*C or C = alpha*A^T*A + beta*C for double precision real matrices.
func (h *CudaBlas) Dsyrk(uplo cublas.Uplo, trans cublas.Transpose, n, k int, alpha float64, a cudart.DevicePtr, lda int, beta float64, c cudart.DevicePtr, ldc int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Dsyrk(h.handle, uplo, trans, n, k, alpha, a, lda, beta, c, ldc)
}

// Csyrk performs the symmetric rank-k update C = alpha*A*A^T + beta*C or C = alpha*A^T*A + beta*C for single precision complex matrices.
func (h *CudaBlas) Csyrk(uplo cublas.Uplo, trans cublas.Transpose, n, k int, alpha complex64, a cudart.DevicePtr, lda int, beta complex64, c cudart.DevicePtr, ldc int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Csyrk(h.handle, uplo, trans, n, k, alpha, a, lda, beta, c, ldc)
}

// Zsyrk performs the symmetric rank-k update C = alpha*A*A^T + beta*C or C = alpha*A^T*A + beta*C for double precision complex matrices.
func (h *CudaBlas) Zsyrk(uplo cublas.Uplo, trans cublas.Transpose, n, k int, alpha complex128, a cudart.DevicePtr, lda int, beta complex128, c cudart.DevicePtr, ldc int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Zsyrk(h.handle, uplo, trans, n, k, alpha, a, lda, beta, c, ldc)
}

// Level 3 BLAS Functions - Symmetric Rank-2K Update Operations

// Ssyr2k performs the symmetric rank-2k update C = alpha*(A*B^T + B*A^T) + beta*C for single precision real matrices.
func (h *CudaBlas) Ssyr2k(uplo cublas.Uplo, trans cublas.Transpose, n, k int, alpha float32, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta float32, c cudart.DevicePtr, ldc int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Ssyr2k(h.handle, uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}

// Dsyr2k performs the symmetric rank-2k update C = alpha*(A*B^T + B*A^T) + beta*C for double precision real matrices.
func (h *CudaBlas) Dsyr2k(uplo cublas.Uplo, trans cublas.Transpose, n, k int, alpha float64, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta float64, c cudart.DevicePtr, ldc int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Dsyr2k(h.handle, uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}

// Csyr2k performs the symmetric rank-2k update C = alpha*(A*B^T + B*A^T) + beta*C for single precision complex matrices.
func (h *CudaBlas) Csyr2k(uplo cublas.Uplo, trans cublas.Transpose, n, k int, alpha complex64, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta complex64, c cudart.DevicePtr, ldc int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Csyr2k(h.handle, uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}

// Zsyr2k performs the symmetric rank-2k update C = alpha*(A*B^T + B*A^T) + beta*C for double precision complex matrices.
func (h *CudaBlas) Zsyr2k(uplo cublas.Uplo, trans cublas.Transpose, n, k int, alpha complex128, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta complex128, c cudart.DevicePtr, ldc int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Zsyr2k(h.handle, uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}

// Level 3 BLAS Functions - Symmetric Rank-KX Update Operations

// Ssyrkx performs the symmetric rank-kx update C = alpha*A*B^T + beta*C for single precision real matrices.
func (h *CudaBlas) Ssyrkx(uplo cublas.Uplo, trans cublas.Transpose, n, k int, alpha float32, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta float32, c cudart.DevicePtr, ldc int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Ssyrkx(h.handle, uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}

// Dsyrkx performs the symmetric rank-kx update C = alpha*A*B^T + beta*C for double precision real matrices.
func (h *CudaBlas) Dsyrkx(uplo cublas.Uplo, trans cublas.Transpose, n, k int, alpha float64, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta float64, c cudart.DevicePtr, ldc int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Dsyrkx(h.handle, uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}

// Csyrkx performs the symmetric rank-kx update C = alpha*A*B^T + beta*C for single precision complex matrices.
func (h *CudaBlas) Csyrkx(uplo cublas.Uplo, trans cublas.Transpose, n, k int, alpha complex64, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta complex64, c cudart.DevicePtr, ldc int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Csyrkx(h.handle, uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}

// Zsyrkx performs the symmetric rank-kx update C = alpha*A*B^T + beta*C for double precision complex matrices.
func (h *CudaBlas) Zsyrkx(uplo cublas.Uplo, trans cublas.Transpose, n, k int, alpha complex128, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta complex128, c cudart.DevicePtr, ldc int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Zsyrkx(h.handle, uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}

// Level 3 BLAS Functions - Triangular Matrix-Matrix Operations

// Strmm performs the triangular matrix-matrix operation C = alpha*op(A)*B or C = alpha*B*op(A) for single precision real matrices.
func (h *CudaBlas) Strmm(side cublas.Side, uplo cublas.Uplo, trans cublas.Transpose, diag cublas.Diag, m, n int, alpha float32, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, c cudart.DevicePtr, ldc int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Strmm(h.handle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, c, ldc)
}

// Dtrmm performs the triangular matrix-matrix operation C = alpha*op(A)*B or C = alpha*B*op(A) for double precision real matrices.
func (h *CudaBlas) Dtrmm(side cublas.Side, uplo cublas.Uplo, trans cublas.Transpose, diag cublas.Diag, m, n int, alpha float64, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, c cudart.DevicePtr, ldc int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Dtrmm(h.handle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, c, ldc)
}

// Ctrmm performs the triangular matrix-matrix operation C = alpha*op(A)*B or C = alpha*B*op(A) for single precision complex matrices.
func (h *CudaBlas) Ctrmm(side cublas.Side, uplo cublas.Uplo, trans cublas.Transpose, diag cublas.Diag, m, n int, alpha complex64, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, c cudart.DevicePtr, ldc int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Ctrmm(h.handle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, c, ldc)
}

// Ztrmm performs the triangular matrix-matrix operation C = alpha*op(A)*B or C = alpha*B*op(A) for double precision complex matrices.
func (h *CudaBlas) Ztrmm(side cublas.Side, uplo cublas.Uplo, trans cublas.Transpose, diag cublas.Diag, m, n int, alpha complex128, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, c cudart.DevicePtr, ldc int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Ztrmm(h.handle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, c, ldc)
}

// Level 3 BLAS Functions - Triangular Matrix Solve Operations

// Strsm solves the triangular matrix equation op(A)*X = alpha*B or X*op(A) = alpha*B for single precision real matrices.
func (h *CudaBlas) Strsm(side cublas.Side, uplo cublas.Uplo, trans cublas.Transpose, diag cublas.Diag, m, n int, alpha float32, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Strsm(h.handle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
}

// Dtrsm solves the triangular matrix equation op(A)*X = alpha*B or X*op(A) = alpha*B for double precision real matrices.
func (h *CudaBlas) Dtrsm(side cublas.Side, uplo cublas.Uplo, trans cublas.Transpose, diag cublas.Diag, m, n int, alpha float64, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Dtrsm(h.handle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
}

// Ctrsm solves the triangular matrix equation op(A)*X = alpha*B or X*op(A) = alpha*B for single precision complex matrices.
func (h *CudaBlas) Ctrsm(side cublas.Side, uplo cublas.Uplo, trans cublas.Transpose, diag cublas.Diag, m, n int, alpha complex64, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Ctrsm(h.handle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
}

// Ztrsm solves the triangular matrix equation op(A)*X = alpha*B or X*op(A) = alpha*B for double precision complex matrices.
func (h *CudaBlas) Ztrsm(side cublas.Side, uplo cublas.Uplo, trans cublas.Transpose, diag cublas.Diag, m, n int, alpha complex128, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Ztrsm(h.handle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
}

// Level 3 BLAS Functions - Hermitian Matrix-Matrix Operations

// Chemm performs the Hermitian matrix-matrix operation C = alpha*A*B + beta*C or C = alpha*B*A + beta*C for single precision complex matrices.
func (h *CudaBlas) Chemm(side cublas.Side, uplo cublas.Uplo, m, n int, alpha complex64, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta complex64, c cudart.DevicePtr, ldc int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Chemm(h.handle, side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc)
}

// Zhemm performs the Hermitian matrix-matrix operation C = alpha*A*B + beta*C or C = alpha*B*A + beta*C for double precision complex matrices.
func (h *CudaBlas) Zhemm(side cublas.Side, uplo cublas.Uplo, m, n int, alpha complex128, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta complex128, c cudart.DevicePtr, ldc int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Zhemm(h.handle, side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc)
}

// Level 3 BLAS Functions - Hermitian Rank-K Update Operations

// Cherk performs the Hermitian rank-k update C = alpha*A*A^H + beta*C for single precision complex matrices.
func (h *CudaBlas) Cherk(uplo cublas.Uplo, trans cublas.Transpose, n, k int, alpha float32, a cudart.DevicePtr, lda int, beta float32, c cudart.DevicePtr, ldc int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Cherk(h.handle, uplo, trans, n, k, alpha, a, lda, beta, c, ldc)
}

// Zherk performs the Hermitian rank-k update C = alpha*A*A^H + beta*C for double precision complex matrices.
func (h *CudaBlas) Zherk(uplo cublas.Uplo, trans cublas.Transpose, n, k int, alpha float64, a cudart.DevicePtr, lda int, beta float64, c cudart.DevicePtr, ldc int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Zherk(h.handle, uplo, trans, n, k, alpha, a, lda, beta, c, ldc)
}

// Level 3 BLAS Functions - Hermitian Rank-2K Update Operations

// Cher2k performs the Hermitian rank-2k update C = alpha*(A*B^H + B*A^H) + beta*C for single precision complex matrices.
func (h *CudaBlas) Cher2k(uplo cublas.Uplo, trans cublas.Transpose, n, k int, alpha complex64, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta float32, c cudart.DevicePtr, ldc int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Cher2k(h.handle, uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}

// Zher2k performs the Hermitian rank-2k update C = alpha*(A*B^H + B*A^H) + beta*C for double precision complex matrices.
func (h *CudaBlas) Zher2k(uplo cublas.Uplo, trans cublas.Transpose, n, k int, alpha complex128, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta float64, c cudart.DevicePtr, ldc int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Zher2k(h.handle, uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}

// Level 3 BLAS Functions - Hermitian Rank-KX Update Operations

// Cherkx performs the Hermitian rank-kx update C = alpha*A*B^H + beta*C for single precision complex matrices.
func (h *CudaBlas) Cherkx(uplo cublas.Uplo, trans cublas.Transpose, n, k int, alpha complex64, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta float32, c cudart.DevicePtr, ldc int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Cherkx(h.handle, uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}

// Zherkx performs the Hermitian rank-kx update C = alpha*A*B^H + beta*C for double precision complex matrices.
func (h *CudaBlas) Zherkx(uplo cublas.Uplo, trans cublas.Transpose, n, k int, alpha complex128, a cudart.DevicePtr, lda int, b cudart.DevicePtr, ldb int, beta float64, c cudart.DevicePtr, ldc int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Zherkx(h.handle, uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}
