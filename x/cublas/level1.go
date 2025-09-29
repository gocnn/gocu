package cublas

import (
	"github.com/gocnn/gocu/cublas"
	"github.com/gocnn/gocu/cudart"
)

// Level 1 BLAS Functions - Index Operations

// Isamax finds the index of the element with maximum absolute value in a single precision real vector.
func (h *CudaBlas) Isamax(n int, x cudart.DevicePtr, incX int) (int, error) {
	h.Lock()
	defer h.Unlock()
	return cublas.Isamax(h.handle, n, x, incX)
}

// Idamax finds the index of the element with maximum absolute value in a double precision real vector.
func (h *CudaBlas) Idamax(n int, x cudart.DevicePtr, incX int) (int, error) {
	h.Lock()
	defer h.Unlock()
	return cublas.Idamax(h.handle, n, x, incX)
}

// Icamax finds the index of the element with maximum absolute value in a single precision complex vector.
func (h *CudaBlas) Icamax(n int, x cudart.DevicePtr, incX int) (int, error) {
	h.Lock()
	defer h.Unlock()
	return cublas.Icamax(h.handle, n, x, incX)
}

// Izamax finds the index of the element with maximum absolute value in a double precision complex vector.
func (h *CudaBlas) Izamax(n int, x cudart.DevicePtr, incX int) (int, error) {
	h.Lock()
	defer h.Unlock()
	return cublas.Izamax(h.handle, n, x, incX)
}

// Isamin finds the index of the element with minimum absolute value in a single precision real vector.
func (h *CudaBlas) Isamin(n int, x cudart.DevicePtr, incX int) (int, error) {
	h.Lock()
	defer h.Unlock()
	return cublas.Isamin(h.handle, n, x, incX)
}

// Idamin finds the index of the element with minimum absolute value in a double precision real vector.
func (h *CudaBlas) Idamin(n int, x cudart.DevicePtr, incX int) (int, error) {
	h.Lock()
	defer h.Unlock()
	return cublas.Idamin(h.handle, n, x, incX)
}

// Icamin finds the index of the element with minimum absolute value in a single precision complex vector.
func (h *CudaBlas) Icamin(n int, x cudart.DevicePtr, incX int) (int, error) {
	h.Lock()
	defer h.Unlock()
	return cublas.Icamin(h.handle, n, x, incX)
}

// Izamin finds the index of the element with minimum absolute value in a double precision complex vector.
func (h *CudaBlas) Izamin(n int, x cudart.DevicePtr, incX int) (int, error) {
	h.Lock()
	defer h.Unlock()
	return cublas.Izamin(h.handle, n, x, incX)
}

// Level 1 BLAS Functions - Sum Operations

// Sasum computes the sum of the absolute values of elements in a single precision real vector.
func (h *CudaBlas) Sasum(n int, x cudart.DevicePtr, incX int) (float32, error) {
	h.Lock()
	defer h.Unlock()
	return cublas.Sasum(h.handle, n, x, incX)
}

// Dasum computes the sum of the absolute values of elements in a double precision real vector.
func (h *CudaBlas) Dasum(n int, x cudart.DevicePtr, incX int) (float64, error) {
	h.Lock()
	defer h.Unlock()
	return cublas.Dasum(h.handle, n, x, incX)
}

// Scasum computes the sum of the absolute values of elements in a single precision complex vector.
func (h *CudaBlas) Scasum(n int, x cudart.DevicePtr, incX int) (float32, error) {
	h.Lock()
	defer h.Unlock()
	return cublas.Scasum(h.handle, n, x, incX)
}

// Dzasum computes the sum of the absolute values of elements in a double precision complex vector.
func (h *CudaBlas) Dzasum(n int, x cudart.DevicePtr, incX int) (float64, error) {
	h.Lock()
	defer h.Unlock()
	return cublas.Dzasum(h.handle, n, x, incX)
}

// Level 1 BLAS Functions - Vector Operations

// Saxpy performs the operation y = alpha*x + y for single precision real vectors.
func (h *CudaBlas) Saxpy(n int, alpha float32, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Saxpy(h.handle, n, alpha, x, incX, y, incY)
}

// Daxpy performs the operation y = alpha*x + y for double precision real vectors.
func (h *CudaBlas) Daxpy(n int, alpha float64, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Daxpy(h.handle, n, alpha, x, incX, y, incY)
}

// Caxpy performs the operation y = alpha*x + y for single precision complex vectors.
func (h *CudaBlas) Caxpy(n int, alpha complex64, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Caxpy(h.handle, n, alpha, x, incX, y, incY)
}

// Zaxpy performs the operation y = alpha*x + y for double precision complex vectors.
func (h *CudaBlas) Zaxpy(n int, alpha complex128, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Zaxpy(h.handle, n, alpha, x, incX, y, incY)
}

// Scopy copies a single precision real vector x to vector y.
func (h *CudaBlas) Scopy(n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Scopy(h.handle, n, x, incX, y, incY)
}

// Dcopy copies a double precision real vector x to vector y.
func (h *CudaBlas) Dcopy(n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Dcopy(h.handle, n, x, incX, y, incY)
}

// Ccopy copies a single precision complex vector x to vector y.
func (h *CudaBlas) Ccopy(n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Ccopy(h.handle, n, x, incX, y, incY)
}

// Zcopy copies a double precision complex vector x to vector y.
func (h *CudaBlas) Zcopy(n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Zcopy(h.handle, n, x, incX, y, incY)
}

// Level 1 BLAS Functions - Dot Product Operations

// Sdot computes the dot product of two single precision real vectors.
func (h *CudaBlas) Sdot(n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) (float32, error) {
	h.Lock()
	defer h.Unlock()
	return cublas.Sdot(h.handle, n, x, incX, y, incY)
}

// Ddot computes the dot product of two double precision real vectors.
func (h *CudaBlas) Ddot(n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) (float64, error) {
	h.Lock()
	defer h.Unlock()
	return cublas.Ddot(h.handle, n, x, incX, y, incY)
}

// Cdotu computes the unconjugated dot product of two single precision complex vectors.
func (h *CudaBlas) Cdotu(n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) (complex64, error) {
	h.Lock()
	defer h.Unlock()
	return cublas.Cdotu(h.handle, n, x, incX, y, incY)
}

// Cdotc computes the conjugated dot product of two single precision complex vectors.
func (h *CudaBlas) Cdotc(n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) (complex64, error) {
	h.Lock()
	defer h.Unlock()
	return cublas.Cdotc(h.handle, n, x, incX, y, incY)
}

// Zdotu computes the unconjugated dot product of two double precision complex vectors.
func (h *CudaBlas) Zdotu(n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) (complex128, error) {
	h.Lock()
	defer h.Unlock()
	return cublas.Zdotu(h.handle, n, x, incX, y, incY)
}

// Zdotc computes the conjugated dot product of two double precision complex vectors.
func (h *CudaBlas) Zdotc(n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) (complex128, error) {
	h.Lock()
	defer h.Unlock()
	return cublas.Zdotc(h.handle, n, x, incX, y, incY)
}

// Level 1 BLAS Functions - Norm Operations

// Snrm2 computes the Euclidean norm of a single precision real vector.
func (h *CudaBlas) Snrm2(n int, x cudart.DevicePtr, incX int) (float32, error) {
	h.Lock()
	defer h.Unlock()
	return cublas.Snrm2(h.handle, n, x, incX)
}

// Dnrm2 computes the Euclidean norm of a double precision real vector.
func (h *CudaBlas) Dnrm2(n int, x cudart.DevicePtr, incX int) (float64, error) {
	h.Lock()
	defer h.Unlock()
	return cublas.Dnrm2(h.handle, n, x, incX)
}

// Scnrm2 computes the Euclidean norm of a single precision complex vector.
func (h *CudaBlas) Scnrm2(n int, x cudart.DevicePtr, incX int) (float32, error) {
	h.Lock()
	defer h.Unlock()
	return cublas.Scnrm2(h.handle, n, x, incX)
}

// Dznrm2 computes the Euclidean norm of a double precision complex vector.
func (h *CudaBlas) Dznrm2(n int, x cudart.DevicePtr, incX int) (float64, error) {
	h.Lock()
	defer h.Unlock()
	return cublas.Dznrm2(h.handle, n, x, incX)
}

// Level 1 BLAS Functions - Rotation Operations

// Srot applies a Givens rotation to single precision real vectors.
func (h *CudaBlas) Srot(n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, c, s float32) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Srot(h.handle, n, x, incX, y, incY, c, s)
}

// Drot applies a Givens rotation to double precision real vectors.
func (h *CudaBlas) Drot(n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, c, s float64) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Drot(h.handle, n, x, incX, y, incY, c, s)
}

// Crot applies a Givens rotation to single precision complex vectors.
func (h *CudaBlas) Crot(n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, c float32, s complex64) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Crot(h.handle, n, x, incX, y, incY, c, s)
}

// Csrot applies a Givens rotation to single precision complex vectors with real parameters.
func (h *CudaBlas) Csrot(n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, c, s float32) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Csrot(h.handle, n, x, incX, y, incY, c, s)
}

// Zrot applies a Givens rotation to double precision complex vectors.
func (h *CudaBlas) Zrot(n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, c float64, s complex128) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Zrot(h.handle, n, x, incX, y, incY, c, s)
}

// Zdrot applies a Givens rotation to double precision complex vectors with real parameters.
func (h *CudaBlas) Zdrot(n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, c, s float64) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Zdrot(h.handle, n, x, incX, y, incY, c, s)
}

// Srotg generates a Givens rotation for single precision real values.
func (h *CudaBlas) Srotg(a, b float32) (c, s float32, err error) {
	h.Lock()
	defer h.Unlock()
	c, s, err = cublas.Srotg(h.handle, a, b)
	return c, s, err
}

// Drotg generates a Givens rotation for double precision real values.
func (h *CudaBlas) Drotg(a, b float64) (c, s float64, err error) {
	h.Lock()
	defer h.Unlock()
	c, s, err = cublas.Drotg(h.handle, a, b)
	return c, s, err
}

// Crotg generates a Givens rotation for single precision complex values.
func (h *CudaBlas) Crotg(a, b complex64) (c float32, s complex64, err error) {
	h.Lock()
	defer h.Unlock()
	c, s, err = cublas.Crotg(h.handle, a, b)
	return c, s, err
}

// Zrotg generates a Givens rotation for double precision complex values.
func (h *CudaBlas) Zrotg(a, b complex128) (c float64, s complex128, err error) {
	h.Lock()
	defer h.Unlock()
	c, s, err = cublas.Zrotg(h.handle, a, b)
	return c, s, err
}

// Srotm applies a modified Givens rotation to single precision real vectors.
func (h *CudaBlas) Srotm(n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, param cublas.SrotmParams) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Srotm(h.handle, n, x, incX, y, incY, param)
}

// Drotm applies a modified Givens rotation to double precision real vectors.
func (h *CudaBlas) Drotm(n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, param cublas.DrotmParams) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Drotm(h.handle, n, x, incX, y, incY, param)
}

// Srotmg generates a modified Givens rotation for single precision real values.
func (h *CudaBlas) Srotmg(d1, d2, x1, y1 float32) (param cublas.SrotmParams, err error) {
	h.Lock()
	defer h.Unlock()
	p, err := cublas.Srotmg(h.handle, d1, d2, x1, y1)
	return p, err
}

// Drotmg generates a modified Givens rotation for double precision real values.
func (h *CudaBlas) Drotmg(d1, d2, x1, y1 float64) (param cublas.DrotmParams, err error) {
	h.Lock()
	defer h.Unlock()
	p, err := cublas.Drotmg(h.handle, d1, d2, x1, y1)
	return p, err
}

// Level 1 BLAS Functions - Scaling Operations

// Sscal scales a single precision real vector by a scalar.
func (h *CudaBlas) Sscal(n int, alpha float32, x cudart.DevicePtr, incX int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Sscal(h.handle, n, alpha, x, incX)
}

// Dscal scales a double precision real vector by a scalar.
func (h *CudaBlas) Dscal(n int, alpha float64, x cudart.DevicePtr, incX int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Dscal(h.handle, n, alpha, x, incX)
}

// Cscal scales a single precision complex vector by a complex scalar.
func (h *CudaBlas) Cscal(n int, alpha complex64, x cudart.DevicePtr, incX int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Cscal(h.handle, n, alpha, x, incX)
}

// Csscal scales a single precision complex vector by a real scalar.
func (h *CudaBlas) Csscal(n int, alpha float32, x cudart.DevicePtr, incX int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Csscal(h.handle, n, alpha, x, incX)
}

// Zscal scales a double precision complex vector by a complex scalar.
func (h *CudaBlas) Zscal(n int, alpha complex128, x cudart.DevicePtr, incX int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Zscal(h.handle, n, alpha, x, incX)
}

// Zdscal scales a double precision complex vector by a real scalar.
func (h *CudaBlas) Zdscal(n int, alpha float64, x cudart.DevicePtr, incX int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Zdscal(h.handle, n, alpha, x, incX)
}

// Level 1 BLAS Functions - Swap Operations

// Sswap swaps the contents of two single precision real vectors.
func (h *CudaBlas) Sswap(n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Sswap(h.handle, n, x, incX, y, incY)
}

// Dswap swaps the contents of two double precision real vectors.
func (h *CudaBlas) Dswap(n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Dswap(h.handle, n, x, incX, y, incY)
}

// Cswap swaps the contents of two single precision complex vectors.
func (h *CudaBlas) Cswap(n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Cswap(h.handle, n, x, incX, y, incY)
}

// Zswap swaps the contents of two double precision complex vectors.
func (h *CudaBlas) Zswap(n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) error {
	h.Lock()
	defer h.Unlock()
	return cublas.Zswap(h.handle, n, x, incX, y, incY)
}
