//go:build !cuda && !cuda11 && !cuda12 && !cuda13

package cublas

import (
	"github.com/gocnn/gocu/cudart"
)

// Level 1 BLAS - Index operations
func Isamax(h *Handle, n int, x cudart.DevicePtr, incX int) (int, error) {
	return 0, ErrNotAvailable
}
func Idamax(h *Handle, n int, x cudart.DevicePtr, incX int) (int, error) {
	return 0, ErrNotAvailable
}
func Icamax(h *Handle, n int, x cudart.DevicePtr, incX int) (int, error) {
	return 0, ErrNotAvailable
}
func Izamax(h *Handle, n int, x cudart.DevicePtr, incX int) (int, error) {
	return 0, ErrNotAvailable
}
func Isamin(h *Handle, n int, x cudart.DevicePtr, incX int) (int, error) {
	return 0, ErrNotAvailable
}
func Idamin(h *Handle, n int, x cudart.DevicePtr, incX int) (int, error) {
	return 0, ErrNotAvailable
}
func Icamin(h *Handle, n int, x cudart.DevicePtr, incX int) (int, error) {
	return 0, ErrNotAvailable
}
func Izamin(h *Handle, n int, x cudart.DevicePtr, incX int) (int, error) {
	return 0, ErrNotAvailable
}

// Level 1 BLAS - Sum operations
func Sasum(h *Handle, n int, x cudart.DevicePtr, incX int) (float32, error) {
	return 0, ErrNotAvailable
}
func Dasum(h *Handle, n int, x cudart.DevicePtr, incX int) (float64, error) {
	return 0, ErrNotAvailable
}
func Scasum(h *Handle, n int, x cudart.DevicePtr, incX int) (float32, error) {
	return 0, ErrNotAvailable
}
func Dzasum(h *Handle, n int, x cudart.DevicePtr, incX int) (float64, error) {
	return 0, ErrNotAvailable
}

// Level 1 BLAS - Axpy operations
func Saxpy(h *Handle, n int, alpha float32, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) error {
	return ErrNotAvailable
}
func Daxpy(h *Handle, n int, alpha float64, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) error {
	return ErrNotAvailable
}
func Caxpy(h *Handle, n int, alpha complex64, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) error {
	return ErrNotAvailable
}
func Zaxpy(h *Handle, n int, alpha complex128, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) error {
	return ErrNotAvailable
}

// Level 1 BLAS - Copy operations
func Scopy(h *Handle, n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) error {
	return ErrNotAvailable
}
func Dcopy(h *Handle, n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) error {
	return ErrNotAvailable
}
func Ccopy(h *Handle, n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) error {
	return ErrNotAvailable
}
func Zcopy(h *Handle, n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) error {
	return ErrNotAvailable
}

// Level 1 BLAS - Dot product operations
func Sdot(h *Handle, n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) (float32, error) {
	return 0, ErrNotAvailable
}
func Ddot(h *Handle, n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) (float64, error) {
	return 0, ErrNotAvailable
}
func Cdotu(h *Handle, n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) (complex64, error) {
	return 0, ErrNotAvailable
}
func Cdotc(h *Handle, n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) (complex64, error) {
	return 0, ErrNotAvailable
}
func Zdotu(h *Handle, n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) (complex128, error) {
	return 0, ErrNotAvailable
}
func Zdotc(h *Handle, n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) (complex128, error) {
	return 0, ErrNotAvailable
}

// Level 1 BLAS - Norm operations
func Snrm2(h *Handle, n int, x cudart.DevicePtr, incX int) (float32, error) {
	return 0, ErrNotAvailable
}
func Dnrm2(h *Handle, n int, x cudart.DevicePtr, incX int) (float64, error) {
	return 0, ErrNotAvailable
}
func Scnrm2(h *Handle, n int, x cudart.DevicePtr, incX int) (float32, error) {
	return 0, ErrNotAvailable
}
func Dznrm2(h *Handle, n int, x cudart.DevicePtr, incX int) (float64, error) {
	return 0, ErrNotAvailable
}

// Level 1 BLAS - Rotation operations
func Srot(h *Handle, n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, c, s float32) error {
	return ErrNotAvailable
}
func Drot(h *Handle, n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, c, s float64) error {
	return ErrNotAvailable
}
func Crot(h *Handle, n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, c float32, s complex64) error {
	return ErrNotAvailable
}
func Csrot(h *Handle, n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, c, s float32) error {
	return ErrNotAvailable
}
func Zrot(h *Handle, n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, c float64, s complex128) error {
	return ErrNotAvailable
}
func Zdrot(h *Handle, n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, c, s float64) error {
	return ErrNotAvailable
}

// Level 1 BLAS - Rotation generation operations
func Srotg(h *Handle, a, b float32) (c, s float32, err error) {
	return 0, 0, ErrNotAvailable
}
func Drotg(h *Handle, a, b float64) (c, s float64, err error) {
	return 0, 0, ErrNotAvailable
}
func Crotg(h *Handle, a, b complex64) (c float32, s complex64, err error) {
	return 0, 0, ErrNotAvailable
}
func Zrotg(h *Handle, a, b complex128) (c float64, s complex128, err error) {
	return 0, 0, ErrNotAvailable
}

// Level 1 BLAS - Modified Givens rotation operations
func Srotm(h *Handle, n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, p SrotmParams) error {
	return ErrNotAvailable
}
func Drotm(h *Handle, n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, p DrotmParams) error {
	return ErrNotAvailable
}
func Srotmg(h *Handle, d1, d2, x1, y1 float32) (p SrotmParams, err error) {
	return SrotmParams{}, ErrNotAvailable
}
func Drotmg(h *Handle, d1, d2, x1, y1 float64) (p DrotmParams, err error) {
	return DrotmParams{}, ErrNotAvailable
}

// Level 1 BLAS - Scale operations
func Sscal(h *Handle, n int, alpha float32, x cudart.DevicePtr, incX int) error {
	return ErrNotAvailable
}
func Dscal(h *Handle, n int, alpha float64, x cudart.DevicePtr, incX int) error {
	return ErrNotAvailable
}
func Cscal(h *Handle, n int, alpha complex64, x cudart.DevicePtr, incX int) error {
	return ErrNotAvailable
}
func Csscal(h *Handle, n int, alpha float32, x cudart.DevicePtr, incX int) error {
	return ErrNotAvailable
}
func Zscal(h *Handle, n int, alpha complex128, x cudart.DevicePtr, incX int) error {
	return ErrNotAvailable
}
func Zdscal(h *Handle, n int, alpha float64, x cudart.DevicePtr, incX int) error {
	return ErrNotAvailable
}

// Level 1 BLAS - Swap operations
func Sswap(h *Handle, n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) error {
	return ErrNotAvailable
}
func Dswap(h *Handle, n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) error {
	return ErrNotAvailable
}
func Cswap(h *Handle, n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) error {
	return ErrNotAvailable
}
func Zswap(h *Handle, n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) error {
	return ErrNotAvailable
}
