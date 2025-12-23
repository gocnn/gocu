//go:build cuda || cuda11 || cuda12 || cuda13

package cublas

/*
#include <cublas_v2.h>
*/
import "C"
import (
	"unsafe"

	"github.com/gocnn/gocu/cudart"
	"github.com/gocnn/gomat/blas"
)

// Isamax finds the index of the element with maximum absolute value in a single precision real vector.
// Returns the 1-based index of the element with maximum absolute value.
func Isamax(h *Handle, n int, x cudart.DevicePtr, incX int) (int, error) {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	var result C.int
	err := Check(C.cublasIsamax(C.cublasHandle_t(h.h), C.int(n), (*C.float)(x), C.int(incX), &result))
	return int(result), err
}

// Idamax finds the index of the element with maximum absolute value in a double precision real vector.
// Returns the 1-based index of the element with maximum absolute value.
func Idamax(h *Handle, n int, x cudart.DevicePtr, incX int) (int, error) {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	var result C.int
	err := Check(C.cublasIdamax(C.cublasHandle_t(h.h), C.int(n), (*C.double)(x), C.int(incX), &result))
	return int(result), err
}

// Icamax finds the index of the element with maximum absolute value in a single precision complex vector.
// Returns the 1-based index of the element with maximum absolute value.
func Icamax(h *Handle, n int, x cudart.DevicePtr, incX int) (int, error) {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	var result C.int
	err := Check(C.cublasIcamax(C.cublasHandle_t(h.h), C.int(n), (*C.cuComplex)(x), C.int(incX), &result))
	return int(result), err
}

// Izamax finds the index of the element with maximum absolute value in a double precision complex vector.
// Returns the 1-based index of the element with maximum absolute value.
func Izamax(h *Handle, n int, x cudart.DevicePtr, incX int) (int, error) {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	var result C.int
	err := Check(C.cublasIzamax(C.cublasHandle_t(h.h), C.int(n), (*C.cuDoubleComplex)(x), C.int(incX), &result))
	return int(result), err
}

// Isamin finds the index of the element with minimum absolute value in a single precision real vector.
// Returns the 1-based index of the element with minimum absolute value.
func Isamin(h *Handle, n int, x cudart.DevicePtr, incX int) (int, error) {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	var result C.int
	err := Check(C.cublasIsamin(C.cublasHandle_t(h.h), C.int(n), (*C.float)(x), C.int(incX), &result))
	return int(result), err
}

// Idamin finds the index of the element with minimum absolute value in a double precision real vector.
// Returns the 1-based index of the element with minimum absolute value.
func Idamin(h *Handle, n int, x cudart.DevicePtr, incX int) (int, error) {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	var result C.int
	err := Check(C.cublasIdamin(C.cublasHandle_t(h.h), C.int(n), (*C.double)(x), C.int(incX), &result))
	return int(result), err
}

// Icamin finds the index of the element with minimum absolute value in a single precision complex vector.
// Returns the 1-based index of the element with minimum absolute value.
func Icamin(h *Handle, n int, x cudart.DevicePtr, incX int) (int, error) {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	var result C.int
	err := Check(C.cublasIcamin(C.cublasHandle_t(h.h), C.int(n), (*C.cuComplex)(x), C.int(incX), &result))
	return int(result), err
}

// Izamin finds the index of the element with minimum absolute value in a double precision complex vector.
// Returns the 1-based index of the element with minimum absolute value.
func Izamin(h *Handle, n int, x cudart.DevicePtr, incX int) (int, error) {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	var result C.int
	err := Check(C.cublasIzamin(C.cublasHandle_t(h.h), C.int(n), (*C.cuDoubleComplex)(x), C.int(incX), &result))
	return int(result), err
}

// Sasum computes the sum of the absolute values of elements in a single precision real vector.
// Returns the sum of absolute values of the vector elements.
func Sasum(h *Handle, n int, x cudart.DevicePtr, incX int) (float32, error) {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	var result C.float
	err := Check(C.cublasSasum(C.cublasHandle_t(h.h), C.int(n), (*C.float)(x), C.int(incX), &result))
	return float32(result), err
}

// Dasum computes the sum of the absolute values of elements in a double precision real vector.
// Returns the sum of absolute values of the vector elements.
func Dasum(h *Handle, n int, x cudart.DevicePtr, incX int) (float64, error) {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	var result C.double
	err := Check(C.cublasDasum(C.cublasHandle_t(h.h), C.int(n), (*C.double)(x), C.int(incX), &result))
	return float64(result), err
}

// Scasum computes the sum of the absolute values of elements in a single precision complex vector.
// Returns the sum of absolute values of the vector elements as a real number.
func Scasum(h *Handle, n int, x cudart.DevicePtr, incX int) (float32, error) {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	var result C.float
	err := Check(C.cublasScasum(C.cublasHandle_t(h.h), C.int(n), (*C.cuComplex)(x), C.int(incX), &result))
	return float32(result), err
}

// Dzasum computes the sum of the absolute values of elements in a double precision complex vector.
// Returns the sum of absolute values of the vector elements as a real number.
func Dzasum(h *Handle, n int, x cudart.DevicePtr, incX int) (float64, error) {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	var result C.double
	err := Check(C.cublasDzasum(C.cublasHandle_t(h.h), C.int(n), (*C.cuDoubleComplex)(x), C.int(incX), &result))
	return float64(result), err
}

// Saxpy performs the vector operation y = alpha*x + y for single precision real vectors.
// The vectors x and y must have at least n elements with the given increments.
func Saxpy(h *Handle, n int, alpha float32, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) error {
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}

	return Check(C.cublasSaxpy(C.cublasHandle_t(h.h), C.int(n), (*C.float)(&alpha), (*C.float)(x), C.int(incX), (*C.float)(y), C.int(incY)))
}

// Daxpy performs the vector operation y = alpha*x + y for double precision real vectors.
// The vectors x and y must have at least n elements with the given increments.
func Daxpy(h *Handle, n int, alpha float64, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) error {
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}

	return Check(C.cublasDaxpy(C.cublasHandle_t(h.h), C.int(n), (*C.double)(&alpha), (*C.double)(x), C.int(incX), (*C.double)(y), C.int(incY)))
}

// Caxpy performs the vector operation y = alpha*x + y for single precision complex vectors.
// The vectors x and y must have at least n elements with the given increments.
func Caxpy(h *Handle, n int, alpha complex64, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) error {
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}

	return Check(C.cublasCaxpy(C.cublasHandle_t(h.h), C.int(n), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(x), C.int(incX), (*C.cuComplex)(y), C.int(incY)))
}

// Zaxpy performs the vector operation y = alpha*x + y for double precision complex vectors.
// The vectors x and y must have at least n elements with the given increments.
func Zaxpy(h *Handle, n int, alpha complex128, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) error {
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}

	return Check(C.cublasZaxpy(C.cublasHandle_t(h.h), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(x), C.int(incX), (*C.cuDoubleComplex)(y), C.int(incY)))
}

// Scopy copies a single precision real vector x to vector y.
// The vectors x and y must have at least n elements with the given increments.
func Scopy(h *Handle, n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) error {
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}

	return Check(C.cublasScopy(C.cublasHandle_t(h.h), C.int(n), (*C.float)(x), C.int(incX), (*C.float)(y), C.int(incY)))
}

// Dcopy copies a double precision real vector x to vector y.
// The vectors x and y must have at least n elements with the given increments.
func Dcopy(h *Handle, n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) error {
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}

	return Check(C.cublasDcopy(C.cublasHandle_t(h.h), C.int(n), (*C.double)(x), C.int(incX), (*C.double)(y), C.int(incY)))
}

// Ccopy copies a single precision complex vector x to vector y.
// The vectors x and y must have at least n elements with the given increments.
func Ccopy(h *Handle, n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) error {
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}

	return Check(C.cublasCcopy(C.cublasHandle_t(h.h), C.int(n), (*C.cuComplex)(x), C.int(incX), (*C.cuComplex)(y), C.int(incY)))
}

// Zcopy copies a double precision complex vector x to vector y.
// The vectors x and y must have at least n elements with the given increments.
func Zcopy(h *Handle, n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) error {
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}

	return Check(C.cublasZcopy(C.cublasHandle_t(h.h), C.int(n), (*C.cuDoubleComplex)(x), C.int(incX), (*C.cuDoubleComplex)(y), C.int(incY)))
}

// Sdot computes the dot product of two single precision real vectors.
// Returns the dot product x^T * y.
func Sdot(h *Handle, n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) (float32, error) {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}

	var result C.float
	err := Check(C.cublasSdot(C.cublasHandle_t(h.h), C.int(n), (*C.float)(x), C.int(incX), (*C.float)(y), C.int(incY), &result))
	return float32(result), err
}

// Ddot computes the dot product of two double precision real vectors.
// Returns the dot product x^T * y.
func Ddot(h *Handle, n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) (float64, error) {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}

	var result C.double
	err := Check(C.cublasDdot(C.cublasHandle_t(h.h), C.int(n), (*C.double)(x), C.int(incX), (*C.double)(y), C.int(incY), &result))
	return float64(result), err
}

// Cdotu computes the unconjugated dot product of two single precision complex vectors.
// Returns the dot product x^T * y (without conjugation).
func Cdotu(h *Handle, n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) (complex64, error) {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}

	var result complex64
	err := Check(C.cublasCdotu(C.cublasHandle_t(h.h), C.int(n), (*C.cuComplex)(x), C.int(incX), (*C.cuComplex)(y), C.int(incY), (*C.cuComplex)(unsafe.Pointer(&result))))
	return result, err
}

// Cdotc computes the conjugated dot product of two single precision complex vectors.
// Returns the dot product x^H * y (with conjugation of x).
func Cdotc(h *Handle, n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) (complex64, error) {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}

	var result complex64
	err := Check(C.cublasCdotc(C.cublasHandle_t(h.h), C.int(n), (*C.cuComplex)(x), C.int(incX), (*C.cuComplex)(y), C.int(incY), (*C.cuComplex)(unsafe.Pointer(&result))))
	return result, err
}

// Zdotu computes the unconjugated dot product of two double precision complex vectors.
// Returns the dot product x^T * y (without conjugation).
func Zdotu(h *Handle, n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) (complex128, error) {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}

	var result complex128
	err := Check(C.cublasZdotu(C.cublasHandle_t(h.h), C.int(n), (*C.cuDoubleComplex)(x), C.int(incX), (*C.cuDoubleComplex)(y), C.int(incY), (*C.cuDoubleComplex)(unsafe.Pointer(&result))))
	return result, err
}

// Zdotc computes the conjugated dot product of two double precision complex vectors.
// Returns the dot product x^H * y (with conjugation of x).
func Zdotc(h *Handle, n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) (complex128, error) {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}

	var result complex128
	err := Check(C.cublasZdotc(C.cublasHandle_t(h.h), C.int(n), (*C.cuDoubleComplex)(x), C.int(incX), (*C.cuDoubleComplex)(y), C.int(incY), (*C.cuDoubleComplex)(unsafe.Pointer(&result))))
	return result, err
}

// Snrm2 computes the Euclidean norm (2-norm) of a single precision real vector.
// Returns the 2-norm: sqrt(sum(x[i]^2)).
func Snrm2(h *Handle, n int, x cudart.DevicePtr, incX int) (float32, error) {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	var result C.float
	err := Check(C.cublasSnrm2(C.cublasHandle_t(h.h), C.int(n), (*C.float)(x), C.int(incX), &result))
	return float32(result), err
}

// Dnrm2 computes the Euclidean norm (2-norm) of a double precision real vector.
// Returns the 2-norm: sqrt(sum(x[i]^2)).
func Dnrm2(h *Handle, n int, x cudart.DevicePtr, incX int) (float64, error) {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	var result C.double
	err := Check(C.cublasDnrm2(C.cublasHandle_t(h.h), C.int(n), (*C.double)(x), C.int(incX), &result))
	return float64(result), err
}

// Scnrm2 computes the Euclidean norm (2-norm) of a single precision complex vector.
// Returns the 2-norm: sqrt(sum(|x[i]|^2)) as a real number.
func Scnrm2(h *Handle, n int, x cudart.DevicePtr, incX int) (float32, error) {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	var result C.float
	err := Check(C.cublasScnrm2(C.cublasHandle_t(h.h), C.int(n), (*C.cuComplex)(x), C.int(incX), &result))
	return float32(result), err
}

// Dznrm2 computes the Euclidean norm (2-norm) of a double precision complex vector.
// Returns the 2-norm: sqrt(sum(|x[i]|^2)) as a real number.
func Dznrm2(h *Handle, n int, x cudart.DevicePtr, incX int) (float64, error) {

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	var result C.double
	err := Check(C.cublasDznrm2(C.cublasHandle_t(h.h), C.int(n), (*C.cuDoubleComplex)(x), C.int(incX), &result))
	return float64(result), err
}

// Srot applies a Givens rotation to single precision real vectors x and y.
// Performs the transformation: x[i] = c*x[i] + s*y[i], y[i] = c*y[i] - s*x[i].
func Srot(h *Handle, n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, c, s float32) error {
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}

	return Check(C.cublasSrot(C.cublasHandle_t(h.h), C.int(n), (*C.float)(x), C.int(incX), (*C.float)(y), C.int(incY), (*C.float)(&c), (*C.float)(&s)))
}

// Drot applies a Givens rotation to double precision real vectors x and y.
// Performs the transformation: x[i] = c*x[i] + s*y[i], y[i] = c*y[i] - s*x[i].
func Drot(h *Handle, n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, c, s float64) error {
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}

	return Check(C.cublasDrot(C.cublasHandle_t(h.h), C.int(n), (*C.double)(x), C.int(incX), (*C.double)(y), C.int(incY), (*C.double)(&c), (*C.double)(&s)))
}

// Crot applies a Givens rotation to single precision complex vectors x and y.
// Performs the transformation: x[i] = c*x[i] + s*y[i], y[i] = c*y[i] - conj(s)*x[i].
func Crot(h *Handle, n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, c float32, s complex64) error {
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}

	return Check(C.cublasCrot(C.cublasHandle_t(h.h), C.int(n), (*C.cuComplex)(x), C.int(incX), (*C.cuComplex)(y), C.int(incY), (*C.float)(&c), (*C.cuComplex)(unsafe.Pointer(&s))))
}

// Csrot applies a real Givens rotation to single precision complex vectors x and y.
// Performs the transformation: x[i] = c*x[i] + s*y[i], y[i] = c*y[i] - s*x[i].
func Csrot(h *Handle, n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, c, s float32) error {
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}

	return Check(C.cublasCsrot(C.cublasHandle_t(h.h), C.int(n), (*C.cuComplex)(x), C.int(incX), (*C.cuComplex)(y), C.int(incY), (*C.float)(&c), (*C.float)(&s)))
}

// Zrot applies a Givens rotation to double precision complex vectors x and y.
// Performs the transformation: x[i] = c*x[i] + s*y[i], y[i] = c*y[i] - conj(s)*x[i].
func Zrot(h *Handle, n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, c float64, s complex128) error {
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}

	return Check(C.cublasZrot(C.cublasHandle_t(h.h), C.int(n), (*C.cuDoubleComplex)(x), C.int(incX), (*C.cuDoubleComplex)(y), C.int(incY), (*C.double)(&c), (*C.cuDoubleComplex)(unsafe.Pointer(&s))))
}

// Zdrot applies a real Givens rotation to double precision complex vectors x and y.
// Performs the transformation: x[i] = c*x[i] + s*y[i], y[i] = c*y[i] - s*x[i].
func Zdrot(h *Handle, n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, c, s float64) error {
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}

	return Check(C.cublasZdrot(C.cublasHandle_t(h.h), C.int(n), (*C.cuDoubleComplex)(x), C.int(incX), (*C.cuDoubleComplex)(y), C.int(incY), (*C.double)(&c), (*C.double)(&s)))
}

// Srotg generates the parameters for a Givens rotation matrix for single precision real values.
// Given input values a and b, computes c, s, and r such that:
// [c  s] [a] = [r]
// [-s c] [b]   [0]
// The function modifies a to contain r, and b to contain z (reconstruction parameter).
func Srotg(h *Handle, a, b float32) (c, s float32, err error) {
	err = Check(C.cublasSrotg(C.cublasHandle_t(h.h), (*C.float)(&a), (*C.float)(&b), (*C.float)(&c), (*C.float)(&s)))
	return c, s, err
}

// Drotg generates the parameters for a Givens rotation matrix for double precision real values.
// Given input values a and b, computes c, s, and r such that:
// [c  s] [a] = [r]
// [-s c] [b]   [0]
// The function modifies a to contain r, and b to contain z (reconstruction parameter).
func Drotg(h *Handle, a, b float64) (c, s float64, err error) {
	err = Check(C.cublasDrotg(C.cublasHandle_t(h.h), (*C.double)(&a), (*C.double)(&b), (*C.double)(&c), (*C.double)(&s)))
	return c, s, err
}

// Crotg generates the parameters for a Givens rotation matrix for single precision complex values.
// Given input values a and b, computes c, s, and r such that:
// [c       s] [a] = [r]
// [-conj(s) c] [b]   [0]
// The function modifies a to contain r, and b to contain z (reconstruction parameter).
func Crotg(h *Handle, a, b complex64) (c float32, s complex64, err error) {
	err = Check(C.cublasCrotg(C.cublasHandle_t(h.h), (*C.cuComplex)(unsafe.Pointer(&a)), (*C.cuComplex)(unsafe.Pointer(&b)), (*C.float)(&c), (*C.cuComplex)(unsafe.Pointer(&s))))
	return c, s, err
}

// Zrotg generates the parameters for a Givens rotation matrix for double precision complex values.
// Given input values a and b, computes c, s, and r such that:
// [c       s] [a] = [r]
// [-conj(s) c] [b]   [0]
// The function modifies a to contain r, and b to contain z (reconstruction parameter).
func Zrotg(h *Handle, a, b complex128) (c float64, s complex128, err error) {
	err = Check(C.cublasZrotg(C.cublasHandle_t(h.h), (*C.cuDoubleComplex)(unsafe.Pointer(&a)), (*C.cuDoubleComplex)(unsafe.Pointer(&b)), (*C.double)(&c), (*C.cuDoubleComplex)(unsafe.Pointer(&s))))
	return c, s, err
}

// Srotm applies a modified Givens rotation to single precision real vectors x and y.
// The rotation is defined by the param array which contains the rotation parameters.
// param[0] contains the flag indicating the form of the H matrix:
//
//	-1.0: H has the form [[H11, H12], [H21, H22]]
//	 0.0: H has the form [[1.0, H12], [H21, 1.0]]
//	 1.0: H has the form [[H11, 1.0], [-1.0, H22]]
//	-2.0: H is the identity matrix
func Srotm(h *Handle, n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, p SrotmParams) error {
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}
	return Check(C.cublasSrotm(C.cublasHandle_t(h.h), C.int(n), (*C.float)(x), C.int(incX), (*C.float)(y), C.int(incY), (*C.float)(unsafe.Pointer(&p))))
}

// Drotm applies a modified Givens rotation to double precision real vectors x and y.
// The rotation is defined by the param array which contains the rotation parameters.
// param[0] contains the flag indicating the form of the H matrix:
//
//	-1.0: H has the form [[H11, H12], [H21, H22]]
//	 0.0: H has the form [[1.0, H12], [H21, 1.0]]
//	 1.0: H has the form [[H11, 1.0], [-1.0, H22]]
//	-2.0: H is the identity matrix
func Drotm(h *Handle, n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, p DrotmParams) error {
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}
	return Check(C.cublasDrotm(C.cublasHandle_t(h.h), C.int(n), (*C.double)(x), C.int(incX), (*C.double)(y), C.int(incY), (*C.double)(unsafe.Pointer(&p))))
}

// Srotmg generates the parameters for a modified Givens rotation matrix for single precision real values.
// Given input values d1, d2, x1, and y1, computes the modified Givens rotation matrix H
// and the parameters stored in the param array. The function modifies d1, d2, and x1.
// param[0] contains the flag indicating the form of the H matrix:
//
//	-1.0: H has the form [[H11, H12], [H21, H22]]
//	 0.0: H has the form [[1.0, H12], [H21, 1.0]]
//	 1.0: H has the form [[H11, 1.0], [-1.0, H22]]
//	-2.0: H is the identity matrix
func Srotmg(h *Handle, d1, d2, x1, y1 float32) (p SrotmParams, err error) {
	err = Check(C.cublasSrotmg(C.cublasHandle_t(h.h), (*C.float)(&d1), (*C.float)(&d2), (*C.float)(&x1), (*C.float)(&y1), (*C.float)(unsafe.Pointer(&p))))
	return p, err
}

// Drotmg generates the parameters for a modified Givens rotation matrix for double precision real values.
// Given input values d1, d2, x1, and y1, computes the modified Givens rotation matrix H
// and the parameters stored in the param array. The function modifies d1, d2, and x1.
// param[0] contains the flag indicating the form of the H matrix:
//
//	-1.0: H has the form [[H11, H12], [H21, H22]]
//	 0.0: H has the form [[1.0, H12], [H21, 1.0]]
//	 1.0: H has the form [[H11, 1.0], [-1.0, H22]]
//	-2.0: H is the identity matrix
func Drotmg(h *Handle, d1, d2, x1, y1 float64) (p DrotmParams, err error) {
	err = Check(C.cublasDrotmg(C.cublasHandle_t(h.h), (*C.double)(&d1), (*C.double)(&d2), (*C.double)(&x1), (*C.double)(&y1), (*C.double)(unsafe.Pointer(&p))))
	return p, err
}

// Sscal scales a single precision real vector by a scalar.
// Performs the operation: x = alpha * x.
func Sscal(h *Handle, n int, alpha float32, x cudart.DevicePtr, incX int) error {
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	return Check(C.cublasSscal(C.cublasHandle_t(h.h), C.int(n), (*C.float)(&alpha), (*C.float)(x), C.int(incX)))
}

// Dscal scales a double precision real vector by a scalar.
// Performs the operation: x = alpha * x.
func Dscal(h *Handle, n int, alpha float64, x cudart.DevicePtr, incX int) error {
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	return Check(C.cublasDscal(C.cublasHandle_t(h.h), C.int(n), (*C.double)(&alpha), (*C.double)(x), C.int(incX)))
}

// Cscal scales a single precision complex vector by a complex scalar.
// Performs the operation: x = alpha * x.
func Cscal(h *Handle, n int, alpha complex64, x cudart.DevicePtr, incX int) error {
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	return Check(C.cublasCscal(C.cublasHandle_t(h.h), C.int(n), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(x), C.int(incX)))
}

// Csscal scales a single precision complex vector by a real scalar.
// Performs the operation: x = alpha * x (where alpha is real).
func Csscal(h *Handle, n int, alpha float32, x cudart.DevicePtr, incX int) error {
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	return Check(C.cublasCsscal(C.cublasHandle_t(h.h), C.int(n), (*C.float)(&alpha), (*C.cuComplex)(x), C.int(incX)))
}

// Zscal scales a double precision complex vector by a complex scalar.
// Performs the operation: x = alpha * x.
func Zscal(h *Handle, n int, alpha complex128, x cudart.DevicePtr, incX int) error {
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	return Check(C.cublasZscal(C.cublasHandle_t(h.h), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(x), C.int(incX)))
}

// Zdscal scales a double precision complex vector by a real scalar.
// Performs the operation: x = alpha * x (where alpha is real).
func Zdscal(h *Handle, n int, alpha float64, x cudart.DevicePtr, incX int) error {
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	return Check(C.cublasZdscal(C.cublasHandle_t(h.h), C.int(n), (*C.double)(&alpha), (*C.cuDoubleComplex)(x), C.int(incX)))
}

// Sswap swaps the elements of two single precision real vectors.
// Performs the operation: x <-> y (exchanges the contents of vectors x and y).
func Sswap(h *Handle, n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) error {
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}

	return Check(C.cublasSswap(C.cublasHandle_t(h.h), C.int(n), (*C.float)(x), C.int(incX), (*C.float)(y), C.int(incY)))
}

// Dswap swaps the elements of two double precision real vectors.
// Performs the operation: x <-> y (exchanges the contents of vectors x and y).
func Dswap(h *Handle, n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) error {
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}

	return Check(C.cublasDswap(C.cublasHandle_t(h.h), C.int(n), (*C.double)(x), C.int(incX), (*C.double)(y), C.int(incY)))
}

// Cswap swaps the elements of two single precision complex vectors.
// Performs the operation: x <-> y (exchanges the contents of vectors x and y).
func Cswap(h *Handle, n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) error {
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}

	return Check(C.cublasCswap(C.cublasHandle_t(h.h), C.int(n), (*C.cuComplex)(x), C.int(incX), (*C.cuComplex)(y), C.int(incY)))
}

// Zswap swaps the elements of two double precision complex vectors.
// Performs the operation: x <-> y (exchanges the contents of vectors x and y).
func Zswap(h *Handle, n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) error {
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}

	return Check(C.cublasZswap(C.cublasHandle_t(h.h), C.int(n), (*C.cuDoubleComplex)(x), C.int(incX), (*C.cuDoubleComplex)(y), C.int(incY)))
}
