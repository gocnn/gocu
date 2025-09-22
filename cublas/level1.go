package cublas

// #include <cublas_v2.h>
import "C"
import (
	"unsafe"

	"github.com/gocnn/gocu/cudart"
	"github.com/gocnn/gomat/blas"
)

// Isamax finds the index of the element with maximum absolute value in a single precision real vector.
// Returns the 1-based index of the element with maximum absolute value.
func (impl *Standard) Isamax(n int, x cudart.DevicePtr, incX int) int {
	if impl.e != nil {
		return 0
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	var result C.int
	impl.e = Check(C.cublasIsamax(C.cublasHandle_t(impl.h), C.int(n), (*C.float)(x), C.int(incX), &result))
	return int(result)
}

// Idamax finds the index of the element with maximum absolute value in a double precision real vector.
// Returns the 1-based index of the element with maximum absolute value.
func (impl *Standard) Idamax(n int, x cudart.DevicePtr, incX int) int {
	if impl.e != nil {
		return 0
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	var result C.int
	impl.e = Check(C.cublasIdamax(C.cublasHandle_t(impl.h), C.int(n), (*C.double)(x), C.int(incX), &result))
	return int(result)
}

// Icamax finds the index of the element with maximum absolute value in a single precision complex vector.
// Returns the 1-based index of the element with maximum absolute value.
func (impl *Standard) Icamax(n int, x cudart.DevicePtr, incX int) int {
	if impl.e != nil {
		return 0
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	var result C.int
	impl.e = Check(C.cublasIcamax(C.cublasHandle_t(impl.h), C.int(n), (*C.cuComplex)(x), C.int(incX), &result))
	return int(result)
}

// Izamax finds the index of the element with maximum absolute value in a double precision complex vector.
// Returns the 1-based index of the element with maximum absolute value.
func (impl *Standard) Izamax(n int, x cudart.DevicePtr, incX int) int {
	if impl.e != nil {
		return 0
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	var result C.int
	impl.e = Check(C.cublasIzamax(C.cublasHandle_t(impl.h), C.int(n), (*C.cuDoubleComplex)(x), C.int(incX), &result))
	return int(result)
}

// Isamin finds the index of the element with minimum absolute value in a single precision real vector.
// Returns the 1-based index of the element with minimum absolute value.
func (impl *Standard) Isamin(n int, x cudart.DevicePtr, incX int) int {
	if impl.e != nil {
		return 0
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	var result C.int
	impl.e = Check(C.cublasIsamin(C.cublasHandle_t(impl.h), C.int(n), (*C.float)(x), C.int(incX), &result))
	return int(result)
}

// Idamin finds the index of the element with minimum absolute value in a double precision real vector.
// Returns the 1-based index of the element with minimum absolute value.
func (impl *Standard) Idamin(n int, x cudart.DevicePtr, incX int) int {
	if impl.e != nil {
		return 0
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	var result C.int
	impl.e = Check(C.cublasIdamin(C.cublasHandle_t(impl.h), C.int(n), (*C.double)(x), C.int(incX), &result))
	return int(result)
}

// Icamin finds the index of the element with minimum absolute value in a single precision complex vector.
// Returns the 1-based index of the element with minimum absolute value.
func (impl *Standard) Icamin(n int, x cudart.DevicePtr, incX int) int {
	if impl.e != nil {
		return 0
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	var result C.int
	impl.e = Check(C.cublasIcamin(C.cublasHandle_t(impl.h), C.int(n), (*C.cuComplex)(x), C.int(incX), &result))
	return int(result)
}

// Izamin finds the index of the element with minimum absolute value in a double precision complex vector.
// Returns the 1-based index of the element with minimum absolute value.
func (impl *Standard) Izamin(n int, x cudart.DevicePtr, incX int) int {
	if impl.e != nil {
		return 0
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	var result C.int
	impl.e = Check(C.cublasIzamin(C.cublasHandle_t(impl.h), C.int(n), (*C.cuDoubleComplex)(x), C.int(incX), &result))
	return int(result)
}

// Sasum computes the sum of the absolute values of elements in a single precision real vector.
// Returns the sum of absolute values of the vector elements.
func (impl *Standard) Sasum(n int, x cudart.DevicePtr, incX int) float32 {
	if impl.e != nil {
		return 0
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	var result C.float
	impl.e = Check(C.cublasSasum(C.cublasHandle_t(impl.h), C.int(n), (*C.float)(x), C.int(incX), &result))
	return float32(result)
}

// Dasum computes the sum of the absolute values of elements in a double precision real vector.
// Returns the sum of absolute values of the vector elements.
func (impl *Standard) Dasum(n int, x cudart.DevicePtr, incX int) float64 {
	if impl.e != nil {
		return 0
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	var result C.double
	impl.e = Check(C.cublasDasum(C.cublasHandle_t(impl.h), C.int(n), (*C.double)(x), C.int(incX), &result))
	return float64(result)
}

// Scasum computes the sum of the absolute values of elements in a single precision complex vector.
// Returns the sum of absolute values of the vector elements as a real number.
func (impl *Standard) Scasum(n int, x cudart.DevicePtr, incX int) float32 {
	if impl.e != nil {
		return 0
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	var result C.float
	impl.e = Check(C.cublasScasum(C.cublasHandle_t(impl.h), C.int(n), (*C.cuComplex)(x), C.int(incX), &result))
	return float32(result)
}

// Dzasum computes the sum of the absolute values of elements in a double precision complex vector.
// Returns the sum of absolute values of the vector elements as a real number.
func (impl *Standard) Dzasum(n int, x cudart.DevicePtr, incX int) float64 {
	if impl.e != nil {
		return 0
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	var result C.double
	impl.e = Check(C.cublasDzasum(C.cublasHandle_t(impl.h), C.int(n), (*C.cuDoubleComplex)(x), C.int(incX), &result))
	return float64(result)
}

// Saxpy performs the vector operation y = alpha*x + y for single precision real vectors.
// The vectors x and y must have at least n elements with the given increments.
func (impl *Standard) Saxpy(n int, alpha float32, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) {
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

	impl.e = Check(C.cublasSaxpy(C.cublasHandle_t(impl.h), C.int(n), (*C.float)(&alpha), (*C.float)(x), C.int(incX), (*C.float)(y), C.int(incY)))
}

// Daxpy performs the vector operation y = alpha*x + y for double precision real vectors.
// The vectors x and y must have at least n elements with the given increments.
func (impl *Standard) Daxpy(n int, alpha float64, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) {
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

	impl.e = Check(C.cublasDaxpy(C.cublasHandle_t(impl.h), C.int(n), (*C.double)(&alpha), (*C.double)(x), C.int(incX), (*C.double)(y), C.int(incY)))
}

// Caxpy performs the vector operation y = alpha*x + y for single precision complex vectors.
// The vectors x and y must have at least n elements with the given increments.
func (impl *Standard) Caxpy(n int, alpha complex64, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) {
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

	impl.e = Check(C.cublasCaxpy(C.cublasHandle_t(impl.h), C.int(n), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(x), C.int(incX), (*C.cuComplex)(y), C.int(incY)))
}

// Zaxpy performs the vector operation y = alpha*x + y for double precision complex vectors.
// The vectors x and y must have at least n elements with the given increments.
func (impl *Standard) Zaxpy(n int, alpha complex128, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) {
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

	impl.e = Check(C.cublasZaxpy(C.cublasHandle_t(impl.h), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(x), C.int(incX), (*C.cuDoubleComplex)(y), C.int(incY)))
}

// Scopy copies a single precision real vector x to vector y.
// The vectors x and y must have at least n elements with the given increments.
func (impl *Standard) Scopy(n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) {
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

	impl.e = Check(C.cublasScopy(C.cublasHandle_t(impl.h), C.int(n), (*C.float)(x), C.int(incX), (*C.float)(y), C.int(incY)))
}

// Dcopy copies a double precision real vector x to vector y.
// The vectors x and y must have at least n elements with the given increments.
func (impl *Standard) Dcopy(n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) {
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

	impl.e = Check(C.cublasDcopy(C.cublasHandle_t(impl.h), C.int(n), (*C.double)(x), C.int(incX), (*C.double)(y), C.int(incY)))
}

// Ccopy copies a single precision complex vector x to vector y.
// The vectors x and y must have at least n elements with the given increments.
func (impl *Standard) Ccopy(n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) {
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

	impl.e = Check(C.cublasCcopy(C.cublasHandle_t(impl.h), C.int(n), (*C.cuComplex)(x), C.int(incX), (*C.cuComplex)(y), C.int(incY)))
}

// Zcopy copies a double precision complex vector x to vector y.
// The vectors x and y must have at least n elements with the given increments.
func (impl *Standard) Zcopy(n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) {
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

	impl.e = Check(C.cublasZcopy(C.cublasHandle_t(impl.h), C.int(n), (*C.cuDoubleComplex)(x), C.int(incX), (*C.cuDoubleComplex)(y), C.int(incY)))
}

// Sdot computes the dot product of two single precision real vectors.
// Returns the dot product x^T * y.
func (impl *Standard) Sdot(n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) float32 {
	if impl.e != nil {
		return 0
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

	var result C.float
	impl.e = Check(C.cublasSdot(C.cublasHandle_t(impl.h), C.int(n), (*C.float)(x), C.int(incX), (*C.float)(y), C.int(incY), &result))
	return float32(result)
}

// Ddot computes the dot product of two double precision real vectors.
// Returns the dot product x^T * y.
func (impl *Standard) Ddot(n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) float64 {
	if impl.e != nil {
		return 0
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

	var result C.double
	impl.e = Check(C.cublasDdot(C.cublasHandle_t(impl.h), C.int(n), (*C.double)(x), C.int(incX), (*C.double)(y), C.int(incY), &result))
	return float64(result)
}

// Cdotu computes the unconjugated dot product of two single precision complex vectors.
// Returns the dot product x^T * y (without conjugation).
func (impl *Standard) Cdotu(n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) complex64 {
	if impl.e != nil {
		return 0
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

	var result complex64
	impl.e = Check(C.cublasCdotu(C.cublasHandle_t(impl.h), C.int(n), (*C.cuComplex)(x), C.int(incX), (*C.cuComplex)(y), C.int(incY), (*C.cuComplex)(unsafe.Pointer(&result))))
	return result
}

// Cdotc computes the conjugated dot product of two single precision complex vectors.
// Returns the dot product x^H * y (with conjugation of x).
func (impl *Standard) Cdotc(n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) complex64 {
	if impl.e != nil {
		return 0
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

	var result complex64
	impl.e = Check(C.cublasCdotc(C.cublasHandle_t(impl.h), C.int(n), (*C.cuComplex)(x), C.int(incX), (*C.cuComplex)(y), C.int(incY), (*C.cuComplex)(unsafe.Pointer(&result))))
	return result
}

// Zdotu computes the unconjugated dot product of two double precision complex vectors.
// Returns the dot product x^T * y (without conjugation).
func (impl *Standard) Zdotu(n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) complex128 {
	if impl.e != nil {
		return 0
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

	var result complex128
	impl.e = Check(C.cublasZdotu(C.cublasHandle_t(impl.h), C.int(n), (*C.cuDoubleComplex)(x), C.int(incX), (*C.cuDoubleComplex)(y), C.int(incY), (*C.cuDoubleComplex)(unsafe.Pointer(&result))))
	return result
}

// Zdotc computes the conjugated dot product of two double precision complex vectors.
// Returns the dot product x^H * y (with conjugation of x).
func (impl *Standard) Zdotc(n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) complex128 {
	if impl.e != nil {
		return 0
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

	var result complex128
	impl.e = Check(C.cublasZdotc(C.cublasHandle_t(impl.h), C.int(n), (*C.cuDoubleComplex)(x), C.int(incX), (*C.cuDoubleComplex)(y), C.int(incY), (*C.cuDoubleComplex)(unsafe.Pointer(&result))))
	return result
}

// Snrm2 computes the Euclidean norm (2-norm) of a single precision real vector.
// Returns the 2-norm: sqrt(sum(x[i]^2)).
func (impl *Standard) Snrm2(n int, x cudart.DevicePtr, incX int) float32 {
	if impl.e != nil {
		return 0
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	var result C.float
	impl.e = Check(C.cublasSnrm2(C.cublasHandle_t(impl.h), C.int(n), (*C.float)(x), C.int(incX), &result))
	return float32(result)
}

// Dnrm2 computes the Euclidean norm (2-norm) of a double precision real vector.
// Returns the 2-norm: sqrt(sum(x[i]^2)).
func (impl *Standard) Dnrm2(n int, x cudart.DevicePtr, incX int) float64 {
	if impl.e != nil {
		return 0
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	var result C.double
	impl.e = Check(C.cublasDnrm2(C.cublasHandle_t(impl.h), C.int(n), (*C.double)(x), C.int(incX), &result))
	return float64(result)
}

// Scnrm2 computes the Euclidean norm (2-norm) of a single precision complex vector.
// Returns the 2-norm: sqrt(sum(|x[i]|^2)) as a real number.
func (impl *Standard) Scnrm2(n int, x cudart.DevicePtr, incX int) float32 {
	if impl.e != nil {
		return 0
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	var result C.float
	impl.e = Check(C.cublasScnrm2(C.cublasHandle_t(impl.h), C.int(n), (*C.cuComplex)(x), C.int(incX), &result))
	return float32(result)
}

// Dznrm2 computes the Euclidean norm (2-norm) of a double precision complex vector.
// Returns the 2-norm: sqrt(sum(|x[i]|^2)) as a real number.
func (impl *Standard) Dznrm2(n int, x cudart.DevicePtr, incX int) float64 {
	if impl.e != nil {
		return 0
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	var result C.double
	impl.e = Check(C.cublasDznrm2(C.cublasHandle_t(impl.h), C.int(n), (*C.cuDoubleComplex)(x), C.int(incX), &result))
	return float64(result)
}

// Srot applies a Givens rotation to single precision real vectors x and y.
// Performs the transformation: x[i] = c*x[i] + s*y[i], y[i] = c*y[i] - s*x[i].
func (impl *Standard) Srot(n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, c, s float32) {
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

	impl.e = Check(C.cublasSrot(C.cublasHandle_t(impl.h), C.int(n), (*C.float)(x), C.int(incX), (*C.float)(y), C.int(incY), (*C.float)(&c), (*C.float)(&s)))
}

// Drot applies a Givens rotation to double precision real vectors x and y.
// Performs the transformation: x[i] = c*x[i] + s*y[i], y[i] = c*y[i] - s*x[i].
func (impl *Standard) Drot(n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, c, s float64) {
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

	impl.e = Check(C.cublasDrot(C.cublasHandle_t(impl.h), C.int(n), (*C.double)(x), C.int(incX), (*C.double)(y), C.int(incY), (*C.double)(&c), (*C.double)(&s)))
}

// Crot applies a Givens rotation to single precision complex vectors x and y.
// Performs the transformation: x[i] = c*x[i] + s*y[i], y[i] = c*y[i] - conj(s)*x[i].
func (impl *Standard) Crot(n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, c float32, s complex64) {
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

	impl.e = Check(C.cublasCrot(C.cublasHandle_t(impl.h), C.int(n), (*C.cuComplex)(x), C.int(incX), (*C.cuComplex)(y), C.int(incY), (*C.float)(&c), (*C.cuComplex)(unsafe.Pointer(&s))))
}

// Csrot applies a real Givens rotation to single precision complex vectors x and y.
// Performs the transformation: x[i] = c*x[i] + s*y[i], y[i] = c*y[i] - s*x[i].
func (impl *Standard) Csrot(n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, c, s float32) {
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

	impl.e = Check(C.cublasCsrot(C.cublasHandle_t(impl.h), C.int(n), (*C.cuComplex)(x), C.int(incX), (*C.cuComplex)(y), C.int(incY), (*C.float)(&c), (*C.float)(&s)))
}

// Zrot applies a Givens rotation to double precision complex vectors x and y.
// Performs the transformation: x[i] = c*x[i] + s*y[i], y[i] = c*y[i] - conj(s)*x[i].
func (impl *Standard) Zrot(n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, c float64, s complex128) {
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

	impl.e = Check(C.cublasZrot(C.cublasHandle_t(impl.h), C.int(n), (*C.cuDoubleComplex)(x), C.int(incX), (*C.cuDoubleComplex)(y), C.int(incY), (*C.double)(&c), (*C.cuDoubleComplex)(unsafe.Pointer(&s))))
}

// Zdrot applies a real Givens rotation to double precision complex vectors x and y.
// Performs the transformation: x[i] = c*x[i] + s*y[i], y[i] = c*y[i] - s*x[i].
func (impl *Standard) Zdrot(n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, c, s float64) {
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

	impl.e = Check(C.cublasZdrot(C.cublasHandle_t(impl.h), C.int(n), (*C.cuDoubleComplex)(x), C.int(incX), (*C.cuDoubleComplex)(y), C.int(incY), (*C.double)(&c), (*C.double)(&s)))
}

// Srotg generates the parameters for a Givens rotation matrix for single precision real values.
// Given input values a and b, computes c, s, and r such that:
// [c  s] [a] = [r]
// [-s c] [b]   [0]
// The function modifies a to contain r, and b to contain z (reconstruction parameter).
func (impl *Standard) Srotg(a, b *float32) (c, s float32) {
	if impl.e != nil {
		return 0, 0
	}

	impl.e = Check(C.cublasSrotg(C.cublasHandle_t(impl.h), (*C.float)(a), (*C.float)(b), (*C.float)(&c), (*C.float)(&s)))
	return c, s
}

// Drotg generates the parameters for a Givens rotation matrix for double precision real values.
// Given input values a and b, computes c, s, and r such that:
// [c  s] [a] = [r]
// [-s c] [b]   [0]
// The function modifies a to contain r, and b to contain z (reconstruction parameter).
func (impl *Standard) Drotg(a, b *float64) (c, s float64) {
	if impl.e != nil {
		return 0, 0
	}

	impl.e = Check(C.cublasDrotg(C.cublasHandle_t(impl.h), (*C.double)(a), (*C.double)(b), (*C.double)(&c), (*C.double)(&s)))
	return c, s
}

// Crotg generates the parameters for a Givens rotation matrix for single precision complex values.
// Given input values a and b, computes c, s, and r such that:
// [c       s] [a] = [r]
// [-conj(s) c] [b]   [0]
// The function modifies a to contain r, and b to contain z (reconstruction parameter).
func (impl *Standard) Crotg(a, b *complex64) (c float32, s complex64) {
	if impl.e != nil {
		return 0, 0
	}

	impl.e = Check(C.cublasCrotg(C.cublasHandle_t(impl.h), (*C.cuComplex)(unsafe.Pointer(a)), (*C.cuComplex)(unsafe.Pointer(b)), (*C.float)(&c), (*C.cuComplex)(unsafe.Pointer(&s))))
	return c, s
}

// Zrotg generates the parameters for a Givens rotation matrix for double precision complex values.
// Given input values a and b, computes c, s, and r such that:
// [c       s] [a] = [r]
// [-conj(s) c] [b]   [0]
// The function modifies a to contain r, and b to contain z (reconstruction parameter).
func (impl *Standard) Zrotg(a, b *complex128) (c float64, s complex128) {
	if impl.e != nil {
		return 0, 0
	}

	impl.e = Check(C.cublasZrotg(C.cublasHandle_t(impl.h), (*C.cuDoubleComplex)(unsafe.Pointer(a)), (*C.cuDoubleComplex)(unsafe.Pointer(b)), (*C.double)(&c), (*C.cuDoubleComplex)(unsafe.Pointer(&s))))
	return c, s
}

// Srotm applies a modified Givens rotation to single precision real vectors x and y.
// The rotation is defined by the param array which contains the rotation parameters.
// param[0] contains the flag indicating the form of the H matrix:
//
//	-1.0: H has the form [[H11, H12], [H21, H22]]
//	 0.0: H has the form [[1.0, H12], [H21, 1.0]]
//	 1.0: H has the form [[H11, 1.0], [-1.0, H22]]
//	-2.0: H is the identity matrix
func (impl *Standard) Srotm(n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, param blas.SrotmParams) {
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
	p := srotmParams{
		flag: float32(param.Flag),
		h:    param.H,
	}

	impl.e = Check(C.cublasSrotm(C.cublasHandle_t(impl.h), C.int(n), (*C.float)(x), C.int(incX), (*C.float)(y), C.int(incY), (*C.float)(unsafe.Pointer(&p))))
}

// Drotm applies a modified Givens rotation to double precision real vectors x and y.
// The rotation is defined by the param array which contains the rotation parameters.
// param[0] contains the flag indicating the form of the H matrix:
//
//	-1.0: H has the form [[H11, H12], [H21, H22]]
//	 0.0: H has the form [[1.0, H12], [H21, 1.0]]
//	 1.0: H has the form [[H11, 1.0], [-1.0, H22]]
//	-2.0: H is the identity matrix
func (impl *Standard) Drotm(n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int, param blas.DrotmParams) {
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
	p := drotmParams{
		flag: float64(param.Flag),
		h:    param.H,
	}

	impl.e = Check(C.cublasDrotm(C.cublasHandle_t(impl.h), C.int(n), (*C.double)(x), C.int(incX), (*C.double)(y), C.int(incY), (*C.double)(unsafe.Pointer(&p))))
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
func (impl *Standard) Srotmg(d1, d2, x1 *float32, y1 float32) (param blas.SrotmParams) {
	if impl.e != nil {
		return
	}
	var p srotmParams
	impl.e = Check(C.cublasSrotmg(C.cublasHandle_t(impl.h), (*C.float)(d1), (*C.float)(d2), (*C.float)(x1), (*C.float)(&y1), (*C.float)(unsafe.Pointer(&p))))
	return blas.SrotmParams{Flag: blas.Flag(p.flag), H: p.h}
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
func (impl *Standard) Drotmg(d1, d2, x1 *float64, y1 float64) (param blas.DrotmParams) {
	if impl.e != nil {
		return
	}
	var p drotmParams
	impl.e = Check(C.cublasDrotmg(C.cublasHandle_t(impl.h), (*C.double)(d1), (*C.double)(d2), (*C.double)(x1), (*C.double)(&y1), (*C.double)(unsafe.Pointer(&p))))
	return blas.DrotmParams{Flag: blas.Flag(p.flag), H: p.h}
}

// Sscal scales a single precision real vector by a scalar.
// Performs the operation: x = alpha * x.
func (impl *Standard) Sscal(n int, alpha float32, x cudart.DevicePtr, incX int) {
	if impl.e != nil {
		return
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	impl.e = Check(C.cublasSscal(C.cublasHandle_t(impl.h), C.int(n), (*C.float)(&alpha), (*C.float)(x), C.int(incX)))
}

// Dscal scales a double precision real vector by a scalar.
// Performs the operation: x = alpha * x.
func (impl *Standard) Dscal(n int, alpha float64, x cudart.DevicePtr, incX int) {
	if impl.e != nil {
		return
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	impl.e = Check(C.cublasDscal(C.cublasHandle_t(impl.h), C.int(n), (*C.double)(&alpha), (*C.double)(x), C.int(incX)))
}

// Cscal scales a single precision complex vector by a complex scalar.
// Performs the operation: x = alpha * x.
func (impl *Standard) Cscal(n int, alpha complex64, x cudart.DevicePtr, incX int) {
	if impl.e != nil {
		return
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	impl.e = Check(C.cublasCscal(C.cublasHandle_t(impl.h), C.int(n), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(x), C.int(incX)))
}

// Csscal scales a single precision complex vector by a real scalar.
// Performs the operation: x = alpha * x (where alpha is real).
func (impl *Standard) Csscal(n int, alpha float32, x cudart.DevicePtr, incX int) {
	if impl.e != nil {
		return
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	impl.e = Check(C.cublasCsscal(C.cublasHandle_t(impl.h), C.int(n), (*C.float)(&alpha), (*C.cuComplex)(x), C.int(incX)))
}

// Zscal scales a double precision complex vector by a complex scalar.
// Performs the operation: x = alpha * x.
func (impl *Standard) Zscal(n int, alpha complex128, x cudart.DevicePtr, incX int) {
	if impl.e != nil {
		return
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	impl.e = Check(C.cublasZscal(C.cublasHandle_t(impl.h), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(x), C.int(incX)))
}

// Zdscal scales a double precision complex vector by a real scalar.
// Performs the operation: x = alpha * x (where alpha is real).
func (impl *Standard) Zdscal(n int, alpha float64, x cudart.DevicePtr, incX int) {
	if impl.e != nil {
		return
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	impl.e = Check(C.cublasZdscal(C.cublasHandle_t(impl.h), C.int(n), (*C.double)(&alpha), (*C.cuDoubleComplex)(x), C.int(incX)))
}

// Sswap swaps the elements of two single precision real vectors.
// Performs the operation: x <-> y (exchanges the contents of vectors x and y).
func (impl *Standard) Sswap(n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) {
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

	impl.e = Check(C.cublasSswap(C.cublasHandle_t(impl.h), C.int(n), (*C.float)(x), C.int(incX), (*C.float)(y), C.int(incY)))
}

// Dswap swaps the elements of two double precision real vectors.
// Performs the operation: x <-> y (exchanges the contents of vectors x and y).
func (impl *Standard) Dswap(n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) {
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

	impl.e = Check(C.cublasDswap(C.cublasHandle_t(impl.h), C.int(n), (*C.double)(x), C.int(incX), (*C.double)(y), C.int(incY)))
}

// Cswap swaps the elements of two single precision complex vectors.
// Performs the operation: x <-> y (exchanges the contents of vectors x and y).
func (impl *Standard) Cswap(n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) {
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

	impl.e = Check(C.cublasCswap(C.cublasHandle_t(impl.h), C.int(n), (*C.cuComplex)(x), C.int(incX), (*C.cuComplex)(y), C.int(incY)))
}

// Zswap swaps the elements of two double precision complex vectors.
// Performs the operation: x <-> y (exchanges the contents of vectors x and y).
func (impl *Standard) Zswap(n int, x cudart.DevicePtr, incX int, y cudart.DevicePtr, incY int) {
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

	impl.e = Check(C.cublasZswap(C.cublasHandle_t(impl.h), C.int(n), (*C.cuDoubleComplex)(x), C.int(incX), (*C.cuDoubleComplex)(y), C.int(incY)))
}
