package gocu

// #include <cuda.h>
import "C"

// MemsetD8 sets size bytes of device memory at dst to the specified 8-bit value.
// Returns an error if the operation fails.
func MemsetD8(dst DevicePtr, value byte, size int64) error {
	return Result(C.cuMemsetD8(C.CUdeviceptr(dst), C.uchar(value), C.size_t(size)))
}

// MemsetD16 sets size 16-bit words of device memory at dst to the specified 16-bit value.
// Size must be a multiple of 2. Returns an error if the operation fails.
func MemsetD16(dst DevicePtr, value uint16, size int64) error {
	return Result(C.cuMemsetD16(C.CUdeviceptr(dst), C.ushort(value), C.size_t(size)))
}

// MemsetD32 sets size 32-bit words of device memory at dst to the specified 32-bit value.
// Size must be a multiple of 4. Returns an error if the operation fails.
func MemsetD32(dst DevicePtr, value uint, size int64) error {
	return Result(C.cuMemsetD32(C.CUdeviceptr(dst), C.uint(value), C.size_t(size)))
}

// MemsetD2D8 sets a 2D device memory region at dst to the specified 8-bit value.
// The region has the given pitch, width, and height. Returns an error if the operation fails.
func MemsetD2D8(dst DevicePtr, pitch, width, height int64, value byte) error {
	return Result(C.cuMemsetD2D8(
		C.CUdeviceptr(dst), C.size_t(pitch),
		C.uchar(value), C.size_t(width), C.size_t(height),
	))
}

// MemsetD2D16 sets a 2D device memory region at dst to the specified 16-bit value.
// The region has the given pitch, width, and height. Width must be a multiple of 2.
// Returns an error if the operation fails.
func MemsetD2D16(dst DevicePtr, pitch, width, height int64, value uint16) error {
	return Result(C.cuMemsetD2D16(
		C.CUdeviceptr(dst), C.size_t(pitch),
		C.ushort(value), C.size_t(width), C.size_t(height),
	))
}

// MemsetD2D32 sets a 2D device memory region at dst to the specified 32-bit value.
// The region has the given pitch, width, and height. Width must be a multiple of 4.
// Returns an error if the operation fails.
func MemsetD2D32(dst DevicePtr, pitch, width, height int64, value uint) error {
	return Result(C.cuMemsetD2D32(
		C.CUdeviceptr(dst), C.size_t(pitch),
		C.uint(value), C.size_t(width), C.size_t(height),
	))
}

// MemsetD8Async sets size bytes of device memory at dst to the specified 8-bit value asynchronously.
// The operation is performed in the specified stream. Returns an error if the operation fails.
func MemsetD8Async(dst DevicePtr, value byte, size int64, stream CUStream) error {
	return Result(C.cuMemsetD8Async(C.CUdeviceptr(dst), C.uchar(value), C.size_t(size), stream.c()))
}

// MemsetD16Async sets size 16-bit words of device memory at dst to the specified 16-bit value asynchronously.
// Size must be a multiple of 2. The operation is performed in the specified stream.
// Returns an error if the operation fails.
func MemsetD16Async(dst DevicePtr, value uint16, size int64, stream CUStream) error {
	return Result(C.cuMemsetD16Async(C.CUdeviceptr(dst), C.ushort(value), C.size_t(size), stream.c()))
}

// MemsetD32Async sets size 32-bit words of device memory at dst to the specified 32-bit value asynchronously.
// Size must be a multiple of 4. The operation is performed in the specified stream.
// Returns an error if the operation fails.
func MemsetD32Async(dst DevicePtr, value uint, size int64, stream CUStream) error {
	return Result(C.cuMemsetD32Async(C.CUdeviceptr(dst), C.uint(value), C.size_t(size), stream.c()))
}

// MemsetD2D8Async sets a 2D device memory region at dst to the specified 8-bit value asynchronously.
// The region has the given pitch, width, and height. The operation is performed in the specified stream.
// Returns an error if the operation fails.
func MemsetD2D8Async(dst DevicePtr, pitch, width, height int64, value byte, stream CUStream) error {
	return Result(C.cuMemsetD2D8Async(
		C.CUdeviceptr(dst), C.size_t(pitch),
		C.uchar(value), C.size_t(width), C.size_t(height),
		stream.c(),
	))
}

// MemsetD2D16Async sets a 2D device memory region at dst to the specified 16-bit value asynchronously.
// The region has the given pitch, width, and height. Width must be a multiple of 2.
// The operation is performed in the specified stream. Returns an error if the operation fails.
func MemsetD2D16Async(dst DevicePtr, pitch, width, height int64, value uint16, stream CUStream) error {
	return Result(C.cuMemsetD2D16Async(
		C.CUdeviceptr(dst), C.size_t(pitch),
		C.ushort(value), C.size_t(width), C.size_t(height),
		stream.c(),
	))
}

// MemsetD2D32Async sets a 2D device memory region at dst to the specified 32-bit value asynchronously.
// The region has the given pitch, width, and height. Width must be a multiple of 4.
// The operation is performed in the specified stream. Returns an error if the operation fails.
func MemsetD2D32Async(dst DevicePtr, pitch, width, height int64, value uint, stream CUStream) error {
	return Result(C.cuMemsetD2D32Async(
		C.CUdeviceptr(dst), C.size_t(pitch),
		C.uint(value), C.size_t(width), C.size_t(height),
		stream.c(),
	))
}
