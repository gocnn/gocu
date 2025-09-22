package gocu

// #include <cuda.h>
import "C"
import "unsafe"

// DevicePtr represents a pointer to device memory, equivalent to CUDA's CUdeviceptr.
type DevicePtr uintptr

// MemAlloc allocates size bytes of device memory and returns a pointer to it.
// Returns an error if allocation fails.
func MemAlloc(size int64) (DevicePtr, error) {
	var dptr C.CUdeviceptr
	err := Check(C.cuMemAlloc(&dptr, C.size_t(size)))
	return DevicePtr(dptr), err
}

// MemFree frees the device memory pointed to by dptr.
// Returns an error if freeing fails. Safe to call on already freed or invalid pointers.
func MemFree(dptr DevicePtr) error {
	return Check(C.cuMemFree(C.CUdeviceptr(dptr)))
}

// Free frees the device memory associated with the pointer.
// It is safe to call multiple times.
func (dptr DevicePtr) Free() error {
	return MemFree(dptr)
}

// Memcpy copies size bytes from src to dst on the current device.
// Requires unified addressing support. For device-to-device copy, see MemcpyDtoD.
func Memcpy(dst, src DevicePtr, size int64) error {
	return Check(C.cuMemcpy(C.CUdeviceptr(dst), C.CUdeviceptr(src), C.size_t(size)))
}

// MemcpyPeer copies size bytes from src on srcCtx to dst on dstCtx across devices.
// Requires peer access to be enabled between the contexts.
func MemcpyPeer(dst DevicePtr, dstCtx Context, src DevicePtr, srcCtx Context, size int64) error {
	return Check(C.cuMemcpyPeer(
		C.CUdeviceptr(dst), dstCtx.c(),
		C.CUdeviceptr(src), srcCtx.c(),
		C.size_t(size),
	))
}

// MemcpyHtoD copies size bytes from host memory (src) to device memory (dst).
func MemcpyHtoD(dst DevicePtr, src unsafe.Pointer, size int64) error {
	return Check(C.cuMemcpyHtoD(C.CUdeviceptr(dst), src, C.size_t(size)))
}

// MemcpyDtoH copies size bytes from device memory (src) to host memory (dst).
func MemcpyDtoH(dst unsafe.Pointer, src DevicePtr, size int64) error {
	return Check(C.cuMemcpyDtoH(dst, C.CUdeviceptr(src), C.size_t(size)))
}

// MemcpyDtoD copies size bytes from src to dst on the same device.
func MemcpyDtoD(dst, src DevicePtr, size int64) error {
	return Check(C.cuMemcpyDtoD(C.CUdeviceptr(dst), C.CUdeviceptr(src), C.size_t(size)))
}

// MemcpyAsync copies size bytes from src to dst on the current device asynchronously.
// The copy is performed in the specified stream.
func MemcpyAsync(dst, src DevicePtr, size int64, stream Stream) error {
	return Check(C.cuMemcpyAsync(
		C.CUdeviceptr(dst), C.CUdeviceptr(src),
		C.size_t(size), stream.c(),
	))
}

// MemcpyPeerAsync copies size bytes from src on srcCtx to dst on dstCtx across devices asynchronously.
// Requires peer access to be enabled. The copy is performed in the specified stream.
func MemcpyPeerAsync(dst DevicePtr, dstCtx Context, src DevicePtr, srcCtx Context, size int64, stream Stream) error {
	return Check(C.cuMemcpyPeerAsync(
		C.CUdeviceptr(dst), dstCtx.c(),
		C.CUdeviceptr(src), srcCtx.c(),
		C.size_t(size), stream.c(),
	))
}

// MemcpyHtoDAsync copies size bytes from host memory (src) to device memory (dst) asynchronously.
// The copy is performed in the specified stream.
func MemcpyHtoDAsync(dst DevicePtr, src unsafe.Pointer, size int64, stream Stream) error {
	return Check(C.cuMemcpyHtoDAsync(C.CUdeviceptr(dst), src, C.size_t(size), stream.c()))
}

// MemcpyDtoHAsync copies size bytes from device memory (src) to host memory (dst) asynchronously.
// The copy is performed in the specified stream.
func MemcpyDtoHAsync(dst unsafe.Pointer, src DevicePtr, size int64, stream Stream) error {
	return Check(C.cuMemcpyDtoHAsync(dst, C.CUdeviceptr(src), C.size_t(size), stream.c()))
}

// MemcpyDtoDAsync copies size bytes from src to dst on the same device asynchronously.
// The copy is performed in the specified stream.
func MemcpyDtoDAsync(dst, src DevicePtr, size int64, stream Stream) error {
	return Check(C.cuMemcpyDtoDAsync(
		C.CUdeviceptr(dst), C.CUdeviceptr(src),
		C.size_t(size), stream.c(),
	))
}

// MemsetD8 sets size bytes of device memory at dst to the specified 8-bit value.
// Returns an error if the operation fails.
func MemsetD8(dst DevicePtr, value byte, size int64) error {
	return Check(C.cuMemsetD8(C.CUdeviceptr(dst), C.uchar(value), C.size_t(size)))
}

// MemsetD16 sets size 16-bit words of device memory at dst to the specified 16-bit value.
// Size must be a multiple of 2. Returns an error if the operation fails.
func MemsetD16(dst DevicePtr, value uint16, size int64) error {
	return Check(C.cuMemsetD16(C.CUdeviceptr(dst), C.ushort(value), C.size_t(size)))
}

// MemsetD32 sets size 32-bit words of device memory at dst to the specified 32-bit value.
// Size must be a multiple of 4. Returns an error if the operation fails.
func MemsetD32(dst DevicePtr, value uint, size int64) error {
	return Check(C.cuMemsetD32(C.CUdeviceptr(dst), C.uint(value), C.size_t(size)))
}

// MemsetD2D8 sets a 2D device memory region at dst to the specified 8-bit value.
// The region has the given pitch, width, and height. Returns an error if the operation fails.
func MemsetD2D8(dst DevicePtr, pitch, width, height int64, value byte) error {
	return Check(C.cuMemsetD2D8(
		C.CUdeviceptr(dst), C.size_t(pitch),
		C.uchar(value), C.size_t(width), C.size_t(height),
	))
}

// MemsetD2D16 sets a 2D device memory region at dst to the specified 16-bit value.
// The region has the given pitch, width, and height. Width must be a multiple of 2.
// Returns an error if the operation fails.
func MemsetD2D16(dst DevicePtr, pitch, width, height int64, value uint16) error {
	return Check(C.cuMemsetD2D16(
		C.CUdeviceptr(dst), C.size_t(pitch),
		C.ushort(value), C.size_t(width), C.size_t(height),
	))
}

// MemsetD2D32 sets a 2D device memory region at dst to the specified 32-bit value.
// The region has the given pitch, width, and height. Width must be a multiple of 4.
// Returns an error if the operation fails.
func MemsetD2D32(dst DevicePtr, pitch, width, height int64, value uint) error {
	return Check(C.cuMemsetD2D32(
		C.CUdeviceptr(dst), C.size_t(pitch),
		C.uint(value), C.size_t(width), C.size_t(height),
	))
}

// MemsetD8Async sets size bytes of device memory at dst to the specified 8-bit value asynchronously.
// The operation is performed in the specified stream. Returns an error if the operation fails.
func MemsetD8Async(dst DevicePtr, value byte, size int64, stream Stream) error {
	return Check(C.cuMemsetD8Async(C.CUdeviceptr(dst), C.uchar(value), C.size_t(size), stream.c()))
}

// MemsetD16Async sets size 16-bit words of device memory at dst to the specified 16-bit value asynchronously.
// Size must be a multiple of 2. The operation is performed in the specified stream.
// Returns an error if the operation fails.
func MemsetD16Async(dst DevicePtr, value uint16, size int64, stream Stream) error {
	return Check(C.cuMemsetD16Async(C.CUdeviceptr(dst), C.ushort(value), C.size_t(size), stream.c()))
}

// MemsetD32Async sets size 32-bit words of device memory at dst to the specified 32-bit value asynchronously.
// Size must be a multiple of 4. The operation is performed in the specified stream.
// Returns an error if the operation fails.
func MemsetD32Async(dst DevicePtr, value uint, size int64, stream Stream) error {
	return Check(C.cuMemsetD32Async(C.CUdeviceptr(dst), C.uint(value), C.size_t(size), stream.c()))
}

// MemsetD2D8Async sets a 2D device memory region at dst to the specified 8-bit value asynchronously.
// The region has the given pitch, width, and height. The operation is performed in the specified stream.
// Returns an error if the operation fails.
func MemsetD2D8Async(dst DevicePtr, pitch, width, height int64, value byte, stream Stream) error {
	return Check(C.cuMemsetD2D8Async(
		C.CUdeviceptr(dst), C.size_t(pitch),
		C.uchar(value), C.size_t(width), C.size_t(height),
		stream.c(),
	))
}

// MemsetD2D16Async sets a 2D device memory region at dst to the specified 16-bit value asynchronously.
// The region has the given pitch, width, and height. Width must be a multiple of 2.
// The operation is performed in the specified stream. Returns an error if the operation fails.
func MemsetD2D16Async(dst DevicePtr, pitch, width, height int64, value uint16, stream Stream) error {
	return Check(C.cuMemsetD2D16Async(
		C.CUdeviceptr(dst), C.size_t(pitch),
		C.ushort(value), C.size_t(width), C.size_t(height),
		stream.c(),
	))
}

// MemsetD2D32Async sets a 2D device memory region at dst to the specified 32-bit value asynchronously.
// The region has the given pitch, width, and height. Width must be a multiple of 4.
// The operation is performed in the specified stream. Returns an error if the operation fails.
func MemsetD2D32Async(dst DevicePtr, pitch, width, height int64, value uint, stream Stream) error {
	return Check(C.cuMemsetD2D32Async(
		C.CUdeviceptr(dst), C.size_t(pitch),
		C.uint(value), C.size_t(width), C.size_t(height),
		stream.c(),
	))
}
