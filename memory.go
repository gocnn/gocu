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
	err := Result(C.cuMemAlloc(&dptr, C.size_t(size)))
	return DevicePtr(dptr), err
}

// MemFree frees the device memory pointed to by dptr.
// Returns an error if freeing fails. Safe to call on already freed or invalid pointers.
func MemFree(dptr DevicePtr) error {
	return Result(C.cuMemFree(C.CUdeviceptr(dptr)))
}

// Free frees the device memory associated with the pointer.
// It is safe to call multiple times.
func (dptr DevicePtr) Free() error {
	return MemFree(dptr)
}

// Memcpy copies size bytes from src to dst on the current device.
// Requires unified addressing support. For device-to-device copy, see MemcpyDtoD.
func Memcpy(dst, src DevicePtr, size int64) error {
	return Result(C.cuMemcpy(C.CUdeviceptr(dst), C.CUdeviceptr(src), C.size_t(size)))
}

// MemcpyAsync copies size bytes from src to dst on the current device asynchronously.
// The copy is performed in the specified stream.
func MemcpyAsync(dst, src DevicePtr, size int64, stream CUStream) error {
	return Result(C.cuMemcpyAsync(
		C.CUdeviceptr(dst), C.CUdeviceptr(src),
		C.size_t(size), stream.c(),
	))
}

// MemcpyPeer copies size bytes from src on srcCtx to dst on dstCtx across devices.
// Requires peer access to be enabled between the contexts.
func MemcpyPeer(dst DevicePtr, dstCtx CUContext, src DevicePtr, srcCtx CUContext, size int64) error {
	return Result(C.cuMemcpyPeer(
		C.CUdeviceptr(dst), dstCtx.c(),
		C.CUdeviceptr(src), srcCtx.c(),
		C.size_t(size),
	))
}

// MemcpyPeerAsync copies size bytes from src on srcCtx to dst on dstCtx across devices asynchronously.
// Requires peer access to be enabled. The copy is performed in the specified stream.
func MemcpyPeerAsync(dst DevicePtr, dstCtx CUContext, src DevicePtr, srcCtx CUContext, size int64, stream CUStream) error {
	return Result(C.cuMemcpyPeerAsync(
		C.CUdeviceptr(dst), dstCtx.c(),
		C.CUdeviceptr(src), srcCtx.c(),
		C.size_t(size), stream.c(),
	))
}

// MemcpyHtoD copies size bytes from host memory (src) to device memory (dst).
func MemcpyHtoD(dst DevicePtr, src unsafe.Pointer, size int64) error {
	return Result(C.cuMemcpyHtoD(C.CUdeviceptr(dst), src, C.size_t(size)))
}

// MemcpyHtoDAsync copies size bytes from host memory (src) to device memory (dst) asynchronously.
// The copy is performed in the specified stream.
func MemcpyHtoDAsync(dst DevicePtr, src unsafe.Pointer, size int64, stream CUStream) error {
	return Result(C.cuMemcpyHtoDAsync(C.CUdeviceptr(dst), src, C.size_t(size), stream.c()))
}

// MemcpyDtoH copies size bytes from device memory (src) to host memory (dst).
func MemcpyDtoH(dst unsafe.Pointer, src DevicePtr, size int64) error {
	return Result(C.cuMemcpyDtoH(dst, C.CUdeviceptr(src), C.size_t(size)))
}

// MemcpyDtoHAsync copies size bytes from device memory (src) to host memory (dst) asynchronously.
// The copy is performed in the specified stream.
func MemcpyDtoHAsync(dst unsafe.Pointer, src DevicePtr, size int64, stream CUStream) error {
	return Result(C.cuMemcpyDtoHAsync(dst, C.CUdeviceptr(src), C.size_t(size), stream.c()))
}

// MemcpyDtoD copies size bytes from src to dst on the same device.
func MemcpyDtoD(dst, src DevicePtr, size int64) error {
	return Result(C.cuMemcpyDtoD(C.CUdeviceptr(dst), C.CUdeviceptr(src), C.size_t(size)))
}

// MemcpyDtoDAsync copies size bytes from src to dst on the same device asynchronously.
// The copy is performed in the specified stream.
func MemcpyDtoDAsync(dst, src DevicePtr, size int64, stream CUStream) error {
	return Result(C.cuMemcpyDtoDAsync(
		C.CUdeviceptr(dst), C.CUdeviceptr(src),
		C.size_t(size), stream.c(),
	))
}
