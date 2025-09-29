package cudart

import (
	"unsafe"

	"github.com/gocnn/gocu/cudart"
)

// Flags for memory copy types
const (
	// Host to Host
	HtoH = cudart.HtoH
	// Host to Device
	HtoD = cudart.HtoD
	// Device to Host
	DtoH = cudart.DtoH
	// Device to Device
	DtoD = cudart.DtoD
	// Auto-detect direction based on pointer addresses
	Auto = cudart.Auto
)

// MallocFor allocates device memory for the same size as the given slice.
// This is more elegant - just pass your slice and get matching device memory.
// Usage: ptr, err := MallocFor(mySlice)
func Malloc[T any](slice []T) (cudart.DevicePtr, error) {
	if len(slice) == 0 {
		return cudart.DevicePtr(nil), nil
	}
	bytes := SliceBytes(slice)
	return cudart.Malloc(bytes)
}

// Free frees memory on the device.
func Free(ptr cudart.DevicePtr) error {
	if ptr == cudart.DevicePtr(nil) {
		return nil
	}
	return cudart.Free(ptr)
}

// MallocHost allocates page-locked host memory for a specific type and count.
func MallocHost[T any](slice []T) (cudart.HostPtr, error) {
	if len(slice) == 0 {
		return cudart.HostPtr(nil), nil
	}
	bytes := SliceBytes(slice)
	return cudart.MallocHost(bytes)
}

// FreeHost frees page-locked host memory.
func FreeHost(ptr cudart.HostPtr) error {
	if ptr == cudart.HostPtr(nil) {
		return nil
	}
	return cudart.FreeHost(ptr)
}

// MallocAndCopy allocates device memory and copies the slice data to it.
// This is a convenience function that combines Malloc + MemcpyHtoD in one call.
// Returns the device pointer with data already copied.
func MallocAndCopy[T any](src []T) (cudart.DevicePtr, error) {
	if len(src) == 0 {
		return cudart.DevicePtr(nil), nil
	}

	// Allocate device memory
	ptr, err := Malloc(src)
	if err != nil {
		return cudart.DevicePtr(nil), err
	}

	// Copy data to device
	err = MemcpyHtoD(ptr, src)
	if err != nil {
		Free(ptr) // Clean up on error
		return cudart.DevicePtr(nil), err
	}

	return ptr, nil
}

// MallocHostAndCopy allocates page-locked host memory and copies the slice data to it.
// Returns both the host pointer and a Go slice pointing to the allocated memory.
func MallocHostAndCopy[T any](src []T) (cudart.HostPtr, []T, error) {
	if len(src) == 0 {
		return cudart.HostPtr(nil), nil, nil
	}

	// Allocate page-locked host memory
	ptr, err := MallocHost(src)
	if err != nil {
		return cudart.HostPtr(nil), nil, err
	}

	// Create a slice that points to the allocated memory
	hostSlice := HostPtrToSlice[T](ptr, len(src))
	copy(hostSlice, src)

	return ptr, hostSlice, nil
}

// CopyAndFree copies device memory to host slice and frees the device memory.
// This is a convenience function for cleanup operations.
func CopyAndFree[T any](dst []T, src cudart.DevicePtr) error {
	err := MemcpyDtoH(dst, src)
	if freeErr := Free(src); freeErr != nil && err == nil {
		err = freeErr
	}
	return err
}

// MallocManaged allocates unified memory for a specific type and count.
func MallocManaged[T any](slice []T) (cudart.DevicePtr, []T, error) {
	if len(slice) == 0 {
		return cudart.DevicePtr(nil), nil, nil
	}
	bytes := SliceBytes(slice)
	ptr, err := cudart.MallocManaged(bytes)
	return cudart.DevicePtr(ptr), HostPtrToSlice[T](cudart.HostPtr(ptr), len(slice)), err
}

// MallocManagedAndCopy allocates unified memory and copies the slice data to it.
// This is a convenience function that combines MallocManaged + copy in one call.
// Returns both the device pointer (for GPU operations) and CPU slice (for CPU access).
func MallocManagedAndCopy[T any](src []T) (cudart.DevicePtr, []T, error) {
	if len(src) == 0 {
		return cudart.DevicePtr(nil), nil, nil
	}

	// Allocate unified memory
	devicePtr, managedSlice, err := MallocManaged(src)
	if err != nil {
		return cudart.DevicePtr(nil), nil, err
	}

	// Copy data to unified memory
	copy(managedSlice, src)

	return devicePtr, managedSlice, nil
}

// MallocPitch allocates pitched memory on the device for optimal 2D access patterns.
func MallocPitch[T any](width, height int) (cudart.DevicePtr, int64, error) {
	if width == 0 || height == 0 {
		return cudart.DevicePtr(nil), 0, nil
	}
	var dummy T
	widthBytes := int64(width) * int64(int(unsafe.Sizeof(dummy)))
	return cudart.MallocPitch(widthBytes, int64(height))
}

// MemGetInfo returns the free and total amount of device memory.
func MemGetInfo() (free, total int64, err error) {
	return cudart.MemGetInfo()
}

// Memcpy copies data between memory locations with specified direction.
// If slice lengths differ, copies min(len(dst), len(src)) elements.
// The direction parameter specifies the copy direction (HtoH, HtoD, DtoH, DtoD, Auto).
func Memcpy[T any](dst, src []T, kind uint) error {
	if len(src) == 0 || len(dst) == 0 {
		return nil
	}
	// Use the smaller of the two slice lengths to prevent buffer overflow
	count := min(len(dst), len(src))

	dstPtr := SliceToHostPtr(dst[:count])
	srcPtr := SliceToHostPtr(src[:count])
	bytes := int64(count) * int64(int(unsafe.Sizeof(src[0])))
	return cudart.Memcpy(unsafe.Pointer(dstPtr), unsafe.Pointer(srcPtr), bytes, kind)
}

// MemcpyAuto copies data with automatic direction detection.
// If slice lengths differ, copies min(len(dst), len(src)) elements.
// CUDA automatically detects the copy direction based on pointer addresses.
func MemcpyAuto[T any](dst, src []T) error {
	if len(src) == 0 || len(dst) == 0 {
		return nil
	}
	// Use the smaller of the two slice lengths to prevent buffer overflow
	count := min(len(dst), len(src))

	dstPtr := SliceToHostPtr(dst[:count])
	srcPtr := SliceToHostPtr(src[:count])
	bytes := int64(count) * int64(int(unsafe.Sizeof(src[0])))
	return cudart.MemcpyAuto(unsafe.Pointer(dstPtr), unsafe.Pointer(srcPtr), bytes)
}

// MemcpyHtoD copies a Go slice from host to device memory.
func MemcpyHtoD[T any](dst cudart.DevicePtr, src []T) error {
	if len(src) == 0 {
		return nil
	}
	srcPtr := SliceToHostPtr(src)
	bytes := SliceBytes(src)
	return cudart.MemcpyHtoD(dst, srcPtr, bytes)
}

// MemcpyDtoH copies data from device to host memory, filling the provided slice.
func MemcpyDtoH[T any](dst []T, src cudart.DevicePtr) error {
	if len(dst) == 0 {
		return nil
	}
	dstPtr := SliceToHostPtr(dst)
	bytes := SliceBytes(dst)
	return cudart.MemcpyDtoH(dstPtr, src, bytes)
}

// MemcpyDtoD copies data between device memory locations.
func MemcpyDtoD[T any](dst, src cudart.DevicePtr, count int) error {
	if count == 0 {
		return nil
	}
	var dummy T
	bytes := int64(count) * int64(int(unsafe.Sizeof(dummy)))
	return cudart.MemcpyDtoD(dst, src, bytes)
}

// MemcpyHtoH copies data between host memory locations using slices.
// If slice lengths differ, copies min(len(dst), len(src)) elements.
// This is equivalent to Go's built-in copy() but uses CUDA's optimized memcpy.
func MemcpyHtoH[T any](dst, src []T) error {
	if len(src) == 0 || len(dst) == 0 {
		return nil
	}
	// Use the smaller of the two slice lengths to prevent buffer overflow
	count := min(len(dst), len(src))

	dstPtr := SliceToHostPtr(dst[:count])
	srcPtr := SliceToHostPtr(src[:count])
	bytes := int64(count) * int64(int(unsafe.Sizeof(src[0])))
	return cudart.MemcpyHtoH(dstPtr, srcPtr, bytes)
}

// MemcpyAsync copies data between memory locations asynchronously with specified direction.
// If slice lengths differ, copies min(len(dst), len(src)) elements.
// The operation is queued in the specified stream and returns immediately.
func MemcpyAsync[T any](dst, src []T, kind uint, stream *cudart.Stream) error {
	if len(src) == 0 || len(dst) == 0 {
		return nil
	}
	// Use the smaller of the two slice lengths to prevent buffer overflow
	count := min(len(dst), len(src))

	dstPtr := SliceToHostPtr(dst[:count])
	srcPtr := SliceToHostPtr(src[:count])
	bytes := int64(count) * int64(int(unsafe.Sizeof(src[0])))
	return cudart.MemcpyAsync(unsafe.Pointer(dstPtr), unsafe.Pointer(srcPtr), bytes, kind, stream)
}

// MemcpyAutoAsync copies data with automatic direction detection asynchronously.
// If slice lengths differ, copies min(len(dst), len(src)) elements.
// CUDA automatically detects the copy direction and queues the operation in the stream.
func MemcpyAutoAsync[T any](dst, src []T, stream *cudart.Stream) error {
	if len(src) == 0 || len(dst) == 0 {
		return nil
	}
	// Use the smaller of the two slice lengths to prevent buffer overflow
	count := min(len(dst), len(src))

	dstPtr := SliceToHostPtr(dst[:count])
	srcPtr := SliceToHostPtr(src[:count])
	bytes := int64(count) * int64(int(unsafe.Sizeof(src[0])))
	return cudart.MemcpyAutoAsync(unsafe.Pointer(dstPtr), unsafe.Pointer(srcPtr), bytes, stream)
}

// MemcpyHtoDAsync copies a Go slice from host to device memory asynchronously.
func MemcpyHtoDAsync[T any](dst cudart.DevicePtr, src []T, stream *cudart.Stream) error {
	if len(src) == 0 {
		return nil
	}
	srcPtr := SliceToHostPtr(src)
	bytes := SliceBytes(src)
	return cudart.MemcpyAsync(unsafe.Pointer(dst), unsafe.Pointer(srcPtr), bytes, HtoD, stream)
}

// MemcpyDtoHAsync copies data from device to host memory asynchronously.
func MemcpyDtoHAsync[T any](dst []T, src cudart.DevicePtr, stream *cudart.Stream) error {
	if len(dst) == 0 {
		return nil
	}
	dstPtr := SliceToHostPtr(dst)
	bytes := SliceBytes(dst)
	return cudart.MemcpyAsync(unsafe.Pointer(dstPtr), unsafe.Pointer(src), bytes, DtoH, stream)
}

// MemcpyDtoDAsync copies data between device memory locations asynchronously.
func MemcpyDtoDAsync[T any](dst, src cudart.DevicePtr, count int, stream *cudart.Stream) error {
	if count == 0 {
		return nil
	}
	var dummy T
	bytes := int64(count) * int64(SliceBytes([]T{dummy})/1)
	return cudart.MemcpyAsync(unsafe.Pointer(dst), unsafe.Pointer(src), bytes, DtoD, stream)
}

// MemcpyHtoHAsync copies data between host memory locations asynchronously.
// If slice lengths differ, copies min(len(dst), len(src)) elements.
// The operation is queued in the specified stream for asynchronous execution.
func MemcpyHtoHAsync[T any](dst, src []T, stream *cudart.Stream) error {
	if len(src) == 0 || len(dst) == 0 {
		return nil
	}
	// Use the smaller of the two slice lengths to prevent buffer overflow
	count := min(len(dst), len(src))

	dstPtr := SliceToHostPtr(dst[:count])
	srcPtr := SliceToHostPtr(src[:count])
	bytes := int64(count) * int64(int(unsafe.Sizeof(src[0])))
	return cudart.MemcpyAsync(unsafe.Pointer(dstPtr), unsafe.Pointer(srcPtr), bytes, HtoH, stream)
}

// Memset sets device memory to a value.
func Memset[T any](dst cudart.DevicePtr, value byte, count int) error {
	if count == 0 {
		return nil
	}
	var dummy T
	bytes := int64(count) * int64(int(unsafe.Sizeof(dummy)))
	return cudart.Memset(dst, value, bytes)
}

// MemsetAsync sets device memory to a value asynchronously.
func MemsetAsync[T any](dst cudart.DevicePtr, value byte, count int, stream *cudart.Stream) error {
	if count == 0 {
		return nil
	}
	var dummy T
	bytes := int64(count) * int64(SliceBytes([]T{dummy})/1)
	return cudart.MemsetAsync(dst, value, bytes, stream)
}
