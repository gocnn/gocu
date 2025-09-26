package cudart

//#include <cuda_runtime.h>
import "C"
import "unsafe"

// DevicePtr represents a pointer to device memory, equivalent to CUDA's void*.
type DevicePtr unsafe.Pointer

// HostPtr represents a pointer to host memory, equivalent to void*.
type HostPtr unsafe.Pointer

// Flags for memory copy types
const (
	// Host to Host
	HtoH = C.cudaMemcpyHostToHost
	// Host to Device
	HtoD = C.cudaMemcpyHostToDevice
	// Device to Host
	DtoH = C.cudaMemcpyDeviceToHost
	// Device to Device
	DtoD = C.cudaMemcpyDeviceToDevice
	// Auto-detect direction based on pointer addresses
	Auto = C.cudaMemcpyDefault
)

// Malloc allocates memory on the device.
func Malloc(bytes int64) (DevicePtr, error) {
	var devptr unsafe.Pointer
	err := Check(C.cudaMalloc(&devptr, C.size_t(bytes)))
	return DevicePtr(devptr), err
}

// Free frees memory on the device.
func Free(ptr DevicePtr) error {
	return Check(C.cudaFree(unsafe.Pointer(ptr)))
}

// MallocHost allocates page-locked host memory.
// Page-locked memory can be accessed by the device directly and enables faster transfers.
func MallocHost(bytes int64) (HostPtr, error) {
	var hostptr unsafe.Pointer
	err := Check(C.cudaMallocHost(&hostptr, C.size_t(bytes)))
	return HostPtr(hostptr), err
}

// FreeHost frees page-locked host memory allocated by MallocHost.
func FreeHost(ptr HostPtr) error {
	return Check(C.cudaFreeHost(unsafe.Pointer(ptr)))
}

// MallocManaged allocates unified memory that can be accessed by both CPU and GPU.
func MallocManaged(bytes int64) (unsafe.Pointer, error) {
	var ptr unsafe.Pointer
	err := Check(C.cudaMallocManaged(&ptr, C.size_t(bytes), C.cudaMemAttachGlobal))
	return ptr, err
}

// MallocPitch allocates pitched memory on the device for optimal 2D access patterns.
func MallocPitch(width, height int64) (DevicePtr, int64, error) {
	var devptr unsafe.Pointer
	var pitch C.size_t
	err := Check(C.cudaMallocPitch(&devptr, &pitch, C.size_t(width), C.size_t(height)))
	return DevicePtr(devptr), int64(pitch), err
}

// MemGetInfo returns the free and total amount of device memory.
func MemGetInfo() (free, total int64, err error) {
	var freeBytes, totalBytes C.size_t
	err = Check(C.cudaMemGetInfo(&freeBytes, &totalBytes))
	return int64(freeBytes), int64(totalBytes), err
}

// Memcpy copies data between memory locations with specified direction.
func Memcpy(dst, src unsafe.Pointer, bytes int64, kind uint) error {
	return Check(C.cudaMemcpy(dst, src, C.size_t(bytes), uint32(kind)))
}

// MemcpyAuto copies data with automatic direction detection.
func MemcpyAuto(dst, src unsafe.Pointer, bytes int64) error {
	return Memcpy(dst, src, bytes, Auto)
}

// MemcpyHtoD copies data from host to device memory.
func MemcpyHtoD(dst DevicePtr, src HostPtr, bytes int64) error {
	return Memcpy(unsafe.Pointer(dst), unsafe.Pointer(src), bytes, HtoD)
}

// MemcpyDtoH copies data from device to host memory.
func MemcpyDtoH(dst HostPtr, src DevicePtr, bytes int64) error {
	return Memcpy(unsafe.Pointer(dst), unsafe.Pointer(src), bytes, DtoH)
}

// MemcpyDtoD copies data from device to device memory.
func MemcpyDtoD(dst, src DevicePtr, bytes int64) error {
	return Memcpy(unsafe.Pointer(dst), unsafe.Pointer(src), bytes, DtoD)
}

// MemcpyHtoH copies data from host to host memory.
func MemcpyHtoH(dst, src HostPtr, bytes int64) error {
	return Memcpy(unsafe.Pointer(dst), unsafe.Pointer(src), bytes, HtoH)
}

// MemcpyAsync copies data between memory locations asynchronously.
func MemcpyAsync(dst, src unsafe.Pointer, bytes int64, kind uint, stream Stream) error {
	return Check(C.cudaMemcpyAsync(dst, src, C.size_t(bytes), uint32(kind), stream.c()))
}

// MemcpyAutoAsync copies data with automatic direction detection asynchronously.
func MemcpyAutoAsync(dst, src unsafe.Pointer, bytes int64, stream Stream) error {
	return MemcpyAsync(dst, src, bytes, Auto, stream)
}

// MemcpyHtoDAsync copies data from host to device memory asynchronously.
func MemcpyHtoDAsync(dst DevicePtr, src HostPtr, bytes int64, stream Stream) error {
	return MemcpyAsync(unsafe.Pointer(dst), unsafe.Pointer(src), bytes, HtoD, stream)
}

// MemcpyDtoHAsync copies data from device to host memory asynchronously.
func MemcpyDtoHAsync(dst HostPtr, src DevicePtr, bytes int64, stream Stream) error {
	return MemcpyAsync(unsafe.Pointer(dst), unsafe.Pointer(src), bytes, DtoH, stream)
}

// MemcpyDtoDAsync copies data from device to device memory asynchronously.
func MemcpyDtoDAsync(dst, src DevicePtr, bytes int64, stream Stream) error {
	return MemcpyAsync(unsafe.Pointer(dst), unsafe.Pointer(src), bytes, DtoD, stream)
}

// MemcpyHtoHAsync copies data from host to host memory asynchronously.
func MemcpyHtoHAsync(dst, src HostPtr, bytes int64, stream Stream) error {
	return MemcpyAsync(unsafe.Pointer(dst), unsafe.Pointer(src), bytes, HtoH, stream)
}

// Memset sets memory to a value.
func Memset(dst DevicePtr, value byte, bytes int64) error {
	return Check(C.cudaMemset(unsafe.Pointer(dst), C.int(value), C.size_t(bytes)))
}

// MemsetAsync sets memory to a value asynchronously.
func MemsetAsync(dst DevicePtr, value byte, bytes int64, stream Stream) error {
	return Check(C.cudaMemsetAsync(unsafe.Pointer(dst), C.int(value), C.size_t(bytes), stream.c()))
}
