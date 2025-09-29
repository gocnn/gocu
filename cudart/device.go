package cudart

/*
#include <cuda_runtime.h>
*/
import "C"

// Device represents a CUDA device ordinal.
type Device int

// DeviceAttr represents CUDA device attributes that can be queried.
type DeviceAttr uint32

// Limit represents CUDA limits that can be queried or set.
type Limit uint32

// Constants for Limit.
const (
	LimitDevRuntimePendingLaunchCount Limit = C.cudaLimitDevRuntimePendingLaunchCount
	LimitDevRuntimeSyncDepth          Limit = C.cudaLimitDevRuntimeSyncDepth
	LimitMallocHeapSize               Limit = C.cudaLimitMallocHeapSize
	LimitMaxL2FetchGranularity        Limit = C.cudaLimitMaxL2FetchGranularity
	LimitPersistingL2CacheSize        Limit = C.cudaLimitPersistingL2CacheSize
	LimitPrintfFifoSize               Limit = C.cudaLimitPrintfFifoSize
	LimitStackSize                    Limit = C.cudaLimitStackSize
)

// GetDevice returns which device is currently being used.
func GetDevice() (Device, error) {
	var device C.int
	err := Check(C.cudaGetDevice(&device))
	return Device(device), err
}

// SetDevice sets the device to be used for GPU executions.
func SetDevice(device Device) error {
	return Check(C.cudaSetDevice(C.int(device)))
}

// GetDeviceCount returns the number of compute-capable devices.
func GetDeviceCount() (int, error) {
	var count C.int
	err := Check(C.cudaGetDeviceCount(&count))
	return int(count), err
}

// GetDeviceAttribute returns information about the device.
func GetDeviceAttribute(attr DeviceAttr, device Device) (int, error) {
	var value C.int
	err := Check(C.cudaDeviceGetAttribute(&value, uint32(attr), C.int(device)))
	return int(value), err
}

// DeviceReset destroys all allocations and resets all state on the current device.
func DeviceReset() error {
	return Check(C.cudaDeviceReset())
}

// DeviceSynchronize waits for compute device to finish.
func DeviceSynchronize() error {
	return Check(C.cudaDeviceSynchronize())
}

// SetDeviceFlags sets flags to be used for device executions.
func SetDeviceFlags(flags uint) error {
	return Check(C.cudaSetDeviceFlags(C.uint(flags)))
}

// GetDeviceFlags gets the flags for the current device.
func GetDeviceFlags() (uint, error) {
	var flags C.uint
	err := Check(C.cudaGetDeviceFlags(&flags))
	return uint(flags), err
}

// DeviceGetLimit returns resource limits.
func DeviceGetLimit(limit Limit) (int64, error) {
	var value C.size_t
	err := Check(C.cudaDeviceGetLimit(&value, uint32(limit)))
	return int64(value), err
}

// DeviceSetLimit sets resource limits.
func DeviceSetLimit(limit Limit, value int64) error {
	return Check(C.cudaDeviceSetLimit(uint32(limit), C.size_t(value)))
}

// GetDeviceProperties returns the properties of a compute device.
func GetDeviceProperties(device Device) (*DeviceProp, error) {
	var prop C.struct_cudaDeviceProp
	var properties DeviceProp
	err := Check(C.cudaGetDeviceProperties(&prop, C.int(device)))
	if err != nil {
		return nil, err
	}
	properties.fromC(&prop)
	return &properties, nil
}
