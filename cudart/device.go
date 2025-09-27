package cudart

/*
#include <cuda_runtime.h>
*/
import "C"

// Device represents a CUDA device ordinal.
type Device int

// CudaDeviceAttr represents CUDA device attributes that can be queried.
type CudaDeviceAttr uint32

// CudaLimit represents CUDA limits that can be queried or set.
type CudaLimit uint32

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
func GetDeviceAttribute(attr CudaDeviceAttr, device Device) (int, error) {
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
func DeviceGetLimit(limit CudaLimit) (int64, error) {
	var value C.size_t
	err := Check(C.cudaDeviceGetLimit(&value, uint32(limit)))
	return int64(value), err
}

// DeviceSetLimit sets resource limits.
func DeviceSetLimit(limit CudaLimit, value int64) error {
	return Check(C.cudaDeviceSetLimit(uint32(limit), C.size_t(value)))
}

// GetDeviceProperties returns the properties of a compute device.
func GetDeviceProperties(device Device) (*CudaDeviceProp, error) {
	var prop C.struct_cudaDeviceProp
	var properties CudaDeviceProp
	err := Check(C.cudaGetDeviceProperties(&prop, C.int(device)))
	if err != nil {
		return nil, err
	}
	properties.fromC(&prop)
	return &properties, nil
}
