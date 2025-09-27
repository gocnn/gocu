package gocu

// #include <cuda.h>
// #include <stdlib.h>
import "C"
import (
	"unsafe"

	"github.com/google/uuid"
)

// Device represents a CUDA device ordinal.
type Device int

// DeviceAttribute represents CUDA device attributes that can be queried. Generated from cuda v12.0 cuda.h.
type CUDeviceAttr int

// DeviceGet retrieves the device with the specified index.
// Returns an error if the index is invalid.
func DeviceGet(idx int) (Device, error) {
	var dev C.CUdevice
	err := Check(C.cuDeviceGet(&dev, C.int(idx)))
	return Device(dev), err
}

// DeviceGetCount returns the number of available CUDA devices.
// Returns an error if the query fails.
func DeviceGetCount() (int, error) {
	var count C.int
	err := Check(C.cuDeviceGetCount(&count))
	return int(count), err
}

// DeviceGetName retrieves the name of the specified device.
// Returns an error if the query fails.
func DeviceGetName(dev Device) (string, error) {
	const size = 256
	buf := make([]byte, size)
	cstr := (*C.char)(unsafe.Pointer(&buf[0]))
	if err := Check(C.cuDeviceGetName(cstr, C.int(size), C.CUdevice(dev))); err != nil {
		return "", err
	}
	return C.GoString(cstr), nil
}

// Name retrieves the name of the device.
// Returns an error if the query fails.
func (dev Device) Name() (string, error) {
	return DeviceGetName(dev)
}

// DeviceGetUUID retrieves the UUID of the specified device.
// Returns an error if the query fails.
func DeviceGetUUID(dev Device) (uuid.UUID, error) {
	var id uuid.UUID
	err := Check(C.cuDeviceGetUuid((*C.CUuuid)(unsafe.Pointer(&id)), C.CUdevice(dev)))
	return id, err
}

// UUID retrieves the UUID of the device.
// Returns an error if the query fails.
func (dev Device) UUID() (uuid.UUID, error) {
	return DeviceGetUUID(dev)
}

// DeviceGetAttribute retrieves the value of the specified device attribute.
// Returns an error if the query fails.
func DeviceGetAttribute(attr CUDeviceAttr, dev Device) (int, error) {
	var val C.int
	err := Check(C.cuDeviceGetAttribute(&val, C.CUdevice_attribute(attr), C.CUdevice(dev)))
	return int(val), err
}

// Attribute retrieves the value of the specified device attribute.
// Returns an error if the query fails.
func (dev Device) Attribute(attr CUDeviceAttr) (int, error) {
	return DeviceGetAttribute(attr, dev)
}

// DeviceTotalMem retrieves the total global memory available on the specified device in bytes.
// Returns an error if the query fails.
func DeviceTotalMem(dev Device) (int64, error) {
	var bytes C.size_t
	err := Check(C.cuDeviceTotalMem(&bytes, C.CUdevice(dev)))
	return int64(bytes), err
}

// TotalMem retrieves the total global memory available on the device in bytes.
// Returns an error if the query fails.
func (dev Device) TotalMem() (int64, error) {
	return DeviceTotalMem(dev)
}
