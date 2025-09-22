package cudart

// #include <cuda_runtime.h>
import "C"

// Device represents a CUDA device ordinal.
type Device int

// Set the device as current.
func SetDevice(device Device) error {
	err := Result(C.cudaSetDevice(C.int(device)))
	return err
}
