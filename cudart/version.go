package cudart

/*
#include <cuda_runtime.h>
*/
import "C"

// DriverGetVersion returns the latest version of CUDA supported by the driver.
func DriverGetVersion() (int, error) {
	var driverVersion C.int
	err := Check(C.cudaDriverGetVersion(&driverVersion))
	return int(driverVersion), err
}

// RuntimeGetVersion returns the CUDA Runtime version.
func RuntimeGetVersion() (int, error) {
	var runtimeVersion C.int
	err := Check(C.cudaRuntimeGetVersion(&runtimeVersion))
	return int(runtimeVersion), err
}
