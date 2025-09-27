package cudart

/*
#include <cuda_runtime.h>
*/
import "C"
import "fmt"

// CudaError represents a CUDA error code.
type CudaError C.enum_cudaError

func (err CudaError) Error() string { return err.String() }
func (err CudaError) String() string {
	if msg, ok := ErrorMessages[err]; ok {
		return msg
	}
	return fmt.Sprintf("UnknownErrorCode:%d", err)
}

func Check(x C.cudaError_t) error {
	err := CudaError(x)
	if err == CudaSuccess {
		return nil
	}
	return err
}
