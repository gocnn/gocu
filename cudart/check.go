package cudart

/*
#include <cuda_runtime.h>
*/
import "C"
import "fmt"

// Error represents a CUDA error code.
type Error C.enum_cudaError

func (err Error) Error() string { return err.String() }
func (err Error) String() string {
	if msg, ok := ErrorMessages[err]; ok {
		return msg
	}
	return fmt.Sprintf("Unknown Error Code:%d", err)
}

func Check(x C.cudaError_t) error {
	err := Error(x)
	if err == CudaSuccess {
		return nil
	}
	return err
}
