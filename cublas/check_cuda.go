//go:build cuda || cuda11 || cuda12 || cuda13

package cublas

/*
#include <cublas_v2.h>
*/
import "C"
import "fmt"

// Status is the cublas Check.
type Status int

func (err Status) Error() string { return err.String() }
func (err Status) String() string {
	if msg, ok := StatusMessages[err]; ok {
		return msg
	}
	return fmt.Sprintf("Unknown Error Code:%d", err)
}

func Check(x C.cublasStatus_t) error {
	err := Status(x)
	if err == CublasStatusSuccess {
		return nil
	}
	return err
}

const (
	CublasStatusSuccess        Status = C.CUBLAS_STATUS_SUCCESS          // The operation completed successfully.
	CublasStatusNotInitialized Status = C.CUBLAS_STATUS_NOT_INITIALIZED  // The cuBLAS library was not initialized. This is usually caused by the lack of a prior cublasCreate() call,
	CublasStatusAllocFailed    Status = C.CUBLAS_STATUS_ALLOC_FAILED     // Resource allocation failed inside the cuBLAS library.
	CublasStatusInvalidValue   Status = C.CUBLAS_STATUS_INVALID_VALUE    // An unsupported value or parameter was passed to the function (a negative vector size, for example).
	CublasStatusArchMismatch   Status = C.CUBLAS_STATUS_ARCH_MISMATCH    // The function requires a feature absent from the device architecture; usually caused by the lack of support for double precision.
	CublasStatusMappingError   Status = C.CUBLAS_STATUS_MAPPING_ERROR    // An access to GPU memory space failed, which is usually caused by a failure to bind a texture.
	CublasStatusExecFailed     Status = C.CUBLAS_STATUS_EXECUTION_FAILED // The GPU program failed to execute. This is often caused by a launch failure of the kernel on the GPU, which can be caused by multiple reasons.
	CublasStatusInternalError  Status = C.CUBLAS_STATUS_INTERNAL_ERROR   // An internal cuBLAS operation failed. This error is usually caused by a cudaMemcpyAsync() failure.
	CublasStatusUnsupported    Status = C.CUBLAS_STATUS_NOT_SUPPORTED    // The functionnality requested is not supported
	CublasStatusLicenceError   Status = C.CUBLAS_STATUS_LICENSE_ERROR    // The functionnality requested requires some license and an error was detected when trying to check the current licensing.
)

var StatusMessages = map[Status]string{
	CublasStatusSuccess:        "Success",
	CublasStatusNotInitialized: "NotInitialized",
	CublasStatusAllocFailed:    "AllocFailed",
	CublasStatusInvalidValue:   "InvalidValue",
	CublasStatusArchMismatch:   "ArchMismatch",
	CublasStatusMappingError:   "MappingError",
	CublasStatusExecFailed:     "ExecFailed",
	CublasStatusInternalError:  "InternalError",
	CublasStatusUnsupported:    "Unsupported",
	CublasStatusLicenceError:   "LicenceError",
}
