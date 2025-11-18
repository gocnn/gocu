//go:build !cuda
// +build !cuda

package cublas

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

func Check(x Status) error {
	if x == CublasStatusSuccess {
		return nil
	}
	return x
}

const (
	CublasStatusSuccess        Status = 0  // The operation completed successfully.
	CublasStatusNotInitialized Status = 1  // The cuBLAS library was not initialized.
	CublasStatusAllocFailed    Status = 3  // Resource allocation failed inside the cuBLAS library.
	CublasStatusInvalidValue   Status = 7  // An unsupported value or parameter was passed to the function.
	CublasStatusArchMismatch   Status = 8  // The function requires a feature absent from the device architecture.
	CublasStatusMappingError   Status = 11 // An access to GPU memory space failed.
	CublasStatusExecFailed     Status = 13 // The GPU program failed to execute.
	CublasStatusInternalError  Status = 14 // An internal cuBLAS operation failed.
	CublasStatusUnsupported    Status = 15 // The functionality requested is not supported.
	CublasStatusLicenceError   Status = 16 // The functionality requested requires some license.
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
