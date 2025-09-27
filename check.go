package gocu

//#include <cuda.h>
import "C"
import "fmt"

type CUResult int

func (err CUResult) Error() string { return err.String() }
func (err CUResult) String() string {
	if msg, ok := CUResultMessages[err]; ok {
		return msg
	}
	return fmt.Sprintf("UnknownErrorCode:%d", err)
}

func Check(x C.CUresult) error {
	err := CUResult(x)
	if err == CudaSuccess {
		return nil
	}
	return err
}
