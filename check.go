package gocu

//#include <cuda.h>
import "C"
import "fmt"

type Result int

func (err Result) Error() string { return err.String() }
func (err Result) String() string {
	if msg, ok := ResultMessages[err]; ok {
		return msg
	}
	return fmt.Sprintf("UnknownErrorCode:%d", err)
}

func Check(x C.CUresult) error {
	err := Result(x)
	if err == CudaSuccess {
		return nil
	}
	return err
}
