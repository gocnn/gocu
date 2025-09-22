package cublas

// #include <cublas_v2.h>
import "C"

type Handle struct {
	h C.cublasHandle_t
}

// Create creates a new CUBLAS handle and returns it.
// The handle is used to manage CUBLAS library context and must be destroyed with Destroy when no longer needed.
func Create() (Handle, error) {
	var handle C.cublasHandle_t
	if err := Check(C.cublasCreate(&handle)); err != nil {
		return Handle{}, err
	}
	return Handle{h: handle}, nil
}

func Destroy(h Handle) error {
	if err := Check(C.cublasDestroy(h.h)); err != nil {
		return err
	}
	return nil
}

func (h Handle) Destroy() error {
	return Destroy(h)
}

// CHandle returns the underlying C cublasHandle_t for use in CUBLAS API calls.
func (h Handle) CHandle() C.cublasHandle_t {
	return h.h
}
