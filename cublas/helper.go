package cublas

/*
#include <cublas_v2.h>
*/
import "C"
import (
	"unsafe"

	"github.com/gocnn/gocu/cudart"
)

type Handle struct {
	h C.cublasHandle_t
}

// Create creates a new CUBLAS handle and returns it.
// The handle is used to manage CUBLAS library context and must be destroyed with Destroy when no longer needed.
func Create() (*Handle, error) {
	var handle C.cublasHandle_t
	if err := Check(C.cublasCreate(&handle)); err != nil {
		return nil, err
	}
	return &Handle{h: handle}, nil
}

func Destroy(h *Handle) error {
	if err := Check(C.cublasDestroy(h.h)); err != nil {
		return err
	}
	return nil
}

func (h *Handle) Destroy() error {
	return Destroy(h)
}

// CHandle returns the underlying C cublasHandle_t for use in CUBLAS API calls.
func (h *Handle) CHandle() C.cublasHandle_t {
	return h.h
}

// SetStream sets the cuBLAS library stream, which will be used to execute all subsequent calls to the cuBLAS library functions.
// If the cuBLAS library stream is not set, all kernels use the default NULL stream.
// This routine can be used to change the stream between kernel launches and then to reset the cuBLAS library stream back to NULL.
// Additionally this function unconditionally resets the cuBLAS library workspace back to the default workspace pool.
func (h Handle) SetStream(stream *cudart.Stream) error {
	return Check(C.cublasSetStream(h.h, C.cudaStream_t(unsafe.Pointer(stream.CStream()))))
}

// GetStream gets the cuBLAS library stream, which is being used to execute all calls to the cuBLAS library functions.
// If the cuBLAS library stream is not set, all kernels use the default NULL stream.
func (h Handle) GetStream() (*cudart.Stream, error) {
	var streamId C.cudaStream_t
	if err := Check(C.cublasGetStream(h.h, &streamId)); err != nil {
		return nil, err
	}
	stream := &cudart.Stream{}
	stream.SetCStream(unsafe.Pointer(streamId))
	return stream, nil
}
