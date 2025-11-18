//go:build !cuda
// +build !cuda

package cublas

import (
	"errors"

	"github.com/gocnn/gocu/cudart"
)

type Handle struct {
	h uintptr
}

// Create creates a new CUBLAS handle and returns it.
// The handle is used to manage CUBLAS library context and must be destroyed with Destroy when no longer needed.
func Create() (*Handle, error) {
	return nil, errors.New("cublas: CUDA support not available")
}

func Destroy(h *Handle) error {
	return errors.New("cublas: CUDA support not available")
}

func (h *Handle) Destroy() error {
	return errors.New("cublas: CUDA support not available")
}

// CHandle returns the underlying handle pointer for use in CUBLAS API calls.
func (h *Handle) CHandle() uintptr {
	return h.h
}

// SetStream sets the cuBLAS library stream, which will be used to execute all subsequent calls to the cuBLAS library functions.
// If the cuBLAS library stream is not set, all kernels use the default NULL stream.
// This routine can be used to change the stream between kernel launches and then to reset the cuBLAS library stream back to NULL.
// Additionally this function unconditionally resets the cuBLAS library workspace back to the default workspace pool.
func (h Handle) SetStream(stream *cudart.Stream) error {
	return errors.New("cublas: CUDA support not available")
}

// GetStream gets the cuBLAS library stream, which is being used to execute all calls to the cuBLAS library functions.
// If the cuBLAS library stream is not set, all kernels use the default NULL stream.
func (h Handle) GetStream() (*cudart.Stream, error) {
	return nil, errors.New("cublas: CUDA support not available")
}
