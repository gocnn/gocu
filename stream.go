package gocu

// #include <cuda.h>
import "C"

// CUStream is a CUDA stream
type CUStream struct{ stream C.CUstream }

// C returns the CUStream as its C version
func (s CUStream) c() C.CUstream { return s.stream }
