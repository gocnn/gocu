package gocu

// #include <cuda.h>
import "C"

// Stream is a CUDA stream
type Stream struct{ stream C.CUstream }

// C returns the Stream as its C version
func (s Stream) c() C.CUstream { return s.stream }
