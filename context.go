package gocu

// #include <cuda.h>
import "C"

// CUContext is a CUDA context
type CUContext struct{ ctx C.CUcontext }

// C returns the CUContext as its C version
func (ctx CUContext) c() C.CUcontext { return ctx.ctx }
