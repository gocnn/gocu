package gocu

// #include <cuda.h>
import "C"

// Context is a CUDA context
type Context struct{ ctx C.CUcontext }

// C returns the Context as its C version
func (ctx Context) c() C.CUcontext { return ctx.ctx }
