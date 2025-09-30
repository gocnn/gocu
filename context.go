//go:build !cuda13

package gocu

/*
#include <cuda.h>
*/
import "C"

// Limit represents context resource limits
type Limit C.CUlimit

// ContextFlag represents context creation flags
type ContextFlag uint32

// Context is a CUDA context
type Context struct{ ctx C.CUcontext }

// CContext returns the Context as its C version
func (ctx *Context) CContext() C.CUcontext { return ctx.ctx }

// CtxCreate creates a CUDA context
func CtxCreate(flags ContextFlag, dev Device) (*Context, error) {
	var ctx C.CUcontext
	result := Check(C.cuCtxCreate(&ctx, C.uint(flags), C.CUdevice(dev)))
	return &Context{ctx: ctx}, result
}

// CtxDestroy destroys a CUDA context
func CtxDestroy(ctx *Context) error {
	return Check(C.cuCtxDestroy(ctx.CContext()))
}

// Destroy destroys the context (method version)
func (ctx *Context) Destroy() error {
	return CtxDestroy(ctx)
}

// CtxPushCurrent pushes a context on the current CPU thread
func CtxPushCurrent(ctx *Context) error {
	return Check(C.cuCtxPushCurrent(ctx.CContext()))
}

// PushCurrent pushes the context on the current CPU thread (method version)
func (ctx *Context) PushCurrent() error {
	return CtxPushCurrent(ctx)
}

// CtxPopCurrent pops the current CUDA context from the current CPU thread
func CtxPopCurrent() (*Context, error) {
	var ctx C.CUcontext
	result := Check(C.cuCtxPopCurrent(&ctx))
	return &Context{ctx: ctx}, result
}

// CtxSetCurrent binds the specified CUDA context to the calling CPU thread
func CtxSetCurrent(ctx *Context) error {
	return Check(C.cuCtxSetCurrent(ctx.CContext()))
}

// SetCurrent binds the context to the calling CPU thread (method version)
func (ctx *Context) SetCurrent() error {
	return CtxSetCurrent(ctx)
}

// CtxGetCurrent returns the CUDA context bound to the calling CPU thread
func CtxGetCurrent() (*Context, error) {
	var ctx C.CUcontext
	result := Check(C.cuCtxGetCurrent(&ctx))
	return &Context{ctx: ctx}, result
}

// CtxGetApiVersion gets the context's API version
func CtxGetApiVersion(ctx *Context) (uint32, error) {
	var version C.uint
	result := Check(C.cuCtxGetApiVersion(ctx.CContext(), &version))
	return uint32(version), result
}

// GetApiVersion gets the context's API version (method version)
func (ctx *Context) GetApiVersion() (uint32, error) {
	return CtxGetApiVersion(ctx)
}

// CtxGetDevice returns the device handle for the current context
func CtxGetDevice() (Device, error) {
	var device C.CUdevice
	result := Check(C.cuCtxGetDevice(&device))
	return Device(device), result
}

// GetDevice returns the device handle for the context (method version)
func (ctx *Context) GetDevice() (Device, error) {
	return CtxGetDevice()
}

// CtxGetFlags returns the flags for the current context
func CtxGetFlags() (uint32, error) {
	var flags C.uint
	result := Check(C.cuCtxGetFlags(&flags))
	return uint32(flags), result
}

// CtxSetFlags sets the flags for the current context
func CtxSetFlags(flags uint32) error {
	return Check(C.cuCtxSetFlags(C.uint(flags)))
}

// CtxGetId returns the unique Id associated with the context
func CtxGetId(ctx *Context) (uint64, error) {
	var ctxId C.ulonglong
	result := Check(C.cuCtxGetId(ctx.CContext(), &ctxId))
	return uint64(ctxId), result
}

// GetId returns the unique Id associated with the context (method version)
func (ctx *Context) GetId() (uint64, error) {
	return CtxGetId(ctx)
}

// CtxGetLimit returns resource limits
func CtxGetLimit(limit Limit) (uint64, error) {
	var pvalue C.size_t
	result := Check(C.cuCtxGetLimit(&pvalue, C.CUlimit(limit)))
	return uint64(pvalue), result
}

// CtxSetLimit sets resource limits
func CtxSetLimit(limit Limit, value uint64) error {
	return Check(C.cuCtxSetLimit(C.CUlimit(limit), C.size_t(value)))
}

// CtxGetStreamPriorityRange returns numerical values that correspond to the least and greatest stream priorities
func CtxGetStreamPriorityRange() (int, int, error) {
	var leastPriority, greatestPriority C.int
	result := Check(C.cuCtxGetStreamPriorityRange(&leastPriority, &greatestPriority))
	return int(leastPriority), int(greatestPriority), result
}

// CtxGetCacheConfig returns the preferred cache configuration for the current context
func CtxGetCacheConfig() (CUfuncCache, error) {
	var config C.CUfunc_cache
	result := Check(C.cuCtxGetCacheConfig(&config))
	return CUfuncCache(config), result
}

// CtxSetCacheConfig sets the preferred cache configuration for the current context
func CtxSetCacheConfig(config CUfuncCache) error {
	return Check(C.cuCtxSetCacheConfig(C.CUfunc_cache(config)))
}

// CtxSynchronize blocks for the current context's tasks to complete
func CtxSynchronize() error {
	return Check(C.cuCtxSynchronize())
}

// Synchronize blocks for the context's tasks to complete (method version)
func (ctx *Context) Synchronize() error {
	return CtxSynchronize()
}

// CtxResetPersistingL2Cache resets all persisting lines in cache to normal status
func CtxResetPersistingL2Cache() error {
	return Check(C.cuCtxResetPersistingL2Cache())
}

// CtxRecordEvent records an event
func CtxRecordEvent(ctx *Context, event *Event) error {
	return Check(C.cuCtxRecordEvent(ctx.CContext(), event.CEvent()))
}

// RecordEvent records an event (method version)
func (ctx *Context) RecordEvent(event *Event) error {
	return CtxRecordEvent(ctx, event)
}

// CtxWaitEvent makes a context wait on an event
func CtxWaitEvent(ctx *Context, event *Event) error {
	return Check(C.cuCtxWaitEvent(ctx.CContext(), event.CEvent()))
}

// WaitEvent makes the context wait on an event (method version)
func (ctx *Context) WaitEvent(event *Event) error {
	return CtxWaitEvent(ctx, event)
}
