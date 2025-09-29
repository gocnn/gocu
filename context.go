//go:build !cuda13

package gocu

/*
#include <cuda.h>
*/
import "C"

// CUlimit represents context resource limits
type CUlimit C.CUlimit

const (
	LimitStackSize                    CUlimit = C.CU_LIMIT_STACK_SIZE                       // GPU thread stack size
	LimitPrintfFifoSize               CUlimit = C.CU_LIMIT_PRINTF_FIFO_SIZE                 // GPU printf FIFO size
	LimitMallocHeapSize               CUlimit = C.CU_LIMIT_MALLOC_HEAP_SIZE                 // GPU malloc heap size
	LimitDevRuntimeSyncDepth          CUlimit = C.CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH           // GPU device runtime synchronization depth
	LimitDevRuntimePendingLaunchCount CUlimit = C.CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT // GPU device runtime pending launch count
	LimitMaxL2FetchGranularity        CUlimit = C.CU_LIMIT_MAX_L2_FETCH_GRANULARITY         // L2 cache fetch granularity
	LimitPersistingL2CacheSize        CUlimit = C.CU_LIMIT_PERSISTING_L2_CACHE_SIZE         // Persisting L2 cache size
)

// CUexecAffinityType represents execution affinity types
type CUexecAffinityType C.CUexecAffinityType

const (
	ExecAffinityTypeSMCount CUexecAffinityType = C.CU_EXEC_AFFINITY_TYPE_SM_COUNT // SM count affinity
	ExecAffinityTypeMax     CUexecAffinityType = C.CU_EXEC_AFFINITY_TYPE_MAX      // Maximum execution affinity type
)

// CUexecAffinityParam represents execution affinity parameters
type CUexecAffinityParam struct {
	Type  CUexecAffinityType
	Param CUexecAffinitySmCount
}

// CUexecAffinitySmCount represents SM count affinity parameters
type CUexecAffinitySmCount struct {
	Val uint32
}

// CUctxCreateParams represents context creation parameters
type CUctxCreateParams struct {
	ExecAffinityParamCount uint32
	ExecAffinityParams     *CUexecAffinityParam
}

// Context creation flags
const (
	CtxSchedAuto          uint32 = C.CU_CTX_SCHED_AUTO           // Automatic scheduling
	CtxSchedSpin          uint32 = C.CU_CTX_SCHED_SPIN           // Spin when waiting for results from the GPU
	CtxSchedYield         uint32 = C.CU_CTX_SCHED_YIELD          // Yield when waiting for results from the GPU
	CtxSchedBlockingSync  uint32 = C.CU_CTX_SCHED_BLOCKING_SYNC  // Use blocking synchronization
	CtxBlockingSync       uint32 = C.CU_CTX_BLOCKING_SYNC        // Use blocking synchronization (deprecated)
	CtxMapHost            uint32 = C.CU_CTX_MAP_HOST             // Support mapped pinned allocations
	CtxLmemResizeToMax    uint32 = C.CU_CTX_LMEM_RESIZE_TO_MAX   // Keep local memory allocation after launch
	CtxCoredumpEnable     uint32 = C.CU_CTX_COREDUMP_ENABLE      // Enable coredump creation
	CtxUserCoredumpEnable uint32 = C.CU_CTX_USER_COREDUMP_ENABLE // Enable user-triggered coredump creation
)

// Context is a CUDA context
type Context struct{ ctx C.CUcontext }

// CContext returns the Context as its C version
func (ctx *Context) CContext() C.CUcontext { return ctx.ctx }

// Context Creation and Destruction

// CtxCreate creates a CUDA context
func CtxCreate(flags uint32, dev Device) (*Context, error) {
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

// Context Stack Management

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

// Context Property Queries

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

// Resource Limit Management

// CtxGetLimit returns resource limits
func CtxGetLimit(limit CUlimit) (uint64, error) {
	var pvalue C.size_t
	result := Check(C.cuCtxGetLimit(&pvalue, C.CUlimit(limit)))
	return uint64(pvalue), result
}

// CtxSetLimit sets resource limits
func CtxSetLimit(limit CUlimit, value uint64) error {
	return Check(C.cuCtxSetLimit(C.CUlimit(limit), C.size_t(value)))
}

// CtxGetStreamPriorityRange returns numerical values that correspond to the least and greatest stream priorities
func CtxGetStreamPriorityRange() (int, int, error) {
	var leastPriority, greatestPriority C.int
	result := Check(C.cuCtxGetStreamPriorityRange(&leastPriority, &greatestPriority))
	return int(leastPriority), int(greatestPriority), result
}

// Execution Affinity

// CtxGetExecAffinity returns the execution affinity setting for the current context
func CtxGetExecAffinity(affinityType CUexecAffinityType) (CUexecAffinityParam, error) {
	var param CUexecAffinityParam
	cParam := C.CUexecAffinityParam{
		_type: C.CUexecAffinityType(affinityType),
	}

	result := Check(C.cuCtxGetExecAffinity(&cParam, C.CUexecAffinityType(affinityType)))
	if result != nil {
		return param, result
	}

	param.Type = CUexecAffinityType(cParam._type)
	// Note: Proper conversion of param union would be needed here

	return param, nil
}

// Cache Configuration

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

// Context Flags

// CtxSetFlags sets the flags for the current context
func CtxSetFlags(flags uint32) error {
	return Check(C.cuCtxSetFlags(C.uint(flags)))
}

// Synchronization

// CtxSynchronize blocks for the current context's tasks to complete
func CtxSynchronize() error {
	return Check(C.cuCtxSynchronize())
}

// Synchronize blocks for the context's tasks to complete (method version)
func (ctx *Context) Synchronize() error {
	return CtxSynchronize()
}

// L2 Cache Management

// CtxResetPersistingL2Cache resets all persisting lines in cache to normal status
func CtxResetPersistingL2Cache() error {
	return Check(C.cuCtxResetPersistingL2Cache())
}

// Event Operations

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
