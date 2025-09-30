package gocu

/*
#include <cuda.h>
*/
import "C"
import (
	"unsafe"
)

// Dim3 represents a 3D dimension for CUDA grids and blocks.
type Dim3 struct {
	X, Y, Z uint32
}

// MakeDim3 creates a Dim3 with the specified dimensions.
func MakeDim3(x, y, z uint32) Dim3 {
	return Dim3{X: x, Y: y, Z: z}
}

// MakeDim3_1D creates a 1D Dim3.
func MakeDim3_1D(x uint32) Dim3 {
	return Dim3{X: x, Y: 1, Z: 1}
}

// MakeDim3_2D creates a 2D Dim3.
func MakeDim3_2D(x, y uint32) Dim3 {
	return Dim3{X: x, Y: y, Z: 1}
}

// CUfunctionAttribute represents function attributes that can be queried
type CUfunctionAttribute C.CUfunction_attribute

const (
	FuncAttributeMaxThreadsPerBlock            CUfunctionAttribute = C.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK
	FuncAttributeSharedSizeBytes               CUfunctionAttribute = C.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES
	FuncAttributeConstSizeBytes                CUfunctionAttribute = C.CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES
	FuncAttributeLocalSizeBytes                CUfunctionAttribute = C.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES
	FuncAttributeNumRegs                       CUfunctionAttribute = C.CU_FUNC_ATTRIBUTE_NUM_REGS
	FuncAttributePtxVersion                    CUfunctionAttribute = C.CU_FUNC_ATTRIBUTE_PTX_VERSION
	FuncAttributeBinaryVersion                 CUfunctionAttribute = C.CU_FUNC_ATTRIBUTE_BINARY_VERSION
	FuncAttributeCacheModeCA                   CUfunctionAttribute = C.CU_FUNC_ATTRIBUTE_CACHE_MODE_CA
	FuncAttributeMaxDynamicSharedSizeBytes     CUfunctionAttribute = C.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
	FuncAttributePreferredSharedMemoryCarveout CUfunctionAttribute = C.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
)

// CUfuncCache represents cache configuration options
type CUfuncCache C.CUfunc_cache

const (
	FuncCachePreferNone   CUfuncCache = C.CU_FUNC_CACHE_PREFER_NONE   // No preference for shared memory or L1 (default)
	FuncCachePreferShared CUfuncCache = C.CU_FUNC_CACHE_PREFER_SHARED // Prefer larger shared memory and smaller L1 cache
	FuncCachePreferL1     CUfuncCache = C.CU_FUNC_CACHE_PREFER_L1     // Prefer larger L1 cache and smaller shared memory
	FuncCachePreferEqual  CUfuncCache = C.CU_FUNC_CACHE_PREFER_EQUAL  // Prefer equal sized L1 cache and shared memory
)

// CUfunctionLoadingState represents function loading state
type CUfunctionLoadingState C.CUfunctionLoadingState

const (
	FunctionLoadingStateUnloaded CUfunctionLoadingState = C.CU_FUNCTION_LOADING_STATE_UNLOADED
	FunctionLoadingStateLoaded   CUfunctionLoadingState = C.CU_FUNCTION_LOADING_STATE_LOADED
)

// CUlaunchConfig represents launch configuration for extended kernel launch
type CUlaunchConfig struct {
	GridDimX       uint32
	GridDimY       uint32
	GridDimZ       uint32
	BlockDimX      uint32
	BlockDimY      uint32
	BlockDimZ      uint32
	SharedMemBytes uint32
	Stream         Stream
}

// Function Information Functions

// FuncGetAttribute returns information about a function
func FuncGetAttribute(function Function, attrib CUfunctionAttribute) (int, error) {
	var pi C.int
	result := Check(C.cuFuncGetAttribute(&pi, C.CUfunction_attribute(attrib), function.CFunction()))
	return int(pi), result
}

// GetAttribute returns information about the function (method version)
func (f Function) GetAttribute(attrib CUfunctionAttribute) (int, error) {
	return FuncGetAttribute(f, attrib)
}

// FuncSetAttribute sets information about a function
func FuncSetAttribute(function Function, attrib CUfunctionAttribute, value int) error {
	return Check(C.cuFuncSetAttribute(function.CFunction(), C.CUfunction_attribute(attrib), C.int(value)))
}

// SetAttribute sets information about the function (method version)
func (f Function) SetAttribute(attrib CUfunctionAttribute, value int) error {
	return FuncSetAttribute(f, attrib, value)
}

// FuncGetModule returns a module handle from a function
func FuncGetModule(function Function) (Module, error) {
	var module C.CUmodule
	result := Check(C.cuFuncGetModule(&module, function.CFunction()))
	return Module{module: module}, result
}

// GetModule returns the module handle from the function (method version)
func (f Function) GetModule() (Module, error) {
	return FuncGetModule(f)
}

// FuncGetName returns the function name for a CUfunction handle
func FuncGetName(function Function) (string, error) {
	var name *C.char
	result := Check(C.cuFuncGetName(&name, function.CFunction()))
	if result != nil {
		return "", result
	}
	return C.GoString(name), nil
}

// GetName returns the function name (method version)
func (f Function) GetName() (string, error) {
	return FuncGetName(f)
}

// FuncGetParamInfo returns the offset and size of a kernel parameter
func FuncGetParamInfo(function Function, paramIndex uint64) (uint64, uint64, error) {
	var paramOffset, paramSize C.size_t
	result := Check(C.cuFuncGetParamInfo(function.CFunction(), C.size_t(paramIndex), &paramOffset, &paramSize))
	return uint64(paramOffset), uint64(paramSize), result
}

// GetParamInfo returns parameter offset and size (method version)
func (f Function) GetParamInfo(paramIndex uint64) (uint64, uint64, error) {
	return FuncGetParamInfo(f, paramIndex)
}

// Function Loading Functions

// FuncIsLoaded returns if the function is loaded
func FuncIsLoaded(function Function) (CUfunctionLoadingState, error) {
	var state C.CUfunctionLoadingState
	result := Check(C.cuFuncIsLoaded(&state, function.CFunction()))
	return CUfunctionLoadingState(state), result
}

// IsLoaded returns if the function is loaded (method version)
func (f Function) IsLoaded() (CUfunctionLoadingState, error) {
	return FuncIsLoaded(f)
}

// FuncLoad loads a function
func FuncLoad(function Function) error {
	return Check(C.cuFuncLoad(function.CFunction()))
}

// Load loads the function (method version)
func (f Function) Load() error {
	return FuncLoad(f)
}

// Cache Configuration Functions

// FuncSetCacheConfig sets the preferred cache configuration for a device function
func FuncSetCacheConfig(function Function, config CUfuncCache) error {
	return Check(C.cuFuncSetCacheConfig(function.CFunction(), C.CUfunc_cache(config)))
}

// SetCacheConfig sets the preferred cache configuration (method version)
func (f Function) SetCacheConfig(config CUfuncCache) error {
	return FuncSetCacheConfig(f, config)
}

// Kernel Launch Functions

// LaunchKernel launches a CUDA function
func LaunchKernel(function Function, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ uint32, sharedMemBytes uint32, stream Stream, kernelParams []unsafe.Pointer, extra []unsafe.Pointer) error {
	// Since Go 1.6, a cgo argument cannot have a Go pointer to Go pointer,
	// so we copy the argument values go C memory first.
	argv := C.malloc(C.size_t(len(kernelParams) * pointerSize))
	argp := C.malloc(C.size_t(len(kernelParams) * pointerSize))
	defer C.free(argv)
	defer C.free(argp)
	for i := range kernelParams {
		*((*unsafe.Pointer)(offset(argp, i))) = offset(argv, i)       // argp[i] = &argv[i]
		*((*uint64)(offset(argv, i))) = *((*uint64)(kernelParams[i])) // argv[i] = *kernelParams[i]
	}

	return Check(C.cuLaunchKernel(
		function.CFunction(),
		C.uint(gridDimX), C.uint(gridDimY), C.uint(gridDimZ),
		C.uint(blockDimX), C.uint(blockDimY), C.uint(blockDimZ),
		C.uint(sharedMemBytes),
		stream.CStream(),
		(*unsafe.Pointer)(argp),
		(*unsafe.Pointer)(nil),
	))
}

func offset(ptr unsafe.Pointer, i int) unsafe.Pointer {
	return unsafe.Pointer(uintptr(ptr) + pointerSize*uintptr(i))
}

const pointerSize = 8

// Launch launches the kernel (method version)
func (f Function) Launch(gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ uint32, sharedMemBytes uint32, stream Stream, kernelParams []unsafe.Pointer, extra []unsafe.Pointer) error {
	return LaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, stream, kernelParams, extra)
}

// LaunchKernelEx launches a CUDA function with extended configuration
func LaunchKernelEx(config CUlaunchConfig, function Function, kernelParams []unsafe.Pointer, extra []unsafe.Pointer) error {
	cConfig := C.CUlaunchConfig{
		gridDimX:       C.uint(config.GridDimX),
		gridDimY:       C.uint(config.GridDimY),
		gridDimZ:       C.uint(config.GridDimZ),
		blockDimX:      C.uint(config.BlockDimX),
		blockDimY:      C.uint(config.BlockDimY),
		blockDimZ:      C.uint(config.BlockDimZ),
		sharedMemBytes: C.uint(config.SharedMemBytes),
		hStream:        config.Stream.CStream(),
	}

	var kernelParamsPtr *unsafe.Pointer
	var extraPtr *unsafe.Pointer

	if len(kernelParams) > 0 {
		kernelParamsPtr = &kernelParams[0]
	}
	if len(extra) > 0 {
		extraPtr = &extra[0]
	}

	return Check(C.cuLaunchKernelEx(&cConfig, function.CFunction(), kernelParamsPtr, extraPtr))
}

// LaunchEx launches the kernel with extended configuration (method version)
func (f Function) LaunchEx(config CUlaunchConfig, kernelParams []unsafe.Pointer, extra []unsafe.Pointer) error {
	return LaunchKernelEx(config, f, kernelParams, extra)
}

// LaunchCooperativeKernel launches a cooperative CUDA kernel
func LaunchCooperativeKernel(function Function, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ uint32, sharedMemBytes uint32, stream Stream, kernelParams []unsafe.Pointer) error {
	var kernelParamsPtr *unsafe.Pointer
	if len(kernelParams) > 0 {
		kernelParamsPtr = &kernelParams[0]
	}

	return Check(C.cuLaunchCooperativeKernel(
		function.CFunction(),
		C.uint(gridDimX), C.uint(gridDimY), C.uint(gridDimZ),
		C.uint(blockDimX), C.uint(blockDimY), C.uint(blockDimZ),
		C.uint(sharedMemBytes),
		stream.CStream(),
		kernelParamsPtr,
	))
}

// LaunchCooperative launches the cooperative kernel (method version)
func (f Function) LaunchCooperative(gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ uint32, sharedMemBytes uint32, stream Stream, kernelParams []unsafe.Pointer) error {
	return LaunchCooperativeKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, stream, kernelParams)
}
