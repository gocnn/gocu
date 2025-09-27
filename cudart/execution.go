package cudart

/*
#include <cuda_runtime.h>
*/
import "C"
import "unsafe"

// Dim3 represents a 3D dimension for CUDA grids and blocks.
type Dim3 struct {
	X, Y, Z uint32
}

// toCDim3 converts Go Dim3 to C dim3.
func (d Dim3) toCDim3() C.dim3 {
	return C.dim3{x: C.uint(d.X), y: C.uint(d.Y), z: C.uint(d.Z)}
}

// FuncAttribute represents CUDA function attributes.
type FuncAttribute int

const (
	FuncAttributeMaxDynamicSharedMemorySize        FuncAttribute = C.cudaFuncAttributeMaxDynamicSharedMemorySize        // Maximum dynamic shared memory size
	FuncAttributePreferredSharedMemoryCarveout     FuncAttribute = C.cudaFuncAttributePreferredSharedMemoryCarveout     // Preferred shared memory-L1 cache split
	FuncAttributeClusterDimMustBeSet               FuncAttribute = C.cudaFuncAttributeClusterDimMustBeSet               // Indicator to enforce valid cluster dimension specification
	FuncAttributeRequiredClusterWidth              FuncAttribute = C.cudaFuncAttributeRequiredClusterWidth              // Required cluster width
	FuncAttributeRequiredClusterHeight             FuncAttribute = C.cudaFuncAttributeRequiredClusterHeight             // Required cluster height
	FuncAttributeRequiredClusterDepth              FuncAttribute = C.cudaFuncAttributeRequiredClusterDepth              // Required cluster depth
	FuncAttributeNonPortableClusterSizeAllowed     FuncAttribute = C.cudaFuncAttributeNonPortableClusterSizeAllowed     // Whether non-portable cluster scheduling policy is supported
	FuncAttributeClusterSchedulingPolicyPreference FuncAttribute = C.cudaFuncAttributeClusterSchedulingPolicyPreference // Required cluster scheduling policy preference
)

// FuncCache represents cache configuration for CUDA functions.
type FuncCache int

const (
	FuncCachePreferNone   FuncCache = C.cudaFuncCachePreferNone   // No preference for shared memory or L1 (default)
	FuncCachePreferShared FuncCache = C.cudaFuncCachePreferShared // Prefer larger shared memory and smaller L1 cache
	FuncCachePreferL1     FuncCache = C.cudaFuncCachePreferL1     // Prefer larger L1 cache and smaller shared memory
	FuncCachePreferEqual  FuncCache = C.cudaFuncCachePreferEqual  // Prefer equal sized L1 cache and shared memory
)

// HostFn represents a host function that can be launched on a stream.
type HostFn func(userData unsafe.Pointer)

// LaunchKernel launches a device function (kernel).
// func: pointer to the kernel function
// gridDim: grid dimensions
// blockDim: block dimensions
// args: array of pointers to kernel arguments
// sharedMem: dynamic shared memory size per thread block in bytes
// stream: stream to launch the kernel on
func LaunchKernel(function unsafe.Pointer, gridDim, blockDim Dim3, args []unsafe.Pointer, sharedMem int64, stream Stream) error {
	var argsPtr *unsafe.Pointer
	if len(args) > 0 {
		argsPtr = &args[0]
	}

	return Check(C.cudaLaunchKernel(
		function,
		gridDim.toCDim3(),
		blockDim.toCDim3(),
		argsPtr,
		C.size_t(sharedMem),
		stream.c(),
	))
}

// LaunchCooperativeKernel launches a device function where thread blocks can cooperate.
// This is used for kernels that require synchronization between thread blocks.
func LaunchCooperativeKernel(function unsafe.Pointer, gridDim, blockDim Dim3, args []unsafe.Pointer, sharedMem int64, stream Stream) error {
	var argsPtr *unsafe.Pointer
	if len(args) > 0 {
		argsPtr = &args[0]
	}

	return Check(C.cudaLaunchCooperativeKernel(
		function,
		gridDim.toCDim3(),
		blockDim.toCDim3(),
		argsPtr,
		C.size_t(sharedMem),
		stream.c(),
	))
}

// LaunchHostFunc enqueues a host function call in a stream.
// The host function will be called when all previously enqueued operations complete.
func LaunchHostFunc(stream Stream, fn HostFn, userData unsafe.Pointer) error {
	// Store the Go function and userData in a way that's safe for CGO
	// We need to use a different approach since we can't pass Go function pointers to C

	// For now, return an error indicating this function needs proper implementation
	// TODO: Implement proper callback mechanism using cgo export functions
	return Check(C.cudaErrorNotSupported)
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
