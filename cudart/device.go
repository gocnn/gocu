package cudart

/*
#include <cuda_runtime.h>
*/
import "C"

// Device represents a CUDA device ordinal.
type Device int

// DeviceAttribute represents CUDA device attributes that can be queried.
// Generated from cudaDeviceAttr enum. Only core attributes compatible with CUDA 10-13 are included.
type DeviceAttribute uint32

const (
	// Basic thread and grid configuration
	MaxThreadsPerBlock DeviceAttribute = C.cudaDevAttrMaxThreadsPerBlock // 1: Maximum number of threads per block
	MaxBlockDimX       DeviceAttribute = C.cudaDevAttrMaxBlockDimX       // 2: Maximum block dimension X
	MaxBlockDimY       DeviceAttribute = C.cudaDevAttrMaxBlockDimY       // 3: Maximum block dimension Y
	MaxBlockDimZ       DeviceAttribute = C.cudaDevAttrMaxBlockDimZ       // 4: Maximum block dimension Z
	MaxGridDimX        DeviceAttribute = C.cudaDevAttrMaxGridDimX        // 5: Maximum grid dimension X
	MaxGridDimY        DeviceAttribute = C.cudaDevAttrMaxGridDimY        // 6: Maximum grid dimension Y
	MaxGridDimZ        DeviceAttribute = C.cudaDevAttrMaxGridDimZ        // 7: Maximum grid dimension Z

	// Memory and registers
	MaxSharedMemoryPerBlock DeviceAttribute = C.cudaDevAttrMaxSharedMemoryPerBlock // 8: Maximum shared memory available per block in bytes
	TotalConstantMemory     DeviceAttribute = C.cudaDevAttrTotalConstantMemory     // 9: Memory available on device for __constant__ variables in a CUDA C kernel in bytes
	WarpSize                DeviceAttribute = C.cudaDevAttrWarpSize                // 10: Warp size in threads
	MaxPitch                DeviceAttribute = C.cudaDevAttrMaxPitch                // 11: Maximum pitch in bytes allowed by memory copies
	MaxRegistersPerBlock    DeviceAttribute = C.cudaDevAttrMaxRegistersPerBlock    // 12: Maximum number of 32-bit registers available per block

	// Performance and hardware
	ClockRate           DeviceAttribute = C.cudaDevAttrClockRate           // 13: Peak clock frequency in kilohertz
	TextureAlignment    DeviceAttribute = C.cudaDevAttrTextureAlignment    // 14: Alignment requirement for textures
	GpuOverlap          DeviceAttribute = C.cudaDevAttrGpuOverlap          // 15: Device can possibly copy memory and execute a kernel concurrently
	MultiProcessorCount DeviceAttribute = C.cudaDevAttrMultiProcessorCount // 16: Number of multiprocessors on device
	KernelExecTimeout   DeviceAttribute = C.cudaDevAttrKernelExecTimeout   // 17: Specifies whether there is a run time limit on kernels
	Integrated          DeviceAttribute = C.cudaDevAttrIntegrated          // 18: Device is integrated with host memory
	CanMapHostMemory    DeviceAttribute = C.cudaDevAttrCanMapHostMemory    // 19: Device can map host memory into CUDA address space
	ComputeMode         DeviceAttribute = C.cudaDevAttrComputeMode         // 20: Compute mode (See cudaComputeMode for details)

	// Core texture dimensions (1D, 2D, 3D, Cubemap)
	MaxTexture1DWidth         DeviceAttribute = C.cudaDevAttrMaxTexture1DWidth         // 21: Maximum 1D texture width
	MaxTexture2DWidth         DeviceAttribute = C.cudaDevAttrMaxTexture2DWidth         // 22: Maximum 2D texture width
	MaxTexture2DHeight        DeviceAttribute = C.cudaDevAttrMaxTexture2DHeight        // 23: Maximum 2D texture height
	MaxTexture3DWidth         DeviceAttribute = C.cudaDevAttrMaxTexture3DWidth         // 24: Maximum 3D texture width
	MaxTexture3DHeight        DeviceAttribute = C.cudaDevAttrMaxTexture3DHeight        // 25: Maximum 3D texture height
	MaxTexture3DDepth         DeviceAttribute = C.cudaDevAttrMaxTexture3DDepth         // 26: Maximum 3D texture depth
	MaxTexture2DLayeredWidth  DeviceAttribute = C.cudaDevAttrMaxTexture2DLayeredWidth  // 27: Maximum 2D layered texture width
	MaxTexture2DLayeredHeight DeviceAttribute = C.cudaDevAttrMaxTexture2DLayeredHeight // 28: Maximum 2D layered texture height
	MaxTexture2DLayeredLayers DeviceAttribute = C.cudaDevAttrMaxTexture2DLayeredLayers // 29: Maximum layers in a 2D layered texture
	TexturePitchAlignment     DeviceAttribute = C.cudaDevAttrTexturePitchAlignment     // 44: Pitch alignment requirement for textures
	MaxTextureCubemapWidth    DeviceAttribute = C.cudaDevAttrMaxTextureCubemapWidth    // 48: Maximum cubemap texture width/height

	// Core surface dimensions (1D, 2D, 3D, Cubemap)
	SurfaceAlignment       DeviceAttribute = C.cudaDevAttrSurfaceAlignment       // 30: Alignment requirement for surfaces
	MaxSurface2DWidth      DeviceAttribute = C.cudaDevAttrMaxSurface2DWidth      // 51: Maximum 2D surface width
	MaxSurface2DHeight     DeviceAttribute = C.cudaDevAttrMaxSurface2DHeight     // 52: Maximum 2D surface height
	MaxSurface3DWidth      DeviceAttribute = C.cudaDevAttrMaxSurface3DWidth      // 53: Maximum 3D surface width
	MaxSurface3DHeight     DeviceAttribute = C.cudaDevAttrMaxSurface3DHeight     // 54: Maximum 3D surface height
	MaxSurface3DDepth      DeviceAttribute = C.cudaDevAttrMaxSurface3DDepth      // 55: Maximum 3D surface depth
	MaxSurfaceCubemapWidth DeviceAttribute = C.cudaDevAttrMaxSurfaceCubemapWidth // 56: Maximum cubemap surface width

	// Concurrency and ECC
	ConcurrentKernels DeviceAttribute = C.cudaDevAttrConcurrentKernels // 31: Device can possibly execute multiple kernels concurrently
	EccEnabled        DeviceAttribute = C.cudaDevAttrEccEnabled        // 32: Device has ECC support enabled

	// PCI and driver
	PciBusId    DeviceAttribute = C.cudaDevAttrPciBusId    // 33: PCI bus ID of the device
	PciDeviceId DeviceAttribute = C.cudaDevAttrPciDeviceId // 34: PCI device ID of the device
	PciDomainId DeviceAttribute = C.cudaDevAttrPciDomainId // 49: PCI domain ID of the device
	TccDriver   DeviceAttribute = C.cudaDevAttrTccDriver   // 35: Device is using TCC driver model

	// Memory and clock
	MemoryClockRate      DeviceAttribute = C.cudaDevAttrMemoryClockRate      // 36: Peak memory clock frequency in kilohertz
	GlobalMemoryBusWidth DeviceAttribute = C.cudaDevAttrGlobalMemoryBusWidth // 37: Global memory bus width in bits
	L2CacheSize          DeviceAttribute = C.cudaDevAttrL2CacheSize          // 38: Size of L2 cache in bytes

	// Advanced threads and async
	MaxThreadsPerMultiProcessor DeviceAttribute = C.cudaDevAttrMaxThreadsPerMultiProcessor // 39: Maximum resident threads per multiprocessor
	AsyncEngineCount            DeviceAttribute = C.cudaDevAttrAsyncEngineCount            // 40: Number of asynchronous engines
	UnifiedAddressing           DeviceAttribute = C.cudaDevAttrUnifiedAddressing           // 41: Device shares a unified address space with the host

	// Compute capability
	ComputeCapabilityMajor DeviceAttribute = C.cudaDevAttrComputeCapabilityMajor // 75: Major compute capability version number
	ComputeCapabilityMinor DeviceAttribute = C.cudaDevAttrComputeCapabilityMinor // 76: Minor compute capability version number

	// Stream and cache support (CUDA 7.5+, stable in 10+)
	StreamPrioritiesSupported        DeviceAttribute = C.cudaDevAttrStreamPrioritiesSupported        // 71: Device supports stream priorities
	GlobalL1CacheSupported           DeviceAttribute = C.cudaDevAttrGlobalL1CacheSupported           // 73: Device supports caching globals in L1
	LocalL1CacheSupported            DeviceAttribute = C.cudaDevAttrLocalL1CacheSupported            // 74: Device supports caching locals in L1
	MaxSharedMemoryPerMultiprocessor DeviceAttribute = C.cudaDevAttrMaxSharedMemoryPerMultiprocessor // 77: Maximum shared memory available per multiprocessor in bytes
	MaxRegistersPerMultiprocessor    DeviceAttribute = C.cudaDevAttrMaxRegistersPerMultiprocessor    // 78: Maximum number of 32-bit registers available per multiprocessor

	// Managed memory and host access (CUDA 6+, stable)
	ManagedMemory        DeviceAttribute = C.cudaDevAttrManagedMemory        // 108: Device can allocate managed memory on this system
	PageableMemoryAccess DeviceAttribute = C.cudaDevAttrPageableMemoryAccess // 109: Device supports coherently accessing pageable memory without calling cudaHostRegister

	// Multi-GPU and atomics (CUDA 8+, stable)
	IsMultiGpuBoard           DeviceAttribute = C.cudaDevAttrIsMultiGpuBoard           // 110: Device is on a multi-GPU board
	MultiGpuBoardGroupID      DeviceAttribute = C.cudaDevAttrMultiGpuBoardGroupID      // 111: Unique identifier for a group of devices on the same multi-GPU board
	HostNativeAtomicSupported DeviceAttribute = C.cudaDevAttrHostNativeAtomicSupported // 112: Link between the device and the host supports native atomic operations
)

// CudaLimit represents CUDA limits that can be queried or set. Generated from cudaLimit enum.
// (Kept as-is, stable across versions)
type CudaLimit uint32

const (
	StackSize                    CudaLimit = C.cudaLimitStackSize                    // GPU thread stack size
	PrintfFifoSize               CudaLimit = C.cudaLimitPrintfFifoSize               // GPU printf FIFO size
	MallocHeapSize               CudaLimit = C.cudaLimitMallocHeapSize               // GPU malloc heap size
	DevRuntimeSyncDepth          CudaLimit = C.cudaLimitDevRuntimeSyncDepth          // GPU device runtime synchronize depth
	DevRuntimePendingLaunchCount CudaLimit = C.cudaLimitDevRuntimePendingLaunchCount // GPU device runtime pending launch count
	MaxL2FetchGranularity        CudaLimit = C.cudaLimitMaxL2FetchGranularity        // A value between 0 and 128 that indicates the maximum fetch granularity of L2 (in Bytes). This is a hint
	PersistingL2CacheSize        CudaLimit = C.cudaLimitPersistingL2CacheSize        // A size in bytes for L2 persisting lines cache size
)

// GetDevice returns which device is currently being used.
func GetDevice() (Device, error) {
	var device C.int
	err := Check(C.cudaGetDevice(&device))
	return Device(device), err
}

// SetDevice sets the device to be used for GPU executions.
func SetDevice(device Device) error {
	return Check(C.cudaSetDevice(C.int(device)))
}

// GetDeviceCount returns the number of compute-capable devices.
func GetDeviceCount() (int, error) {
	var count C.int
	err := Check(C.cudaGetDeviceCount(&count))
	return int(count), err
}

// GetDeviceAttribute returns information about the device.
func GetDeviceAttribute(attr DeviceAttribute, device Device) (int, error) {
	var value C.int
	err := Check(C.cudaDeviceGetAttribute(&value, uint32(attr), C.int(device)))
	return int(value), err
}

// DeviceReset destroys all allocations and resets all state on the current device.
func DeviceReset() error {
	return Check(C.cudaDeviceReset())
}

// DeviceSynchronize waits for compute device to finish.
func DeviceSynchronize() error {
	return Check(C.cudaDeviceSynchronize())
}

// SetDeviceFlags sets flags to be used for device executions.
func SetDeviceFlags(flags uint) error {
	return Check(C.cudaSetDeviceFlags(C.uint(flags)))
}

// GetDeviceFlags gets the flags for the current device.
func GetDeviceFlags() (uint, error) {
	var flags C.uint
	err := Check(C.cudaGetDeviceFlags(&flags))
	return uint(flags), err
}

// DeviceGetLimit returns resource limits.
func DeviceGetLimit(limit CudaLimit) (int64, error) {
	var value C.size_t
	err := Check(C.cudaDeviceGetLimit(&value, uint32(limit)))
	return int64(value), err
}

// DeviceSetLimit sets resource limits.
func DeviceSetLimit(limit CudaLimit, value int64) error {
	return Check(C.cudaDeviceSetLimit(uint32(limit), C.size_t(value)))
}

// GetDeviceProperties returns the properties of a compute device.
func GetDeviceProperties(device Device) (*DeviceProperties, error) {
	var prop C.struct_cudaDeviceProp
	var properties DeviceProperties
	err := Check(C.cudaGetDeviceProperties(&prop, C.int(device)))
	if err != nil {
		return nil, err
	}
	properties.fromC(&prop)
	return &properties, nil
}
