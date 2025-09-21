package gocu

//#include <cuda.h>
import "C"
import (
	"unsafe"

	"github.com/google/uuid"
)

// CUDA Device number.
type Device int

// Returns the device ordinal for the specified device.
func DeviceGet(ordinal int) (device Device, err error) {
	Cordinal := C.int(ordinal)
	var Cdevice C.CUdevice
	err = Result(C.cuDeviceGet(&Cdevice, Cordinal))
	device = Device(Cdevice)
	return
}

// Returns the number of compute devices available.
func DeviceGetCount() (count int, err error) {
	var Ccount C.int
	err = Result(C.cuDeviceGetCount(&Ccount))
	count = int(Ccount)
	return
}

// Gets the name of the device.
func DeviceGetName(dev Device) (name string, err error) {
	size := 256
	buf := make([]byte, 256)
	cstr := C.CString(string(buf))
	defer C.free(unsafe.Pointer(cstr))
	if err := Result(C.cuDeviceGetName(cstr, C.int(size), C.CUdevice(dev))); err != nil {
		return "", err
	}
	return C.GoString(cstr), nil
}

// Gets the name of the device.
func (dev Device) Name() (name string, err error) {
	return DeviceGetName(dev)
}

// Returns the UUID of the device.
func DeviceGetUuid(d Device) (retVal uuid.UUID, err error) {
	ptr := &retVal
	if err = Result(C.cuDeviceGetUuid((*C.CUuuid)(unsafe.Pointer(ptr)), C.CUdevice(d))); err != nil {
		return retVal, err
	}
	return retVal, nil
}

// Returns the UUID of the device.
func (d Device) UUID() (retVal uuid.UUID, err error) {
	return DeviceGetUuid(d)
}

// Returns the compute capability of the device.
//
// Deprecated: This function uses the deprecated cuDeviceComputeCapability CUDA API,
// which was deprecated as of CUDA 5.0. Use DeviceGetAttribute with
// ComputeCapabilityMajor and ComputeCapabilityMinor attributes instead:
//
//	major, err := dev.Attribute(gocu.ComputeCapabilityMajor)
//	minor, err := dev.Attribute(gocu.ComputeCapabilityMinor)
func DeviceComputeCapability(device Device) (major, minor int, err error) {
	major, err = DeviceGetAttribute(ComputeCapabilityMajor, device)
	if err != nil {
		return
	}
	minor, err = DeviceGetAttribute(ComputeCapabilityMinor, device)
	return
}

// Returns the compute capability of the device.
//
// Deprecated: This method uses DeviceComputeCapability which relies on deprecated CUDA API.
// Use Device.Attribute with ComputeCapabilityMajor and ComputeCapabilityMinor instead.
func (device Device) ComputeCapability() (major, minor int, err error) {
	return DeviceComputeCapability(device)
}

// Gets the value of a device attribute.
func DeviceGetAttribute(attrib DeviceAttribute, dev Device) (attr int, err error) {
	var Cattr C.int
	err = Result(C.cuDeviceGetAttribute(&Cattr, C.CUdevice_attribute(attrib), C.CUdevice(dev)))
	attr = int(Cattr)
	return
}

// Gets the value of a device attribute.
func (dev Device) Attribute(attrib DeviceAttribute) (attr int, err error) {
	return DeviceGetAttribute(attrib, dev)
}

// Returns the total amount of global memory available on the device.
func DeviceTotalMem(device Device) (bytes int64, err error) {
	Cdev := C.CUdevice(device)
	var Cbytes C.size_t
	err = Result(C.cuDeviceTotalMem(&Cbytes, Cdev))
	bytes = int64(Cbytes)
	return
}

// Returns the total amount of global memory available on the device.
func (device Device) TotalMem() (bytes int64, err error) {
	return DeviceTotalMem(device)
}

// DeviceAttribute represents the device attributes that the user can query CUDA for.
type DeviceAttribute int

const (
	MaxThreadsPerBlock                 DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK                   // Maximum number of threads per block
	MaxBlockDimX                       DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X                         // Maximum block dimension X
	MaxBlockDimY                       DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y                         // Maximum block dimension Y
	MaxBlockDimZ                       DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z                         // Maximum block dimension Z
	MaxGridDimX                        DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X                          // Maximum grid dimension X
	MaxGridDimY                        DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y                          // Maximum grid dimension Y
	MaxGridDimZ                        DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z                          // Maximum grid dimension Z
	MaxSharedMemoryPerBlock            DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK             // Maximum shared memory available per block in bytes
	SharedMemoryPerBlock               DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK                 // Deprecated, use CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK
	TotalConstantMemory                DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY                   // Memory available on device for __constant__ variables in a CUDA C kernel in bytes
	WarpSize                           DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_WARP_SIZE                               // Warp size in threads
	MaxPitch                           DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAX_PITCH                               // Maximum pitch in bytes allowed by memory copies
	MaxRegistersPerBlock               DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK                 // Maximum number of 32-bit registers available per block
	RegistersPerBlock                  DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK                     // Deprecated, use CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK
	ClockRate                          DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_CLOCK_RATE                              // Typical clock frequency in kilohertz
	TextureAlignment                   DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT                       // Alignment requirement for textures
	GpuOverlap                         DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_GPU_OVERLAP                             // Device can possibly copy memory and execute a kernel concurrently. Deprecated. Use instead CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT.
	MultiprocessorCount                DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT                    // Number of multiprocessors on device
	KernelExecTimeout                  DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT                     // Specifies whether there is a run time limit on kernels
	Integrated                         DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_INTEGRATED                              // Device is integrated with host memory
	CanMapHostMemory                   DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY                     // Device can map host memory into CUDA address space
	ComputeMode                        DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_COMPUTE_MODE                            // Compute mode (See CUcomputemode for details)
	MaximumTexture1dWidth              DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH                 // Maximum 1D texture width
	MaximumTexture2dWidth              DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH                 // Maximum 2D texture width
	MaximumTexture2dHeight             DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT                // Maximum 2D texture height
	MaximumTexture3dWidth              DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH                 // Maximum 3D texture width
	MaximumTexture3dHeight             DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT                // Maximum 3D texture height
	MaximumTexture3dDepth              DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH                 // Maximum 3D texture depth
	MaximumTexture2dLayeredWidth       DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH         // Maximum 2D layered texture width
	MaximumTexture2dLayeredHeight      DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT        // Maximum 2D layered texture height
	MaximumTexture2dLayeredLayers      DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS        // Maximum layers in a 2D layered texture
	MaximumTexture2dArrayWidth         DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH           // Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH
	MaximumTexture2dArrayHeight        DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT          // Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT
	MaximumTexture2dArrayNumslices     DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES       // Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS
	SurfaceAlignment                   DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT                       // Alignment requirement for surfaces
	ConcurrentKernels                  DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS                      // Device can possibly execute multiple kernels concurrently
	EccEnabled                         DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_ECC_ENABLED                             // Device has ECC support enabled
	PciBusID                           DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_PCI_BUS_ID                              // PCI bus ID of the device
	PciDeviceID                        DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID                           // PCI device ID of the device
	TccDriver                          DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_TCC_DRIVER                              // Device is using TCC driver model
	MemoryClockRate                    DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE                       // Peak memory clock frequency in kilohertz
	GlobalMemoryBusWidth               DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH                 // Global memory bus width in bits
	L2CacheSize                        DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE                           // Size of L2 cache in bytes
	MaxThreadsPerMultiprocessor        DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR          // Maximum resident threads per multiprocessor
	AsyncEngineCount                   DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT                      // Number of asynchronous engines
	UnifiedAddressing                  DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING                      // Device shares a unified address space with the host
	MaximumTexture1dLayeredWidth       DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH         // Maximum 1D layered texture width
	MaximumTexture1dLayeredLayers      DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS        // Maximum layers in a 1D layered texture
	CanTex2dGather                     DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER                        // Deprecated, do not use.
	MaximumTexture2dGatherWidth        DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH          // Maximum 2D texture width if CUDA_ARRAY3D_TEXTURE_GATHER is set
	MaximumTexture2dGatherHeight       DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT         // Maximum 2D texture height if CUDA_ARRAY3D_TEXTURE_GATHER is set
	MaximumTexture3dWidthAlternate     DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE       // Alternate maximum 3D texture width
	MaximumTexture3dHeightAlternate    DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE      // Alternate maximum 3D texture height
	MaximumTexture3dDepthAlternate     DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE       // Alternate maximum 3D texture depth
	PciDomainID                        DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID                           // PCI domain ID of the device
	TexturePitchAlignment              DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT                 // Pitch alignment requirement for textures
	MaximumTexturecubemapWidth         DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH            // Maximum cubemap texture width/height
	MaximumTexturecubemapLayeredWidth  DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH    // Maximum cubemap layered texture width/height
	MaximumTexturecubemapLayeredLayers DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS   // Maximum layers in a cubemap layered texture
	MaximumSurface1dWidth              DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH                 // Maximum 1D surface width
	MaximumSurface2dWidth              DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH                 // Maximum 2D surface width
	MaximumSurface2dHeight             DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT                // Maximum 2D surface height
	MaximumSurface3dWidth              DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH                 // Maximum 3D surface width
	MaximumSurface3dHeight             DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT                // Maximum 3D surface height
	MaximumSurface3dDepth              DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH                 // Maximum 3D surface depth
	MaximumSurface1dLayeredWidth       DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH         // Maximum 1D layered surface width
	MaximumSurface1dLayeredLayers      DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS        // Maximum layers in a 1D layered surface
	MaximumSurface2dLayeredWidth       DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH         // Maximum 2D layered surface width
	MaximumSurface2dLayeredHeight      DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT        // Maximum 2D layered surface height
	MaximumSurface2dLayeredLayers      DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS        // Maximum layers in a 2D layered surface
	MaximumSurfacecubemapWidth         DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH            // Maximum cubemap surface width
	MaximumSurfacecubemapLayeredWidth  DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH    // Maximum cubemap layered surface width
	MaximumSurfacecubemapLayeredLayers DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS   // Maximum layers in a cubemap layered surface
	MaximumTexture1dLinearWidth        DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH          // Maximum 1D linear texture width
	MaximumTexture2dLinearWidth        DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH          // Maximum 2D linear texture width
	MaximumTexture2dLinearHeight       DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT         // Maximum 2D linear texture height
	MaximumTexture2dLinearPitch        DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH          // Maximum 2D linear texture pitch in bytes
	MaximumTexture2dMipmappedWidth     DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH       // Maximum mipmapped 2D texture width
	MaximumTexture2dMipmappedHeight    DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT      // Maximum mipmapped 2D texture height
	ComputeCapabilityMajor             DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR                // Major compute capability version number
	ComputeCapabilityMinor             DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR                // Minor compute capability version number
	MaximumTexture1dMipmappedWidth     DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH       // Maximum mipmapped 1D texture width
	StreamPrioritiesSupported          DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED             // Device supports stream priorities
	GlobalL1CacheSupported             DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED               // Device supports caching globals in L1
	LocalL1CacheSupported              DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED                // Device supports caching locals in L1
	MaxSharedMemoryPerMultiprocessor   DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR    // Maximum shared memory available per multiprocessor in bytes
	MaxRegistersPerMultiprocessor      DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR        // Maximum number of 32-bit registers available per multiprocessor
	ManagedMemory                      DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY                          // Device can allocate managed memory on this system
	MultiGpuBoard                      DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD                         // Device is on a multi-GPU board
	MultiGpuBoardGroupID               DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID                // Unique id for a group of devices on the same multi-GPU board
	HostNativeAtomicSupported          DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED            // Link between the device and the host supports native atomic operations (this is a placeholder attribute, and is not supported on any current hardware)
	SingleToDoublePrecisionPerfRatio   DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO   // Ratio of single precision performance (in floating-point operations per second) to double precision performance
	PageableMemoryAccess               DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS                  // Device supports coherently accessing pageable memory without calling cudaHostRegister on it
	ConcurrentManagedAccess            DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS               // Device can coherently access managed memory concurrently with the CPU
	ComputePreemptionSupported         DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED            // Device supports compute preemption.
	CanUseHostPointerForRegisteredMem  DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM // Device can access host registered memory at the same virtual address as the CPU

)
