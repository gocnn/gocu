package gocu

// #include <cuda.h>
// #include <stdlib.h>
import "C"
import (
	"unsafe"

	"github.com/google/uuid"
)

// Device represents a CUDA device ordinal.
type Device int

// DeviceAttribute represents CUDA device attributes that can be queried. Generated from cuda v12.0 cuda.h.
type DeviceAttribute int

const (
	MaxThreadsPerBlock                     DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK                        // Maximum number of threads per block
	MaxBlockDimX                           DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X                              // Maximum block dimension X
	MaxBlockDimY                           DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y                              // Maximum block dimension Y
	MaxBlockDimZ                           DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z                              // Maximum block dimension Z
	MaxGridDimX                            DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X                               // Maximum grid dimension X
	MaxGridDimY                            DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y                               // Maximum grid dimension Y
	MaxGridDimZ                            DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z                               // Maximum grid dimension Z
	MaxSharedMemoryPerBlock                DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK                  // Maximum shared memory available per block in bytes
	SharedMemoryPerBlock                   DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK                      // Deprecated, use CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK
	TotalConstantMemory                    DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY                        // Memory available on device for __constant__ variables in a CUDA C kernel in bytes
	WarpSize                               DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_WARP_SIZE                                    // Warp size in threads
	MaxPitch                               DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAX_PITCH                                    // Maximum pitch in bytes allowed by memory copies
	MaxRegistersPerBlock                   DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK                      // Maximum number of 32-bit registers available per block
	RegistersPerBlock                      DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK                          // Deprecated, use CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK
	ClockRate                              DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_CLOCK_RATE                                   // Typical clock frequency in kilohertz
	TextureAlignment                       DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT                            // Alignment requirement for textures
	GpuOverlap                             DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_GPU_OVERLAP                                  // Device can possibly copy memory and execute a kernel concurrently. Deprecated. Use instead CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT.
	MultiprocessorCount                    DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT                         // Number of multiprocessors on device
	KernelExecTimeout                      DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT                          // Specifies whether there is a run time limit on kernels
	Integrated                             DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_INTEGRATED                                   // Device is integrated with host memory
	CanMapHostMemory                       DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY                          // Device can map host memory into CUDA address space
	ComputeMode                            DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_COMPUTE_MODE                                 // Compute mode (See ::CUcomputemode for details)
	MaximumTexture1dWidth                  DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH                      // Maximum 1D texture width
	MaximumTexture2dWidth                  DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH                      // Maximum 2D texture width
	MaximumTexture2dHeight                 DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT                     // Maximum 2D texture height
	MaximumTexture3dWidth                  DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH                      // Maximum 3D texture width
	MaximumTexture3dHeight                 DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT                     // Maximum 3D texture height
	MaximumTexture3dDepth                  DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH                      // Maximum 3D texture depth
	MaximumTexture2dLayeredWidth           DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH              // Maximum 2D layered texture width
	MaximumTexture2dLayeredHeight          DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT             // Maximum 2D layered texture height
	MaximumTexture2dLayeredLayers          DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS             // Maximum layers in a 2D layered texture
	MaximumTexture2dArrayWidth             DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH                // Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH
	MaximumTexture2dArrayHeight            DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT               // Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT
	MaximumTexture2dArrayNumslices         DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES            // Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS
	SurfaceAlignment                       DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT                            // Alignment requirement for surfaces
	ConcurrentKernels                      DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS                           // Device can possibly execute multiple kernels concurrently
	EccEnabled                             DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_ECC_ENABLED                                  // Device has ECC support enabled
	PciBusId                               DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_PCI_BUS_ID                                   // PCI bus ID of the device
	PciDeviceId                            DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID                                // PCI device ID of the device
	TccDriver                              DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_TCC_DRIVER                                   // Device is using TCC driver model
	MemoryClockRate                        DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE                            // Peak memory clock frequency in kilohertz
	GlobalMemoryBusWidth                   DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH                      // Global memory bus width in bits
	L2CacheSize                            DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE                                // Size of L2 cache in bytes
	MaxThreadsPerMultiprocessor            DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR               // Maximum resident threads per multiprocessor
	AsyncEngineCount                       DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT                           // Number of asynchronous engines
	UnifiedAddressing                      DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING                           // Device shares a unified address space with the host
	MaximumTexture1dLayeredWidth           DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH              // Maximum 1D layered texture width
	MaximumTexture1dLayeredLayers          DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS             // Maximum layers in a 1D layered texture
	CanTex2dGather                         DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER                             // Deprecated, do not use.
	MaximumTexture2dGatherWidth            DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH               // Maximum 2D texture width if CUDA_ARRAY3D_TEXTURE_GATHER is set
	MaximumTexture2dGatherHeight           DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT              // Maximum 2D texture height if CUDA_ARRAY3D_TEXTURE_GATHER is set
	MaximumTexture3dWidthAlternate         DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE            // Alternate maximum 3D texture width
	MaximumTexture3dHeightAlternate        DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE           // Alternate maximum 3D texture height
	MaximumTexture3dDepthAlternate         DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE            // Alternate maximum 3D texture depth
	PciDomainId                            DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID                                // PCI domain ID of the device
	TexturePitchAlignment                  DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT                      // Pitch alignment requirement for textures
	MaximumTexturecubemapWidth             DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH                 // Maximum cubemap texture width/height
	MaximumTexturecubemapLayeredWidth      DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH         // Maximum cubemap layered texture width/height
	MaximumTexturecubemapLayeredLayers     DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS        // Maximum layers in a cubemap layered texture
	MaximumSurface1dWidth                  DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH                      // Maximum 1D surface width
	MaximumSurface2dWidth                  DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH                      // Maximum 2D surface width
	MaximumSurface2dHeight                 DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT                     // Maximum 2D surface height
	MaximumSurface3dWidth                  DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH                      // Maximum 3D surface width
	MaximumSurface3dHeight                 DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT                     // Maximum 3D surface height
	MaximumSurface3dDepth                  DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH                      // Maximum 3D surface depth
	MaximumSurface1dLayeredWidth           DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH              // Maximum 1D layered surface width
	MaximumSurface1dLayeredLayers          DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS             // Maximum layers in a 1D layered surface
	MaximumSurface2dLayeredWidth           DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH              // Maximum 2D layered surface width
	MaximumSurface2dLayeredHeight          DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT             // Maximum 2D layered surface height
	MaximumSurface2dLayeredLayers          DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS             // Maximum layers in a 2D layered surface
	MaximumSurfacecubemapWidth             DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH                 // Maximum cubemap surface width
	MaximumSurfacecubemapLayeredWidth      DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH         // Maximum cubemap layered surface width
	MaximumSurfacecubemapLayeredLayers     DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS        // Maximum layers in a cubemap layered surface
	MaximumTexture1dLinearWidth            DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH               // Deprecated, do not use. Use cudaDeviceGetTexture1DLinearMaxWidth() or cuDeviceGetTexture1DLinearMaxWidth() instead.
	MaximumTexture2dLinearWidth            DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH               // Maximum 2D linear texture width
	MaximumTexture2dLinearHeight           DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT              // Maximum 2D linear texture height
	MaximumTexture2dLinearPitch            DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH               // Maximum 2D linear texture pitch in bytes
	MaximumTexture2dMipmappedWidth         DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH            // Maximum mipmapped 2D texture width
	MaximumTexture2dMipmappedHeight        DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT           // Maximum mipmapped 2D texture height
	ComputeCapabilityMajor                 DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR                     // Major compute capability version number
	ComputeCapabilityMinor                 DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR                     // Minor compute capability version number
	MaximumTexture1dMipmappedWidth         DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH            // Maximum mipmapped 1D texture width
	StreamPrioritiesSupported              DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED                  // Device supports stream priorities
	GlobalL1CacheSupported                 DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED                    // Device supports caching globals in L1
	LocalL1CacheSupported                  DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED                     // Device supports caching locals in L1
	MaxSharedMemoryPerMultiprocessor       DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR         // Maximum shared memory available per multiprocessor in bytes
	MaxRegistersPerMultiprocessor          DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR             // Maximum number of 32-bit registers available per multiprocessor
	ManagedMemory                          DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY                               // Device can allocate managed memory on this system
	MultiGpuBoard                          DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD                              // Device is on a multi-GPU board
	MultiGpuBoardGroupId                   DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID                     // Unique id for a group of devices on the same multi-GPU board
	HostNativeAtomicSupported              DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED                 // Link between the device and the host supports native atomic operations (this is a placeholder attribute, and is not supported on any current hardware)
	SingleToDoublePrecisionPerfRatio       DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO        // Ratio of single precision performance (in floating-point operations per second) to double precision performance
	PageableMemoryAccess                   DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS                       // Device supports coherently accessing pageable memory without calling cudaHostRegister on it
	ConcurrentManagedAccess                DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS                    // Device can coherently access managed memory concurrently with the CPU
	ComputePreemptionSupported             DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED                 // Device supports compute preemption.
	CanUseHostPointerForRegisteredMem      DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM      // Device can access host registered memory at the same virtual address as the CPU
	CanUseStreamMemOpsV1                   DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1                    // Deprecated, along with v1 MemOps API, ::cuStreamBatchMemOp and related APIs are supported.
	CanUse64BitStreamMemOpsV1              DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS_V1             // Deprecated, along with v1 MemOps API, 64-bit operations are supported in ::cuStreamBatchMemOp and related APIs.
	CanUseStreamWaitValueNorV1             DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V1             // Deprecated, along with v1 MemOps API, ::CU_STREAM_WAIT_VALUE_NOR is supported.
	CooperativeLaunch                      DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH                           // Device supports launching cooperative kernels via ::cuLaunchCooperativeKernel
	CooperativeMultiDeviceLaunch           DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH              // Deprecated, ::cuLaunchCooperativeKernelMultiDevice is deprecated.
	MaxSharedMemoryPerBlockOptin           DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN            // Maximum optin shared memory per block
	CanFlushRemoteWrites                   DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES                      // The ::CU_STREAM_WAIT_VALUE_FLUSH flag and the ::CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES MemOp are supported on the device. See \ref CUDA_MEMOP for additional details.
	HostRegisterSupported                  DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED                      // Device supports host memory registration via ::cudaHostRegister.
	PageableMemoryAccessUsesHostPageTables DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES // Device accesses pageable memory via the host's page tables.
	DirectManagedMemAccessFromHost         DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST          // The host can directly access managed memory on the device without migration.
	VirtualAddressManagementSupported      DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED         // Deprecated, Use CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED
	VirtualMemoryManagementSupported       DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED          // Device supports virtual memory management APIs like ::cuMemAddressReserve, ::cuMemCreate, ::cuMemMap and related APIs
	HandleTypePosixFileDescriptorSupported DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED  // Device supports exporting memory to a posix file descriptor with ::cuMemExportToShareableHandle, if requested via ::cuMemCreate
	HandleTypeWin32HandleSupported         DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED           // Device supports exporting memory to a Win32 NT handle with ::cuMemExportToShareableHandle, if requested via ::cuMemCreate
	HandleTypeWin32KmtHandleSupported      DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED       // Device supports exporting memory to a Win32 KMT handle with ::cuMemExportToShareableHandle, if requested via ::cuMemCreate
	MaxBlocksPerMultiprocessor             DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR                // Maximum number of blocks per multiprocessor
	GenericCompressionSupported            DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED                // Device supports compression of memory
	MaxPersistingL2CacheSize               DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE                 // Maximum L2 persisting lines capacity setting in bytes.
	MaxAccessPolicyWindowSize              DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE                // Maximum value of CUaccessPolicyWindow::num_bytes.
	GpuDirectRdmaWithCudaVmmSupported      DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED      // Device supports specifying the GPUDirect RDMA flag with ::cuMemCreate
	ReservedSharedMemoryPerBlock           DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK             // Shared memory reserved by CUDA driver per block in bytes
	SparseCudaArraySupported               DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED                  // Device supports sparse CUDA arrays and sparse CUDA mipmapped arrays
	ReadOnlyHostRegisterSupported          DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED            // Device supports using the ::cuMemHostRegister flag ::CU_MEMHOSTERGISTER_READ_ONLY to register memory that must be mapped as read-only to the GPU
	TimelineSemaphoreInteropSupported      DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED         // External timeline semaphore interop is supported on the device
	MemoryPoolsSupported                   DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED                       // Device supports using the ::cuMemAllocAsync and ::cuMemPool family of APIs
	GpuDirectRdmaSupported                 DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED                    // Device supports GPUDirect RDMA APIs, like nvidia_p2p_get_pages (see https://docs.nvidia.com/cuda/gpudirect-rdma for more information)
	GpuDirectRdmaFlushWritesOptions        DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS         // The returned attribute shall be interpreted as a bitmask, where the individual bits are described by the ::CUflushGPUDirectRDMAWritesOptions enum
	GpuDirectRdmaWritesOrdering            DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING              // GPUDirect RDMA writes to the device do not need to be flushed for consumers within the scope indicated by the returned attribute. See ::CUGPUDirectRDMAWritesOrdering for the numerical values returned here.
	MempoolSupportedHandleTypes            DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES               // Handle types supported with mempool based IPC
	ClusterLaunch                          DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH                               // Indicates device supports cluster launch
	DeferredMappingCudaArraySupported      DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED        // Device supports deferred mapping CUDA arrays and CUDA mipmapped arrays
	CanUse64BitStreamMemOps                DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS                // 64-bit operations are supported in ::cuStreamBatchMemOp and related MemOp APIs.
	CanUseStreamWaitValueNor               DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR                // ::CU_STREAM_WAIT_VALUE_NOR is supported by MemOp APIs.
	DmaBufSupported                        DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED                            // Device supports buffer sharing with dma_buf mechanism.
	IpcEventSupported                      DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED                          // Device supports IPC Events.
	MemSyncDomainCount                     DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MEM_SYNC_DOMAIN_COUNT                        // Number of memory domains the device supports.
	TensorMapAccessSupported               DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED                  // Device supports accessing memory using Tensor Map.
	UnifiedFunctionPointers                DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS                    // Device supports unified function pointers.                                   // Maximum value of device attributes
)

// DeviceGet retrieves the device with the specified index.
// Returns an error if the index is invalid.
func DeviceGet(idx int) (Device, error) {
	var dev C.CUdevice
	err := Check(C.cuDeviceGet(&dev, C.int(idx)))
	return Device(dev), err
}

// DeviceGetCount returns the number of available CUDA devices.
// Returns an error if the query fails.
func DeviceGetCount() (int, error) {
	var count C.int
	err := Check(C.cuDeviceGetCount(&count))
	return int(count), err
}

// DeviceGetName retrieves the name of the specified device.
// Returns an error if the query fails.
func DeviceGetName(dev Device) (string, error) {
	const size = 256
	buf := make([]byte, size)
	cstr := (*C.char)(unsafe.Pointer(&buf[0]))
	defer C.free(unsafe.Pointer(cstr))
	if err := Check(C.cuDeviceGetName(cstr, C.int(size), C.CUdevice(dev))); err != nil {
		return "", err
	}
	return C.GoString(cstr), nil
}

// Name retrieves the name of the device.
// Returns an error if the query fails.
func (dev Device) Name() (string, error) {
	return DeviceGetName(dev)
}

// DeviceGetUUID retrieves the UUID of the specified device.
// Returns an error if the query fails.
func DeviceGetUUID(dev Device) (uuid.UUID, error) {
	var id uuid.UUID
	err := Check(C.cuDeviceGetUuid((*C.CUuuid)(unsafe.Pointer(&id)), C.CUdevice(dev)))
	return id, err
}

// UUID retrieves the UUID of the device.
// Returns an error if the query fails.
func (dev Device) UUID() (uuid.UUID, error) {
	return DeviceGetUUID(dev)
}

// DeviceGetAttribute retrieves the value of the specified device attribute.
// Returns an error if the query fails.
func DeviceGetAttribute(attr DeviceAttribute, dev Device) (int, error) {
	var val C.int
	err := Check(C.cuDeviceGetAttribute(&val, C.CUdevice_attribute(attr), C.CUdevice(dev)))
	return int(val), err
}

// Attribute retrieves the value of the specified device attribute.
// Returns an error if the query fails.
func (dev Device) Attribute(attr DeviceAttribute) (int, error) {
	return DeviceGetAttribute(attr, dev)
}

// DeviceTotalMem retrieves the total global memory available on the specified device in bytes.
// Returns an error if the query fails.
func DeviceTotalMem(dev Device) (int64, error) {
	var bytes C.size_t
	err := Check(C.cuDeviceTotalMem(&bytes, C.CUdevice(dev)))
	return int64(bytes), err
}

// TotalMem retrieves the total global memory available on the device in bytes.
// Returns an error if the query fails.
func (dev Device) TotalMem() (int64, error) {
	return DeviceTotalMem(dev)
}
