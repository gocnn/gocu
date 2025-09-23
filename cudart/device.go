package cudart

// #include <cuda_runtime.h>
import "C"
import (
	"unsafe"

	"github.com/google/uuid"
)

// Device represents a CUDA device ordinal.
type Device int

// DeviceAttribute represents CUDA device attributes that can be queried. Generated from cudaDeviceAttr enum.
type DeviceAttribute uint32

const (
	MaxThreadsPerBlock                     DeviceAttribute = C.cudaDevAttrMaxThreadsPerBlock                     // Maximum number of threads per block
	MaxBlockDimX                           DeviceAttribute = C.cudaDevAttrMaxBlockDimX                           // Maximum block dimension X
	MaxBlockDimY                           DeviceAttribute = C.cudaDevAttrMaxBlockDimY                           // Maximum block dimension Y
	MaxBlockDimZ                           DeviceAttribute = C.cudaDevAttrMaxBlockDimZ                           // Maximum block dimension Z
	MaxGridDimX                            DeviceAttribute = C.cudaDevAttrMaxGridDimX                            // Maximum grid dimension X
	MaxGridDimY                            DeviceAttribute = C.cudaDevAttrMaxGridDimY                            // Maximum grid dimension Y
	MaxGridDimZ                            DeviceAttribute = C.cudaDevAttrMaxGridDimZ                            // Maximum grid dimension Z
	MaxSharedMemoryPerBlock                DeviceAttribute = C.cudaDevAttrMaxSharedMemoryPerBlock                // Maximum shared memory available per block in bytes
	TotalConstantMemory                    DeviceAttribute = C.cudaDevAttrTotalConstantMemory                    // Memory available on device for __constant__ variables in a CUDA C kernel in bytes
	WarpSize                               DeviceAttribute = C.cudaDevAttrWarpSize                               // Warp size in threads
	MaxPitch                               DeviceAttribute = C.cudaDevAttrMaxPitch                               // Maximum pitch in bytes allowed by memory copies
	MaxRegistersPerBlock                   DeviceAttribute = C.cudaDevAttrMaxRegistersPerBlock                   // Maximum number of 32-bit registers available per block
	ClockRate                              DeviceAttribute = C.cudaDevAttrClockRate                              // Peak clock frequency in kilohertz
	TextureAlignment                       DeviceAttribute = C.cudaDevAttrTextureAlignment                       // Alignment requirement for textures
	GpuOverlap                             DeviceAttribute = C.cudaDevAttrGpuOverlap                             // Device can possibly copy memory and execute a kernel concurrently
	MultiProcessorCount                    DeviceAttribute = C.cudaDevAttrMultiProcessorCount                    // Number of multiprocessors on device
	KernelExecTimeout                      DeviceAttribute = C.cudaDevAttrKernelExecTimeout                      // Specifies whether there is a run time limit on kernels
	Integrated                             DeviceAttribute = C.cudaDevAttrIntegrated                             // Device is integrated with host memory
	CanMapHostMemory                       DeviceAttribute = C.cudaDevAttrCanMapHostMemory                       // Device can map host memory into CUDA address space
	ComputeMode                            DeviceAttribute = C.cudaDevAttrComputeMode                            // Compute mode (See cudaComputeMode for details)
	MaxTexture1DWidth                      DeviceAttribute = C.cudaDevAttrMaxTexture1DWidth                      // Maximum 1D texture width
	MaxTexture2DWidth                      DeviceAttribute = C.cudaDevAttrMaxTexture2DWidth                      // Maximum 2D texture width
	MaxTexture2DHeight                     DeviceAttribute = C.cudaDevAttrMaxTexture2DHeight                     // Maximum 2D texture height
	MaxTexture3DWidth                      DeviceAttribute = C.cudaDevAttrMaxTexture3DWidth                      // Maximum 3D texture width
	MaxTexture3DHeight                     DeviceAttribute = C.cudaDevAttrMaxTexture3DHeight                     // Maximum 3D texture height
	MaxTexture3DDepth                      DeviceAttribute = C.cudaDevAttrMaxTexture3DDepth                      // Maximum 3D texture depth
	MaxTexture2DLayeredWidth               DeviceAttribute = C.cudaDevAttrMaxTexture2DLayeredWidth               // Maximum 2D layered texture width
	MaxTexture2DLayeredHeight              DeviceAttribute = C.cudaDevAttrMaxTexture2DLayeredHeight              // Maximum 2D layered texture height
	MaxTexture2DLayeredLayers              DeviceAttribute = C.cudaDevAttrMaxTexture2DLayeredLayers              // Maximum layers in a 2D layered texture
	SurfaceAlignment                       DeviceAttribute = C.cudaDevAttrSurfaceAlignment                       // Alignment requirement for surfaces
	ConcurrentKernels                      DeviceAttribute = C.cudaDevAttrConcurrentKernels                      // Device can possibly execute multiple kernels concurrently
	EccEnabled                             DeviceAttribute = C.cudaDevAttrEccEnabled                             // Device has ECC support enabled
	PciBusId                               DeviceAttribute = C.cudaDevAttrPciBusId                               // PCI bus ID of the device
	PciDeviceId                            DeviceAttribute = C.cudaDevAttrPciDeviceId                            // PCI device ID of the device
	TccDriver                              DeviceAttribute = C.cudaDevAttrTccDriver                              // Device is using TCC driver model
	MemoryClockRate                        DeviceAttribute = C.cudaDevAttrMemoryClockRate                        // Peak memory clock frequency in kilohertz
	GlobalMemoryBusWidth                   DeviceAttribute = C.cudaDevAttrGlobalMemoryBusWidth                   // Global memory bus width in bits
	L2CacheSize                            DeviceAttribute = C.cudaDevAttrL2CacheSize                            // Size of L2 cache in bytes
	MaxThreadsPerMultiProcessor            DeviceAttribute = C.cudaDevAttrMaxThreadsPerMultiProcessor            // Maximum resident threads per multiprocessor
	AsyncEngineCount                       DeviceAttribute = C.cudaDevAttrAsyncEngineCount                       // Number of asynchronous engines
	UnifiedAddressing                      DeviceAttribute = C.cudaDevAttrUnifiedAddressing                      // Device shares a unified address space with the host
	MaxTexture1DLayeredWidth               DeviceAttribute = C.cudaDevAttrMaxTexture1DLayeredWidth               // Maximum 1D layered texture width
	MaxTexture1DLayeredLayers              DeviceAttribute = C.cudaDevAttrMaxTexture1DLayeredLayers              // Maximum layers in a 1D layered texture
	MaxTexture2DGatherWidth                DeviceAttribute = C.cudaDevAttrMaxTexture2DGatherWidth                // Maximum 2D texture width if cudaArrayTextureGather is set
	MaxTexture2DGatherHeight               DeviceAttribute = C.cudaDevAttrMaxTexture2DGatherHeight               // Maximum 2D texture height if cudaArrayTextureGather is set
	MaxTexture3DWidthAlt                   DeviceAttribute = C.cudaDevAttrMaxTexture3DWidthAlt                   // Alternate maximum 3D texture width
	MaxTexture3DHeightAlt                  DeviceAttribute = C.cudaDevAttrMaxTexture3DHeightAlt                  // Alternate maximum 3D texture height
	MaxTexture3DDepthAlt                   DeviceAttribute = C.cudaDevAttrMaxTexture3DDepthAlt                   // Alternate maximum 3D texture depth
	PciDomainId                            DeviceAttribute = C.cudaDevAttrPciDomainId                            // PCI domain ID of the device
	TexturePitchAlignment                  DeviceAttribute = C.cudaDevAttrTexturePitchAlignment                  // Pitch alignment requirement for textures
	MaxTextureCubemapWidth                 DeviceAttribute = C.cudaDevAttrMaxTextureCubemapWidth                 // Maximum cubemap texture width/height
	MaxTextureCubemapLayeredWidth          DeviceAttribute = C.cudaDevAttrMaxTextureCubemapLayeredWidth          // Maximum cubemap layered texture width/height
	MaxTextureCubemapLayeredLayers         DeviceAttribute = C.cudaDevAttrMaxTextureCubemapLayeredLayers         // Maximum layers in a cubemap layered texture
	MaxSurface1DWidth                      DeviceAttribute = C.cudaDevAttrMaxSurface1DWidth                      // Maximum 1D surface width
	MaxSurface2DWidth                      DeviceAttribute = C.cudaDevAttrMaxSurface2DWidth                      // Maximum 2D surface width
	MaxSurface2DHeight                     DeviceAttribute = C.cudaDevAttrMaxSurface2DHeight                     // Maximum 2D surface height
	MaxSurface3DWidth                      DeviceAttribute = C.cudaDevAttrMaxSurface3DWidth                      // Maximum 3D surface width
	MaxSurface3DHeight                     DeviceAttribute = C.cudaDevAttrMaxSurface3DHeight                     // Maximum 3D surface height
	MaxSurface3DDepth                      DeviceAttribute = C.cudaDevAttrMaxSurface3DDepth                      // Maximum 3D surface depth
	MaxSurface1DLayeredWidth               DeviceAttribute = C.cudaDevAttrMaxSurface1DLayeredWidth               // Maximum 1D layered surface width
	MaxSurface1DLayeredLayers              DeviceAttribute = C.cudaDevAttrMaxSurface1DLayeredLayers              // Maximum layers in a 1D layered surface
	MaxSurface2DLayeredWidth               DeviceAttribute = C.cudaDevAttrMaxSurface2DLayeredWidth               // Maximum 2D layered surface width
	MaxSurface2DLayeredHeight              DeviceAttribute = C.cudaDevAttrMaxSurface2DLayeredHeight              // Maximum 2D layered surface height
	MaxSurface2DLayeredLayers              DeviceAttribute = C.cudaDevAttrMaxSurface2DLayeredLayers              // Maximum layers in a 2D layered surface
	MaxSurfaceCubemapWidth                 DeviceAttribute = C.cudaDevAttrMaxSurfaceCubemapWidth                 // Maximum cubemap surface width
	MaxSurfaceCubemapLayeredWidth          DeviceAttribute = C.cudaDevAttrMaxSurfaceCubemapLayeredWidth          // Maximum cubemap layered surface width
	MaxSurfaceCubemapLayeredLayers         DeviceAttribute = C.cudaDevAttrMaxSurfaceCubemapLayeredLayers         // Maximum layers in a cubemap layered surface
	MaxTexture1DLinearWidth                DeviceAttribute = C.cudaDevAttrMaxTexture1DLinearWidth                // Maximum 1D linear texture width
	MaxTexture2DLinearWidth                DeviceAttribute = C.cudaDevAttrMaxTexture2DLinearWidth                // Maximum 2D linear texture width
	MaxTexture2DLinearHeight               DeviceAttribute = C.cudaDevAttrMaxTexture2DLinearHeight               // Maximum 2D linear texture height
	MaxTexture2DLinearPitch                DeviceAttribute = C.cudaDevAttrMaxTexture2DLinearPitch                // Maximum 2D linear texture pitch in bytes
	MaxTexture2DMipmappedWidth             DeviceAttribute = C.cudaDevAttrMaxTexture2DMipmappedWidth             // Maximum mipmapped 2D texture width
	MaxTexture2DMipmappedHeight            DeviceAttribute = C.cudaDevAttrMaxTexture2DMipmappedHeight            // Maximum mipmapped 2D texture height
	ComputeCapabilityMajor                 DeviceAttribute = C.cudaDevAttrComputeCapabilityMajor                 // Major compute capability version number
	ComputeCapabilityMinor                 DeviceAttribute = C.cudaDevAttrComputeCapabilityMinor                 // Minor compute capability version number
	MaxTexture1DMipmappedWidth             DeviceAttribute = C.cudaDevAttrMaxTexture1DMipmappedWidth             // Maximum mipmapped 1D texture width
	StreamPrioritiesSupported              DeviceAttribute = C.cudaDevAttrStreamPrioritiesSupported              // Device supports stream priorities
	GlobalL1CacheSupported                 DeviceAttribute = C.cudaDevAttrGlobalL1CacheSupported                 // Device supports caching globals in L1
	LocalL1CacheSupported                  DeviceAttribute = C.cudaDevAttrLocalL1CacheSupported                  // Device supports caching locals in L1
	MaxSharedMemoryPerMultiprocessor       DeviceAttribute = C.cudaDevAttrMaxSharedMemoryPerMultiprocessor       // Maximum shared memory available per multiprocessor in bytes
	MaxRegistersPerMultiprocessor          DeviceAttribute = C.cudaDevAttrMaxRegistersPerMultiprocessor          // Maximum number of 32-bit registers available per multiprocessor
	ManagedMemory                          DeviceAttribute = C.cudaDevAttrManagedMemory                          // Device can allocate managed memory on this system
	IsMultiGpuBoard                        DeviceAttribute = C.cudaDevAttrIsMultiGpuBoard                        // Device is on a multi-GPU board
	MultiGpuBoardGroupID                   DeviceAttribute = C.cudaDevAttrMultiGpuBoardGroupID                   // Unique identifier for a group of devices on the same multi-GPU board
	HostNativeAtomicSupported              DeviceAttribute = C.cudaDevAttrHostNativeAtomicSupported              // Link between the device and the host supports native atomic operations
	SingleToDoublePrecisionPerfRatio       DeviceAttribute = C.cudaDevAttrSingleToDoublePrecisionPerfRatio       // Ratio of single precision performance to double precision performance
	PageableMemoryAccess                   DeviceAttribute = C.cudaDevAttrPageableMemoryAccess                   // Device supports coherently accessing pageable memory without calling cudaHostRegister
	ConcurrentManagedAccess                DeviceAttribute = C.cudaDevAttrConcurrentManagedAccess                // Device can coherently access managed memory concurrently with the CPU
	ComputePreemptionSupported             DeviceAttribute = C.cudaDevAttrComputePreemptionSupported             // Device supports Compute Preemption
	CanUseHostPointerForRegisteredMem      DeviceAttribute = C.cudaDevAttrCanUseHostPointerForRegisteredMem      // Device can access host registered memory at the same virtual address as the CPU
	CooperativeLaunch                      DeviceAttribute = C.cudaDevAttrCooperativeLaunch                      // Device supports launching cooperative kernels via cudaLaunchCooperativeKernel
	MaxSharedMemoryPerBlockOptin           DeviceAttribute = C.cudaDevAttrMaxSharedMemoryPerBlockOptin           // The maximum optin shared memory per block
	CanFlushRemoteWrites                   DeviceAttribute = C.cudaDevAttrCanFlushRemoteWrites                   // Device supports flushing of outstanding remote writes
	HostRegisterSupported                  DeviceAttribute = C.cudaDevAttrHostRegisterSupported                  // Device supports host memory registration via cudaHostRegister
	PageableMemoryAccessUsesHostPageTables DeviceAttribute = C.cudaDevAttrPageableMemoryAccessUsesHostPageTables // Device accesses pageable memory via the host's page tables
	DirectManagedMemAccessFromHost         DeviceAttribute = C.cudaDevAttrDirectManagedMemAccessFromHost         // Host can directly access managed memory on the device without migration
	MaxBlocksPerMultiprocessor             DeviceAttribute = C.cudaDevAttrMaxBlocksPerMultiprocessor             // Maximum number of blocks per multiprocessor
	MaxPersistingL2CacheSize               DeviceAttribute = C.cudaDevAttrMaxPersistingL2CacheSize               // Maximum L2 persisting lines capacity setting in bytes
	MaxAccessPolicyWindowSize              DeviceAttribute = C.cudaDevAttrMaxAccessPolicyWindowSize              // Maximum value of cudaAccessPolicyWindow::num_bytes
	ReservedSharedMemoryPerBlock           DeviceAttribute = C.cudaDevAttrReservedSharedMemoryPerBlock           // Shared memory reserved by CUDA driver per block in bytes
	SparseCudaArraySupported               DeviceAttribute = C.cudaDevAttrSparseCudaArraySupported               // Device supports sparse CUDA arrays and sparse CUDA mipmapped arrays
	HostRegisterReadOnlySupported          DeviceAttribute = C.cudaDevAttrHostRegisterReadOnlySupported          // Device supports using the cudaHostRegister flag cudaHostRegisterReadOnly
	TimelineSemaphoreInteropSupported      DeviceAttribute = C.cudaDevAttrTimelineSemaphoreInteropSupported      // External timeline semaphore interop is supported on the device
	MemoryPoolsSupported                   DeviceAttribute = C.cudaDevAttrMemoryPoolsSupported                   // Device supports using the cudaMallocAsync and cudaMemPool family of APIs
	GPUDirectRDMASupported                 DeviceAttribute = C.cudaDevAttrGPUDirectRDMASupported                 // Device supports GPUDirect RDMA APIs
	GPUDirectRDMAFlushWritesOptions        DeviceAttribute = C.cudaDevAttrGPUDirectRDMAFlushWritesOptions        // The returned attribute shall be interpreted as a bitmask
	GPUDirectRDMAWritesOrdering            DeviceAttribute = C.cudaDevAttrGPUDirectRDMAWritesOrdering            // GPUDirect RDMA writes to the device do not need to be flushed
	MemoryPoolSupportedHandleTypes         DeviceAttribute = C.cudaDevAttrMemoryPoolSupportedHandleTypes         // Handle types supported with mempool based IPC
	ClusterLaunch                          DeviceAttribute = C.cudaDevAttrClusterLaunch                          // Indicates device supports cluster launch
	DeferredMappingCudaArraySupported      DeviceAttribute = C.cudaDevAttrDeferredMappingCudaArraySupported      // Device supports deferred mapping CUDA arrays and CUDA mipmapped arrays
	// IpcEventSupport                        DeviceAttribute = C.cudaDevAttrIpcEventSupport                        // Device supports IPC Events
	// MemSyncDomainCount                     DeviceAttribute = C.cudaDevAttrMemSyncDomainCount                     // Number of memory synchronization domains the device supports
)

// CudaLimit represents CUDA limits that can be queried or set. Generated from cudaLimit enum.
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

// DeviceProperties represents CUDA device properties from cudaDeviceProp.
type DeviceProperties struct {
	Name                                   string
	UUID                                   uuid.UUID
	LUID                                   [8]byte
	LUIDDeviceNodeMask                     uint32
	TotalGlobalMem                         int64
	SharedMemPerBlock                      int64
	RegsPerBlock                           int32
	WarpSize                               int32
	MemPitch                               int64
	MaxThreadsPerBlock                     int32
	MaxThreadsDim                          [3]int32
	MaxGridSize                            [3]int32
	ClockRate                              int32
	TotalConstMem                          int64
	Major                                  int32
	Minor                                  int32
	TextureAlignment                       int64
	TexturePitchAlignment                  int64
	DeviceOverlap                          int32
	MultiProcessorCount                    int32
	KernelExecTimeoutEnabled               int32
	Integrated                             int32
	CanMapHostMemory                       int32
	ComputeMode                            int32
	MaxTexture1D                           int32
	MaxTexture1DMipmap                     int32
	MaxTexture1DLinear                     int32
	MaxTexture2D                           [2]int32
	MaxTexture2DMipmap                     [2]int32
	MaxTexture2DLinear                     [3]int32
	MaxTexture2DGather                     [2]int32
	MaxTexture3D                           [3]int32
	MaxTexture3DAlt                        [3]int32
	MaxTextureCubemap                      int32
	MaxTexture1DLayered                    [2]int32
	MaxTexture2DLayered                    [3]int32
	MaxTextureCubemapLayered               [2]int32
	MaxSurface1D                           int32
	MaxSurface2D                           [2]int32
	MaxSurface3D                           [3]int32
	MaxSurface1DLayered                    [2]int32
	MaxSurface2DLayered                    [3]int32
	MaxSurfaceCubemap                      int32
	MaxSurfaceCubemapLayered               [2]int32
	SurfaceAlignment                       int64
	ConcurrentKernels                      int32
	ECCEnabled                             int32
	PCIBusID                               int32
	PCIDeviceID                            int32
	PCIDomainID                            int32
	TccDriver                              int32
	AsyncEngineCount                       int32
	UnifiedAddressing                      int32
	MemoryClockRate                        int32
	MemoryBusWidth                         int32
	L2CacheSize                            int32
	PersistingL2CacheMaxSize               int32
	MaxThreadsPerMultiProcessor            int32
	StreamPrioritiesSupported              int32
	GlobalL1CacheSupported                 int32
	LocalL1CacheSupported                  int32
	SharedMemPerMultiprocessor             int64
	RegsPerMultiprocessor                  int32
	ManagedMemory                          int32
	IsMultiGpuBoard                        int32
	MultiGpuBoardGroupID                   int32
	HostNativeAtomicSupported              int32
	SingleToDoublePrecisionPerfRatio       int32
	PageableMemoryAccess                   int32
	ConcurrentManagedAccess                int32
	ComputePreemptionSupported             int32
	CanUseHostPointerForRegisteredMem      int32
	CooperativeLaunch                      int32
	CooperativeMultiDeviceLaunch           int32
	SharedMemPerBlockOptin                 int64
	PageableMemoryAccessUsesHostPageTables int32
	DirectManagedMemAccessFromHost         int32
	MaxBlocksPerMultiProcessor             int32
	AccessPolicyMaxWindowSize              int32
	ReservedSharedMemPerBlock              int64
}

// GetDeviceProperties returns the properties of a compute device.
func GetDeviceProperties(device Device) (*DeviceProperties, error) {
	var prop C.struct_cudaDeviceProp
	err := Check(C.cudaGetDeviceProperties(&prop, C.int(device)))
	if err != nil {
		return nil, err
	}

	// Convert C struct to Go struct
	properties := &DeviceProperties{
		Name:                                   C.GoString(&prop.name[0]),
		TotalGlobalMem:                         int64(prop.totalGlobalMem),
		SharedMemPerBlock:                      int64(prop.sharedMemPerBlock),
		RegsPerBlock:                           int32(prop.regsPerBlock),
		WarpSize:                               int32(prop.warpSize),
		MemPitch:                               int64(prop.memPitch),
		MaxThreadsPerBlock:                     int32(prop.maxThreadsPerBlock),
		ClockRate:                              int32(prop.clockRate),
		TotalConstMem:                          int64(prop.totalConstMem),
		Major:                                  int32(prop.major),
		Minor:                                  int32(prop.minor),
		TextureAlignment:                       int64(prop.textureAlignment),
		TexturePitchAlignment:                  int64(prop.texturePitchAlignment),
		DeviceOverlap:                          int32(prop.deviceOverlap),
		MultiProcessorCount:                    int32(prop.multiProcessorCount),
		KernelExecTimeoutEnabled:               int32(prop.kernelExecTimeoutEnabled),
		Integrated:                             int32(prop.integrated),
		CanMapHostMemory:                       int32(prop.canMapHostMemory),
		ComputeMode:                            int32(prop.computeMode),
		MaxTexture1D:                           int32(prop.maxTexture1D),
		MaxTexture1DMipmap:                     int32(prop.maxTexture1DMipmap),
		MaxTexture1DLinear:                     int32(prop.maxTexture1DLinear),
		MaxTextureCubemap:                      int32(prop.maxTextureCubemap),
		SurfaceAlignment:                       int64(prop.surfaceAlignment),
		ConcurrentKernels:                      int32(prop.concurrentKernels),
		ECCEnabled:                             int32(prop.ECCEnabled),
		PCIBusID:                               int32(prop.pciBusID),
		PCIDeviceID:                            int32(prop.pciDeviceID),
		PCIDomainID:                            int32(prop.pciDomainID),
		TccDriver:                              int32(prop.tccDriver),
		AsyncEngineCount:                       int32(prop.asyncEngineCount),
		UnifiedAddressing:                      int32(prop.unifiedAddressing),
		MemoryClockRate:                        int32(prop.memoryClockRate),
		MemoryBusWidth:                         int32(prop.memoryBusWidth),
		L2CacheSize:                            int32(prop.l2CacheSize),
		PersistingL2CacheMaxSize:               int32(prop.persistingL2CacheMaxSize),
		MaxThreadsPerMultiProcessor:            int32(prop.maxThreadsPerMultiProcessor),
		StreamPrioritiesSupported:              int32(prop.streamPrioritiesSupported),
		GlobalL1CacheSupported:                 int32(prop.globalL1CacheSupported),
		LocalL1CacheSupported:                  int32(prop.localL1CacheSupported),
		SharedMemPerMultiprocessor:             int64(prop.sharedMemPerMultiprocessor),
		RegsPerMultiprocessor:                  int32(prop.regsPerMultiprocessor),
		ManagedMemory:                          int32(prop.managedMemory),
		IsMultiGpuBoard:                        int32(prop.isMultiGpuBoard),
		MultiGpuBoardGroupID:                   int32(prop.multiGpuBoardGroupID),
		HostNativeAtomicSupported:              int32(prop.hostNativeAtomicSupported),
		SingleToDoublePrecisionPerfRatio:       int32(prop.singleToDoublePrecisionPerfRatio),
		PageableMemoryAccess:                   int32(prop.pageableMemoryAccess),
		ConcurrentManagedAccess:                int32(prop.concurrentManagedAccess),
		ComputePreemptionSupported:             int32(prop.computePreemptionSupported),
		CanUseHostPointerForRegisteredMem:      int32(prop.canUseHostPointerForRegisteredMem),
		CooperativeLaunch:                      int32(prop.cooperativeLaunch),
		CooperativeMultiDeviceLaunch:           int32(prop.cooperativeMultiDeviceLaunch),
		SharedMemPerBlockOptin:                 int64(prop.sharedMemPerBlockOptin),
		PageableMemoryAccessUsesHostPageTables: int32(prop.pageableMemoryAccessUsesHostPageTables),
		DirectManagedMemAccessFromHost:         int32(prop.directManagedMemAccessFromHost),
		MaxBlocksPerMultiProcessor:             int32(prop.maxBlocksPerMultiProcessor),
		AccessPolicyMaxWindowSize:              int32(prop.accessPolicyMaxWindowSize),
		ReservedSharedMemPerBlock:              int64(prop.reservedSharedMemPerBlock),
		LUIDDeviceNodeMask:                     uint32(prop.luidDeviceNodeMask),
	}

	// Copy UUID and LUID
	copy(properties.UUID[:], C.GoBytes(unsafe.Pointer(&prop.uuid), 16))
	copy(properties.LUID[:], C.GoBytes(unsafe.Pointer(&prop.luid), 8))

	// Copy array fields
	for i := 0; i < 3; i++ {
		properties.MaxThreadsDim[i] = int32(prop.maxThreadsDim[i])
		properties.MaxGridSize[i] = int32(prop.maxGridSize[i])
		if i < 2 {
			properties.MaxTexture2D[i] = int32(prop.maxTexture2D[i])
			properties.MaxTexture2DMipmap[i] = int32(prop.maxTexture2DMipmap[i])
			properties.MaxTexture2DGather[i] = int32(prop.maxTexture2DGather[i])
			properties.MaxTexture1DLayered[i] = int32(prop.maxTexture1DLayered[i])
			properties.MaxTextureCubemapLayered[i] = int32(prop.maxTextureCubemapLayered[i])
			properties.MaxSurface2D[i] = int32(prop.maxSurface2D[i])
			properties.MaxSurface1DLayered[i] = int32(prop.maxSurface1DLayered[i])
			properties.MaxSurfaceCubemapLayered[i] = int32(prop.maxSurfaceCubemapLayered[i])
		}
		if i < 3 {
			properties.MaxTexture2DLinear[i] = int32(prop.maxTexture2DLinear[i])
			properties.MaxTexture3D[i] = int32(prop.maxTexture3D[i])
			properties.MaxTexture3DAlt[i] = int32(prop.maxTexture3DAlt[i])
			properties.MaxTexture2DLayered[i] = int32(prop.maxTexture2DLayered[i])
			properties.MaxSurface3D[i] = int32(prop.maxSurface3D[i])
			properties.MaxSurface2DLayered[i] = int32(prop.maxSurface2DLayered[i])
		}
	}

	return properties, nil
}
