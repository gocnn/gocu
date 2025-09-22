package cudart

//#include <cuda_runtime.h>
import "C"
import "fmt"

type cudaError int

func (err cudaError) Error() string { return err.String() }
func (err cudaError) String() string {
	if msg, ok := resString[err]; ok {
		return msg
	}
	return fmt.Sprintf("UnknownErrorCode:%d", err)
}

func Result(x C.cudaError_t) error {
	err := cudaError(x)
	if err == Success {
		return nil
	}
	return err
}

const (
	Success                        cudaError = C.cudaSuccess
	InvalidValue                   cudaError = C.cudaErrorInvalidValue
	MemoryAllocation               cudaError = C.cudaErrorMemoryAllocation
	InitializationError            cudaError = C.cudaErrorInitializationError
	CudartUnloading                cudaError = C.cudaErrorCudartUnloading
	ProfilerDisabled               cudaError = C.cudaErrorProfilerDisabled
	ProfilerNotInitialized         cudaError = C.cudaErrorProfilerNotInitialized
	ProfilerAlreadyStarted         cudaError = C.cudaErrorProfilerAlreadyStarted
	ProfilerAlreadyStopped         cudaError = C.cudaErrorProfilerAlreadyStopped
	InvalidConfiguration           cudaError = C.cudaErrorInvalidConfiguration
	InvalidPitchValue              cudaError = C.cudaErrorInvalidPitchValue
	InvalidSymbol                  cudaError = C.cudaErrorInvalidSymbol
	InvalidHostPointer             cudaError = C.cudaErrorInvalidHostPointer
	InvalidDevicePointer           cudaError = C.cudaErrorInvalidDevicePointer
	InvalidTexture                 cudaError = C.cudaErrorInvalidTexture
	InvalidTextureBinding          cudaError = C.cudaErrorInvalidTextureBinding
	InvalidChannelDescriptor       cudaError = C.cudaErrorInvalidChannelDescriptor
	InvalidMemcpyDirection         cudaError = C.cudaErrorInvalidMemcpyDirection
	AddressOfConstant              cudaError = C.cudaErrorAddressOfConstant
	TextureFetchFailed             cudaError = C.cudaErrorTextureFetchFailed
	TextureNotBound                cudaError = C.cudaErrorTextureNotBound
	SynchronizationError           cudaError = C.cudaErrorSynchronizationError
	InvalidFilterSetting           cudaError = C.cudaErrorInvalidFilterSetting
	InvalidNormSetting             cudaError = C.cudaErrorInvalidNormSetting
	MixedDeviceExecution           cudaError = C.cudaErrorMixedDeviceExecution
	NotYetImplemented              cudaError = C.cudaErrorNotYetImplemented
	MemoryValueTooLarge            cudaError = C.cudaErrorMemoryValueTooLarge
	StubLibrary                    cudaError = C.cudaErrorStubLibrary
	InsufficientDriver             cudaError = C.cudaErrorInsufficientDriver
	CallRequiresNewerDriver        cudaError = C.cudaErrorCallRequiresNewerDriver
	InvalidSurface                 cudaError = C.cudaErrorInvalidSurface
	DuplicateVariableName          cudaError = C.cudaErrorDuplicateVariableName
	DuplicateTextureName           cudaError = C.cudaErrorDuplicateTextureName
	DuplicateSurfaceName           cudaError = C.cudaErrorDuplicateSurfaceName
	DevicesUnavailable             cudaError = C.cudaErrorDevicesUnavailable
	IncompatibleDriverContext      cudaError = C.cudaErrorIncompatibleDriverContext
	MissingConfiguration           cudaError = C.cudaErrorMissingConfiguration
	PriorLaunchFailure             cudaError = C.cudaErrorPriorLaunchFailure
	LaunchMaxDepthExceeded         cudaError = C.cudaErrorLaunchMaxDepthExceeded
	LaunchFileScopedTex            cudaError = C.cudaErrorLaunchFileScopedTex
	LaunchFileScopedSurf           cudaError = C.cudaErrorLaunchFileScopedSurf
	SyncDepthExceeded              cudaError = C.cudaErrorSyncDepthExceeded
	LaunchPendingCountExceeded     cudaError = C.cudaErrorLaunchPendingCountExceeded
	InvalidDeviceFunction          cudaError = C.cudaErrorInvalidDeviceFunction
	NoDevice                       cudaError = C.cudaErrorNoDevice
	InvalidDevice                  cudaError = C.cudaErrorInvalidDevice
	DeviceNotLicensed              cudaError = C.cudaErrorDeviceNotLicensed
	SoftwareValidityNotEstablished cudaError = C.cudaErrorSoftwareValidityNotEstablished
	StartupFailure                 cudaError = C.cudaErrorStartupFailure
	InvalidKernelImage             cudaError = C.cudaErrorInvalidKernelImage
	DeviceUninitialized            cudaError = C.cudaErrorDeviceUninitialized
	MapBufferObjectFailed          cudaError = C.cudaErrorMapBufferObjectFailed
	UnmapBufferObjectFailed        cudaError = C.cudaErrorUnmapBufferObjectFailed
	ArrayIsMapped                  cudaError = C.cudaErrorArrayIsMapped
	AlreadyMapped                  cudaError = C.cudaErrorAlreadyMapped
	NoKernelImageForDevice         cudaError = C.cudaErrorNoKernelImageForDevice
	AlreadyAcquired                cudaError = C.cudaErrorAlreadyAcquired
	NotMapped                      cudaError = C.cudaErrorNotMapped
	NotMappedAsArray               cudaError = C.cudaErrorNotMappedAsArray
	NotMappedAsPointer             cudaError = C.cudaErrorNotMappedAsPointer
	ECCUncorrectable               cudaError = C.cudaErrorECCUncorrectable
	UnsupportedLimit               cudaError = C.cudaErrorUnsupportedLimit
	DeviceAlreadyInUse             cudaError = C.cudaErrorDeviceAlreadyInUse
	PeerAccessUnsupported          cudaError = C.cudaErrorPeerAccessUnsupported
	InvalidPtx                     cudaError = C.cudaErrorInvalidPtx
	InvalidGraphicsContext         cudaError = C.cudaErrorInvalidGraphicsContext
	NvlinkUncorrectable            cudaError = C.cudaErrorNvlinkUncorrectable
	JitCompilerNotFound            cudaError = C.cudaErrorJitCompilerNotFound
	UnsupportedPtxVersion          cudaError = C.cudaErrorUnsupportedPtxVersion
	JitCompilationDisabled         cudaError = C.cudaErrorJitCompilationDisabled
	UnsupportedExecAffinity        cudaError = C.cudaErrorUnsupportedExecAffinity
	InvalidSource                  cudaError = C.cudaErrorInvalidSource
	FileNotFound                   cudaError = C.cudaErrorFileNotFound
	SharedObjectSymbolNotFound     cudaError = C.cudaErrorSharedObjectSymbolNotFound
	SharedObjectInitFailed         cudaError = C.cudaErrorSharedObjectInitFailed
	OperatingSystem                cudaError = C.cudaErrorOperatingSystem
	InvalidResourceHandle          cudaError = C.cudaErrorInvalidResourceHandle
	IllegalState                   cudaError = C.cudaErrorIllegalState
	SymbolNotFound                 cudaError = C.cudaErrorSymbolNotFound
	NotReady                       cudaError = C.cudaErrorNotReady
	IllegalAddress                 cudaError = C.cudaErrorIllegalAddress
	LaunchOutOfResources           cudaError = C.cudaErrorLaunchOutOfResources
	LaunchTimeout                  cudaError = C.cudaErrorLaunchTimeout
	LaunchIncompatibleTexturing    cudaError = C.cudaErrorLaunchIncompatibleTexturing
	PeerAccessAlreadyEnabled       cudaError = C.cudaErrorPeerAccessAlreadyEnabled
	PeerAccessNotEnabled           cudaError = C.cudaErrorPeerAccessNotEnabled
	SetOnActiveProcess             cudaError = C.cudaErrorSetOnActiveProcess
	ContextIsDestroyed             cudaError = C.cudaErrorContextIsDestroyed
	Assert                         cudaError = C.cudaErrorAssert
	TooManyPeers                   cudaError = C.cudaErrorTooManyPeers
	HostMemoryAlreadyRegistered    cudaError = C.cudaErrorHostMemoryAlreadyRegistered
	HostMemoryNotRegistered        cudaError = C.cudaErrorHostMemoryNotRegistered
	HardwareStackError             cudaError = C.cudaErrorHardwareStackError
	IllegalInstruction             cudaError = C.cudaErrorIllegalInstruction
	MisalignedAddress              cudaError = C.cudaErrorMisalignedAddress
	InvalidAddressSpace            cudaError = C.cudaErrorInvalidAddressSpace
	InvalidPc                      cudaError = C.cudaErrorInvalidPc
	LaunchFailure                  cudaError = C.cudaErrorLaunchFailure
	CooperativeLaunchTooLarge      cudaError = C.cudaErrorCooperativeLaunchTooLarge
	NotPermitted                   cudaError = C.cudaErrorNotPermitted
	NotSupported                   cudaError = C.cudaErrorNotSupported
	SystemNotReady                 cudaError = C.cudaErrorSystemNotReady
	SystemDriverMismatch           cudaError = C.cudaErrorSystemDriverMismatch
	CompatNotSupportedOnDevice     cudaError = C.cudaErrorCompatNotSupportedOnDevice
	MpsConnectionFailed            cudaError = C.cudaErrorMpsConnectionFailed
	MpsRpcFailure                  cudaError = C.cudaErrorMpsRpcFailure
	MpsServerNotReady              cudaError = C.cudaErrorMpsServerNotReady
	MpsMaxClientsReached           cudaError = C.cudaErrorMpsMaxClientsReached
	MpsMaxConnectionsReached       cudaError = C.cudaErrorMpsMaxConnectionsReached
	MpsClientTerminated            cudaError = C.cudaErrorMpsClientTerminated
	CdpNotSupported                cudaError = C.cudaErrorCdpNotSupported
	CdpVersionMismatch             cudaError = C.cudaErrorCdpVersionMismatch
	StreamCaptureUnsupported       cudaError = C.cudaErrorStreamCaptureUnsupported
	StreamCaptureInvalidated       cudaError = C.cudaErrorStreamCaptureInvalidated
	StreamCaptureMerge             cudaError = C.cudaErrorStreamCaptureMerge
	StreamCaptureUnmatched         cudaError = C.cudaErrorStreamCaptureUnmatched
	StreamCaptureUnjoined          cudaError = C.cudaErrorStreamCaptureUnjoined
	StreamCaptureIsolation         cudaError = C.cudaErrorStreamCaptureIsolation
	StreamCaptureImplicit          cudaError = C.cudaErrorStreamCaptureImplicit
	CapturedEvent                  cudaError = C.cudaErrorCapturedEvent
	StreamCaptureWrongThread       cudaError = C.cudaErrorStreamCaptureWrongThread
	Timeout                        cudaError = C.cudaErrorTimeout
	GraphExecUpdateFailure         cudaError = C.cudaErrorGraphExecUpdateFailure
	ExternalDevice                 cudaError = C.cudaErrorExternalDevice
	InvalidClusterSize             cudaError = C.cudaErrorInvalidClusterSize
	Unknown                        cudaError = C.cudaErrorUnknown
	ApiFailureBase                 cudaError = C.cudaErrorApiFailureBase
)

var resString = map[cudaError]string{
	Success:                        "Success",
	InvalidValue:                   "InvalidValue",
	MemoryAllocation:               "MemoryAllocation",
	InitializationError:            "InitializationError",
	CudartUnloading:                "CudartUnloading",
	ProfilerDisabled:               "ProfilerDisabled",
	ProfilerNotInitialized:         "ProfilerNotInitialized",
	ProfilerAlreadyStarted:         "ProfilerAlreadyStarted",
	ProfilerAlreadyStopped:         "ProfilerAlreadyStopped",
	InvalidConfiguration:           "InvalidConfiguration",
	InvalidPitchValue:              "InvalidPitchValue",
	InvalidSymbol:                  "InvalidSymbol",
	InvalidHostPointer:             "InvalidHostPointer",
	InvalidDevicePointer:           "InvalidDevicePointer",
	InvalidTexture:                 "InvalidTexture",
	InvalidTextureBinding:          "InvalidTextureBinding",
	InvalidChannelDescriptor:       "InvalidChannelDescriptor",
	InvalidMemcpyDirection:         "InvalidMemcpyDirection",
	AddressOfConstant:              "AddressOfConstant",
	TextureFetchFailed:             "TextureFetchFailed",
	TextureNotBound:                "TextureNotBound",
	SynchronizationError:           "SynchronizationError",
	InvalidFilterSetting:           "InvalidFilterSetting",
	InvalidNormSetting:             "InvalidNormSetting",
	MixedDeviceExecution:           "MixedDeviceExecution",
	NotYetImplemented:              "NotYetImplemented",
	MemoryValueTooLarge:            "MemoryValueTooLarge",
	StubLibrary:                    "StubLibrary",
	InsufficientDriver:             "InsufficientDriver",
	CallRequiresNewerDriver:        "CallRequiresNewerDriver",
	InvalidSurface:                 "InvalidSurface",
	DuplicateVariableName:          "DuplicateVariableName",
	DuplicateTextureName:           "DuplicateTextureName",
	DuplicateSurfaceName:           "DuplicateSurfaceName",
	DevicesUnavailable:             "DevicesUnavailable",
	IncompatibleDriverContext:      "IncompatibleDriverContext",
	MissingConfiguration:           "MissingConfiguration",
	PriorLaunchFailure:             "PriorLaunchFailure",
	LaunchMaxDepthExceeded:         "LaunchMaxDepthExceeded",
	LaunchFileScopedTex:            "LaunchFileScopedTex",
	LaunchFileScopedSurf:           "LaunchFileScopedSurf",
	SyncDepthExceeded:              "SyncDepthExceeded",
	LaunchPendingCountExceeded:     "LaunchPendingCountExceeded",
	InvalidDeviceFunction:          "InvalidDeviceFunction",
	NoDevice:                       "NoDevice",
	InvalidDevice:                  "InvalidDevice",
	DeviceNotLicensed:              "DeviceNotLicensed",
	SoftwareValidityNotEstablished: "SoftwareValidityNotEstablished",
	StartupFailure:                 "StartupFailure",
	InvalidKernelImage:             "InvalidKernelImage",
	DeviceUninitialized:            "DeviceUninitialized",
	MapBufferObjectFailed:          "MapBufferObjectFailed",
	UnmapBufferObjectFailed:        "UnmapBufferObjectFailed",
	ArrayIsMapped:                  "ArrayIsMapped",
	AlreadyMapped:                  "AlreadyMapped",
	NoKernelImageForDevice:         "NoKernelImageForDevice",
	AlreadyAcquired:                "AlreadyAcquired",
	NotMapped:                      "NotMapped",
	NotMappedAsArray:               "NotMappedAsArray",
	NotMappedAsPointer:             "NotMappedAsPointer",
	ECCUncorrectable:               "ECCUncorrectable",
	UnsupportedLimit:               "UnsupportedLimit",
	DeviceAlreadyInUse:             "DeviceAlreadyInUse",
	PeerAccessUnsupported:          "PeerAccessUnsupported",
	InvalidPtx:                     "InvalidPtx",
	InvalidGraphicsContext:         "InvalidGraphicsContext",
	NvlinkUncorrectable:            "NvlinkUncorrectable",
	JitCompilerNotFound:            "JitCompilerNotFound",
	UnsupportedPtxVersion:          "UnsupportedPtxVersion",
	JitCompilationDisabled:         "JitCompilationDisabled",
	UnsupportedExecAffinity:        "UnsupportedExecAffinity",
	InvalidSource:                  "InvalidSource",
	FileNotFound:                   "FileNotFound",
	SharedObjectSymbolNotFound:     "SharedObjectSymbolNotFound",
	SharedObjectInitFailed:         "SharedObjectInitFailed",
	OperatingSystem:                "OperatingSystem",
	InvalidResourceHandle:          "InvalidResourceHandle",
	IllegalState:                   "IllegalState",
	SymbolNotFound:                 "SymbolNotFound",
	NotReady:                       "NotReady",
	IllegalAddress:                 "IllegalAddress",
	LaunchOutOfResources:           "LaunchOutOfResources",
	LaunchTimeout:                  "LaunchTimeout",
	LaunchIncompatibleTexturing:    "LaunchIncompatibleTexturing",
	PeerAccessAlreadyEnabled:       "PeerAccessAlreadyEnabled",
	PeerAccessNotEnabled:           "PeerAccessNotEnabled",
	SetOnActiveProcess:             "SetOnActiveProcess",
	ContextIsDestroyed:             "ContextIsDestroyed",
	Assert:                         "Assert",
	TooManyPeers:                   "TooManyPeers",
	HostMemoryAlreadyRegistered:    "HostMemoryAlreadyRegistered",
	HostMemoryNotRegistered:        "HostMemoryNotRegistered",
	HardwareStackError:             "HardwareStackError",
	IllegalInstruction:             "IllegalInstruction",
	MisalignedAddress:              "MisalignedAddress",
	InvalidAddressSpace:            "InvalidAddressSpace",
	InvalidPc:                      "InvalidPc",
	LaunchFailure:                  "LaunchFailure",
	CooperativeLaunchTooLarge:      "CooperativeLaunchTooLarge",
	NotPermitted:                   "NotPermitted",
	NotSupported:                   "NotSupported",
	SystemNotReady:                 "SystemNotReady",
	SystemDriverMismatch:           "SystemDriverMismatch",
	CompatNotSupportedOnDevice:     "CompatNotSupportedOnDevice",
	MpsConnectionFailed:            "MpsConnectionFailed",
	MpsRpcFailure:                  "MpsRpcFailure",
	MpsServerNotReady:              "MpsServerNotReady",
	MpsMaxClientsReached:           "MpsMaxClientsReached",
	MpsMaxConnectionsReached:       "MpsMaxConnectionsReached",
	MpsClientTerminated:            "MpsClientTerminated",
	CdpNotSupported:                "CdpNotSupported",
	CdpVersionMismatch:             "CdpVersionMismatch",
	StreamCaptureUnsupported:       "StreamCaptureUnsupported",
	StreamCaptureInvalidated:       "StreamCaptureInvalidated",
	StreamCaptureMerge:             "StreamCaptureMerge",
	StreamCaptureUnmatched:         "StreamCaptureUnmatched",
	StreamCaptureUnjoined:          "StreamCaptureUnjoined",
	StreamCaptureIsolation:         "StreamCaptureIsolation",
	StreamCaptureImplicit:          "StreamCaptureImplicit",
	CapturedEvent:                  "CapturedEvent",
	StreamCaptureWrongThread:       "StreamCaptureWrongThread",
	Timeout:                        "Timeout",
	GraphExecUpdateFailure:         "GraphExecUpdateFailure",
	ExternalDevice:                 "ExternalDevice",
	InvalidClusterSize:             "InvalidClusterSize",
	Unknown:                        "Unknown",
	ApiFailureBase:                 "ApiFailureBase",
}
