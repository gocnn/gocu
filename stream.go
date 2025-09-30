package gocu

/*
#include <cuda.h>
*/
import "C"

// Stream creation flags
type StreamFlags uint32

const (
	StreamDefault     StreamFlags = 0x0 // Default stream flag
	StreamNonBlocking StreamFlags = 0x1 // Stream does not synchronize with stream 0 (the NULL stream)
)

// CUStreamCaptureMode is the mode for stream capture
type CUStreamCaptureMode C.CUstreamCaptureMode

const (
	StreamCaptureModeGlobal      CUStreamCaptureMode = C.CU_STREAM_CAPTURE_MODE_GLOBAL
	StreamCaptureModeThreadLocal CUStreamCaptureMode = C.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL
	StreamCaptureModeRelaxed     CUStreamCaptureMode = C.CU_STREAM_CAPTURE_MODE_RELAXED
)

// EventWaitFlags are flags to be used with stream wait event operations
type EventWaitFlags uint32

const (
	EventWaitDefault  EventWaitFlags = C.CU_EVENT_WAIT_DEFAULT  // Default event wait flag
	EventWaitExternal EventWaitFlags = C.CU_EVENT_WAIT_EXTERNAL // Event is captured as external node during stream capture
)

// MemAttachFlags are flags to be used with memory attach operations
type MemAttachFlags uint32

const (
	MemAttachGlobal MemAttachFlags = C.CU_MEM_ATTACH_GLOBAL // Memory can be accessed by any stream on any device
	MemAttachHost   MemAttachFlags = C.CU_MEM_ATTACH_HOST   // Memory cannot be accessed by any stream on any device
	MemAttachSingle MemAttachFlags = C.CU_MEM_ATTACH_SINGLE // Memory can only be accessed by a single stream on the associated device
)

// Stream capture status
type CUStreamCaptureStatus C.CUstreamCaptureStatus

const (
	StreamCaptureStatusNone        CUStreamCaptureStatus = C.CU_STREAM_CAPTURE_STATUS_NONE
	StreamCaptureStatusActive      CUStreamCaptureStatus = C.CU_STREAM_CAPTURE_STATUS_ACTIVE
	StreamCaptureStatusInvalidated CUStreamCaptureStatus = C.CU_STREAM_CAPTURE_STATUS_INVALIDATED
)

// Stream is a CUDA stream
type Stream struct{ stream C.CUstream }

// CStream returns the Stream as its C version
func (s *Stream) CStream() C.CUstream { return s.stream }

// StreamCreate creates a new CUDA stream
func StreamCreate(flags StreamFlags) (*Stream, error) {
	var stream C.CUstream
	result := Check(C.cuStreamCreate(&stream, C.uint(flags)))
	return &Stream{stream: stream}, result
}

// StreamCreateWithPriority creates a new CUDA stream with specified priority
func StreamCreateWithPriority(flags StreamFlags, priority int) (*Stream, error) {
	var stream C.CUstream
	result := Check(C.cuStreamCreateWithPriority(&stream, C.uint(flags), C.int(priority)))
	return &Stream{stream: stream}, result
}

// StreamDestroy destroys a CUDA stream
func StreamDestroy(stream *Stream) error {
	result := Check(C.cuStreamDestroy(stream.CStream()))
	return result
}

// Destroy destroys the stream (method version)
func (s *Stream) Destroy() error {
	return StreamDestroy(s)
}

// StreamSynchronize waits until all tasks in the stream are completed
func StreamSynchronize(stream *Stream) error {
	return Check(C.cuStreamSynchronize(stream.CStream()))
}

// Synchronize waits until all tasks in the stream are completed (method version)
func (s *Stream) Synchronize() error {
	return StreamSynchronize(s)
}

// StreamQuery determines the status of a compute stream
func StreamQuery(stream *Stream) error {
	return Check(C.cuStreamQuery(stream.CStream()))
}

// Query determines the status of the stream (method version)
func (s *Stream) Query() error {
	return StreamQuery(s)
}

// StreamGetFlags queries the flags of a given stream
func StreamGetFlags(stream *Stream) (StreamFlags, error) {
	var flags C.uint
	result := Check(C.cuStreamGetFlags(stream.CStream(), &flags))
	return StreamFlags(flags), result
}

// GetFlags queries the flags of the stream (method version)
func (s *Stream) GetFlags() (StreamFlags, error) {
	return StreamGetFlags(s)
}

// StreamGetPriority queries the priority of a given stream
func StreamGetPriority(stream *Stream) (int, error) {
	var priority C.int
	result := Check(C.cuStreamGetPriority(stream.CStream(), &priority))
	return int(priority), result
}

// GetPriority queries the priority of the stream (method version)
func (s *Stream) GetPriority() (int, error) {
	return StreamGetPriority(s)
}

// StreamGetDevice returns the device handle of the stream
func StreamGetDevice(stream *Stream) (Device, error) {
	var device C.CUdevice
	result := Check(C.cuStreamGetDevice(stream.CStream(), &device))
	return Device(device), result
}

// GetDevice returns the device handle of the stream (method version)
func (s *Stream) GetDevice() (Device, error) {
	return StreamGetDevice(s)
}

// StreamWaitEvent makes a compute stream wait on an event
func StreamWaitEvent(stream *Stream, event Event, flags EventWaitFlags) error {
	return Check(C.cuStreamWaitEvent(stream.CStream(), event.CEvent(), C.uint(flags)))
}

// WaitEvent makes the stream wait on an event (method version)
func (s *Stream) WaitEvent(event Event, flags EventWaitFlags) error {
	return StreamWaitEvent(s, event, flags)
}

// StreamAttachMemAsync attaches memory to a stream asynchronously
func StreamAttachMemAsync(stream *Stream, dptr DevicePtr, length uint64, flags MemAttachFlags) error {
	return Check(C.cuStreamAttachMemAsync(stream.CStream(), C.CUdeviceptr(uintptr(dptr)), C.size_t(length), C.uint(flags)))
}

// AttachMemAsync attaches memory to the stream asynchronously (method version)
func (s *Stream) AttachMemAsync(dptr DevicePtr, length uint64, flags MemAttachFlags) error {
	return StreamAttachMemAsync(s, dptr, length, flags)
}

// StreamBeginCapture begins graph capture on a stream
func StreamBeginCapture(stream *Stream, mode CUStreamCaptureMode) error {
	return Check(C.cuStreamBeginCapture(stream.CStream(), C.CUstreamCaptureMode(mode)))
}

// BeginCapture begins graph capture on the stream (method version)
func (s *Stream) BeginCapture(mode CUStreamCaptureMode) error {
	return StreamBeginCapture(s, mode)
}

// StreamEndCapture ends capture on a stream, returning the captured graph
func StreamEndCapture(stream *Stream) (*Graph, error) {
	var graph C.CUgraph
	result := Check(C.cuStreamEndCapture(stream.CStream(), &graph))
	return &Graph{graph: graph}, result
}

// EndCapture ends capture on the stream, returning the captured graph (method version)
func (s *Stream) EndCapture() (*Graph, error) {
	return StreamEndCapture(s)
}

// StreamIsCapturing returns a stream's capture status
func StreamIsCapturing(stream *Stream) (CUStreamCaptureStatus, error) {
	var status C.CUstreamCaptureStatus
	return CUStreamCaptureStatus(status), Check(C.cuStreamIsCapturing(stream.CStream(), &status))
}

// IsCapturing returns the stream's capture status (method version)
func (s *Stream) IsCapturing() (CUStreamCaptureStatus, error) {
	return StreamIsCapturing(s)
}
