package gocu

// #include <cuda.h>
import "C"

// Stream is a CUDA stream
type Stream struct{ stream C.CUstream }

// C returns the Stream as its C version
func (s Stream) c() C.CUstream { return s.stream }

// StreamCreate creates a new CUDA stream
func StreamCreate(flags uint32) (Stream, error) {
	var stream C.CUstream
	result := Check(C.cuStreamCreate(&stream, C.uint(flags)))
	return Stream{stream: stream}, result
}

// StreamCreateWithPriority creates a new CUDA stream with specified priority
func StreamCreateWithPriority(flags uint32, priority int) (Stream, error) {
	var stream C.CUstream
	result := Check(C.cuStreamCreateWithPriority(&stream, C.uint(flags), C.int(priority)))
	return Stream{stream: stream}, result
}

// StreamDestroy destroys a CUDA stream
func StreamDestroy(stream Stream) error {
	result := Check(C.cuStreamDestroy(stream.c()))
	return result
}

// Destroy destroys the stream (method version)
func (s Stream) Destroy() error {
	return StreamDestroy(s)
}

// StreamSynchronize waits until all tasks in the stream are completed
func StreamSynchronize(stream Stream) error {
	return Check(C.cuStreamSynchronize(stream.c()))
}

// Synchronize waits until all tasks in the stream are completed (method version)
func (s Stream) Synchronize() error {
	return StreamSynchronize(s)
}

// StreamQuery determines the status of a compute stream
func StreamQuery(stream Stream) error {
	return Check(C.cuStreamQuery(stream.c()))
}

// Query determines the status of the stream (method version)
func (s Stream) Query() error {
	return StreamQuery(s)
}

// StreamGetFlags queries the flags of a given stream
func StreamGetFlags(stream Stream) (uint32, error) {
	var flags C.uint
	result := Check(C.cuStreamGetFlags(stream.c(), &flags))
	return uint32(flags), result
}

// GetFlags queries the flags of the stream (method version)
func (s Stream) GetFlags() (uint32, error) {
	return StreamGetFlags(s)
}

// StreamGetPriority queries the priority of a given stream
func StreamGetPriority(stream Stream) (int, error) {
	var priority C.int
	result := Check(C.cuStreamGetPriority(stream.c(), &priority))
	return int(priority), result
}

// GetPriority queries the priority of the stream (method version)
func (s Stream) GetPriority() (int, error) {
	return StreamGetPriority(s)
}

// // StreamGetDevice returns the device handle of the stream
// func StreamGetDevice(stream Stream) (Device, error) {
// 	var device C.CUdevice
// 	result := Check(C.cuStreamGetDevice(stream.c(), &device))
// 	return Device{device: device}, result
// }

// // GetDevice returns the device handle of the stream (method version)
// func (s Stream) GetDevice() (Device, error) {
// 	return StreamGetDevice(s)
// }

// // StreamWaitEvent makes a compute stream wait on an event
// func StreamWaitEvent(stream Stream, event CUEvent, flags uint32) error {
// 	return Check(C.cuStreamWaitEvent(stream.c(), event.c(), C.uint(flags)))
// }

// // WaitEvent makes the stream wait on an event (method version)
// func (s Stream) WaitEvent(event CUEvent, flags uint32) error {
// 	return StreamWaitEvent(s, event, flags)
// }

// // StreamAttachMemAsync attaches memory to a stream asynchronously
// func StreamAttachMemAsync(stream Stream, dptr DevicePtr, length uint64, flags uint32) error {
// 	return Check(C.cuStreamAttachMemAsync(stream.c(), C.CUdeviceptr(uintptr(dptr)), C.size_t(length), C.uint(flags)))
// }

// // AttachMemAsync attaches memory to the stream asynchronously (method version)
// func (s Stream) AttachMemAsync(dptr DevicePtr, length uint64, flags uint32) error {
// 	return StreamAttachMemAsync(s, dptr, length, flags)
// }

// // StreamBeginCapture begins graph capture on a stream
// func StreamBeginCapture(stream Stream, mode CUStreamCaptureMode) error {
// 	return Check(C.cuStreamBeginCapture(stream.c(), C.CUstreamCaptureMode(mode)))
// }

// // BeginCapture begins graph capture on the stream (method version)
// func (s Stream) BeginCapture(mode CUStreamCaptureMode) error {
// 	return StreamBeginCapture(s, mode)
// }

// // StreamEndCapture ends capture on a stream, returning the captured graph
// func StreamEndCapture(stream Stream) (CUGraph, error) {
// 	var graph C.CUgraph
// 	result := Check(C.cuStreamEndCapture(stream.c(), &graph))
// 	return CUGraph{graph: graph}, result
// }

// // EndCapture ends capture on the stream, returning the captured graph (method version)
// func (s Stream) EndCapture() (CUGraph, error) {
// 	return StreamEndCapture(s)
// }

// // StreamIsCapturing returns a stream's capture status
// func StreamIsCapturing(stream Stream) (CUStreamCaptureStatus, error) {
// 	var status C.CUstreamCaptureStatus
// 	return CUStreamCaptureStatus(status), Check(C.cuStreamIsCapturing(stream.c(), &status))
// }

// // IsCapturing returns the stream's capture status (method version)
// func (s Stream) IsCapturing() (CUStreamCaptureStatus, error) {
// 	return StreamIsCapturing(s)
// }

// // Stream creation flags
// const (
// 	StreamDefault     = 0x0 // Default stream flag
// 	StreamNonBlocking = 0x1 // Stream does not synchronize with stream 0 (the NULL stream)
// )

// // Stream capture modes
// type CUStreamCaptureMode C.CUstreamCaptureMode

// const (
// 	StreamCaptureModeGlobal      CUStreamCaptureMode = C.CU_STREAM_CAPTURE_MODE_GLOBAL
// 	StreamCaptureModeThreadLocal CUStreamCaptureMode = C.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL
// 	StreamCaptureModeRelaxed     CUStreamCaptureMode = C.CU_STREAM_CAPTURE_MODE_RELAXED
// )

// // Stream capture status
// type CUStreamCaptureStatus C.CUstreamCaptureStatus

// const (
// 	StreamCaptureStatusNone        CUStreamCaptureStatus = C.CU_STREAM_CAPTURE_STATUS_NONE
// 	StreamCaptureStatusActive      CUStreamCaptureStatus = C.CU_STREAM_CAPTURE_STATUS_ACTIVE
// 	StreamCaptureStatusInvalidated CUStreamCaptureStatus = C.CU_STREAM_CAPTURE_STATUS_INVALIDATED
// )
