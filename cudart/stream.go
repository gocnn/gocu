package cudart

/*
#include <cuda_runtime.h>
*/
import "C"
import "unsafe"

// StreamFlags represents flags for stream creation.
type StreamFlags uint

const (
	StreamDefault     StreamFlags = C.cudaStreamDefault     // Default stream creation flag
	StreamNonBlocking StreamFlags = C.cudaStreamNonBlocking // Non-blocking stream flag
)

// Stream represents a CUDA stream for asynchronous operations.
type Stream struct {
	stream C.cudaStream_t
}

// CStream returns the underlying C.cudaStream_t value.
func (s *Stream) CStream() C.cudaStream_t {
	return s.stream
}

// SetCStream sets the underlying C.cudaStream_t value.
func (s *Stream) SetCStream(stream unsafe.Pointer) {
	s.stream = C.cudaStream_t(stream)
}

// StreamCreate creates an asynchronous stream.
func StreamCreate() (*Stream, error) {
	var stream C.cudaStream_t
	err := Check(C.cudaStreamCreate(&stream))
	return &Stream{stream: stream}, err
}

// StreamCreateWithFlags creates an asynchronous stream with specified flags.
func StreamCreateWithFlags(flags StreamFlags) (*Stream, error) {
	var stream C.cudaStream_t
	err := Check(C.cudaStreamCreateWithFlags(&stream, C.uint(flags)))
	return &Stream{stream: stream}, err
}

// StreamCreateWithPriority creates an asynchronous stream with specified priority.
// Lower numbers represent higher priorities.
func StreamCreateWithPriority(flags StreamFlags, priority int) (*Stream, error) {
	var stream C.cudaStream_t
	err := Check(C.cudaStreamCreateWithPriority(&stream, C.uint(flags), C.int(priority)))
	return &Stream{stream: stream}, err
}

// StreamDestroy destroys and cleans up an asynchronous stream.
func StreamDestroy(stream *Stream) error {
	return Check(C.cudaStreamDestroy(stream.CStream()))
}

// Destroy destroys and cleans up the stream.
func (s *Stream) Destroy() error {
	return StreamDestroy(s)
}

// StreamSynchronize waits for stream tasks to complete.
func StreamSynchronize(stream *Stream) error {
	return Check(C.cudaStreamSynchronize(stream.CStream()))
}

// Synchronize waits for the stream tasks to complete.
func (s *Stream) Synchronize() error {
	return StreamSynchronize(s)
}

// StreamQuery queries an asynchronous stream for completion Check.
// Returns nil if all operations have completed, or an error if operations are still pending.
func StreamQuery(stream *Stream) error {
	return Check(C.cudaStreamQuery(stream.CStream()))
}

// Query queries the stream for completion Check.
func (s *Stream) Query() error {
	return StreamQuery(s)
}

// StreamGetFlags queries the flags of a stream.
func StreamGetFlags(stream *Stream) (StreamFlags, error) {
	var flags C.uint
	err := Check(C.cudaStreamGetFlags(stream.CStream(), &flags))
	return StreamFlags(flags), err
}

// GetFlags queries the flags of the stream.
func (s *Stream) GetFlags() (StreamFlags, error) {
	return StreamGetFlags(s)
}

// StreamGetPriority queries the priority of a stream.
func StreamGetPriority(stream *Stream) (int, error) {
	var priority C.int
	err := Check(C.cudaStreamGetPriority(stream.CStream(), &priority))
	return int(priority), err
}

// GetPriority queries the priority of the stream.
func (s *Stream) GetPriority() (int, error) {
	return StreamGetPriority(s)
}

// StreamWaitEvent makes a compute stream wait on an event.
func StreamWaitEvent(stream *Stream, event *Event, flags uint) error {
	return Check(C.cudaStreamWaitEvent(stream.CStream(), event.CEvent(), C.uint(flags)))
}

// WaitEvent makes the stream wait on an event.
func (s *Stream) WaitEvent(event *Event, flags uint) error {
	return StreamWaitEvent(s, event, flags)
}
