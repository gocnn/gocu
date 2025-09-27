package cudart

/*
#include <cuda_runtime.h>
*/
import "C"

// Event represents a CUDA event for synchronization and timing.
type Event struct {
	event C.cudaEvent_t
}

// c returns the underlying C.cudaEvent_t value.
func (e Event) c() C.cudaEvent_t {
	return e.event
}

// EventFlags represents flags for event creation.
type EventFlags uint

const (
	EventDefault        EventFlags = C.cudaEventDefault        // Default event creation flag
	EventBlockingSync   EventFlags = C.cudaEventBlockingSync   // Event uses blocking synchronization
	EventDisableTiming  EventFlags = C.cudaEventDisableTiming  // Event will not record timing data
	EventInterprocess   EventFlags = C.cudaEventInterprocess   // Event is suitable for interprocess use
	EventRecordExternal EventFlags = C.cudaEventRecordExternal // Event is being recorded by external means
)

// EventCreate creates an event object.
func EventCreate() (Event, error) {
	var event C.cudaEvent_t
	err := Check(C.cudaEventCreate(&event))
	return Event{event: event}, err
}

// EventCreateWithFlags creates an event object with the specified flags.
func EventCreateWithFlags(flags EventFlags) (Event, error) {
	var event C.cudaEvent_t
	err := Check(C.cudaEventCreateWithFlags(&event, C.uint(flags)))
	return Event{event: event}, err
}

// EventDestroy destroys an event object.
func EventDestroy(event Event) error {
	return Check(C.cudaEventDestroy(event.c()))
}

// Destroy destroys the event object.
func (e Event) Destroy() error {
	return EventDestroy(e)
}

// EventRecord records an event on the specified stream.
// If stream is nil, records on the default stream.
func EventRecord(event Event, stream *Stream) error {
	if stream == nil {
		return Check(C.cudaEventRecord(event.c(), C.cudaStream_t(nil)))
	}
	return Check(C.cudaEventRecord(event.c(), stream.c()))
}

// Record records the event on the specified stream.
func (e Event) Record(stream *Stream) error {
	return EventRecord(e, stream)
}

// EventRecordWithFlags records an event with flags on the specified stream.
func EventRecordWithFlags(event Event, stream *Stream, flags uint) error {
	if stream == nil {
		return Check(C.cudaEventRecordWithFlags(event.c(), C.cudaStream_t(nil), C.uint(flags)))
	}
	return Check(C.cudaEventRecordWithFlags(event.c(), stream.c(), C.uint(flags)))
}

// RecordWithFlags records the event with flags on the specified stream.
func (e Event) RecordWithFlags(stream *Stream, flags uint) error {
	return EventRecordWithFlags(e, stream, flags)
}

// EventSynchronize waits for an event to complete.
func EventSynchronize(event Event) error {
	return Check(C.cudaEventSynchronize(event.c()))
}

// Synchronize waits for the event to complete.
func (e Event) Synchronize() error {
	return EventSynchronize(e)
}

// EventQuery queries an event's Check.
// Returns nil if the event has completed, or an error if it's still pending.
func EventQuery(event Event) error {
	return Check(C.cudaEventQuery(event.c()))
}

// Query queries the event's Check.
func (e Event) Query() error {
	return EventQuery(e)
}

// EventElapsedTime computes the elapsed time between two events in milliseconds.
// Both events must have been recorded before calling this function.
func EventElapsedTime(start, end Event) (float32, error) {
	var ms C.float
	err := Check(C.cudaEventElapsedTime(&ms, start.c(), end.c()))
	return float32(ms), err
}

// ElapsedTime computes the elapsed time from this event to the end event.
func (e Event) ElapsedTime(end Event) (float32, error) {
	return EventElapsedTime(e, end)
}
