package gocu

// #include <cuda.h>
import "C"

// EventFlags are flags to be used with event creation
type EventFlags byte

const (
	DefaultEvent      EventFlags = C.CU_EVENT_DEFAULT        // Default event flag
	BlockingSyncEvent EventFlags = C.CU_EVENT_BLOCKING_SYNC  // Event uses blocking synchronization
	DisableTiming     EventFlags = C.CU_EVENT_DISABLE_TIMING // Event will not record timing data
	InterprocessEvent EventFlags = C.CU_EVENT_INTERPROCESS   // Event is suitable for interprocess use. DisableTiming must be set
)

// EventRecordFlags are flags to be used with event recording
type EventRecordFlags uint32

const (
	EventRecordDefault  EventRecordFlags = C.CU_EVENT_RECORD_DEFAULT  // Default event record flag
	EventRecordExternal EventRecordFlags = C.CU_EVENT_RECORD_EXTERNAL // Event is captured as external node during stream capture
)

// Event represents a CUDA event
type Event struct {
	event C.CUevent
}

func (e Event) c() C.CUevent { return e.event }

// EventCreate creates a new CUDA event
func EventCreate(flags EventFlags) (Event, error) {
	var event C.CUevent
	result := Check(C.cuEventCreate(&event, C.uint(flags)))
	return Event{event: event}, result
}

// EventDestroy destroys a CUDA event
func EventDestroy(event Event) error {
	return Check(C.cuEventDestroy(event.c()))
}

// Destroy destroys the event (method version)
func (e Event) Destroy() error {
	return EventDestroy(e)
}

// EventElapsedTime computes the elapsed time between two events
func EventElapsedTime(start, end Event) (float32, error) {
	var milliseconds C.float
	result := Check(C.cuEventElapsedTime(&milliseconds, start.c(), end.c()))
	return float32(milliseconds), result
}

// ElapsedTime computes the elapsed time from this event to the end event (method version)
func (e Event) ElapsedTime(end Event) (float32, error) {
	return EventElapsedTime(e, end)
}

// EventQuery queries an event's status
func EventQuery(event Event) error {
	return Check(C.cuEventQuery(event.c()))
}

// Query queries the event's status (method version)
func (e Event) Query() error {
	return EventQuery(e)
}

// EventRecord records an event on the specified stream
func EventRecord(event Event, stream Stream) error {
	return Check(C.cuEventRecord(event.c(), stream.c()))
}

// Record records the event on the specified stream (method version)
func (e Event) Record(stream Stream) error {
	return EventRecord(e, stream)
}

// EventRecordWithFlags records an event on the specified stream with flags
func EventRecordWithFlags(event Event, stream Stream, flags EventRecordFlags) error {
	return Check(C.cuEventRecordWithFlags(event.c(), stream.c(), C.uint(flags)))
}

// RecordWithFlags records the event on the specified stream with flags (method version)
func (e Event) RecordWithFlags(stream Stream, flags EventRecordFlags) error {
	return EventRecordWithFlags(e, stream, flags)
}

// EventSynchronize waits for an event to complete
func EventSynchronize(event Event) error {
	return Check(C.cuEventSynchronize(event.c()))
}

// Synchronize waits for the event to complete (method version)
func (e Event) Synchronize() error {
	return EventSynchronize(e)
}
