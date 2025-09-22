package cudart

import "unsafe"

// SliceToHostPtr converts a Go slice to HostPtr for the first element.
// This simplifies the common pattern: cudart.HostPtr(unsafe.Pointer(&slice[0]))
func SliceToHostPtr[T any](slice []T) HostPtr {
	if len(slice) == 0 {
		return HostPtr(nil)
	}
	return HostPtr(unsafe.Pointer(&slice[0]))
}

// SliceBytes returns the size in bytes of a Go slice.
// This simplifies calculating the byte size for memory operations.
func SliceBytes[T any](slice []T) int64 {
	if len(slice) == 0 {
		return 0
	}
	return int64(len(slice) * int(unsafe.Sizeof(slice[0])))
}
