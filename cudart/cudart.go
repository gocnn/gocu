package cudart

//go:generate go run generate.go

import "unsafe"

// SliceToHostPtr converts a Go slice to HostPtr for the first element.
// This simplifies the common pattern: cudart.HostPtr(unsafe.Pointer(&slice[0]))
func SliceToHostPtr[T any](slice []T) HostPtr {
	if len(slice) == 0 {
		return HostPtr(nil)
	}
	return HostPtr(unsafe.Pointer(&slice[0]))
}

// HostPtrToSlice converts a HostPtr to a Go slice with the specified length.
// This is useful for creating slices that point to allocated memory.
// WARNING: The caller must ensure the memory is valid for the entire slice length.
func HostPtrToSlice[T any](ptr HostPtr, length int) []T {
	if ptr == HostPtr(nil) || length == 0 {
		return nil
	}
	return unsafe.Slice((*T)(ptr), length)
}

// SliceToPtr converts a Go slice to unsafe.Pointer for the first element.
// This simplifies the common pattern: unsafe.Pointer(&slice[0])
func SliceToPtr[T any](slice []T) unsafe.Pointer {
	if len(slice) == 0 {
		return nil
	}
	return unsafe.Pointer(&slice[0])
}

// PtrToSlice converts a pointer to a Go slice with the specified length.
// This is useful for creating slices that point to allocated memory.
// WARNING: The caller must ensure the memory is valid for the entire slice length.
func PtrToSlice[T any](ptr unsafe.Pointer, length int) []T {
	if ptr == nil || length == 0 {
		return nil
	}
	return unsafe.Slice((*T)(ptr), length)
}

// SliceBytes returns the size in bytes of a Go slice.
// This simplifies calculating the byte size for memory operations.
func SliceBytes[T any](slice []T) int64 {
	if len(slice) == 0 {
		return 0
	}
	return int64(len(slice) * int(unsafe.Sizeof(slice[0])))
}
