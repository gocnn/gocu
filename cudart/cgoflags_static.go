//go:build static

package cudart

// This file provides CGO flags for static linking with CUDART libraries.

/*
#cgo LDFLAGS: -lcudart_static

#cgo linux LDFLAGS: -L/usr/local/cuda/lib64
#cgo linux CFLAGS: -I/usr/local/cuda/include

#cgo windows LDFLAGS: -LC:/cuda/lib/x64
#cgo windows CFLAGS: -IC:/cuda/include
*/
import "C"
