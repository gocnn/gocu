//go:build cuda || cuda11 || cuda12 || cuda13

package cublas

// This file provides CGO flags to find CUDA libraries and headers.

/*
#cgo CFLAGS: -g -O3
#cgo LDFLAGS: -lcublas

#cgo linux LDFLAGS: -L/usr/local/cuda/lib64
#cgo linux CFLAGS: -I/usr/local/cuda/include

#cgo windows LDFLAGS: -LC:/cuda/lib/x64
#cgo windows CFLAGS: -IC:/cuda/include
*/
import "C"
