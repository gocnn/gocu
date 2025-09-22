package cublas

// This file provides CGO flags to find CUDA libraries and headers.

// #cgo CFLAGS: -g -O3
//#cgo LDFLAGS:-lcublas
//
//#cgo linux LDFLAGS:-L/usr/local/cuda/lib64
//#cgo linux CFLAGS: -I/usr/local/cuda/include
//
//#cgo windows LDFLAGS: -LC:/cuda/v12.0/lib/x64
//#cgo windows CFLAGS: -IC:/cuda/v12.0/include
import "C"
