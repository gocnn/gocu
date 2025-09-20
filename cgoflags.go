package gocu

/*
#cgo LDFLAGS: -lcuda -lcudart
#cgo linux LDFLAGS:-L/usr/local/cuda/lib64
#cgo linux CFLAGS: -I/usr/local/cuda/include
#cgo windows LDFLAGS: -LC:/cuda/lib/x64 -LC:\Users\14388\Desktop\qntx\gocu
#cgo windows CFLAGS: -IC:/cuda/include
*/
import "C"
