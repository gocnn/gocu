package gocu

//go:generate go run generate.go

/*
#include <cuda.h>
*/
import "C"
import (
	"fmt"
	"os"
)

const initHtml = "https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__INITIALIZE.html"

func init() {
	// Given that the flags must be 0, the CUDA driver is initialized at the package level
	// http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__INITIALIZE.html
	if err := Check(C.cuInit(C.uint(0))); err != nil {
		fmt.Printf("Error in initialization, please refer to %q for details on: %+v\n", initHtml, err)
		os.Exit(1)
	}

}

// Version returns the version of the CUDA driver
func Version() int {
	var v C.int
	if err := Check(C.cuDriverGetVersion(&v)); err != nil {
		return -1
	}
	return int(v)
}
