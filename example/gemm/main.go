package main

import (
	"fmt"
	"unsafe"

	"github.com/gocnn/gocu/cublas"
	"github.com/gocnn/gocu/cudart"
)

func main() {
	const (
		M, N, K = 4, 4, 4
		alpha   = 1.0
		beta    = 0.0
	)

	A := []float32{
		1, 2, 3, 4,
		2, 3, 4, 5,
		3, 4, 5, 6,
		4, 5, 6, 7,
	}
	B := []float32{
		2, 0, 1, 0,
		0, 2, 0, 1,
		1, 0, 2, 0,
		0, 1, 0, 2,
	}
	C := make([]float32, M*N)

	devA, err := cudart.Malloc(int64(len(A) * 4))
	if err != nil {
		panic(err)
	}
	devB, err := cudart.Malloc(int64(len(B) * 4))
	if err != nil {
		panic(err)
	}
	devC, err := cudart.Malloc(int64(len(C) * 4))
	if err != nil {
		panic(err)
	}
	defer cudart.Free(devA)
	defer cudart.Free(devB)
	defer cudart.Free(devC)

	if err := cudart.MemcpyHtoD(devA, cudart.HostPtr(unsafe.Pointer(&A[0])), int64(len(A)*4)); err != nil {
		panic(err)
	}
	if err := cudart.MemcpyHtoD(devB, cudart.HostPtr(unsafe.Pointer(&B[0])), int64(len(B)*4)); err != nil {
		panic(err)
	}

	handle, err := cublas.Create()
	if err != nil {
		panic(err)
	}
	defer handle.Destroy()

	err = cublas.Sgemm(handle, cublas.NoTrans, cublas.NoTrans, M, N, K, alpha, devA, M, devB, K, beta, devC, M)
	if err != nil {
		panic(err)
	}

	if err := cudart.MemcpyDtoH(cudart.HostPtr(unsafe.Pointer(&C[0])), devC, int64(len(C)*4)); err != nil {
		panic(err)
	}

	fmt.Println("Result matrix C:")
	for i := range M {
		for j := range N {
			fmt.Printf("%8.1f ", C[i*N+j])
		}
		fmt.Println()
	}
}
