package main

import (
	"fmt"

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

	devA, err := cudart.Malloc(cudart.SliceBytes(A))
	if err != nil {
		panic(err)
	}
	devB, err := cudart.Malloc(cudart.SliceBytes(B))
	if err != nil {
		panic(err)
	}
	devC, err := cudart.Malloc(cudart.SliceBytes(C))
	if err != nil {
		panic(err)
	}
	defer cudart.Free(devA)
	defer cudart.Free(devB)
	defer cudart.Free(devC)

	if err := cudart.MemcpyHtoD(devA, cudart.SliceToHostPtr(A), cudart.SliceBytes(A)); err != nil {
		panic(err)
	}
	if err := cudart.MemcpyHtoD(devB, cudart.SliceToHostPtr(B), cudart.SliceBytes(B)); err != nil {
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

	if err := cudart.MemcpyDtoH(cudart.SliceToHostPtr(C), devC, cudart.SliceBytes(C)); err != nil {
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
