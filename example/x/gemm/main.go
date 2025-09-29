package main

import (
	"fmt"

	"github.com/gocnn/gocu/x/cublas"
	"github.com/gocnn/gocu/x/cudart"
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

	devA, _ := cudart.MallocAndCopy(A)
	devB, _ := cudart.MallocAndCopy(B)
	devC, _ := cudart.Malloc(C)
	defer cudart.Free(devA)
	defer cudart.Free(devB)
	defer cudart.Free(devC)

	blas := cublas.New()
	defer blas.Close()
	blas.Sgemm(cublas.NoTrans, cublas.NoTrans, M, N, K, alpha, devA, M, devB, K, beta, devC, M)

	cudart.MemcpyDtoH(C, devC)

	for i := range M {
		for j := range N {
			fmt.Printf("%8.1f ", C[i*N+j])
		}
		fmt.Println()
	}
}
