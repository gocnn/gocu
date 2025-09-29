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

	devA, _, _ := cudart.MallocManagedAndCopy([]float32{
		1, 2, 3, 4,
		2, 3, 4, 5,
		3, 4, 5, 6,
		4, 5, 6, 7,
	})
	devB, _, _ := cudart.MallocManagedAndCopy([]float32{
		2, 0, 1, 0,
		0, 2, 0, 1,
		1, 0, 2, 0,
		0, 1, 0, 2,
	})
	devC, hostC, _ := cudart.MallocManaged(make([]float32, M*N))
	defer cudart.Free(devA)
	defer cudart.Free(devB)
	defer cudart.Free(devC)

	blas := cublas.New()
	defer blas.Close()
	blas.Sgemm(cublas.NoTrans, cublas.NoTrans, M, N, K, alpha, devA, M, devB, K, beta, devC, M)

	cudart.DeviceSynchronize()

	for i := range M {
		for j := range N {
			fmt.Printf("%8.1f ", hostC[i*N+j])
		}
		fmt.Println()
	}
}
