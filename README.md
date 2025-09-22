# gocu

Go bindings for the CUDA Driver and Runtime APIs, cuBLAS, and cuDNN.

## Quick Start

This example demonstrates GPU-accelerated matrix multiplication using CUBLAS. It shows how to:

- Allocate GPU memory and transfer data between CPU and GPU
- Perform high-performance matrix operations on the GPU
- Use the convenient helper functions for memory management

For the complete example, see [example/cudblas/gemm](example/cudblas/gemm/main.go).

```go
package main

import (
 "fmt"

 "github.com/gocnn/gocu/cublas"
 "github.com/gocnn/gocu/cudart"
)

func main() {
    // Host data (column-major storage)
    A := []float32{
    1, 2, 3, 4,
    2, 3, 4, 5,
    3, 4, 5, 6,
    4, 5, 6, 7,
    } // 4x4 matrix
    B := []float32{
    2, 0, 1, 0,
    0, 2, 0, 1,
    1, 0, 2, 0,
    0, 1, 0, 2,
    } // 4x4 matrix
    C := make([]float32, 16) // 4x4 result

    // GPU memory
    devA, _ := cudart.Malloc(cudart.SliceBytes(A))
    devB, _ := cudart.Malloc(cudart.SliceBytes(B))
    devC, _ := cudart.Malloc(cudart.SliceBytes(C))
    defer cudart.Free(devA)
    defer cudart.Free(devB)
    defer cudart.Free(devC)

    // Copy to GPU
    cudart.MemcpyHtoD(devA, cudart.SliceToHostPtr(A), cudart.SliceBytes(A))
    cudart.MemcpyHtoD(devB, cudart.SliceToHostPtr(B), cudart.SliceBytes(B))

    // Matrix multiplication: C = A * B
    handler := cublas.New()
    defer handler.Close()
    handler.Sgemm(cublas.NoTranspose, cublas.NoTranspose, 4, 4, 4,
    1.0, devA, 4, devB, 4, 0.0, devC, 4)

    // Copy result back
    cudart.MemcpyDtoH(cudart.SliceToHostPtr(C), devC, cudart.SliceBytes(C))
    fmt.Printf("Result: %v\n", C)
    // [ 5  8  7 10
    //   8 11 10 13
    //  11 14 13 16
    //  14 17 16 19]
}
```

## Install

First, verify that your system has CUDA support:

```sh
go install github.com/gocnn/gocu/cmd/gocu
gocu
```

If the detection passes, install the library:

```sh
go install github.com/gocnn/gocu
```

## License

BSD 3-Clause License
