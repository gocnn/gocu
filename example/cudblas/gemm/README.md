# Matrix Multiplication with CUDA in Go

This README provides a step-by-step guide to performing matrix multiplication using CUDA in Go with the `gocnn/gocu` library. The example demonstrates a 4x4 matrix multiplication using the `cublas.Sgemm` function.

## Prerequisites

- Go installed (version 1.16 or later recommended).
- CUDA toolkit installed with a compatible NVIDIA GPU.
- `gocnn/gocu` library installed (`go get github.com/gocnn/gocu`).
- Basic understanding of Go programming and matrix operations.

## Steps

1. **Set Up Package and Imports**  
   Define the main package and import necessary libraries: `fmt` for output, `github.com/gocnn/gocu/cublas` for CUDA BLAS operations, and `github.com/gocnn/gocu/cudart` for CUDA runtime functions.  

   ```go
   package main

   import (
       "fmt"

       "github.com/gocnn/gocu/cublas"
       "github.com/gocnn/gocu/cudart"
   )
   ```

2. **Define Matrix Dimensions and Scalars**  
   Set constants for matrix dimensions (`M`, `N`, `K`) and scalars (`alpha`, `beta`) for the matrix multiplication operation \( C = \alpha \cdot A \cdot B + \beta \cdot C \).  
   Example: `M = 4`, `N = 4`, `K = 4`, `alpha = 1.0`, `beta = 0.0`.  

   ```go
   const (
       M, N, K = 4, 4, 4
       alpha   = 1.0
       beta    = 0.0
   )
   ```

3. **Initialize Input Matrices**  
   Create two input matrices `A` and `B` as `[]float32` slices, each with \( M \times K \) and \( K \times N \) elements, respectively.  
   Initialize an output matrix `C` with \( M \times N \) elements (e.g., `make([]float32, M*N)`).  

   ```go
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
   ```

4. **Allocate GPU Memory**  
   Use `cudart.Malloc` to allocate GPU memory for matrices `A`, `B`, and `C` based on their slice sizes. The helper function `cudart.SliceBytes(slice)` computes the total byte size of a slice by multiplying its length by the element size (e.g., 4 bytes for `float32`), simplifying memory allocation without manual calculations.  
   Without this helper, you would manually compute the size like `uintptr(len(A)) * 4` (assuming `float32` is 4 bytes).  
   Handle errors for each allocation and defer memory deallocation with `cudart.Free`.  

   ```go
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
   ```

   *Alternative code without `SliceBytes` helper:*  

   ```go
   devA, err := cudart.Malloc(int64(len(A) * 4))  // 4 bytes per float32
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
   ```

5. **Copy Data to GPU**  
   Transfer matrices `A` and `B` from host (CPU) to device (GPU) using `cudart.MemcpyHtoD`. The helper function `cudart.SliceToHostPtr(slice)` converts a Go slice to a raw host pointer (using `unsafe.Pointer(&slice[0])`), which is required for CUDA memory operations on host data.  
   Without this helper, you would use `unsafe.Pointer(&A[0])` directly, but it requires importing `unsafe` and ensuring the slice is non-nil and contiguous.  
   Ensure error handling for each memory copy operation.  

   ```go
   if err := cudart.MemcpyHtoD(devA, cudart.SliceToHostPtr(A), cudart.SliceBytes(A)); err != nil {
       panic(err)
   }
   if err := cudart.MemcpyHtoD(devB, cudart.SliceToHostPtr(B), cudart.SliceBytes(B)); err != nil {
       panic(err)
   }
   ```

   *Alternative code without `SliceToHostPtr` and `SliceBytes` helpers:*  

   ```go
   import "unsafe"  // Add this import if not already present

   if err := cudart.MemcpyHtoD(devA, cudart.HostPtr(unsafe.Pointer(&A[0])), int64(len(A)*4)); err != nil {
       panic(err)
   }
   if err := cudart.MemcpyHtoD(devB, cudart.HostPtr(unsafe.Pointer(&B[0])), int64(len(B)*4)); err != nil {
       panic(err)
   }
   ```

6. **Perform Matrix Multiplication**  
   Create a `cublas` handle with `cublas.New()` and defer its closure with `impl.Close()`.  
   Call `impl.Sgemm` with parameters: transpose options, dimensions, scalars, device pointers, and leading dimensions.  
   Check for errors using `impl.Err()`.  

   ```go
   impl := cublas.New()
   defer impl.Close()

   impl.Sgemm(cublas.NoTranspose, cublas.NoTranspose, M, N, K, alpha, devA, M, devB, K, beta, devC, M)

   if err := impl.Err(); err != nil {
       panic(err)
   }
   ```

7. **Copy Result Back to Host**  
   Transfer the result matrix `C` from GPU to CPU using `cudart.MemcpyDtoH`.  
   Handle any errors during the transfer.  

   ```go
   if err := cudart.MemcpyDtoH(cudart.SliceToHostPtr(C), devC, cudart.SliceBytes(C)); err != nil {
       panic(err)
   }
   ```

   *Note:* The same helpers from step 5 apply here for consistency.

8. **Display the Result**  
   Iterate over the output matrix `C` and print its elements in a formatted manner (e.g., row by row).  

   ```go
   fmt.Println("Result matrix C:")
   for i := range M {
       for j := range N {
           fmt.Printf("%8.1f ", C[i*N+j])
       }
       fmt.Println()
   }
   ```

## Example Output

For the provided matrices `A` and `B`, the resulting matrix `C` is printed as a 4x4 matrix with formatted floating-point values.

```text
Result matrix C:
     5.0      8.0     11.0     14.0 
     8.0     11.0     14.0     17.0
     7.0     10.0     13.0     16.0
    10.0     13.0     16.0     19.0
```
