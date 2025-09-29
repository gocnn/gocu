# JIT Compilation with CUDA in Go

This README provides a step-by-step guide to Just-In-Time (JIT) compilation and execution of CUDA kernels in Go using the `gocnn/gocu` library. The example demonstrates loading a PTX kernel at runtime and executing vector addition on the GPU.

## Prerequisites

- Go installed (version 1.16 or later recommended).
- CUDA toolkit installed with a compatible NVIDIA GPU.
- `gocnn/gocu` library installed (`go get github.com/gocnn/gocu`).
- Basic understanding of Go programming and CUDA kernels.

## Compile CUDA Kernel to PTX

Before running the Go program, compile the CUDA kernel source file to PTX format:

```bash
nvcc -ptx add.cu -o add.ptx
```

This command compiles `add.cu` to `add.ptx` which will be embedded into the Go binary at compile time.

## Steps

1. **Set Up Package and Imports**  
   Define the main package and import necessary libraries: `fmt` for output, `unsafe` for pointer operations, and `github.com/gocnn/gocu` for CUDA Driver API functions.  

   ```go
   package main
   
   import (
       _ "embed"
       "fmt"
       "unsafe"
   
       "github.com/gocnn/gocu"
   )
   ```

2. **Embed PTX Kernel**  
   Use Go's `//go:embed` directive to embed the compiled PTX file directly into the binary. This eliminates the need for external file dependencies at runtime.  

   ```go
   //go:embed add.ptx
   var ptx string
   ```

3. **Initialize CUDA Context**  
   Get the first CUDA device and create a context for GPU operations. The context manages GPU resources and must be destroyed when finished.  

   ```go
   var n int64 = 1024
   
   device, _ := gocu.DeviceGet(0)
   ctx, _ := gocu.CtxCreate(gocu.CtxSchedAuto, device)
   defer ctx.Destroy()
   ```

4. **Load PTX Module**  
   Load the PTX kernel code into a CUDA module using `ModuleLoadData`. The module acts as a container for GPU functions and must be unloaded when finished.  

   ```go
   module, _ := gocu.ModuleLoadData([]byte(ptx))
   defer module.Unload()
   ```

5. **Get Kernel Function**  
   Retrieve the kernel function from the module using its mangled name. The name `_Z3addPiPKiS1_i` is the C++ mangled version of the `add` function.  
   *Note: To use a simple name like "add", compile the kernel with `extern "C"` linkage.*  

   ```go
   function, _ := module.GetFunction("add")
   ```

6. **Prepare Host Data**  
   Create input arrays `A` and `B` with test data, and an output array `C` for results. Initialize the input arrays with sequential values for verification.  

   ```go
   hostA := make([]int32, n)
   hostB := make([]int32, n)
   hostC := make([]int32, n)
   
   for i := range hostA {
       hostA[i] = int32(i)
       hostB[i] = int32(i * 2)
   }
   ```

7. **Allocate GPU Memory**  
   Allocate device memory for all three arrays using `gocu.MemAlloc`. Calculate the total size in bytes (4 bytes per `int32`).  
   Defer memory cleanup to ensure proper resource management.  

   ```go
   size := int64(n * 4)
   devA, _ := gocu.MemAlloc(size)
   devB, _ := gocu.MemAlloc(size)
   devC, _ := gocu.MemAlloc(size)
   defer gocu.MemFree(devA)
   defer gocu.MemFree(devB)
   defer gocu.MemFree(devC)
   ```

8. **Copy Data to GPU**  
   Transfer input arrays from host to device memory using `gocu.MemcpyHtoD`. Use `unsafe.Pointer` to convert Go slice pointers to raw memory addresses.  

   ```go
   gocu.MemcpyHtoD(devA, unsafe.Pointer(&hostA[0]), size)
   gocu.MemcpyHtoD(devB, unsafe.Pointer(&hostB[0]), size)
   ```

9. **Prepare Kernel Arguments**  
   Create an array of `unsafe.Pointer` containing addresses of all kernel parameters. The order must match the kernel function signature.  

   ```go
   args := []unsafe.Pointer{
       unsafe.Pointer(&devC),  // output array
       unsafe.Pointer(&devA),  // input array A
       unsafe.Pointer(&devB),  // input array B
       unsafe.Pointer(&n),     // array size
   }
   ```

10. **Launch Kernel**  
    Execute the kernel using `gocu.LaunchKernel` with grid and block dimensions. Calculate grid size to ensure all elements are processed: `(n+255)/256` blocks of 256 threads each.  

    ```go
    gocu.LaunchKernel(
        function,
        (uint32)(n+255)/256, 1, 1,  // grid dimensions
        256, 1, 1,                  // block dimensions
        0,                          // shared memory size
        gocu.Stream{},              // default stream
        args,                       // kernel arguments
        nil,                        // extra parameters
    )
    ```

11. **Synchronize and Copy Results**  
    Wait for kernel completion using `gocu.CtxSynchronize`, then copy results back to host memory using `gocu.MemcpyDtoH`.  

    ```go
    gocu.CtxSynchronize()
    gocu.MemcpyDtoH(unsafe.Pointer(&hostC[0]), devC, size)
    ```

12. **Display Results**  
    Print the first 10 results to verify correct computation. Each element should equal `A[i] + B[i]`.  

    ```go
    fmt.Println("Results:")
    for i := range 10 {
        fmt.Printf("%d + %d = %d\n", hostA[i], hostB[i], hostC[i])
    }
    ```

## Key Concepts

- **JIT Compilation**: PTX code is loaded and compiled at runtime, enabling dynamic kernel deployment.
- **PTX Embedding**: Using `//go:embed` eliminates external file dependencies.
- **Memory Management**: Proper allocation, copying, and cleanup of GPU memory.
- **Kernel Launch**: Grid and block dimensions determine parallel execution strategy.
- **Synchronization**: Essential for ensuring kernel completion before accessing results.

## Expected Output

```sh
Results:
0 + 0 = 0
1 + 2 = 3
2 + 4 = 6
3 + 6 = 9
4 + 8 = 12
5 + 10 = 15
6 + 12 = 18
7 + 14 = 21
8 + 16 = 24
9 + 18 = 27
```

## Notes

- The kernel function name `_Z3addPiPKiS1_i` is C++ name mangling. Use `extern "C"` in CUDA source to get simple names.
- Grid size calculation `(n+255)/256` ensures all array elements are processed even when `n` is not divisible by 256.
- Always synchronize before accessing GPU computation results on the host.
