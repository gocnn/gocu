# gocu

Go bindings for the CUDA Driver and Runtime APIs, cuBLAS, and cuDNN.

## Quick Start

See [example/cudblas/gemm](example/cudblas/gemm/) for a complete GPU-accelerated matrix multiplication example using CUBLAS.

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
