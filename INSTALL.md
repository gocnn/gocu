# GOCU Installation Guide

A minimalist guide for installing GOCU, a Go library for NVIDIA GPU computing with CUDA Runtime API and CUBLAS bindings.

## System Requirements

- **NVIDIA GPU**: Required
- **OS**:
  - Windows: Fully supported
  - Linux: Fully supported
  - macOS: Limited support (untested on older NVIDIA GPU systems)
- **CUDA Toolkit**:
  - 11.x: Supported
  - 12.x: Supported
  - 13.0: Unsupported (linking issues)

## Prerequisites

1. **CUDA Toolkit**:
   - Download: [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
   - Install per NVIDIA's instructions for your OS.

2. **cuDNN (Optional)**:
   - For deep learning, download: [developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn)
   - Match version to CUDA Toolkit and follow NVIDIA's guide.

## Installation

### Windows

1. Install CUDA Toolkit per NVIDIA's instructions.
2. Create symbolic link:

   ```cmd
   mklink /D C:\cuda "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0"
   ```

3. Install and test GOCU:

   ```bash
   go install github.com/gocnn/gocu/cmd/gocu
   gocu
   ```

### Linux

1. Install CUDA Toolkit per NVIDIA's instructions.
2. Install and test GOCU:

   ```bash
   go install github.com/gocnn/gocu/cmd/gocu
   gocu
   ```

   CUDA detection is automatic.

## Custom Linking

For non-standard CUDA paths:

```bash
export CGO_CFLAGS="-I/your/custom/cuda/include"
export CGO_LDFLAGS="-L/your/custom/cuda/lib64"
go build your-app.go
```

Or set permanently:

```bash
go env -w CGO_CFLAGS="-I/your/custom/cuda/include"
go env -w CGO_LDFLAGS="-L/your/custom/cuda/lib64"
```

## Verification

Run:

```bash
gocu
```

If successful, you should see output similar to:

```bash
$ gocu
          _____                   _______                   _____                    _____
         /\    \                 /::\    \                 /\    \                  /\    \
        /::\    \               /::::\    \               /::\    \                /::\____\
       /::::\    \             /::::::\    \             /::::\    \              /:::/    /
      /::::::\    \           /::::::::\    \           /::::::\    \            /:::/    /
     /:::/\:::\    \         /:::/~~\:::\    \         /:::/\:::\    \          /:::/    /
    /:::/  \:::\    \       /:::/    \:::\    \       /:::/  \:::\    \        /:::/    /
   /:::/    \:::\    \     /:::/    / \:::\    \     /:::/    \:::\    \      /:::/    /
  /:::/    / \:::\    \   /:::/____/   \:::\____\   /:::/    / \:::\    \    /:::/    /      _____
 /:::/    /   \:::\ ___\ |:::|    |     |:::|    | /:::/    /   \:::\    \  /:::/____/      /\    \
/:::/____/  ___\:::|    ||:::|____|     |:::|    |/:::/____/     \:::\____\|:::|    /      /::\____\
\:::\    \ /\  /:::|____| \:::\    \   /:::/    / \:::\    \      \::/    /|:::|____\     /:::/    /
 \:::\    /::\ \::/    /   \:::\    \ /:::/    /   \:::\    \      \/____/  \:::\    \   /:::/    /
  \:::\   \:::\ \/____/     \:::\    /:::/    /     \:::\    \               \:::\    \ /:::/    /
   \:::\   \:::\____\        \:::\__/:::/    /       \:::\    \               \:::\    /:::/    /
    \:::\  /:::/    /         \::::::::/    /         \:::\    \               \:::\__/:::/    /
     \:::\/:::/    /           \::::::/    /           \:::\    \               \::::::::/    /
      \::::::/    /             \::::/    /             \:::\    \               \::::::/    /
       \::::/    /               \::/____/               \:::\____\               \::::/    /
        \::/____/                 ~~                      \::/    /                \::/____/
                                                           \/____/                  ~~
Detected 1 CUDA Capable device(s)

Device 0: "NVIDIA GeForce RTX 4060 Ti"
  CUDA Driver Version / Runtime Version          13.0 / 13.0
  CUDA Capability Major/Minor version number:    8.9
  Total amount of global memory:                 16380 MBytes (17175150592 bytes)
  GPU Clock rate:                                2535 MHz (2.54 GHz)
  Memory Clock rate:                             9001 Mhz
  Memory Bus Width:                              128-bit
  L2 Cache Size:                                 33554432 bytes
  Max Texture Dimension Size (x,y,z)             1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Max Layered Texture Size (dim) x layers        1D=(131072) x 2048, 2D=(32768, 32768) x 2048
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  1536
  Maximum number of threads per block:           1024
  Maximum sizes of each dimension of a block:    1024 x 1024 x 64
  Maximum sizes of each dimension of a grid:     2147483647 x 65535 x 65535
  Maximum memory pitch:                          2147483647 bytes
```

This output confirms that:

- CUDA is properly installed and detected
- GOCU can successfully interface with CUDA drivers
- Your GPU specifications are correctly identified
- All device attributes are accessible

If you see this output, GOCU is installed and working correctly. You can now proceed to use GOCU in your projects.
