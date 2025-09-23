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

Expect system info and CUDA device details.
