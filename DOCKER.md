# Docker (Recommended for Quick Start)

## Image Contents

The GOCU Docker image is based on `nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04` and includes:

- **Ubuntu 24.04**: Latest LTS Ubuntu base system
- **CUDA 13.0.1**: Full CUDA development environment with cuDNN
- **Development Tools**: Golang-Go, OpenBLAS, build-essential, and GOCU detection program

This provides a complete, ready-to-use environment for CUDA development with Go.

## Prerequisites

- NVIDIA GPU with updated drivers (CUDA 13.0+ compatible)
- Docker installed

## Quick Test

Test GOCU functionality with a single command:

```bash
docker run --gpus all -it ghcr.io/gocnn/gocu:latest
```

If successful, you should see the GOCU ASCII art and your GPU information, confirming that GOCU is working correctly in the containerized environment.

## Development Environment

### Windows (PowerShell)

```powershell
docker run --gpus all -v ${PWD}:/app -it ghcr.io/gocnn/gocu:latest /bin/bash
```

### Linux

```bash
docker run --gpus all -v $(pwd):/app -it ghcr.io/gocnn/gocu:latest /bin/bash
```

## Build Your Own Image

You can also build the Docker image yourself using the [`Dockerfile`](Dockerfile) in the root directory:

```bash
# Clone the repository
git clone https://github.com/gocnn/gocu.git
cd gocu

# Build the Docker image
docker build -t gocu:local .

# Test the locally built image
docker run --gpus all -it gocu:local
