# GOCU Installation Guide

A minimalist guide for installing GOCU, a Go library for NVIDIA GPU computing with CUDA Runtime API and CUBLAS bindings.
> **Quick Start with Docker**: For a hassle-free setup, consider using our pre-configured Docker environment. See [DOCKER.md](DOCKER.md) for details.

## System Requirements

- **NVIDIA GPU**: Required
- **OS**:
  - Windows: Fully supported
  - Linux: Fully supported
- **CUDA Toolkit**:
  - 11.x: Supported
  - 12.x: Supported
  - 13.0: Supported (Only supported on Linux)

## Prerequisites

1. **C Compiler (Required for CGO)**:

   **Windows**:
   - Install MinGW-w64 via Chocolatey (recommended):

     ```cmd
     choco install mingw
     ```

   - Or use MSYS2: [msys2.org](https://www.msys2.org/)

   - Ensure `gcc` is in your PATH: `gcc --version`

   **Linux**:
   - Ubuntu/Debian: `sudo apt install build-essential`
   - CentOS/RHEL: `sudo yum groupinstall "Development Tools"`
   - Arch: `sudo pacman -S base-devel`

2. **CUDA Toolkit**:
   - Download: [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
   - Install per NVIDIA's instructions for your OS.

   **⚠️ Important**: CUDA Toolkit version must be compatible with your CUDA Driver version，see [compatibility table.](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions)

3. **cuDNN (Optional)**:

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
   go install github.com/gocnn/gocu/cmd/gocu@latest
   gocu
   ```

> **Windows Users**: Run PowerShell as Administrator and execute this script to auto-detect GOCU dependencies and create symbolic links, simplifying setup. If checks pass, skip manual link creation and use GOCU directly:
>
> ```powershell
> Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://raw.githubusercontent.com/gocnn/gocu/main/setup.ps1'))
> ```

### Linux

1. Install CUDA Toolkit per NVIDIA's instructions.
2. Install and test GOCU:

   ```bash
   go install github.com/gocnn/gocu/cmd/gocu@latest
   gocu
   ```

   CUDA detection is automatic.

## Custom Linking

For non-standard CUDA paths:

```bash
export CGO_CFLAGS="-I/your/custom/cuda/include"
export CGO_LDFLAGS="-L/your/custom/cuda/lib64"
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

Device 0: "NVIDIA GeForce RTX 4090"
  CUDA Driver Version / Runtime Version          13.0 / 11.8
  CUDA Capability Major/Minor version number:    8.9
  Total amount of global memory:                 24081 MBytes (25250627584 bytes)
  (128) Multiprocessors, (128) CUDA Cores/MP:     16384 CUDA Cores
  GPU Max Clock rate:                            2520 MHz (2.52 GHz)
  Memory Clock rate:                             10501 Mhz
  Memory Bus Width:                              384-bit
  L2 Cache Size:                                 75497472 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  1536
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 2 copy engine(s)
  Run time limit on kernels:                     No
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        No
  CUDA Device Driver Mode (TCC or WDDM):         Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)

Result = PASS
```

This output confirms that:

- CUDA is properly installed and detected
- GOCU can successfully interface with CUDA drivers
- Your GPU specifications are correctly identified
- All device attributes are accessible

If you see this output, GOCU is installed and working correctly. You can now proceed to use GOCU in your projects.
