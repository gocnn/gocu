# GOCU Installation Guide

A minimalist guide for installing GOCU, a Go library for NVIDIA GPU computing with CUDA Runtime API and CUBLAS bindings.
> **Quick Start with Docker**: For a hassle-free setup, consider using our pre-configured Docker environment. See [DOCKER.md](DOCKER.md) for details.

## System Requirements

- **NVIDIA GPU**: Required
- **OS**:
  - Windows: Fully supported
  - Linux: Fully supported
- **CUDA Toolkit**:
  - 10.x and below: Untested (you may test at your own risk)
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

> **Windows Users**: Run PowerShell as Administrator and execute this script to auto-detect GOCU dependencies and create symbolic links, simplifying setup. If checks pass, skip manual link creation and use GOCU directly:
>
> ```powershell
> Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://raw.githubusercontent.com/gocnn/gocu/main/setup.ps1'))
> ```

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
  General Device Information:
    CUDA Driver Version / Runtime Version:       13.0 / 13.0
    CUDA Capability Major/Minor Version:         8.9
    Device UUID:                                 aacd8952-e80e-f6cb-810a-86ca6aa7e7c5
    PCI Domain ID / Bus ID / Device ID:          0 / 56 / 0
    LUID / Device Node Mask:                      / 0x00000000
    Multi-GPU Board / Group ID:                  No / 0
    Integrated GPU:                              No
    TCC Driver Mode:                             No
  Compute and Processing Capabilities:
    Multiprocessors (SMs):                       128
    CUDA Cores per SM / Total Cores:             128 / 16384
    Warp Size:                                   32 threads
    Max Threads per Block:                       1024
    Max Threads per Multiprocessor:              1536
    Max Blocks per Multiprocessor:               24
    Max Thread Block Dimensions (x,y,z):         (1024, 1024, 64)
    Max Grid Dimensions (x,y,z):                 (2147483647, 65535, 65535)
    Registers per Block / per Multiprocessor:    65536 / 65536
    Async Engine Count:                          2
  Memory Information:
    Total Global Memory:                         24081 MiB (25250627584 bytes)
    Total Constant Memory:                       65536 bytes
    Shared Memory per Block / Opt-in:            49152 / 101376 bytes
    Shared Memory per Multiprocessor:            102400 bytes
    Reserved Shared Memory per Block:            1024 bytes
    L2 Cache Size:                               73728 KiB
    Persisting L2 Cache Max Size:                51904512 bytes
    Memory Bus Width:                            384 bits
    Maximum Memory Pitch:                        2147483647 bytes
  Texture and Surface Limits:
    Texture Alignment / Pitch Alignment:         512 / 32 bytes
    Surface Alignment:                           512 bytes
    Max Texture 1D / 1D Mipmap / 1D Layered:     131072 / 32768 / (32768, 2048)
    Max Texture 2D / 2D Mipmap / 2D Gather:      (131072, 65536) / (32768, 32768) / (32768, 32768)
    Max Texture 2D Layered / 2D Linear:          (32768, 32768, 2048) / (131072, 65000, 2097120)
    Max Texture 3D / 3D Alternate:               (16384, 16384, 16384) / (8192, 8192, 32768)
    Max Texture Cubemap / Layered:               32768 / (32768, 2046)
    Max Surface 1D / 1D Layered:                 32768 / (32768, 2048)
    Max Surface 2D / 2D Layered:                 (131072, 65536) / (32768, 32768, 2048)
    Max Surface 3D:                              (16384, 16384, 16384)
    Max Surface Cubemap / Layered:               32768 / (32768, 2046)
  Feature Support:
    ECC Enabled:                                 No
    Unified Addressing:                          Yes
    Managed Memory:                              Yes
    Concurrent Managed Access:                   Yes
    Direct Managed Memory Access from Host:      No
    Pageable Memory Access:                      No
    Pageable Memory Uses Host Page Tables:       No
    Can Map Host Memory:                         Yes
    Can Use Host Pointer for Registered Mem:     Yes
    Host Native Atomic Supported:                No
    Global L1 Cache Supported:                   Yes
    Local L1 Cache Supported:                    Yes
    Concurrent Kernels:                          Yes
    Cooperative Launch:                          Yes
    Compute Preemption Supported:                Yes
    Stream Priorities Supported:                 Yes
    Access Policy Max Window Size:               134213632

Result = PASS
```

This output confirms that:

- CUDA is properly installed and detected
- GOCU can successfully interface with CUDA drivers
- Your GPU specifications are correctly identified
- All device attributes are accessible

If you see this output, GOCU is installed and working correctly. You can now proceed to use GOCU in your projects.
