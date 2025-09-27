// cudatest tests CUDA availability and device info using CUDA Runtime API.
package main

import (
	"fmt"
	"os"

	"github.com/gocnn/gocu/cudart"
)

func main() {
	fmt.Println(`          _____                   _______                   _____                    _____
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
                                                           \/____/                  ~~`)

	// Get CUDA Runtime version
	runtimeVer, err := cudart.RuntimeGetVersion()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error getting runtime version: %v\n", err)
		os.Exit(1)
	}

	// Get CUDA Driver version
	driverVer, err := cudart.DriverGetVersion()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error getting driver version: %v\n", err)
		os.Exit(1)
	}

	count, err := cudart.GetDeviceCount()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error getting device count: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Detected %d CUDA Capable device(s)\n\n", count)

	for i := 0; i < count; i++ {
		dev := cudart.Device(i)
		displayDeviceInfo(dev, i, driverVer, runtimeVer)
	}

	fmt.Println("Result = PASS")
}

func displayDeviceInfo(dev cudart.Device, deviceId int, driverVer, runtimeVer int) {
	// Get device properties
	props, err := cudart.GetDeviceProperties(dev)
	if err != nil {
		fmt.Printf("Device %d: Error getting properties: %v\n", deviceId, err)
		return
	}

	fmt.Printf("Device %d: \"%s\"\n", deviceId, props.Name)

	// General Device Information
	fmt.Println("  General Device Information:")
	fmt.Printf("    CUDA Driver Version / Runtime Version:       %d.%d / %d.%d\n",
		driverVer/1000, (driverVer%1000)/10, runtimeVer/1000, (runtimeVer%1000)/10)
	fmt.Printf("    CUDA Capability Major/Minor Version:         %d.%d\n", props.Major, props.Minor)
	fmt.Printf("    Device UUID:                                 %s\n", props.Uuid.String())
	fmt.Printf("    PCI Domain ID / Bus ID / Device ID:          %d / %d / %d\n", props.PciDomainId, props.PciBusId, props.PciDeviceId)
	fmt.Printf("    LUID / Device Node Mask:                     %s / 0x%08X\n", props.Luid, props.LuidDeviceNodeMask)
	fmt.Printf("    Multi-GPU Board / Group ID:                  %s / %d\n", boolToYesNo(props.IsMultiGpuBoard != 0), props.MultiGpuBoardGroupId)
	fmt.Printf("    Integrated GPU:                              %s\n", boolToYesNo(props.Integrated != 0))
	fmt.Printf("    TCC Driver Mode:                             %s\n", boolToYesNo(props.TccDriver != 0))

	// Compute and Processing Capabilities
	fmt.Println("  Compute and Processing Capabilities:")
	fmt.Printf("    Multiprocessors (SMs):                       %d\n", props.MultiProcessorCount)
	fmt.Printf("    CUDA Cores per SM / Total Cores:             %d / %d\n",
		getCoresPerMP(int32(props.Major), int32(props.Minor)),
		props.MultiProcessorCount*getCoresPerMP(int32(props.Major), int32(props.Minor)))
	fmt.Printf("    Warp Size:                                   %d threads\n", props.WarpSize)
	fmt.Printf("    Max Threads per Block:                       %d\n", props.MaxThreadsPerBlock)
	fmt.Printf("    Max Threads per Multiprocessor:              %d\n", props.MaxThreadsPerMultiProcessor)
	fmt.Printf("    Max Blocks per Multiprocessor:               %d\n", props.MaxBlocksPerMultiProcessor)
	fmt.Printf("    Max Thread Block Dimensions (x,y,z):         (%d, %d, %d)\n",
		props.MaxThreadsDim[0], props.MaxThreadsDim[1], props.MaxThreadsDim[2])
	fmt.Printf("    Max Grid Dimensions (x,y,z):                 (%d, %d, %d)\n",
		props.MaxGridSize[0], props.MaxGridSize[1], props.MaxGridSize[2])
	fmt.Printf("    Registers per Block / per Multiprocessor:    %d / %d\n", props.RegsPerBlock, props.RegsPerMultiprocessor)
	fmt.Printf("    Async Engine Count:                          %d\n", props.AsyncEngineCount)

	// Memory Information
	fmt.Println("  Memory Information:")
	fmt.Printf("    Total Global Memory:                         %.0f MiB (%d bytes)\n",
		float64(props.TotalGlobalMem)/(1024*1024), props.TotalGlobalMem)
	fmt.Printf("    Total Constant Memory:                       %d bytes\n", props.TotalConstMem)
	fmt.Printf("    Shared Memory per Block / Opt-in:            %d / %d bytes\n", props.SharedMemPerBlock, props.SharedMemPerBlockOptin)
	fmt.Printf("    Shared Memory per Multiprocessor:            %d bytes\n", props.SharedMemPerMultiprocessor)
	fmt.Printf("    Reserved Shared Memory per Block:            %d bytes\n", props.ReservedSharedMemPerBlock)
	fmt.Printf("    L2 Cache Size:                               %d KiB\n", props.L2cacheSize/1024)
	fmt.Printf("    Persisting L2 Cache Max Size:                %d bytes\n", props.PersistingL2cacheMaxSize)
	fmt.Printf("    Memory Bus Width:                            %d bits\n", props.MemoryBusWidth)
	fmt.Printf("    Maximum Memory Pitch:                        %d bytes\n", props.MemPitch)

	// Texture and Surface Limits
	fmt.Println("  Texture and Surface Limits:")
	fmt.Printf("    Texture Alignment / Pitch Alignment:         %d / %d bytes\n", props.TextureAlignment, props.TexturePitchAlignment)
	fmt.Printf("    Surface Alignment:                           %d bytes\n", props.SurfaceAlignment)
	fmt.Printf("    Max Texture 1D / 1D Mipmap / 1D Layered:     %d / %d / (%d, %d)\n",
		props.MaxTexture1d, props.MaxTexture1dmipmap, props.MaxTexture1dlayered[0], props.MaxTexture1dlayered[1])
	fmt.Printf("    Max Texture 2D / 2D Mipmap / 2D Gather:      (%d, %d) / (%d, %d) / (%d, %d)\n",
		props.MaxTexture2d[0], props.MaxTexture2d[1],
		props.MaxTexture2dmipmap[0], props.MaxTexture2dmipmap[1],
		props.MaxTexture2dgather[0], props.MaxTexture2dgather[1])
	fmt.Printf("    Max Texture 2D Layered / 2D Linear:          (%d, %d, %d) / (%d, %d, %d)\n",
		props.MaxTexture2dlayered[0], props.MaxTexture2dlayered[1], props.MaxTexture2dlayered[2],
		props.MaxTexture2dlinear[0], props.MaxTexture2dlinear[1], props.MaxTexture2dlinear[2])
	fmt.Printf("    Max Texture 3D / 3D Alternate:               (%d, %d, %d) / (%d, %d, %d)\n",
		props.MaxTexture3d[0], props.MaxTexture3d[1], props.MaxTexture3d[2],
		props.MaxTexture3dalt[0], props.MaxTexture3dalt[1], props.MaxTexture3dalt[2])
	fmt.Printf("    Max Texture Cubemap / Layered:               %d / (%d, %d)\n",
		props.MaxTextureCubemap, props.MaxTextureCubemapLayered[0], props.MaxTextureCubemapLayered[1])
	fmt.Printf("    Max Surface 1D / 1D Layered:                 %d / (%d, %d)\n",
		props.MaxSurface1d, props.MaxSurface1dlayered[0], props.MaxSurface1dlayered[1])
	fmt.Printf("    Max Surface 2D / 2D Layered:                 (%d, %d) / (%d, %d, %d)\n",
		props.MaxSurface2d[0], props.MaxSurface2d[1],
		props.MaxSurface2dlayered[0], props.MaxSurface2dlayered[1], props.MaxSurface2dlayered[2])
	fmt.Printf("    Max Surface 3D:                              (%d, %d, %d)\n",
		props.MaxSurface3d[0], props.MaxSurface3d[1], props.MaxSurface3d[2])
	fmt.Printf("    Max Surface Cubemap / Layered:               %d / (%d, %d)\n",
		props.MaxSurfaceCubemap, props.MaxSurfaceCubemapLayered[0], props.MaxSurfaceCubemapLayered[1])

	// Feature Support
	fmt.Println("  Feature Support:")
	fmt.Printf("    ECC Enabled:                                 %s\n", boolToYesNo(props.Eccenabled != 0))
	fmt.Printf("    Unified Addressing:                          %s\n", boolToYesNo(props.UnifiedAddressing != 0))
	fmt.Printf("    Managed Memory:                              %s\n", boolToYesNo(props.ManagedMemory != 0))
	fmt.Printf("    Concurrent Managed Access:                   %s\n", boolToYesNo(props.ConcurrentManagedAccess != 0))
	fmt.Printf("    Direct Managed Memory Access from Host:      %s\n", boolToYesNo(props.DirectManagedMemAccessFromHost != 0))
	fmt.Printf("    Pageable Memory Access:                      %s\n", boolToYesNo(props.PageableMemoryAccess != 0))
	fmt.Printf("    Pageable Memory Uses Host Page Tables:       %s\n", boolToYesNo(props.PageableMemoryAccessUsesHostPageTables != 0))
	fmt.Printf("    Can Map Host Memory:                         %s\n", boolToYesNo(props.CanMapHostMemory != 0))
	fmt.Printf("    Can Use Host Pointer for Registered Mem:     %s\n", boolToYesNo(props.CanUseHostPointerForRegisteredMem != 0))
	fmt.Printf("    Host Native Atomic Supported:                %s\n", boolToYesNo(props.HostNativeAtomicSupported != 0))
	fmt.Printf("    Global L1 Cache Supported:                   %s\n", boolToYesNo(props.GlobalL1cacheSupported != 0))
	fmt.Printf("    Local L1 Cache Supported:                    %s\n", boolToYesNo(props.LocalL1cacheSupported != 0))
	fmt.Printf("    Concurrent Kernels:                          %s\n", boolToYesNo(props.ConcurrentKernels != 0))
	fmt.Printf("    Cooperative Launch:                          %s\n", boolToYesNo(props.CooperativeLaunch != 0))
	fmt.Printf("    Compute Preemption Supported:                %s\n", boolToYesNo(props.ComputePreemptionSupported != 0))
	fmt.Printf("    Stream Priorities Supported:                 %s\n", boolToYesNo(props.StreamPrioritiesSupported != 0))
	fmt.Printf("    Access Policy Max Window Size:               %d\n", props.AccessPolicyMaxWindowSize)

	fmt.Println()
}

// getCoresPerMP returns the number of CUDA cores per multiprocessor for different compute capabilities
func getCoresPerMP(major, minor int32) int {
	switch major {
	case 2: // Fermi
		if minor == 1 {
			return 48
		}
		return 32
	case 3: // Kepler
		return 192
	case 5: // Maxwell
		return 128
	case 6: // Pascal
		if minor == 0 {
			return 64
		}
		return 128
	case 7: // Volta, Turing
		if minor == 0 {
			return 64
		}
		return 64
	case 8: // Ampere
		if minor == 0 {
			return 64
		}
		return 128
	case 9: // Ada Lovelace, Hopper
		return 128
	default:
		return 128 // Default for unknown architectures
	}
}

// boolToYesNo converts boolean to "Yes"/"No" string
func boolToYesNo(b bool) string {
	if b {
		return "Yes"
	}
	return "No"
}
