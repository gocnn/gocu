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

	// CUDA Driver/Runtime Version
	fmt.Printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
		driverVer/1000, (driverVer%1000)/10, runtimeVer/1000, (runtimeVer%1000)/10)

	// CUDA Capability
	fmt.Printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
		props.Major, props.Minor)

	// Total Global Memory
	fmt.Printf("  Total amount of global memory:                 %.0f MBytes (%d bytes)\n",
		float64(props.TotalGlobalMem)/(1024*1024), props.TotalGlobalMem)

	// Multiprocessor Count
	fmt.Printf("  (%2d) Multiprocessors, (%3d) CUDA Cores/MP:     %d CUDA Cores\n",
		props.MultiProcessorCount, getCoresPerMP(int32(props.Major), int32(props.Minor)),
		int(props.MultiProcessorCount)*getCoresPerMP(int32(props.Major), int32(props.Minor)))

	// L2 Cache Size
	fmt.Printf("  L2 Cache Size:                                 %d bytes\n",
		props.L2cacheSize)

	// Maximum Texture Dimensions
	fmt.Printf("  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, %d), 3D=(%d, %d, %d)\n",
		props.MaxTexture1d, props.MaxTexture2d[0], props.MaxTexture2d[1],
		props.MaxTexture3d[0], props.MaxTexture3d[1], props.MaxTexture3d[2])

	// Total Constant Memory
	fmt.Printf("  Total amount of constant memory:               %d bytes\n",
		props.TotalConstMem)

	// Shared Memory per Block
	fmt.Printf("  Total amount of shared memory per block:       %d bytes\n",
		props.SharedMemPerBlock)

	// Registers per Block
	fmt.Printf("  Total number of registers available per block: %d\n",
		props.RegsPerBlock)

	// Warp Size
	fmt.Printf("  Warp size:                                     %d\n",
		props.WarpSize)

	// Maximum Threads per Multiprocessor
	fmt.Printf("  Maximum number of threads per multiprocessor:  %d\n",
		props.MaxThreadsPerMultiProcessor)

	// Maximum Threads per Block
	fmt.Printf("  Maximum number of threads per block:           %d\n",
		props.MaxThreadsPerBlock)

	// Maximum Block Dimensions
	fmt.Printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
		props.MaxThreadsDim[0], props.MaxThreadsDim[1], props.MaxThreadsDim[2])

	// Maximum Grid Dimensions
	fmt.Printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
		props.MaxGridSize[0], props.MaxGridSize[1], props.MaxGridSize[2])

	// Maximum Memory Pitch
	fmt.Printf("  Maximum memory pitch:                          %d bytes\n",
		props.MemPitch)

	// Texture Alignment
	fmt.Printf("  Texture alignment:                             %d bytes\n",
		props.TextureAlignment)

	// Integrated GPU
	if props.Integrated != 0 {
		fmt.Printf("  Integrated GPU sharing Host Memory:            Yes\n")
	} else {
		fmt.Printf("  Integrated GPU sharing Host Memory:            No\n")
	}

	// Host Memory Mapping
	if props.CanMapHostMemory != 0 {
		fmt.Printf("  Support host page-locked memory mapping:       Yes\n")
	} else {
		fmt.Printf("  Support host page-locked memory mapping:       No\n")
	}

	// Alignment requirement for Surfaces
	fmt.Printf("  Alignment requirement for Surfaces:            %s\n",
		boolToYesNo(props.SurfaceAlignment != 0))

	// Concurrent Kernel Execution
	fmt.Printf("  Device has ECC support:                        %s\n",
		boolToYesNo(props.Eccenabled != 0))

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
