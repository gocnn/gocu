// cudatest tests CUDA availability and device info.
package main

import (
	"fmt"
	"os"

	"github.com/gocnn/gocu"
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

	ver := gocu.Version()
	fmt.Printf("Detected %d CUDA Capable device(s)\n\n", getDeviceCount())

	count, err := gocu.DeviceGetCount()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error getting device count: %v\n", err)
		os.Exit(1)
	}

	for i := range count {
		dev := gocu.Device(i)
		displayDeviceInfo(dev, i, ver)
	}
}

func getDeviceCount() int {
	count, _ := gocu.DeviceGetCount()
	return count
}

func displayDeviceInfo(dev gocu.Device, deviceId int, cudaVersion int) {
	name, err := dev.Name()
	if err != nil {
		name = "Unknown Device"
	}

	fmt.Printf("Device %d: \"%s\"\n", deviceId, name)

	// CUDA Driver/Runtime Version
	fmt.Printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
		cudaVersion/1000, (cudaVersion%1000)/10, cudaVersion/1000, (cudaVersion%1000)/10)

	// Compute Capability
	maj, _ := dev.Attribute(gocu.ComputeCapabilityMajor)
	min, _ := dev.Attribute(gocu.ComputeCapabilityMinor)
	fmt.Printf("  CUDA Capability Major/Minor version number:    %d.%d\n", maj, min)

	// Total Global Memory
	mem, _ := dev.TotalMem()
	fmt.Printf("  Total amount of global memory:                 %.0f MBytes (%d bytes)\n",
		float64(mem)/(1024*1024), mem)

	// GPU Clock Rate
	clockRate, _ := dev.Attribute(gocu.ClockRate)
	fmt.Printf("  GPU Clock rate:                                %d MHz (%.2f GHz)\n",
		clockRate/1000, float64(clockRate)/1000000)

	// Memory Clock Rate
	memClockRate, _ := dev.Attribute(gocu.MemoryClockRate)
	fmt.Printf("  Memory Clock rate:                             %d Mhz\n", memClockRate/1000)

	// Memory Bus Width
	memBusWidth, _ := dev.Attribute(gocu.GlobalMemoryBusWidth)
	fmt.Printf("  Memory Bus Width:                              %d-bit\n", memBusWidth)

	// L2 Cache Size
	l2CacheSize, _ := dev.Attribute(gocu.L2CacheSize)
	fmt.Printf("  L2 Cache Size:                                 %d bytes\n", l2CacheSize)

	// Texture Dimensions
	tex1D, _ := dev.Attribute(gocu.MaximumTexture1dWidth)
	tex2DW, _ := dev.Attribute(gocu.MaximumTexture2dWidth)
	tex2DH, _ := dev.Attribute(gocu.MaximumTexture2dHeight)
	tex3DW, _ := dev.Attribute(gocu.MaximumTexture3dWidth)
	tex3DH, _ := dev.Attribute(gocu.MaximumTexture3dHeight)
	tex3DD, _ := dev.Attribute(gocu.MaximumTexture3dDepth)
	fmt.Printf("  Max Texture Dimension Size (x,y,z)             1D=(%d), 2D=(%d, %d), 3D=(%d, %d, %d)\n",
		tex1D, tex2DW, tex2DH, tex3DW, tex3DH, tex3DD)

	// Layered Texture Size
	tex2DLayeredW, _ := dev.Attribute(gocu.MaximumTexture2dLayeredWidth)
	tex2DLayeredH, _ := dev.Attribute(gocu.MaximumTexture2dLayeredHeight)
	tex2DLayeredL, _ := dev.Attribute(gocu.MaximumTexture2dLayeredLayers)
	fmt.Printf("  Max Layered Texture Size (dim) x layers        1D=(%d) x %d, 2D=(%d, %d) x %d\n",
		tex1D, 2048, tex2DLayeredW, tex2DLayeredH, tex2DLayeredL)

	// Total Constant Memory
	constMem, _ := dev.Attribute(gocu.TotalConstantMemory)
	fmt.Printf("  Total amount of constant memory:               %d bytes\n", constMem)

	// Shared Memory per Block
	sharedMem, _ := dev.Attribute(gocu.MaxSharedMemoryPerBlock)
	fmt.Printf("  Total amount of shared memory per block:       %d bytes\n", sharedMem)

	// Registers per Block
	registers, _ := dev.Attribute(gocu.MaxRegistersPerBlock)
	fmt.Printf("  Total number of registers available per block: %d\n", registers)

	// Warp Size
	warpSize, _ := dev.Attribute(gocu.WarpSize)
	fmt.Printf("  Warp size:                                     %d\n", warpSize)

	// Maximum Threads per Multiprocessor
	maxThreadsMP, _ := dev.Attribute(gocu.MaxThreadsPerMultiprocessor)
	fmt.Printf("  Maximum number of threads per multiprocessor:  %d\n", maxThreadsMP)

	// Maximum Threads per Block
	maxThreadsBlock, _ := dev.Attribute(gocu.MaxThreadsPerBlock)
	fmt.Printf("  Maximum number of threads per block:           %d\n", maxThreadsBlock)

	// Maximum Block Dimensions
	maxBlockX, _ := dev.Attribute(gocu.MaxBlockDimX)
	maxBlockY, _ := dev.Attribute(gocu.MaxBlockDimY)
	maxBlockZ, _ := dev.Attribute(gocu.MaxBlockDimZ)
	fmt.Printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
		maxBlockX, maxBlockY, maxBlockZ)

	// Maximum Grid Dimensions
	maxGridX, _ := dev.Attribute(gocu.MaxGridDimX)
	maxGridY, _ := dev.Attribute(gocu.MaxGridDimY)
	maxGridZ, _ := dev.Attribute(gocu.MaxGridDimZ)
	fmt.Printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
		maxGridX, maxGridY, maxGridZ)

	// Maximum Memory Pitch
	maxPitch, _ := dev.Attribute(gocu.MaxPitch)
	fmt.Printf("  Maximum memory pitch:                          %d bytes\n", maxPitch)

	fmt.Println()
}
