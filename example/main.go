package main

import (
	"fmt"
	"log"
	"unsafe"

	"github.com/gocnn/gocu"
)

func main() {
	fmt.Println("CUDA Go Example - Device Information and Memory Operations")
	fmt.Println("=========================================================")

	// Initialize CUDA
	gocu.Init(0)
	fmt.Println("✓ CUDA initialized successfully")

	// Get number of devices
	deviceCount := gocu.DeviceGetCount()
	fmt.Printf("✓ Found %d CUDA device(s)\n", deviceCount)

	if deviceCount == 0 {
		log.Fatal("No CUDA devices found")
	}

	// Get first device
	device := gocu.DeviceGet(0)
	fmt.Printf("✓ Using device 0: %s\n", device.Name())

	// Get device compute capability
	major, minor := device.ComputeCapability()
	fmt.Printf("✓ Compute capability: %d.%d\n", major, minor)

	// Get device properties
	props := device.Properties()
	fmt.Printf("✓ Device properties:\n")
	fmt.Printf("  - Max threads per block: %d\n", props.MaxThreadsPerBlock)
	fmt.Printf("  - Max block dimensions: [%d, %d, %d]\n",
		props.MaxThreadsDim[0], props.MaxThreadsDim[1], props.MaxThreadsDim[2])
	fmt.Printf("  - Max grid dimensions: [%d, %d, %d]\n",
		props.MaxGridSize[0], props.MaxGridSize[1], props.MaxGridSize[2])
	fmt.Printf("  - Shared memory per block: %d bytes\n", props.SharedMemPerBlock)
	fmt.Printf("  - Total constant memory: %d bytes\n", props.TotalConstantMemory)
	fmt.Printf("  - Warp size: %d\n", props.SIMDWidth)
	fmt.Printf("  - Clock rate: %d kHz\n", props.ClockRate)

	// Create CUDA context
	ctx := gocu.CtxCreate(gocu.CTX_SCHED_AUTO, device)
	fmt.Println("✓ CUDA context created")

	// Demonstrate memory operations
	demonstrateMemoryOperations()

	// Clean up
	ctx.Destroy()
	fmt.Println("✓ CUDA context destroyed")
	fmt.Println("✓ Example completed successfully!")
}

func demonstrateMemoryOperations() {
	fmt.Println("\n--- Basic Memory Operations ---")

	// Allocate device memory (1024 floats = 4096 bytes)
	const size = 1024
	const byteSize = size * 4 // 4 bytes per float32

	devicePtr := gocu.MemAlloc(byteSize)
	fmt.Printf("✓ Allocated %d bytes on device at address 0x%x\n", byteSize, uintptr(devicePtr))

	// Create host data
	hostData := make([]float32, size)
	for i := range hostData {
		hostData[i] = float32(i) * 0.5
	}
	fmt.Printf("✓ Created host data with %d elements\n", len(hostData))

	// Copy host to device
	gocu.MemcpyHtoD(devicePtr, unsafe.Pointer(&hostData[0]), byteSize)
	fmt.Println("✓ Copied data from host to device")

	// Copy device to host (verify)
	resultData := make([]float32, size)
	gocu.MemcpyDtoH(unsafe.Pointer(&resultData[0]), devicePtr, byteSize)
	fmt.Println("✓ Copied data from device to host")

	// Verify data integrity
	matches := true
	for i := 0; i < size && i < 10; i++ { // Check first 10 elements
		if hostData[i] != resultData[i] {
			matches = false
			break
		}
	}

	if matches {
		fmt.Println("✓ Data integrity verified (first 10 elements match)")
		fmt.Printf("  Sample values: [%.1f, %.1f, %.1f, %.1f, %.1f]\n",
			resultData[0], resultData[1], resultData[2], resultData[3], resultData[4])
	} else {
		fmt.Println("✗ Data integrity check failed")
	}

	// Free device memory
	gocu.MemFree(devicePtr)
	fmt.Println("✓ Device memory freed")
}
