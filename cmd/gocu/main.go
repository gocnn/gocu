// cudatest tests CUDA availability and device info.
package main

import (
	"fmt"
	"os"

	"github.com/gocnn/gocu"
)

func main() {
	ver := gocu.Version()
	fmt.Printf("CUDA Version: %d.%d\n", ver/1000, ver%1000)

	count, err := gocu.DeviceGetCount()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error getting device count: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("CUDA devices: %d\n\n", count)

	for i := range count {
		dev := gocu.Device(i)

		name, err := dev.Name()
		if err != nil {
			continue // Skip device on error.
		}

		cr, err := dev.Attribute(gocu.ClockRate)
		if err != nil {
			cr = 0
		}

		mem, err := dev.TotalMem()
		if err != nil {
			mem = 0
		}

		maj, err := dev.Attribute(gocu.ComputeCapabilityMajor)
		if err != nil {
			maj = 0
		}

		min, err := dev.Attribute(gocu.ComputeCapabilityMinor)
		if err != nil {
			min = 0
		}

		fmt.Printf("Device %d\n========\n", i)
		fmt.Printf("Name      : %q\n", name)
		fmt.Printf("Clock Rate: %d kHz\n", cr)
		fmt.Printf("Memory    : %d bytes\n", mem)
		fmt.Printf("Compute   : %d.%d\n\n", maj, min)
	}
}
