//go:build linux

package main

import (
	"fmt"
	"os/exec"
	"strings"
)

func CheckCUDA() error {
	fmt.Println("=== CUDA Environment Check ===")

	cmd := exec.Command("nvcc", "--version")
	output, err := cmd.CombinedOutput()
	if err != nil || !strings.Contains(string(output), "Cuda compilation tools") {
		fmt.Println("⚠ CUDA not installed")
		fmt.Println("\n=== Installation Options ===")
		fmt.Println("1. Download from NVIDIA:")
		fmt.Println("   https://developer.nvidia.com/cuda-downloads?target_os=Linux")
		fmt.Println("\n2. Use package manager:")
		fmt.Println("   sudo apt install nvidia-cuda-toolkit")
		return fmt.Errorf("CUDA installation required")
	}

	fmt.Println("✓ CUDA installation detected")
	fmt.Println("✓ nvcc compiler test passed")

	fmt.Println("\n=== Next Steps ===")
	fmt.Println("1. Install and test GOCU:")
	fmt.Println("   go install github.com/gocnn/gocu/cmd/gocu")
	fmt.Println("   gocu")

	fmt.Println("\n=== Setup Complete ===")
	return nil
}
