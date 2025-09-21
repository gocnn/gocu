//go:build windows

package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

func CheckCUDA() error {
	cudaDir := `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA`
	entries, err := os.ReadDir(cudaDir)
	if err != nil {
		return fmt.Errorf("CUDA not installed, please download from https://developer.nvidia.com/cuda-downloads?target_os=Windows")
	}

	for _, entry := range entries {
		if entry.IsDir() {
			cudaPath := filepath.Join(cudaDir, entry.Name())

			fmt.Println("=== CUDA Environment Check ===")
			fmt.Printf("✓ CUDA installation detected: %s\n", cudaPath)

			// Test nvcc compiler
			cmd := exec.Command("nvcc", "--version")
			output, err := cmd.CombinedOutput()
			if err != nil || !strings.Contains(string(output), "Cuda compilation tools") {
				fmt.Println("⚠ nvcc compiler not found in PATH")
				fmt.Println("  Please add CUDA bin directory to your system PATH")
			} else {
				fmt.Println("✓ nvcc compiler test passed")
			}

			fmt.Println("\n=== Next Steps ===")
			fmt.Println("1. Create CUDA symlink (run as administrator):")
			fmt.Printf("   mklink /D C:\\cuda \"%s\"\n", cudaPath)

			fmt.Println("\n2. Install and test GOCU:")
			fmt.Println("   go install github.com/gocnn/gocu/cmd/gocu")
			fmt.Println("   gocu")

			fmt.Println("\n=== Setup Complete ===")
			return nil
		}
	}

	return fmt.Errorf("CUDA directory %s exists but has no version subdirectories", cudaDir)
}
