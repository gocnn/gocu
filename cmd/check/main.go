// cuda-check verifies CUDA installation and provides symlink command on Windows.
package main

import (
	"fmt"
	"os"
)

func main() {
	if err := CheckCUDA(); err != nil {
		fmt.Fprintf(os.Stderr, "GoCU CUDA Check Error: %v\n", err)
		os.Exit(1)
	}
}
