//go:build !windows && !linux

package main

import "fmt"

func CheckCUDA() error {
	return fmt.Errorf("Unsupported OS")
}
