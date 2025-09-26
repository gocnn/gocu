package cudart

import "github.com/gocnn/gocu/cudart"

// DeviceSynchronize waits for compute device to finish.
func DeviceSynchronize() error {
	return cudart.DeviceSynchronize()
}
