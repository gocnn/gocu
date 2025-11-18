package cublas

import "errors"

var ErrNotAvailable = errors.New("cublas: CUDA support not available")
