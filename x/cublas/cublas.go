package cublas

import (
	"sync"

	"github.com/gocnn/gocu/cublas"
	"github.com/gocnn/gocu/cudart"
)

// Type aliases for parameter structures
type SrotmParams = cublas.SrotmParams
type DrotmParams = cublas.DrotmParams

// Type aliases for enums and constants
type Order = cublas.Order
type Transpose = cublas.Transpose
type Side = cublas.Side
type Diag = cublas.Diag
type Uplo = cublas.Uplo

// Order constants
const (
	RowMajor = cublas.RowMajor
	ColMajor = cublas.ColMajor
)

// Transpose constants
const (
	NoTrans   = cublas.NoTrans
	Trans     = cublas.Trans
	ConjTrans = cublas.ConjTrans
)

// Side constants
const (
	Left  = cublas.Left
	Right = cublas.Right
)

// Diag constants
const (
	NonUnit = cublas.NonUnit
	Unit    = cublas.Unit
)

// Uplo constants
const (
	Upper = cublas.Upper
	Lower = cublas.Lower
)

type CudaBlas struct {
	handle *cublas.Handle
	sync.Mutex
}

func New() *CudaBlas {
	handle, err := cublas.Create()
	if err != nil {
		panic(err)
	}
	handler := &CudaBlas{handle: handle}
	return handler
}

func (h *CudaBlas) Close() error {
	h.Lock()
	defer h.Unlock()
	if err := h.handle.Destroy(); err != nil {
		return err
	}
	return nil
}

func (h *CudaBlas) Handle() *cublas.Handle {
	return h.handle
}

// SetStream sets the stream to be used by the BLAS handler.
func (h *CudaBlas) SetStream(stream *cudart.Stream) {
	h.Lock()
	defer h.Unlock()
	h.handle.SetStream(stream)
}

// GetStream gets the stream being used by the BLAS handler.
func (h *CudaBlas) GetStream() (*cudart.Stream, error) {
	h.Lock()
	defer h.Unlock()
	return h.handle.GetStream()
}
