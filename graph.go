package gocu

/*
#include <cuda.h>
*/
import "C"

// CUGraph is a CUDA graph
type CUGraph struct {
	graph C.CUgraph
}

// C returns the CUGraph as its C version
func (g CUGraph) c() C.CUgraph { return g.graph }

// CUGraphCreate creates a new CUDA graph
func CUGraphCreate(flags uint32) (CUGraph, error) {
	var graph C.CUgraph
	result := Check(C.cuGraphCreate(&graph, C.uint(flags)))
	return CUGraph{graph: graph}, result
}

// cuGraphDestroy ( CUgraph hGraph )
// Destroys a graph.
func CUGraphDestroy(graph CUGraph) error {
	return Check(C.cuGraphDestroy(graph.c()))
}

// Destroy destroys the graph (method version)
func (g CUGraph) Destroy() error {
	return Check(C.cuGraphDestroy(g.c()))
}
