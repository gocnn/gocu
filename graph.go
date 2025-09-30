package gocu

/*
#include <cuda.h>
*/
import "C"

// Graph is a CUDA graph
type Graph struct {
	graph C.CUgraph
}

// CGraph returns the Graph as its C version
func (g *Graph) CGraph() C.CUgraph { return g.graph }

// CUGraphCreate creates a new CUDA graph
func CUGraphCreate(flags uint32) (*Graph, error) {
	var graph C.CUgraph
	result := Check(C.cuGraphCreate(&graph, C.uint(flags)))
	return &Graph{graph: graph}, result
}

// Destroys a graph.
func CUGraphDestroy(graph *Graph) error {
	return Check(C.cuGraphDestroy(graph.CGraph()))
}

// Destroy destroys the graph (method version)
func (g *Graph) Destroy() error {
	return Check(C.cuGraphDestroy(g.CGraph()))
}
