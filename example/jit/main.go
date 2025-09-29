package main

import (
	_ "embed"
	"fmt"
	"unsafe"

	"github.com/gocnn/gocu"
)

//go:embed add.ptx
var ptx string

func main() {
	var n int64 = 1024

	device, _ := gocu.DeviceGet(0)
	ctx, _ := gocu.CtxCreate(gocu.CtxSchedAuto, device)
	defer ctx.Destroy()
	module, _ := gocu.ModuleLoadData([]byte(ptx))
	defer module.Unload()
	function, _ := module.GetFunction("add")

	hostA := make([]int32, n)
	hostB := make([]int32, n)
	hostC := make([]int32, n)

	for i := range hostA {
		hostA[i] = int32(i)
		hostB[i] = int32(i * 2)
	}

	size := int64(n * 4)
	devA, _ := gocu.MemAlloc(size)
	devB, _ := gocu.MemAlloc(size)
	devC, _ := gocu.MemAlloc(size)
	defer gocu.MemFree(devA)
	defer gocu.MemFree(devB)
	defer gocu.MemFree(devC)

	gocu.MemcpyHtoD(devA, unsafe.Pointer(&hostA[0]), size)
	gocu.MemcpyHtoD(devB, unsafe.Pointer(&hostB[0]), size)

	args := []unsafe.Pointer{
		unsafe.Pointer(&devC),
		unsafe.Pointer(&devA),
		unsafe.Pointer(&devB),
		unsafe.Pointer(&n),
	}

	gocu.LaunchKernel(
		function,
		(uint32)(n+255)/256, 1, 1,
		256, 1, 1,
		0,
		gocu.Stream{},
		args,
		nil,
	)
	gocu.CtxSynchronize()

	gocu.MemcpyDtoH(unsafe.Pointer(&hostC[0]), devC, size)

	fmt.Println("Results:")
	for i := range 100 {
		fmt.Printf("%d + %d = %d\n", hostA[i], hostB[i], hostC[i])
	}
}
