//go:build ignore

package main

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"

	"github.com/gocnn/gocu/internal/codegen"
)

var (
	Package   = "cudart"
	HeaderDir = filepath.Join("..", "internal", "cuda")
	Versions  = []codegen.Version{
		{Version: "v11.8", BuildTag: "cuda11"},
		{Version: "v12.9", BuildTag: "cuda12"},
		{Version: "v13.0", BuildTag: "cuda13"},
	}
)

func main() {
	devicePropCfg := codegen.Config{
		Package:      Package,
		Filename:     "cuda_device_prop",
		Versions:     Versions,
		HeaderDir:    HeaderDir,
		HeaderFile:   "driver_types.h",
		Include:      "cuda_runtime.h",
		StructName:   "CudaDeviceProp",
		CGoType:      "struct_cudaDeviceProp",
		StartString:  "struct __device_builtin__ cudaDeviceProp",
		TemplatePath: filepath.Join("..", "internal", "tmpl", "cuda_struct.tmpl"),
		IsEnum:       false,
		FieldRegex:   regexp.MustCompile(`^\s*([\w\s*]+)\s+([\w]+)(\[[\d\w]*\])?\s*;\s*(?:/\*\*?\s*(.*?)\s*\*/)?$`),
	}
	errorCfg := codegen.Config{
		Package:      Package,
		Filename:     "cuda_error",
		Versions:     Versions,
		HeaderDir:    HeaderDir,
		HeaderFile:   "driver_types.h",
		Include:      "cuda_runtime.h",
		StructName:   "CudaError",
		CGoType:      "",
		StartString:  "enum __device_builtin__ cudaError",
		TemplatePath: filepath.Join("..", "internal", "tmpl", "cuda_enum.tmpl"),
		IsEnum:       true,
		FieldRegex:   regexp.MustCompile(`^\s*(\w+)\s*=\s*(0x[0-9a-fA-F]+|\d+),?\s*(?:/\*\*?\s*(.*?)\s*\*/)?$`),
	}
	deviceAttrCfg := codegen.Config{
		Package:      Package,
		Filename:     "cuda_device_attr",
		Versions:     Versions,
		HeaderDir:    HeaderDir,
		HeaderFile:   "driver_types.h",
		Include:      "cuda_runtime.h",
		StructName:   "CudaDeviceAttr",
		CGoType:      "",
		StartString:  "enum __device_builtin__ cudaDeviceAttr",
		TemplatePath: filepath.Join("..", "internal", "tmpl", "cuda_enum.tmpl"),
		IsEnum:       true,
		FieldRegex:   regexp.MustCompile(`^\s*(\w+)\s*=\s*(0x[0-9a-fA-F]+|\d+),?\s*(?:/\*\*?\s*(.*?)\s*\*/)?$`),
	}

	if err := codegen.Generate(devicePropCfg); err != nil {
		fmt.Fprintf(os.Stderr, "DeviceProp: %v\n", err)
		os.Exit(1)
	}
	if err := codegen.Generate(errorCfg); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	if err := codegen.Generate(deviceAttrCfg); err != nil {
		fmt.Fprintf(os.Stderr, "DeviceAttr: %v\n", err)
		os.Exit(1)
	}
}
