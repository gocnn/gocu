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
	Package   = "gocu"
	HeaderDir = filepath.Join("internal", "cuda")
	Versions  = []codegen.Version{
		{Version: "v11.8", BuildTag: "cuda11"},
		{Version: "v12.9", BuildTag: "cuda12"},
		{Version: "v13.0", BuildTag: "cuda13"},
	}
)

func main() {
	cuResultCfg := codegen.Config{
		Package:      Package,
		Filename:     "cu_result",
		Versions:     Versions,
		HeaderDir:    HeaderDir,
		HeaderFile:   "cuda.h",
		Include:      "cuda.h",
		StructName:   "CUResult",
		CGoType:      "",
		StartString:  "typedef enum cudaError_enum",
		TemplatePath: filepath.Join("internal", "tmpl", "cuda_enum.go.tmpl"),
		IsEnum:       true,
		FieldRegex:   regexp.MustCompile(`^\s*(\w+)\s*=\s*(0x[0-9a-fA-F]+|\d+),?\s*(?:/\*\*<?\s*(.*?)\s*\*/)?$`),
	}
	deviceAttrCfg := codegen.Config{
		Package:      Package,
		Filename:     "cu_device_attr",
		Versions:     Versions,
		HeaderDir:    HeaderDir,
		HeaderFile:   "cuda.h",
		Include:      "cuda.h",
		StructName:   "CUDeviceAttr",
		CGoType:      "",
		StartString:  "typedef enum CUdevice_attribute_enum",
		TemplatePath: filepath.Join("internal", "tmpl", "cuda_enum.go.tmpl"),
		IsEnum:       true,
		FieldRegex:   regexp.MustCompile(`^\s*(\w+)\s*=\s*(0x[0-9a-fA-F]+|\d+),?\s*(?:/\*\*<?\s*(.*?)\s*\*/)?$`),
	}

	if err := codegen.Generate(cuResultCfg); err != nil {
		fmt.Fprintf(os.Stderr, "CUResult: %v\n", err)
		os.Exit(1)
	}
	if err := codegen.Generate(deviceAttrCfg); err != nil {
		fmt.Fprintf(os.Stderr, "DeviceAttr: %v\n", err)
		os.Exit(1)
	}
}
