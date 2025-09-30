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
		Filename:     "result",
		Versions:     Versions,
		HeaderDir:    HeaderDir,
		HeaderFile:   "cuda.h",
		Include:      "cuda.h",
		StructName:   "Result",
		CGoType:      "",
		StartString:  "typedef enum cudaError_enum",
		TemplatePath: filepath.Join("internal", "tmpl", "cuda_enum.tmpl"),
		IsEnum:       true,
		FieldRegex:   regexp.MustCompile(`^\s*(\w+)\s*=\s*(0x[0-9a-fA-F]+|\d+),?\s*(?:/\*\*<?\s*(.*?)\s*\*/)?$`),
	}
	deviceAttrCfg := codegen.Config{
		Package:      Package,
		Filename:     "device_attr",
		Versions:     Versions,
		HeaderDir:    HeaderDir,
		HeaderFile:   "cuda.h",
		Include:      "cuda.h",
		StructName:   "DeviceAttr",
		CGoType:      "",
		StartString:  "typedef enum CUdevice_attribute_enum",
		TemplatePath: filepath.Join("internal", "tmpl", "cuda_enum.tmpl"),
		IsEnum:       true,
		FieldRegex:   regexp.MustCompile(`^\s*(\w+)\s*=\s*(0x[0-9a-fA-F]+|\d+),?\s*(?:/\*\*<?\s*(.*?)\s*\*/)?$`),
	}
	limitCfg := codegen.Config{
		Package:      Package,
		Filename:     "limit",
		Versions:     Versions,
		HeaderDir:    HeaderDir,
		HeaderFile:   "cuda.h",
		Include:      "cuda.h",
		StructName:   "Limit",
		CGoType:      "",
		StartString:  "typedef enum CUlimit_enum",
		TemplatePath: filepath.Join("internal", "tmpl", "cuda_enum.tmpl"),
		IsEnum:       true,
		FieldRegex:   regexp.MustCompile(`^\s*(\w+)\s*=\s*(0x[0-9a-fA-F]+|\d+),?\s*(?:/\*\*<?\s*(.*?)\s*\*/)?$`),
	}
	contextFlagCfg := codegen.Config{
		Package:      Package,
		Filename:     "context_flag",
		Versions:     Versions,
		HeaderDir:    HeaderDir,
		HeaderFile:   "cuda.h",
		Include:      "cuda.h",
		StructName:   "ContextFlag",
		CGoType:      "",
		StartString:  "typedef enum CUctx_flags_enum",
		TemplatePath: filepath.Join("internal", "tmpl", "cuda_enum.tmpl"),
		IsEnum:       true,
		FieldRegex:   regexp.MustCompile(`^\s*(\w+)\s*=\s*(0x[0-9a-fA-F]+|\d+),?\s*(?:/\*\*<?\s*(.*?)\s*\*/)?$`),
	}
	jitOptionCfg := codegen.Config{
		Package:      Package,
		Filename:     "jit_option",
		Versions:     Versions,
		HeaderDir:    HeaderDir,
		HeaderFile:   "cuda.h",
		Include:      "cuda.h",
		StructName:   "JitOption",
		CGoType:      "",
		StartString:  "typedef enum CUjit_option_enum",
		TemplatePath: filepath.Join("internal", "tmpl", "cuda_enum.tmpl"),
		IsEnum:       true,
		FieldRegex:   regexp.MustCompile(`^\s*(\w+)\s*(?:=\s*(0x[0-9a-fA-F]+|\d+))?,?\s*(?:/\*\*<?\s*(.*?)\s*\*/)?$`),
	}

	if err := codegen.Generate(cuResultCfg); err != nil {
		fmt.Fprintf(os.Stderr, "CUResult: %v\n", err)
		os.Exit(1)
	}
	if err := codegen.Generate(deviceAttrCfg); err != nil {
		fmt.Fprintf(os.Stderr, "DeviceAttr: %v\n", err)
		os.Exit(1)
	}
	if err := codegen.Generate(limitCfg); err != nil {
		fmt.Fprintf(os.Stderr, "Limit: %v\n", err)
		os.Exit(1)
	}
	if err := codegen.Generate(contextFlagCfg); err != nil {
		fmt.Fprintf(os.Stderr, "ContextFlag: %v\n", err)
		os.Exit(1)
	}
	if err := codegen.Generate(jitOptionCfg); err != nil {
		fmt.Fprintf(os.Stderr, "JitOption: %v\n", err)
		os.Exit(1)
	}
}
