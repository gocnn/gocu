//go:build ignore
// +build ignore

package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/gocnn/gocu/internal/codegen"
)

func main() {
	deviceTypeCfg := codegen.Config{
		Package:      "cudart",
		Filename:     "device_type",
		HeaderDir:    filepath.Join("..", "internal", "cuda"),
		StructName:   "DeviceProperties",
		TemplatePath: filepath.Join("..", "internal", "tmpl", "device.go.tmpl"),
		Versions: []codegen.Version{
			{Version: "v11.8", BuildTag: "cuda11"},
			{Version: "v12.9", BuildTag: "cuda12"},
			{Version: "v13.0", BuildTag: "cuda13"},
		},
	}
	checkTypeCfg := codegen.Config{
		Package:      "cudart",
		Filename:     "check_type",
		HeaderDir:    filepath.Join("..", "internal", "cuda"),
		StructName:   "CudaError",
		TemplatePath: filepath.Join("..", "internal", "tmpl", "error.go.tmpl"),
		Versions: []codegen.Version{
			{Version: "v11.8", BuildTag: "cuda11"},
			{Version: "v12.9", BuildTag: "cuda12"},
			{Version: "v13.0", BuildTag: "cuda13"},
		},
	}

	if err := generateDeviceType(deviceTypeCfg); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	if err := generateCheckType(checkTypeCfg); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

func generateDeviceType(cfg codegen.Config) error {
	// Parse fields for all CUDA versions.
	versionFields, err := codegen.ParseAllVersions(cfg.HeaderDir, cfg.Versions, "cudaDeviceProp", nil, false)
	if err != nil {
		return fmt.Errorf("parsing versions: %w", err)
	}

	// Generate default file with common fields.
	commonFields := codegen.FindCommonFields(versionFields)
	if err := codegen.GenerateFile(cfg.Filename+".go", codegen.StructDef{
		Package:  cfg.Package,
		BuildTag: codegen.BuildDefaultTag(cfg.Versions),
		Fields:   commonFields,
	}, cfg, false); err != nil {
		return fmt.Errorf("generating default file: %w", err)
	}

	// Generate version-specific files.
	for _, ver := range cfg.Versions {
		if err := codegen.GenerateFile(
			fmt.Sprintf("%s_%s.go", cfg.Filename, ver.BuildTag),
			codegen.StructDef{
				Package:  cfg.Package,
				BuildTag: ver.BuildTag,
				Fields:   versionFields[ver.Version],
			},
			cfg,
			false,
		); err != nil {
			return fmt.Errorf("generating file for %s: %w", ver.Version, err)
		}
	}

	return nil
}

func generateCheckType(cfg codegen.Config) error {
	// Parse enum values for all CUDA versions.
	versionFields, err := codegen.ParseAllVersions(cfg.HeaderDir, cfg.Versions, "cudaError", nil, true)
	if err != nil {
		return fmt.Errorf("parsing versions: %w", err)
	}

	// Generate default file with common enum values.
	commonFields := codegen.FindCommonFields(versionFields)
	if err := codegen.GenerateFile(cfg.Filename+".go", codegen.StructDef{
		Package:  cfg.Package,
		BuildTag: codegen.BuildDefaultTag(cfg.Versions),
		Fields:   commonFields,
	}, cfg, true); err != nil {
		return fmt.Errorf("generating default file: %w", err)
	}

	// Generate version-specific files.
	for _, ver := range cfg.Versions {
		if err := codegen.GenerateFile(
			fmt.Sprintf("%s_%s.go", cfg.Filename, ver.BuildTag),
			codegen.StructDef{
				Package:  cfg.Package,
				BuildTag: ver.BuildTag,
				Fields:   versionFields[ver.Version],
			},
			cfg,
			true,
		); err != nil {
			return fmt.Errorf("generating file for %s: %w", ver.Version, err)
		}
	}

	return nil
}
