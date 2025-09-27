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
	devicePropCfg := codegen.Config{
		Package:      "cudart",
		Filename:     "cuda_device_prop",
		HeaderDir:    filepath.Join("..", "internal", "cuda"),
		StructName:   "CudaDeviceProp",
		TemplatePath: filepath.Join("..", "internal", "tmpl", "cuda_device_prop.go.tmpl"),
		Versions: []codegen.Version{
			{Version: "v11.8", BuildTag: "cuda11"},
			{Version: "v12.9", BuildTag: "cuda12"},
			{Version: "v13.0", BuildTag: "cuda13"},
		},
	}
	checkErrorCfg := codegen.Config{
		Package:      "cudart",
		Filename:     "cuda_error",
		HeaderDir:    filepath.Join("..", "internal", "cuda"),
		StructName:   "CudaError",
		TemplatePath: filepath.Join("..", "internal", "tmpl", "cuda_enum.go.tmpl"),
		Versions: []codegen.Version{
			{Version: "v11.8", BuildTag: "cuda11"},
			{Version: "v12.9", BuildTag: "cuda12"},
			{Version: "v13.0", BuildTag: "cuda13"},
		},
	}
	deviceAttrCfg := codegen.Config{
		Package:      "cudart",
		Filename:     "cuda_device_attr",
		HeaderDir:    filepath.Join("..", "internal", "cuda"),
		StructName:   "CudaDeviceAttr",
		TemplatePath: filepath.Join("..", "internal", "tmpl", "cuda_enum.go.tmpl"),
		Versions: []codegen.Version{
			{Version: "v11.8", BuildTag: "cuda11"},
			{Version: "v12.9", BuildTag: "cuda12"},
			{Version: "v13.0", BuildTag: "cuda13"},
		},
	}
	deviceLimitCfg := codegen.Config{
		Package:      "cudart",
		Filename:     "cuda_limit",
		HeaderDir:    filepath.Join("..", "internal", "cuda"),
		StructName:   "CudaLimit",
		TemplatePath: filepath.Join("..", "internal", "tmpl", "cuda_enum.go.tmpl"),
		Versions: []codegen.Version{
			{Version: "v11.8", BuildTag: "cuda11"},
			{Version: "v12.9", BuildTag: "cuda12"},
			{Version: "v13.0", BuildTag: "cuda13"},
		},
	}

	if err := generateDeviceProp(devicePropCfg); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	if err := generateCheckError(checkErrorCfg); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	if err := generateDeviceAttr(deviceAttrCfg); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	if err := generateDeviceLimit(deviceLimitCfg); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

func generateDeviceProp(cfg codegen.Config) error {
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

func generateCheckError(cfg codegen.Config) error {
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

func generateDeviceAttr(cfg codegen.Config) error {
	// Parse enum values for all CUDA versions.
	versionFields, err := codegen.ParseAllVersions(cfg.HeaderDir, cfg.Versions, "cudaDeviceAttr", nil, true)
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

func generateDeviceLimit(cfg codegen.Config) error {
	// Parse enum values for all CUDA versions.
	versionFields, err := codegen.ParseAllVersions(cfg.HeaderDir, cfg.Versions, "cudaLimit", nil, true)
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
