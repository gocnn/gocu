package codegen

import "regexp"

// Version defines a version and its build tag.
type Version struct {
	Version  string
	BuildTag string
}

// Config holds configuration for code generation.
type Config struct {
	Package      string
	Filename     string
	Versions     []Version
	HeaderDir    string
	HeaderFile   string         // e.g., "cuda.h"
	Include      string         // e.g., "cuda_runtime.h"
	StructName   string         // Go name, e.g., "DeviceProp" or "CudaError"
	CGoType      string         // CGo type, e.g., "struct_cudaDeviceProp" (for structs)
	StartString  string         // Start marker, e.g., "struct __device_builtin__ cudaDeviceProp"
	EndString    string         // End marker, e.g., "};" (relaxed to any line containing "}")
	FieldRegex   *regexp.Regexp // Optional: custom regex for fields
	TemplatePath string
	IsEnum       bool
}

// Field represents a field in a C header (struct or enum).
type Field struct {
	Name string // C name (e.g., cudaSuccess)
	Type string // For structs: C type; for enums: value (e.g., "0")
	Doc  string // Optional: documentation from header
}

// StructDef holds metadata for generating a Go struct or enum.
type StructDef struct {
	Package  string
	BuildTag string
	Fields   []Field
}

// TemplateField holds field data for the template.
type TemplateField struct {
	Name      string // Original C name (e.g., cudaSuccess)
	GoName    string // Go-style name (e.g., Success)
	GoType    string // For structs: Go type; for enums: base type (e.g., CudaError)
	CType     string // For structs: C type; for enums: empty or value
	FromCExpr string // For structs: conversion expr; for enums: C constant (e.g., C.cudaSuccess)
	Doc       string // Optional: documentation
}

// TemplateData prepares data for the Go template.
type TemplateData struct {
	StructDef
	StructName string // For structs: struct name; for enums: type name (e.g., CudaError)
	CGoType    string // CGo type for structs (e.g., struct_cudaDeviceProp)
	Include    string // Header include (e.g., cuda_runtime.h)
	Fields     []TemplateField
}
