package codegen

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

// TemplateData prepares data for the Go template.
type TemplateData struct {
	StructDef
	StructName string // For structs: struct name; for enums: type name (e.g., cudaError)
	Fields     []TemplateField
}

// TemplateField holds field data for the template.
type TemplateField struct {
	Name      string // Original C name (e.g., cudaSuccess)
	GoName    string // Go-style name (e.g., Success)
	GoType    string // For structs: Go type; for enums: base type (e.g., cudaError)
	CType     string // For structs: C type; for enums: empty or value
	FromCExpr string // For structs: conversion expr; for enums: C constant (e.g., C.cudaSuccess)
	Doc       string // Optional: documentation
}

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
	StructName   string
	TemplatePath string
}
