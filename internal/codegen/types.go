package codegen

// Field represents a field in a C header struct.
type Field struct {
	Name string
	Type string
}

// StructDef holds metadata for generating a Go struct.
type StructDef struct {
	Package  string
	BuildTag string
	Fields   []Field
}

// TemplateData prepares data for the Go struct template.
type TemplateData struct {
	StructDef
	StructName string
	Fields     []TemplateField
}

// TemplateField holds field data for the template.
type TemplateField struct {
	Name      string // Original C field name (e.g., access_policy_max_window_size)
	GoName    string // Go-style camelCase name (e.g., AccessPolicyMaxWindowSize)
	GoType    string
	CType     string
	FromCExpr string
}
