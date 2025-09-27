package codegen

import (
	"fmt"
	"os"
	"sort"
	"strings"
	"text/template"
)

// GenerateFile writes a Go file using the provided template and struct definition.
func GenerateFile(filename string, def StructDef, cfg Config, isEnum bool) error {
	f, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("creating %s: %w", filename, err)
	}
	defer f.Close()

	data := TemplateData{
		StructDef:  def,
		StructName: cfg.StructName,
	}
	for _, f := range def.Fields {
		goType := ToGoType(f.Type, isEnum)
		data.Fields = append(data.Fields, TemplateField{
			Name:      f.Name,
			GoName:    ToGoFieldName(f.Name),
			GoType:    goType,
			CType:     f.Type,
			FromCExpr: ToFromCExpr(f.Name, f.Type, goType, isEnum),
			Doc:       f.Doc,
		})
	}
	sort.Slice(data.Fields, func(i, j int) bool {
		return data.Fields[i].Name < data.Fields[j].Name
	})

	tmplContent, err := os.ReadFile(cfg.TemplatePath)
	if err != nil {
		return fmt.Errorf("reading template %s: %w", cfg.TemplatePath, err)
	}

	tmpl, err := template.New("struct").Funcs(template.FuncMap{
		"hasUUID": func(fields []TemplateField) bool {
			for _, f := range fields {
				if f.GoType == "uuid.UUID" {
					return true
				}
			}
			return false
		},
	}).Parse(string(tmplContent))
	if err != nil {
		return fmt.Errorf("parsing template: %w", err)
	}

	return tmpl.Execute(f, data)
}

// BuildDefaultTag constructs the build tag for the default struct.
func BuildDefaultTag(versions []Version) string {
	var tags []string
	for _, ver := range versions {
		tags = append(tags, "!"+ver.BuildTag)
	}
	return strings.Join(tags, " ")
}
