package codegen

import (
	"fmt"
	"os"
	"sort"
	"strings"
	"text/template"
)

// BuildDefaultTag constructs the build tag for the default struct.
func BuildDefaultTag(versions []Version) string {
	var tags []string
	for _, v := range versions {
		tags = append(tags, "!"+v.BuildTag)
	}
	return strings.Join(tags, " ")
}

// GenerateFile writes a Go file using the provided template and struct definition.
func GenerateFile(cfg Config, def StructDef) error {
	f, err := os.Create(cfg.Filename)
	if err != nil {
		return fmt.Errorf("creating %s: %w", cfg.Filename, err)
	}
	defer f.Close()

	data := TemplateData{
		StructDef:  def,
		StructName: cfg.StructName,
		CGoType:    cfg.CGoType,
		Include:    cfg.Include,
	}
	goBaseType := cfg.StructName // For enums
	for _, fld := range def.Fields {
		goType := ToGoType(fld.Type, cfg.IsEnum, goBaseType)
		data.Fields = append(data.Fields, TemplateField{
			Name:      fld.Name,
			GoName:    ToGoFieldName(fld.Name),
			GoType:    goType,
			CType:     fld.Type,
			FromCExpr: ToFromCExpr(fld.Name, fld.Type, goType, cfg.IsEnum),
			Doc:       fld.Doc,
		})
	}
	sort.Slice(data.Fields, func(i, j int) bool { return data.Fields[i].Name < data.Fields[j].Name })

	tmplContent, err := os.ReadFile(cfg.TemplatePath)
	if err != nil {
		return fmt.Errorf("reading template %s: %w", cfg.TemplatePath, err)
	}

	tmpl, err := template.New("def").Funcs(template.FuncMap{
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
