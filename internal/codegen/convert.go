package codegen

import (
	"fmt"
	"strings"
	"unicode"
)

// ToGoFieldName converts a C-style field name to Go-style UpperCamelCase.
func ToGoFieldName(name string) string {
	name = strings.TrimSpace(name)
	if name == "" {
		return ""
	}

	var result strings.Builder
	var word strings.Builder
	isSnakeCase := strings.Contains(name, "_")

	for i, r := range name {
		if isSnakeCase && r == '_' {
			if word.Len() > 0 {
				runes := []rune(strings.ToLower(word.String()))
				if len(runes) > 0 {
					runes[0] = unicode.ToUpper(runes[0])
					result.WriteString(string(runes))
				}
				word.Reset()
			}
			continue
		}
		if !isSnakeCase && i > 0 && unicode.IsUpper(r) && unicode.IsLower(rune(name[i-1])) {
			runes := []rune(strings.ToLower(word.String()))
			if len(runes) > 0 {
				runes[0] = unicode.ToUpper(runes[0])
				result.WriteString(string(runes))
			}
			word.Reset()
		}
		word.WriteRune(r)
	}
	if word.Len() > 0 {
		runes := []rune(strings.ToLower(word.String()))
		if len(runes) > 0 {
			runes[0] = unicode.ToUpper(runes[0])
			result.WriteString(string(runes))
		}
	}

	return result.String()
}

// ToGoType converts a C type to a Go type.
func ToGoType(cType string, isEnum bool, goBaseType string) string {
	if isEnum {
		return goBaseType
	}
	switch {
	case strings.HasPrefix(cType, "char["):
		return "string"
	case strings.HasPrefix(cType, "int["):
		return "[]int"
	case strings.HasPrefix(cType, "size_t["):
		return "[]uint64"
	case cType == "int":
		return "int"
	case cType == "unsigned int":
		return "uint32"
	case cType == "size_t":
		return "uint64"
	case cType == "cudaUUID_t":
		return "uuid.UUID" // Import handled in template
	default:
		return fmt.Sprintf("unknown /* %s */", cType)
	}
}

// ToFromCExpr generates the expression to convert a C field to a Go field.
func ToFromCExpr(fieldName, cType, goType string, isEnum bool) string {
	if isEnum {
		return "C." + fieldName // Use fieldName (enum name) not cType (enum value)
	}
	switch {
	case strings.HasPrefix(cType, "char["):
		return fmt.Sprintf("C.GoString(&prop.%s[0])", fieldName)
	case strings.HasPrefix(cType, "int["):
		size := strings.Trim(cType[strings.Index(cType, "[")+1:], "]")
		return fmt.Sprintf("intSlice(&prop.%s[0], %s)", fieldName, size)
	case strings.HasPrefix(cType, "size_t["):
		size := strings.Trim(cType[strings.Index(cType, "[")+1:], "]")
		return fmt.Sprintf("uint64Slice(&prop.%s[0], %s)", fieldName, size)
	case cType == "int":
		return fmt.Sprintf("int(prop.%s)", fieldName)
	case cType == "unsigned int":
		return fmt.Sprintf("uint32(prop.%s)", fieldName)
	case cType == "size_t":
		return fmt.Sprintf("uint64(prop.%s)", fieldName)
	case cType == "cudaUUID_t":
		return fmt.Sprintf("*(*uuid.UUID)(unsafe.Pointer(&prop.%s))", fieldName)
	default:
		return fmt.Sprintf("/* unsupported type %s */ prop.%s", cType, fieldName)
	}
}
