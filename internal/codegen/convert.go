package codegen

import (
	"fmt"
	"strings"
)

// ToGoFieldName converts a C-style field name to Go-style UpperCamelCase.
func ToGoFieldName(name string) string {
	// Handle snake case
	if strings.Contains(name, "_") {
		words := strings.Split(strings.TrimSpace(name), "_")
		for i, word := range words {
			if word == "" {
				continue
			}
			word = strings.ToLower(word)
			words[i] = strings.ToUpper(string(word[0])) + word[1:]
		}
		return strings.Join(words, "")
	}

	// Handle camelCase or run-together lowercase
	var words []string
	var currentWord strings.Builder
	var prevIsUpper bool

	for i, r := range name {
		isUpper := r >= 'A' && r <= 'Z'
		isDigit := r >= '0' && r <= '9'

		if i == 0 {
			currentWord.WriteRune(r)
			prevIsUpper = isUpper
			continue
		}

		if isUpper && !prevIsUpper || isDigit && currentWord.Len() > 0 {
			words = append(words, currentWord.String())
			currentWord.Reset()
		}
		currentWord.WriteRune(r)
		prevIsUpper = isUpper
	}
	if currentWord.Len() > 0 {
		words = append(words, currentWord.String())
	}

	for i, word := range words {
		if word == "" {
			continue
		}
		word = strings.ToLower(word)
		words[i] = strings.ToUpper(string(word[0])) + word[1:]
	}
	return strings.Join(words, "")
}

// ToGoType converts a C type to a Go type (for enums, returns the base type).
func ToGoType(cType string, isEnum bool) string {
	if isEnum {
		// Handle different enum types
		switch cType {
		case "cudaError":
			return "CudaError"
		case "cudaDeviceAttr":
			return "DeviceAttribute"
		case "cudaLimit":
			return "CudaLimit"
		default:
			return "cudaError" // Default fallback
		}
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
		return "uuid.UUID"
	default:
		return "unknown /* " + cType + " */"
	}
}

// ToFromCExpr generates the expression to convert a C field to a Go field.
func ToFromCExpr(fieldName, cType, goType string, isEnum bool) string {
	if isEnum {
		return "C." + fieldName
	}
	switch {
	case strings.HasPrefix(cType, "char["):
		return "C.GoString(&prop." + fieldName + "[0])"
	case strings.HasPrefix(cType, "int["):
		size := strings.Trim(cType[strings.Index(cType, "[")+1:], "]")
		return fmt.Sprintf("intSlice(&prop.%s[0], %s)", fieldName, size)
	case strings.HasPrefix(cType, "size_t["):
		size := strings.Trim(cType[strings.Index(cType, "[")+1:], "]")
		return fmt.Sprintf("uint64Slice(&prop.%s[0], %s)", fieldName, size)
	case cType == "int":
		return "int(prop." + fieldName + ")"
	case cType == "unsigned int":
		return "uint32(prop." + fieldName + ")"
	case cType == "size_t":
		return "uint64(prop." + fieldName + ")"
	case cType == "cudaUUID_t":
		return "*(*uuid.UUID)(unsafe.Pointer(&prop." + fieldName + "))"
	default:
		return "/* unsupported type " + cType + " */ prop." + fieldName
	}
}
