package codegen

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// ParseHeader parses fields from a C header file for structs or enums.
func ParseHeader(cfg Config, path string) ([]Field, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("opening %s: %w", path, err)
	}
	defer file.Close()

	var fields []Field
	inDef := false
	var currentDoc strings.Builder
	reField := cfg.FieldRegex
	if reField == nil {
		if cfg.IsEnum {
			reField = regexp.MustCompile(`^\s*(\w+)\s*=\s*(0x[0-9a-fA-F]+|\d+),?\s*(?:/\*\*<?\s*(.*?)\s*\*/)?$`)
		} else {
			reField = regexp.MustCompile(`^\s*([\w\s]+)\s+([\w]+)(\[[\d\w]+\])?\s*;\s*(?:/\*\*<?\s*(.*?)\s*\*/)?$`)
		}
	}

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		// Collect multi-line documentation
		if prefix, ok := strings.CutPrefix(line, "/**"); ok {
			currentDoc.WriteString(strings.TrimSpace(prefix))
			currentDoc.WriteRune(' ')
			continue
		}
		if prefix, ok := strings.CutPrefix(line, " *"); ok {
			currentDoc.WriteString(strings.TrimSpace(prefix))
			currentDoc.WriteRune(' ')
			continue
		}
		if prefix, ok := strings.CutPrefix(line, " */"); ok {
			currentDoc.WriteString(strings.TrimSpace(prefix))
			continue
		}

		// Detect start
		if !inDef && strings.Contains(line, cfg.StartString) {
			inDef = true
			currentDoc.Reset()
			continue
		}

		// Detect end (relaxed to any line containing "}")
		if inDef && strings.Contains(line, "}") {
			break
		}
		if !inDef {
			continue
		}

		// Parse field
		matches := reField.FindStringSubmatch(line)
		if len(matches) < 3 {
			continue
		}

		var fieldType, fieldName string
		if cfg.IsEnum {
			// For enums: matches[1] is name, matches[2] is value
			fieldName = strings.TrimSpace(matches[1]) // enum name (e.g., cudaSuccess)
			fieldType = strings.TrimSpace(matches[2]) // enum value (e.g., "0")
		} else {
			// For structs: matches[1] is type, matches[2] is name
			fieldType = strings.TrimSpace(matches[1])
			fieldName = strings.TrimSpace(matches[2])
		}

		doc := strings.TrimSpace(currentDoc.String())
		if len(matches) >= 4 && matches[3] != "" && !cfg.IsEnum {
			fieldType += matches[3]
		}
		if len(matches) >= 5 && matches[4] != "" {
			doc = strings.TrimSpace(matches[4])
		} else if len(matches) >= 4 && matches[3] != "" && cfg.IsEnum {
			doc = strings.TrimSpace(matches[3])
		}
		fields = append(fields, Field{
			Name: fieldName,
			Type: fieldType,
			Doc:  doc,
		})
		currentDoc.Reset()
	}

	if cfg.IsEnum {
		fields = filterDeprecatedDuplicates(fields)
	}
	return fields, scanner.Err()
}

// filterDeprecatedDuplicates removes deprecated enum values that conflict with non-deprecated ones.
func filterDeprecatedDuplicates(fields []Field) []Field {
	valueMap := make(map[string][]Field)
	for _, f := range fields {
		valueMap[f.Type] = append(valueMap[f.Type], f)
	}

	var result []Field
	for _, group := range valueMap {
		var nonDep []Field
		for _, f := range group {
			if !isDeprecated(f) {
				nonDep = append(nonDep, f)
			}
		}
		if len(nonDep) > 0 {
			result = append(result, nonDep...)
		} else {
			result = append(result, group...)
		}
	}
	sort.Slice(result, func(i, j int) bool { return result[i].Name < result[j].Name })
	return result
}

// isDeprecated checks if a field is marked as deprecated.
func isDeprecated(field Field) bool {
	doc := strings.ToLower(field.Doc)
	keywords := []string{"deprecated", "do not use", "obsolete", "superseded", "legacy"}
	for _, kw := range keywords {
		if strings.Contains(doc, kw) {
			return true
		}
	}
	return false
}

// ParseAllVersions parses fields for multiple versions of a header file.
func ParseAllVersions(cfg Config) (map[string][]Field, error) {
	versionFields := make(map[string][]Field)
	for _, ver := range cfg.Versions {
		path := filepath.Join(cfg.HeaderDir, ver.Version, cfg.HeaderFile)
		fields, err := ParseHeader(cfg, path)
		if err != nil {
			return nil, fmt.Errorf("version %s: %w", ver.Version, err)
		}
		versionFields[ver.Version] = fields
	}
	return versionFields, nil
}

// FindCommonFields identifies fields present in all versions with identical types/values.
func FindCommonFields(versionFields map[string][]Field, isEnum bool) []Field {
	if len(versionFields) == 0 {
		return nil
	}
	var refVer string
	var refFields []Field
	for v, f := range versionFields {
		refVer = v
		refFields = f
		break
	}

	var common []Field
	for _, f := range refFields {
		commonInAll := true
		for ver, fields := range versionFields {
			if ver == refVer {
				continue
			}
			found := false
			for _, vf := range fields {
				// For enums, only compare names since values may differ between versions
				// For structs, compare both name and type
				var matches bool
				if isEnum {
					matches = vf.Name == f.Name
				} else {
					matches = vf.Name == f.Name && vf.Type == f.Type
				}

				if matches {
					found = true
					break
				}
			}
			if !found {
				commonInAll = false
				break
			}
		}
		if commonInAll {
			common = append(common, f)
		}
	}
	sort.Slice(common, func(i, j int) bool { return common[i].Name < common[j].Name })
	return common
}
