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

// ParseHeader extracts fields from a C header file.
func ParseHeader(path, structName string, fieldRegex *regexp.Regexp) ([]Field, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("opening %s: %w", path, err)
	}
	defer file.Close()

	var fields []Field
	inStruct := false
	reField := fieldRegex
	if reField == nil {
		reField = regexp.MustCompile(`^\s*([\w\s]+)\s+([\w]+)(\[[\d\w]+\])?\s*;\s*(?:/\*\*.*)?$`)
	}

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		if strings.Contains(line, "struct __device_builtin__ "+structName) {
			inStruct = true
			continue
		}
		if inStruct && strings.Contains(line, "};") {
			break
		}
		if !inStruct {
			continue
		}

		matches := reField.FindStringSubmatch(line)
		if len(matches) < 3 {
			continue
		}

		fieldType := strings.TrimSpace(matches[1])
		if len(matches) >= 4 && matches[3] != "" {
			fieldType += matches[3]
		}
		fields = append(fields, Field{
			Name: matches[2],
			Type: fieldType,
		})
	}
	return fields, scanner.Err()
}

// ParseAllVersions parses fields for multiple versions of a header file.
func ParseAllVersions(headerDir string, versions []Version, structName string, fieldRegex *regexp.Regexp) (map[string][]Field, error) {
	versionFields := make(map[string][]Field)
	for _, ver := range versions {
		path := filepath.Join(headerDir, ver.Version, "driver_types.h")
		fields, err := ParseHeader(path, structName, fieldRegex)
		if err != nil {
			return nil, fmt.Errorf("version %s: %w", ver.Version, err)
		}
		versionFields[ver.Version] = fields
	}
	return versionFields, nil
}

// FindCommonFields identifies fields present in all versions with identical types.
func FindCommonFields(versionFields map[string][]Field) []Field {
	if len(versionFields) == 0 {
		return nil
	}

	var refVersion string
	var refFields []Field
	for v, f := range versionFields {
		refVersion, refFields = v, f
		break
	}

	var common []Field
	for _, f := range refFields {
		if isCommonField(f, versionFields, refVersion) {
			common = append(common, f)
		}
	}

	sort.Slice(common, func(i, j int) bool {
		return common[i].Name < common[j].Name
	})
	return common
}

// isCommonField checks if a field exists in all versions with the same type.
func isCommonField(field Field, versionFields map[string][]Field, refVersion string) bool {
	for version, fields := range versionFields {
		if version == refVersion {
			continue
		}
		found := false
		for _, vf := range fields {
			if vf.Name == field.Name && vf.Type == field.Type {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}
