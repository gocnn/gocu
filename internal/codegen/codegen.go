package codegen

import (
	"fmt"
)

func Generate(cfg Config) error {
	versionFields, err := ParseAllVersions(cfg)
	if err != nil {
		return fmt.Errorf("parsing versions: %w", err)
	}

	commonFields := FindCommonFields(versionFields)

	defaultCfg := cfg
	defaultCfg.Filename += ".go"
	if err := GenerateFile(defaultCfg, StructDef{
		Package:  cfg.Package,
		BuildTag: BuildDefaultTag(cfg.Versions),
		Fields:   commonFields,
	}); err != nil {
		return fmt.Errorf("generating default: %w", err)
	}

	for _, ver := range cfg.Versions {
		verCfg := cfg
		verCfg.Filename = fmt.Sprintf("%s_%s.go", cfg.Filename, ver.BuildTag)
		if err := GenerateFile(verCfg, StructDef{
			Package:  cfg.Package,
			BuildTag: ver.BuildTag,
			Fields:   versionFields[ver.Version],
		}); err != nil {
			return fmt.Errorf("generating %s: %w", ver.Version, err)
		}
	}
	return nil
}
