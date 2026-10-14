package cli

import (
	"errors"
	"strings"

	"bob/internal/sandbox"
)

// MountConfig holds validated host mount configuration for the sandbox.
type MountConfig struct {
	HostPath string
	ReadOnly bool
}

// ParseMountSpec parses a CLI mount specification of the form "PATH[:ro|:rw]".
// If no mode suffix is specified, ReadOnly defaults to true.
// Paths can contain colons; only the final ":ro" or ":rw" is treated as the mode specifier.
// Resolves "~" to user home directory and relative paths against cwd.
// Evaluates symlinks and verifies that the target exists and is a directory.
// Rejects root ("/"), user home directory, blocked system paths, and any path that equals,
// contains, or is contained by any protected path.
func ParseMountSpec(spec, cwd string, protectedPaths []string) (*MountConfig, error) {
	if strings.TrimSpace(spec) == "" {
		return nil, errors.New("mount specification cannot be empty")
	}

	readOnly := true
	rawPath := spec
	if strings.HasSuffix(spec, ":ro") {
		rawPath = strings.TrimSuffix(spec, ":ro")
		readOnly = true
	} else if strings.HasSuffix(spec, ":rw") {
		rawPath = strings.TrimSuffix(spec, ":rw")
		readOnly = false
	}

	if strings.TrimSpace(rawPath) == "" {
		return nil, errors.New("mount path cannot be empty")
	}

	canonicalHost, err := sandbox.ValidateWorkspaceDir(rawPath, cwd, protectedPaths)
	if err != nil {
		return nil, err
	}

	return &MountConfig{
		HostPath: canonicalHost,
		ReadOnly: readOnly,
	}, nil
}
