package cli

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"bob/internal/config"
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
// Rejects root ("/") and any path that equals, contains, or is contained by any protected path.
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

	// Expand leading ~/
	expanded, err := config.ExpandPath(rawPath)
	if err != nil {
		return nil, fmt.Errorf("failed to expand mount path %q: %w", rawPath, err)
	}

	// Resolve relative path against cwd
	if !filepath.IsAbs(expanded) {
		if cwd == "" {
			var err error
			cwd, err = os.Getwd()
			if err != nil {
				return nil, fmt.Errorf("failed to get working directory: %w", err)
			}
		}
		expanded = filepath.Join(cwd, expanded)
	}
	expanded = filepath.Clean(expanded)

	// Canonicalize and evaluate symlinks
	canonicalHost, err := filepath.EvalSymlinks(expanded)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve mount path %q: %w", rawPath, err)
	}

	// Verify target exists and is a directory
	stat, err := os.Stat(canonicalHost)
	if err != nil {
		return nil, fmt.Errorf("mount path %q does not exist: %w", canonicalHost, err)
	}
	if !stat.IsDir() {
		return nil, fmt.Errorf("mount path %q is not a directory", canonicalHost)
	}

	if canonicalHost == "/" {
		return nil, errors.New("mounting root directory '/' is forbidden")
	}

	for _, p := range protectedPaths {
		if p == "" {
			continue
		}
		pExpanded, expErr := config.ExpandPath(p)
		if expErr != nil {
			pExpanded = p
		}
		pCanon := canonicalizeProtected(pExpanded)
		if isSubpath(canonicalHost, pCanon) {
			return nil, fmt.Errorf("mount path %q contains protected directory %q", canonicalHost, pCanon)
		}
		if isSubpath(pCanon, canonicalHost) {
			return nil, fmt.Errorf("mount path %q is inside protected directory %q", canonicalHost, pCanon)
		}
	}

	return &MountConfig{
		HostPath: canonicalHost,
		ReadOnly: readOnly,
	}, nil
}

func canonicalizeProtected(path string) string {
	abs, err := filepath.Abs(path)
	if err != nil {
		abs = filepath.Clean(path)
	}
	// Walk up until we find an existing directory to EvalSymlinks, then append missing parts
	cur := abs
	var missingParts []string
	for cur != "/" && cur != "." {
		if _, err := os.Stat(cur); err == nil {
			break
		}
		missingParts = append([]string{filepath.Base(cur)}, missingParts...)
		cur = filepath.Dir(cur)
	}
	realCur, err := filepath.EvalSymlinks(cur)
	if err == nil {
		cur = realCur
	}
	for _, part := range missingParts {
		cur = filepath.Join(cur, part)
	}
	return filepath.Clean(cur)
}

func isSubpath(parent, child string) bool {
	rel, err := filepath.Rel(parent, child)
	if err != nil {
		return false
	}
	if rel == "." {
		return true
	}
	return !strings.HasPrefix(rel, ".."+string(filepath.Separator)) && rel != ".."
}
