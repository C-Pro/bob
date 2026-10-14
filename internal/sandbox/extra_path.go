package sandbox

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// ResolvedExtraPath contains the validated host directory to mount and the
// corresponding bin directory to prepend to the sandbox $PATH.
type ResolvedExtraPath struct {
	HostMountPath  string // Directory to bind-mount read-only via --ro-bind
	SandboxBinPath string // Directory to prepend to $PATH inside sandbox
}

// BlockedHostPathRoots defines host system roots that must never be mounted
// as extra paths (nor any of their descendants), preventing exposure of
// host process state, devices, sockets, or sensitive system configuration.
var BlockedHostPathRoots = []string{
	"/proc",
	"/sys",
	"/dev",
	"/run",
	"/var/run",
	"/root",
	"/etc",
	"/boot",
}

// BlockedHostExactPaths defines exact host paths that conflict with Bob's default mounts.
var BlockedHostExactPaths = []string{
	"/",
	"/usr",
	"/bin",
	"/sbin",
	"/lib",
	"/lib64",
	"/tmp",
	"/var/tmp",
}

func isBlockedHostPath(p string) (string, bool) {
	for _, exact := range BlockedHostExactPaths {
		if p == exact {
			return exact, true
		}
	}
	for _, root := range BlockedHostPathRoots {
		if p == root || IsSubpath(root, p) {
			return root, true
		}
	}
	return "", false
}

// IsBlockedHostPath reports whether path p matches a blocked host root, exact path, or descendant.
func IsBlockedHostPath(p string) (string, bool) {
	return isBlockedHostPath(p)
}

// DefaultSensitivePaths returns standard sensitive host paths that must never
// be mounted or exposed into a sandbox (e.g. credentials, tokens, keys).
func DefaultSensitivePaths() []string {
	var paths []string
	home, err := os.UserHomeDir()
	if err == nil && home != "" {
		paths = append(paths,
			filepath.Join(home, ".ssh"),
			filepath.Join(home, ".gnupg"),
			filepath.Join(home, ".aws"),
			filepath.Join(home, ".netrc"),
			filepath.Join(home, ".git-credentials"),
			filepath.Join(home, ".config"),
		)
	}
	return paths
}

// ExpandHome expands leading '~/' or '~' with the user's home directory.
func ExpandHome(p string) (string, error) {
	if !strings.HasPrefix(p, "~") {
		return p, nil
	}
	if p != "~" && !strings.HasPrefix(p, "~/") {
		return p, nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("cannot expand home directory: %w", err)
	}
	if p == "~" {
		return home, nil
	}
	return filepath.Join(home, p[2:]), nil
}

// IsSubpath reports whether child is equal to parent or resides inside parent.
// Comparisons are component-aware using filepath.Rel.
func IsSubpath(parent, child string) bool {
	rel, err := filepath.Rel(parent, child)
	if err != nil {
		return false
	}
	if rel == "." {
		return true
	}
	return !strings.HasPrefix(rel, ".."+string(filepath.Separator)) && rel != ".."
}

// CanonicalizeProtected resolves symlinks on the deepest existing ancestor of a
// protected path, appending any non-existing suffix to ensure accurate exclusion.
func CanonicalizeProtected(path string) string {
	abs, err := filepath.Abs(path)
	if err != nil {
		abs = filepath.Clean(path)
	}
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

// ResolveExtraPath validates a raw path specification for an extra sandbox mount.
// It verifies:
//  1. Path is non-empty and contains no colons or newlines (which break $PATH).
//  2. Tilde and relative paths are expanded securely.
//  3. The target exists, is a directory, and symlinks are evaluated.
//  4. The path is not root ('/'), user $HOME, or equal/ancestor/descendant of any protected path.
//  5. If the path has a bin/ subdirectory, the root is mounted and root/bin is prepended to $PATH.
//     If the path is already a bin directory with a valid toolchain parent, the toolchain parent
func ResolveExtraPath(rawPath, cwd string, protectedPaths []string) (*ResolvedExtraPath, error) {
	if strings.ContainsAny(rawPath, "\n\r") {
		return nil, fmt.Errorf("extra path %q contains invalid characters (newlines)", rawPath)
	}

	trimmed := strings.TrimSpace(rawPath)
	if trimmed == "" {
		return nil, errors.New("extra path cannot be empty")
	}

	if strings.ContainsRune(trimmed, ':') {
		return nil, fmt.Errorf("extra path %q contains invalid characters (colon)", trimmed)
	}

	expanded, err := ExpandHome(trimmed)
	if err != nil {
		return nil, fmt.Errorf("failed to expand extra path %q: %w", trimmed, err)
	}

	if !filepath.IsAbs(expanded) {
		if cwd == "" {
			var err error
			cwd, err = os.Getwd()
			if err != nil {
				return nil, fmt.Errorf("failed to get working directory for extra path %q: %w", trimmed, err)
			}
		}
		expanded = filepath.Join(cwd, expanded)
	}
	expanded = filepath.Clean(expanded)

	canonical, err := filepath.EvalSymlinks(expanded)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, fmt.Errorf("extra path %q does not exist: %w", trimmed, err)
		}
		return nil, fmt.Errorf("failed to resolve extra path %q: %w", trimmed, err)
	}

	stat, err := os.Stat(canonical)
	if err != nil {
		return nil, fmt.Errorf("extra path %q does not exist: %w", canonical, err)
	}
	if !stat.IsDir() {
		return nil, fmt.Errorf("extra path %q is not a directory", canonical)
	}

	if canonical == "/" {
		return nil, errors.New("mounting root directory '/' as extra path is forbidden")
	}

	home, err := os.UserHomeDir()
	if err != nil || home == "" {
		return nil, errors.New("failed to determine user home directory")
	}
	realHome, err := filepath.EvalSymlinks(home)
	if err != nil {
		realHome = home
	}
	if canonical == home || canonical == realHome {
		return nil, errors.New("mounting user home directory as extra path is forbidden")
	}

	// Validate canonical path against all protected paths
	for _, p := range protectedPaths {
		if p == "" {
			continue
		}
		pCanon := CanonicalizeProtected(p)
		if IsSubpath(canonical, pCanon) {
			return nil, fmt.Errorf("extra path %q contains protected path %q", canonical, pCanon)
		}
		if IsSubpath(pCanon, canonical) {
			return nil, fmt.Errorf("extra path %q is inside protected path %q", canonical, pCanon)
		}
	}

	mountRoot := canonical
	binPath := canonical

	// If target directory has a bin/ subdirectory (e.g. toolchain root), mount root and add bin to PATH.
	binSubdir := filepath.Join(canonical, "bin")
	if fi, err := os.Stat(binSubdir); err == nil && fi.IsDir() {
		binCanon, err := filepath.EvalSymlinks(binSubdir)
		if err != nil {
			return nil, fmt.Errorf("failed to resolve bin subdirectory %q: %w", binSubdir, err)
		}
		if !IsSubpath(canonical, binCanon) {
			return nil, fmt.Errorf("bin subdirectory %q symlink escapes mount root %q", binCanon, canonical)
		}
		binPath = binCanon
	}

	// Final check on mountRoot against blocked system paths
	if blocked, isBlocked := isBlockedHostPath(mountRoot); isBlocked {
		return nil, fmt.Errorf("extra path mount %q conflicts with system directory %s", mountRoot, blocked)
	}

	if strings.ContainsAny(mountRoot, ":\n\r") {
		return nil, fmt.Errorf("resolved mount path %q contains invalid characters (colons or newlines)", mountRoot)
	}
	if strings.ContainsAny(binPath, ":\n\r") {
		return nil, fmt.Errorf("resolved bin path %q contains invalid characters (colons or newlines)", binPath)
	}

	return &ResolvedExtraPath{
		HostMountPath:  mountRoot,
		SandboxBinPath: binPath,
	}, nil
}
