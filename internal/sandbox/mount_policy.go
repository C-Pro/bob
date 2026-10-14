package sandbox

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// ValidateWorkspaceDir canonicalizes and validates a candidate workspace directory,
// verifying that it exists, is a directory, does not conflict with blocked system directories,
// is not the user's home directory, and does not overlap with any protected paths.
func ValidateWorkspaceDir(rawPath, cwd string, protectedPaths []string) (string, error) {
	if strings.TrimSpace(rawPath) == "" {
		return "", errors.New("workspace directory cannot be empty")
	}

	// Expand leading ~ or ~/
	expanded, err := ExpandHome(rawPath)
	if err != nil {
		return "", fmt.Errorf("failed to expand workspace directory %q: %w", rawPath, err)
	}

	// Resolve relative path against cwd
	if !filepath.IsAbs(expanded) {
		if cwd == "" {
			var err error
			cwd, err = os.Getwd()
			if err != nil {
				return "", fmt.Errorf("failed to get working directory: %w", err)
			}
		}
		expanded = filepath.Join(cwd, expanded)
	}
	expanded = filepath.Clean(expanded)

	// Canonicalize and evaluate symlinks
	canonicalHost, err := filepath.EvalSymlinks(expanded)
	if err != nil {
		if os.IsNotExist(err) {
			return "", fmt.Errorf("failed to resolve workspace directory %q: path does not exist: %w", rawPath, err)
		}
		return "", fmt.Errorf("failed to resolve workspace directory %q: %w", rawPath, err)
	}

	// Verify target exists and is a directory
	stat, err := os.Stat(canonicalHost)
	if err != nil {
		return "", fmt.Errorf("workspace directory %q does not exist: %w", canonicalHost, err)
	}
	if !stat.IsDir() {
		return "", fmt.Errorf("workspace directory %q is not a directory", canonicalHost)
	}

	if canonicalHost == "/" {
		return "", errors.New("mounting root directory '/' is forbidden")
	}

	// Reject user's home directory explicitly
	home, err := os.UserHomeDir()
	if err == nil && home != "" {
		realHome, _ := filepath.EvalSymlinks(home)
		if canonicalHost == home || (realHome != "" && canonicalHost == realHome) {
			return "", fmt.Errorf("mounting user home directory %q is forbidden; mount a specific project directory instead", canonicalHost)
		}
	}

	// Check blocked system paths (/proc, /sys, /dev, /run, /var/run, /root, /etc, /boot, /usr, /bin, /sbin, /lib, /tmp, etc.)
	if blocked, isBlocked := isBlockedHostPath(canonicalHost); isBlocked {
		return "", fmt.Errorf("workspace directory %q conflicts with system directory %s", canonicalHost, blocked)
	}

	for _, p := range protectedPaths {
		if p == "" {
			continue
		}
		pExpanded, expErr := ExpandHome(p)
		if expErr != nil {
			pExpanded = p
		}
		pCanon := CanonicalizeProtected(pExpanded)
		if IsSubpath(canonicalHost, pCanon) {
			return "", fmt.Errorf("workspace directory %q contains protected directory %q", canonicalHost, pCanon)
		}
		if IsSubpath(pCanon, canonicalHost) {
			return "", fmt.Errorf("workspace directory %q is inside protected directory %q", canonicalHost, pCanon)
		}
	}

	return canonicalHost, nil
}
