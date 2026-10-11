package sandbox

import (
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// AllowedToolchainNames defines standard development toolchain names
// that are permitted to be auto-discovered into the sandbox.
// Security: Tools holding credentials or agent tokens (e.g. gh, gcloud, aws,
// claude, codex, opencode) are deliberately excluded.
var AllowedToolchainNames = map[string]bool{
	"go":            true,
	"node":          true,
	"nodejs":        true,
	"python":        true,
	"python3":       true,
	"rust":          true,
	"cargo":         true,
	"bun":           true,
	"deno":          true,
	"golangci-lint": true,
	"zig":           true,
	"ruby":          true,
	"java":          true,
}

// DedicatedToolchainNames defines standalone utility toolchains (linters, formatters, etc.)
// that take precedence over general language SDK/compiler toolchains in the sandbox PATH.
var DedicatedToolchainNames = map[string]bool{
	"golangci-lint": true,
}

// DiscoverToolchains scans the host environment and standard version managers
// (such as mise, pyenv, and cargo) for supported developer toolchains and returns
// validated host mount paths suitable for sandbox ExtraPaths.
//
// Dedicated utility toolchains (e.g. golangci-lint) are partitioned to precede general
// language SDK toolchains (e.g. go, rust, python) in the returned list.
// Each candidate path is validated via ResolveExtraPath against protectedPaths
// and blocked roots.
func DiscoverToolchains(pathEnv, cwd string, protectedPaths []string) []string {
	if pathEnv == "" {
		pathEnv = os.Getenv("PATH")
	}

	home, err := os.UserHomeDir()
	if err != nil || home == "" {
		return nil
	}

	var utilityCandidates []string
	var generalCandidates []string
	seenMounts := make(map[string]bool)
	seenTools := make(map[string]bool)

	addCandidate := func(candidate, toolName string) {
		resolved, err := ResolveExtraPath(candidate, cwd, protectedPaths)
		if err != nil {
			return
		}
		if !seenMounts[resolved.HostMountPath] {
			seenMounts[resolved.HostMountPath] = true
			if DedicatedToolchainNames[toolName] {
				utilityCandidates = append(utilityCandidates, resolved.HostMountPath)
			} else {
				generalCandidates = append(generalCandidates, resolved.HostMountPath)
			}
			if toolName != "" {
				seenTools[toolName] = true
			}
		}
	}

	// 1. Discover active toolchains from pathEnv (host $PATH)
	miseInstalls := filepath.Join(home, ".local", "share", "mise", "installs")
	pyenvVersions := filepath.Join(home, ".pyenv", "versions")
	cargoBin := filepath.Join(home, ".cargo", "bin")

	for _, entry := range filepath.SplitList(pathEnv) {
		entry = strings.TrimSpace(entry)
		if entry == "" {
			continue
		}

		// Check if entry is under mise installs
		if IsSubpath(miseInstalls, entry) {
			rel, err := filepath.Rel(miseInstalls, entry)
			if err == nil && rel != "." && !strings.HasPrefix(rel, "..") {
				parts := strings.Split(rel, string(filepath.Separator))
				toolName := strings.ToLower(parts[0])
				if AllowedToolchainNames[toolName] {
					candidate := entry
					if filepath.Base(entry) == "bin" && IsSubpath(miseInstalls, filepath.Dir(entry)) {
						candidate = filepath.Dir(entry)
					}
					addCandidate(candidate, toolName)
				}
			}
			continue
		}

		// Check if entry is under pyenv versions
		if IsSubpath(pyenvVersions, entry) {
			addCandidate(entry, "python")
			continue
		}

		// Check if entry is cargo bin
		if entry == cargoBin || filepath.Clean(entry) == cargoBin {
			addCandidate(entry, "cargo")
			continue
		}
	}

	// 2. Discover from ~/.local/share/mise/installs directory directly
	if fi, err := os.Stat(miseInstalls); err == nil && fi.IsDir() {
		entries, err := os.ReadDir(miseInstalls)
		if err == nil {
			for _, entry := range entries {
				if !entry.IsDir() {
					continue
				}
				toolName := strings.ToLower(entry.Name())
				if !AllowedToolchainNames[toolName] || seenTools[toolName] {
					continue
				}

				toolDir := filepath.Join(miseInstalls, entry.Name())
				candidate := findBestMiseVersionDir(toolDir, toolName)
				if candidate != "" {
					addCandidate(candidate, toolName)
				}
			}
		}
	}

	// 3. Fallback: Check ~/.cargo/bin if rust/cargo wasn't found in PATH
	if !seenTools["cargo"] && !seenTools["rust"] {
		if fi, err := os.Stat(cargoBin); err == nil && fi.IsDir() {
			addCandidate(cargoBin, "cargo")
		}
	}

	return append(utilityCandidates, generalCandidates...)
}

// findBestMiseVersionDir locates the active or newest version directory under a mise tool folder.
func findBestMiseVersionDir(toolDir, toolName string) string {
	// Preferred priority symlinks/names
	for _, preferred := range []string{"latest", "current", "lts"} {
		prefPath := filepath.Join(toolDir, preferred)
		if fi, err := os.Stat(prefPath); err == nil && fi.IsDir() {
			return resolveMiseBinaryPath(prefPath, toolName)
		}
	}

	entries, err := os.ReadDir(toolDir)
	if err != nil {
		return ""
	}

	var versionDirs []string
	for _, e := range entries {
		name := e.Name()
		if strings.HasPrefix(name, ".") {
			continue
		}
		fullPath := filepath.Join(toolDir, name)
		if fi, err := os.Stat(fullPath); err == nil && fi.IsDir() {
			versionDirs = append(versionDirs, fullPath)
		}
	}

	if len(versionDirs) == 0 {
		return ""
	}

	// Sort versions descending (lexicographical/semver approximation)
	sort.Slice(versionDirs, func(i, j int) bool {
		return versionDirs[i] > versionDirs[j]
	})

	return resolveMiseBinaryPath(versionDirs[0], toolName)
}

// resolveMiseBinaryPath returns the toolchain root or inner executable directory.
func resolveMiseBinaryPath(dir, toolName string) string {
	// If dir has a bin/ subdirectory, return dir (ResolveExtraPath will promote bin/)
	if fi, err := os.Stat(filepath.Join(dir, "bin")); err == nil && fi.IsDir() {
		return dir
	}

	// If the binary exists directly in dir, return dir
	if fi, err := os.Stat(filepath.Join(dir, toolName)); err == nil && !fi.IsDir() {
		return dir
	}

	// Some mise tools extract into a nested directory (e.g. golangci-lint/2.13.2/golangci-lint-2.13.2-linux-amd64)
	entries, err := os.ReadDir(dir)
	if err == nil {
		for _, e := range entries {
			if e.IsDir() {
				sub := filepath.Join(dir, e.Name())
				if fi, err := os.Stat(filepath.Join(sub, toolName)); err == nil && !fi.IsDir() {
					return sub
				}
				if fi, err := os.Stat(filepath.Join(sub, "bin")); err == nil && fi.IsDir() {
					return sub
				}
			}
		}
	}

	return dir
}
