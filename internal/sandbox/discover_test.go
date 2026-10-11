package sandbox

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDiscoverToolchains_PathEnvFiltering(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)

	miseDir := filepath.Join(home, ".local", "share", "mise", "installs")

	// Create allowed tools
	goDir := filepath.Join(miseDir, "go", "1.27.1")
	require.NoError(t, os.MkdirAll(filepath.Join(goDir, "bin"), 0o755))
	require.NoError(t, os.MkdirAll(filepath.Join(goDir, "src"), 0o755))

	nodeDir := filepath.Join(miseDir, "node", "24.21.0")
	require.NoError(t, os.MkdirAll(filepath.Join(nodeDir, "bin"), 0o755))

	// Create disallowed tools (contain credentials/tokens)
	ghDir := filepath.Join(miseDir, "gh", "2.100.0")
	require.NoError(t, os.MkdirAll(filepath.Join(ghDir, "bin"), 0o755))

	claudeDir := filepath.Join(miseDir, "claude", "latest")
	require.NoError(t, os.MkdirAll(filepath.Join(claudeDir, "bin"), 0o755))

	codexDir := filepath.Join(miseDir, "codex", "latest")
	require.NoError(t, os.MkdirAll(filepath.Join(codexDir, "bin"), 0o755))

	// Build mock PATH
	pathEnv := stringsJoinPath(
		filepath.Join(goDir, "bin"),
		filepath.Join(nodeDir, "bin"),
		filepath.Join(ghDir, "bin"),
		filepath.Join(claudeDir, "bin"),
		filepath.Join(codexDir, "bin"),
		"/usr/bin",
	)

	discovered := DiscoverToolchains(pathEnv, t.TempDir(), nil)

	// Verify only allowed tools (go, node) are discovered
	assert.Contains(t, discovered, goDir)
	assert.Contains(t, discovered, nodeDir)
	assert.NotContains(t, discovered, ghDir)
	assert.NotContains(t, discovered, claudeDir)
	assert.NotContains(t, discovered, codexDir)
	assert.Len(t, discovered, 2)
}

func TestDiscoverToolchains_FromMiseDirectory(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)

	miseDir := filepath.Join(home, ".local", "share", "mise", "installs")

	// Create go versions: 1.25.0 and latest -> 1.27.0
	goOld := filepath.Join(miseDir, "go", "1.25.0")
	require.NoError(t, os.MkdirAll(filepath.Join(goOld, "bin"), 0o755))

	goLatest := filepath.Join(miseDir, "go", "1.27.0")
	require.NoError(t, os.MkdirAll(filepath.Join(goLatest, "bin"), 0o755))
	require.NoError(t, os.Symlink(goLatest, filepath.Join(miseDir, "go", "latest")))

	// Create golangci-lint with nested directory structure
	lintDir := filepath.Join(miseDir, "golangci-lint", "2.13.2", "golangci-lint-2.13.2-linux-amd64")
	require.NoError(t, os.MkdirAll(lintDir, 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(lintDir, "golangci-lint"), []byte("#!/bin/sh\n"), 0o755))

	// Empty PATH so discovery relies purely on directory scanning
	discovered := DiscoverToolchains("/usr/bin:/bin", t.TempDir(), nil)

	assert.Contains(t, discovered, goLatest)
	assert.Contains(t, discovered, lintDir)
	assert.NotContains(t, discovered, goOld)
}

func TestDiscoverToolchains_ProtectedPathRejection(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)

	miseDir := filepath.Join(home, ".local", "share", "mise", "installs")
	goDir := filepath.Join(miseDir, "go", "1.27.1")
	require.NoError(t, os.MkdirAll(filepath.Join(goDir, "bin"), 0o755))

	// Protect goDir explicitly
	protected := []string{goDir}

	pathEnv := filepath.Join(goDir, "bin")
	discovered := DiscoverToolchains(pathEnv, t.TempDir(), protected)

	assert.Empty(t, discovered)
}

func TestDiscoverToolchains_RealHostEnvironment(t *testing.T) {
	// If mise is installed on host, verify that Go is automatically discovered
	home, err := os.UserHomeDir()
	if err != nil {
		t.Skip("no home dir")
	}
	miseGo := filepath.Join(home, ".local", "share", "mise", "installs", "go")
	if fi, err := os.Stat(miseGo); err != nil || !fi.IsDir() {
		t.Skip("mise go not present on host")
	}

	discovered := DiscoverToolchains("", t.TempDir(), nil)
	assert.NotEmpty(t, discovered)

	foundGo := false
	for _, p := range discovered {
		if strings.Contains(p, "/mise/installs/go") {
			foundGo = true
			break
		}
	}
	assert.True(t, foundGo, "expected to discover mise go install, got: %v", discovered)
}

func TestDiscoverToolchains_DedicatedUtilityPrecedence(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)

	miseDir := filepath.Join(home, ".local", "share", "mise", "installs")

	// Set up go install
	goDir := filepath.Join(miseDir, "go", "1.27.1")
	require.NoError(t, os.MkdirAll(filepath.Join(goDir, "bin"), 0o755))

	// Set up golangci-lint install
	lintDir := filepath.Join(miseDir, "golangci-lint", "2.13.2", "golangci-lint-2.13.2-linux-amd64")
	require.NoError(t, os.MkdirAll(lintDir, 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(lintDir, "golangci-lint"), []byte("#!/bin/sh\n"), 0o755))

	// Host PATH deliberately lists go BEFORE golangci-lint
	pathEnv := stringsJoinPath(
		filepath.Join(goDir, "bin"),
		lintDir,
		"/usr/bin",
	)

	discovered := DiscoverToolchains(pathEnv, t.TempDir(), nil)
	require.Len(t, discovered, 2)

	// Verify golangci-lint (dedicated utility) precedes go (general SDK) regardless of PATH order
	assert.Equal(t, lintDir, discovered[0], "expected dedicated utility to take first precedence")
	assert.Equal(t, goDir, discovered[1], "expected general language SDK to follow dedicated utilities")
}

func stringsJoinPath(elems ...string) string {
	res := ""
	for i, e := range elems {
		if i > 0 {
			res += string(filepath.ListSeparator)
		}
		res += e
	}
	return res
}
