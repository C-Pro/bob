package cli

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestParseMountSpec(t *testing.T) {
	tmpDir := t.TempDir()
	canonicalTmp, err := filepath.EvalSymlinks(tmpDir)
	require.NoError(t, err)

	workDir := filepath.Join(canonicalTmp, "workspace")
	require.NoError(t, os.Mkdir(workDir, 0755))

	configDir := filepath.Join(canonicalTmp, "config", "bob")
	require.NoError(t, os.MkdirAll(configDir, 0755))
	configFile := filepath.Join(configDir, "config.yaml")
	require.NoError(t, os.WriteFile(configFile, []byte("llm:\n  model: test\n"), 0600))

	dataDir := filepath.Join(canonicalTmp, "data", "bob")
	require.NoError(t, os.MkdirAll(dataDir, 0755))

	protected := []string{configDir, dataDir}

	t.Run("valid ro explicit", func(t *testing.T) {
		m, err := ParseMountSpec(workDir+":ro", "", protected)
		require.NoError(t, err)
		assert.Equal(t, workDir, m.HostPath)
		assert.True(t, m.ReadOnly)
	})

	t.Run("valid rw explicit", func(t *testing.T) {
		m, err := ParseMountSpec(workDir+":rw", "", protected)
		require.NoError(t, err)
		assert.Equal(t, workDir, m.HostPath)
		assert.False(t, m.ReadOnly)
	})

	t.Run("default ro mode", func(t *testing.T) {
		m, err := ParseMountSpec(workDir, "", protected)
		require.NoError(t, err)
		assert.Equal(t, workDir, m.HostPath)
		assert.True(t, m.ReadOnly)
	})

	t.Run("path with colons", func(t *testing.T) {
		colonDir := filepath.Join(canonicalTmp, "dir:with:colons")
		require.NoError(t, os.Mkdir(colonDir, 0755))

		m, err := ParseMountSpec(colonDir+":rw", "", protected)
		require.NoError(t, err)
		assert.Equal(t, colonDir, m.HostPath)
		assert.False(t, m.ReadOnly)

		m2, err := ParseMountSpec(colonDir, "", protected)
		require.NoError(t, err)
		assert.Equal(t, colonDir, m2.HostPath)
		assert.True(t, m2.ReadOnly)
	})

	t.Run("relative path resolution against cwd", func(t *testing.T) {
		subDir := filepath.Join(workDir, "sub")
		require.NoError(t, os.Mkdir(subDir, 0755))

		m, err := ParseMountSpec("./sub:ro", workDir, protected)
		require.NoError(t, err)
		assert.Equal(t, subDir, m.HostPath)
		assert.True(t, m.ReadOnly)

		mDot, err := ParseMountSpec(".:rw", workDir, protected)
		require.NoError(t, err)
		assert.Equal(t, workDir, mDot.HostPath)
		assert.False(t, mDot.ReadOnly)
	})

	t.Run("symlink resolution", func(t *testing.T) {
		symlinkTarget := filepath.Join(workDir, "real_target")
		require.NoError(t, os.Mkdir(symlinkTarget, 0755))
		symlinkPath := filepath.Join(canonicalTmp, "symlink_dir")
		require.NoError(t, os.Symlink(symlinkTarget, symlinkPath))

		m, err := ParseMountSpec(symlinkPath+":ro", "", protected)
		require.NoError(t, err)
		assert.Equal(t, symlinkTarget, m.HostPath)
	})

	t.Run("empty or whitespace spec", func(t *testing.T) {
		_, err := ParseMountSpec("", "", protected)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "cannot be empty")

		_, err = ParseMountSpec("   ", "", protected)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "cannot be empty")
	})

	t.Run("suffix only without path", func(t *testing.T) {
		_, err := ParseMountSpec(":ro", "", protected)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "cannot be empty")

		_, err = ParseMountSpec(":rw", "", protected)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "cannot be empty")
	})

	t.Run("non-existent directory", func(t *testing.T) {
		_, err := ParseMountSpec(filepath.Join(canonicalTmp, "does_not_exist"), "", protected)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed to resolve")
	})

	t.Run("path is file not directory", func(t *testing.T) {
		fileTarget := filepath.Join(workDir, "somefile.txt")
		require.NoError(t, os.WriteFile(fileTarget, []byte("hello"), 0644))

		_, err := ParseMountSpec(fileTarget, "", protected)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "not a directory")
	})

	t.Run("reject root directory", func(t *testing.T) {
		_, err := ParseMountSpec("/", "", protected)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "forbidden")
	})

	t.Run("reject protected path itself", func(t *testing.T) {
		_, err := ParseMountSpec(configDir, "", protected)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "protected directory")
	})

	t.Run("reject descendant of protected path", func(t *testing.T) {
		subConfig := filepath.Join(configDir, "subdir")
		require.NoError(t, os.Mkdir(subConfig, 0755))

		_, err := ParseMountSpec(subConfig, "", protected)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "is inside protected directory")
	})

	t.Run("reject ancestor of protected path", func(t *testing.T) {
		// canonicalTmp contains configDir and dataDir
		_, err := ParseMountSpec(canonicalTmp, "", protected)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "contains protected directory")
	})

	t.Run("reject symlink pointing into protected path", func(t *testing.T) {
		symlinkToConfig := filepath.Join(canonicalTmp, "link_to_config")
		require.NoError(t, os.Symlink(configDir, symlinkToConfig))

		_, err := ParseMountSpec(symlinkToConfig, "", protected)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "protected directory")
	})

	t.Run("protected path not yet created on disk", func(t *testing.T) {
		uncreatedProtected := filepath.Join(canonicalTmp, "uncreated_parent", "bob_uncreated")
		// neither uncreated_parent nor bob_uncreated exist yet

		mountAttempt := filepath.Join(canonicalTmp, "uncreated_parent")
		require.NoError(t, os.Mkdir(mountAttempt, 0755))

		_, err := ParseMountSpec(mountAttempt, "", []string{uncreatedProtected})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "contains protected directory")
	})
}
