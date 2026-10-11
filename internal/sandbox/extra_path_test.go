package sandbox

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestResolveExtraPath(t *testing.T) {
	tmpDir := t.TempDir()
	realTmp, err := filepath.EvalSymlinks(tmpDir)
	require.NoError(t, err)

	// Setup mock directories
	standaloneBin := filepath.Join(realTmp, "standalone_bin")
	require.NoError(t, os.MkdirAll(standaloneBin, 0o755))

	toolchainRoot := filepath.Join(realTmp, "toolchain")
	toolchainBin := filepath.Join(toolchainRoot, "bin")
	toolchainLib := filepath.Join(toolchainRoot, "lib")
	require.NoError(t, os.MkdirAll(toolchainBin, 0o755))
	require.NoError(t, os.MkdirAll(toolchainLib, 0o755))

	protectedDir := filepath.Join(realTmp, "protected")
	require.NoError(t, os.MkdirAll(protectedDir, 0o755))

	protectedSub := filepath.Join(protectedDir, "sub")
	require.NoError(t, os.MkdirAll(protectedSub, 0o755))

	t.Run("empty path", func(t *testing.T) {
		_, err := ResolveExtraPath("", realTmp, nil)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "cannot be empty")

		_, err = ResolveExtraPath("   ", realTmp, nil)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "cannot be empty")
	})

	t.Run("invalid characters", func(t *testing.T) {
		_, err := ResolveExtraPath(standaloneBin+":/bin", realTmp, nil)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "contains invalid characters")

		_, err = ResolveExtraPath(standaloneBin+"\n", realTmp, nil)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "contains invalid characters")
	})

	t.Run("non-existent path", func(t *testing.T) {
		_, err := ResolveExtraPath(filepath.Join(realTmp, "does_not_exist"), realTmp, nil)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "does not exist")
	})

	t.Run("file instead of directory", func(t *testing.T) {
		filePath := filepath.Join(realTmp, "testfile")
		require.NoError(t, os.WriteFile(filePath, []byte("content"), 0o644))

		_, err := ResolveExtraPath(filePath, realTmp, nil)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "is not a directory")
	})

	t.Run("root directory forbidden", func(t *testing.T) {
		_, err := ResolveExtraPath("/", realTmp, nil)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "mounting root directory")
	})

	t.Run("user home forbidden", func(t *testing.T) {
		home, err := os.UserHomeDir()
		if err == nil && home != "" {
			_, err := ResolveExtraPath("~", realTmp, nil)
			assert.Error(t, err)
			assert.Contains(t, err.Error(), "user home directory as extra path is forbidden")

			_, err = ResolveExtraPath(home, realTmp, nil)
			assert.Error(t, err)
			assert.Contains(t, err.Error(), "user home directory as extra path is forbidden")
		}
	})

	t.Run("protected path containment", func(t *testing.T) {
		// Inside protected directory
		_, err := ResolveExtraPath(protectedSub, realTmp, []string{protectedDir})
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "is inside protected path")

		// Exactly protected directory
		_, err = ResolveExtraPath(protectedDir, realTmp, []string{protectedDir})
		assert.Error(t, err)

		// Contains protected directory (ancestor)
		_, err = ResolveExtraPath(realTmp, realTmp, []string{protectedDir})
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "contains protected path")
	})

	t.Run("uncreated protected path", func(t *testing.T) {
		candidateDir := filepath.Join(realTmp, "existing_candidate")
		require.NoError(t, os.MkdirAll(candidateDir, 0o755))
		uncreated := filepath.Join(candidateDir, "nested_uncreated_secret")

		_, err := ResolveExtraPath(candidateDir, realTmp, []string{uncreated})
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "contains protected path")
	})

	t.Run("standalone bin directory", func(t *testing.T) {
		resolved, err := ResolveExtraPath(standaloneBin, realTmp, []string{protectedDir})
		require.NoError(t, err)
		assert.Equal(t, standaloneBin, resolved.HostMountPath)
		assert.Equal(t, standaloneBin, resolved.SandboxBinPath)
	})

	t.Run("toolchain root with bin subdir", func(t *testing.T) {
		resolved, err := ResolveExtraPath(toolchainRoot, realTmp, []string{protectedDir})
		require.NoError(t, err)
		assert.Equal(t, toolchainRoot, resolved.HostMountPath)
		assert.Equal(t, toolchainBin, resolved.SandboxBinPath)
	})

	t.Run("toolchain bin promoted to root with lib", func(t *testing.T) {
		resolved, err := ResolveExtraPath(toolchainBin, realTmp, []string{protectedDir})
		require.NoError(t, err)
		assert.Equal(t, toolchainRoot, resolved.HostMountPath)
		assert.Equal(t, toolchainBin, resolved.SandboxBinPath)
	})

	t.Run("relative path resolution", func(t *testing.T) {
		resolved, err := ResolveExtraPath("standalone_bin", realTmp, []string{protectedDir})
		require.NoError(t, err)
		assert.Equal(t, standaloneBin, resolved.HostMountPath)
		assert.Equal(t, standaloneBin, resolved.SandboxBinPath)
	})

	t.Run("symlink resolution", func(t *testing.T) {
		symlinkTarget := filepath.Join(realTmp, "symlink_bin")
		require.NoError(t, os.Symlink(standaloneBin, symlinkTarget))

		resolved, err := ResolveExtraPath(symlinkTarget, realTmp, []string{protectedDir})
		require.NoError(t, err)
		assert.Equal(t, standaloneBin, resolved.HostMountPath)
		assert.Equal(t, standaloneBin, resolved.SandboxBinPath)
	})

	t.Run("escaping bin symlink rejected", func(t *testing.T) {
		escapeRoot := filepath.Join(realTmp, "escape_toolchain")
		require.NoError(t, os.MkdirAll(escapeRoot, 0o755))
		outsideBin := filepath.Join(realTmp, "outside_bin")
		require.NoError(t, os.MkdirAll(outsideBin, 0o755))
		require.NoError(t, os.Symlink(outsideBin, filepath.Join(escapeRoot, "bin")))

		_, err := ResolveExtraPath(escapeRoot, realTmp, nil)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "escapes mount root")
	})

	t.Run("conflict with blocked system directory", func(t *testing.T) {
		for _, blocked := range []string{"/proc", "/dev", "/sys"} {
			if _, err := os.Stat(blocked); err == nil {
				_, err := ResolveExtraPath(blocked, realTmp, nil)
				assert.Error(t, err)
				assert.Contains(t, err.Error(), "conflicts with system directory")
			}
		}
	})
}
