package webfetch

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestHeuristicsStaticSites(t *testing.T) {
	files, err := filepath.Glob("testdata/static/*.html")
	require.NoError(t, err)
	require.Equal(t, 10, len(files), "expected exactly 10 static HTML fixtures")

	for _, file := range files {
		t.Run(filepath.Base(file), func(t *testing.T) {
			data, err := os.ReadFile(file)
			require.NoError(t, err)

			res, err := ParseReadability(data, "https://example.com/"+filepath.Base(file))
			require.NoError(t, err, "readability should parse static site without error")
			require.NotNil(t, res)
			assert.NotEmpty(t, res.Title)
			assert.GreaterOrEqual(t, len(strings.TrimSpace(res.TextContent)), MinTextLengthThreshold)

			needsFallback, reason := NeedsDynamicFallback(res, err, string(data))
			assert.False(t, needsFallback, "static site should not require dynamic fallback: reason=%s", reason)
		})
	}
}

func TestHeuristicsDynamicSites(t *testing.T) {
	files, err := filepath.Glob("testdata/dynamic/*.html")
	require.NoError(t, err)
	require.Equal(t, 10, len(files), "expected exactly 10 dynamic SPA HTML fixtures")

	for _, file := range files {
		t.Run(filepath.Base(file), func(t *testing.T) {
			data, err := os.ReadFile(file)
			require.NoError(t, err)

			res, parseErr := ParseReadability(data, "https://example.com/"+filepath.Base(file))
			needsFallback, reason := NeedsDynamicFallback(res, parseErr, string(data))

			assert.True(t, needsFallback, "dynamic SPA site must trigger fallback")
			assert.NotEmpty(t, reason)
		})
	}
}

func TestHeuristicsBotProtectionSites(t *testing.T) {
	files, err := filepath.Glob("testdata/bot_protection/*.html")
	require.NoError(t, err)
	require.Equal(t, 10, len(files), "expected exactly 10 bot protection HTML fixtures")

	for _, file := range files {
		t.Run(filepath.Base(file), func(t *testing.T) {
			data, err := os.ReadFile(file)
			require.NoError(t, err)

			res, parseErr := ParseReadability(data, "https://example.com/"+filepath.Base(file))
			needsFallback, reason := NeedsDynamicFallback(res, parseErr, string(data))

			assert.True(t, needsFallback, "bot protection page must trigger fallback")
			assert.NotEmpty(t, reason)
		})
	}
}
