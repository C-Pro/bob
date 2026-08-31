package backup

import (
	"bytes"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestHeaderWriteAndRead(t *testing.T) {
	salt := []byte("1234567890123456")
	payload := []byte("encrypted-payload-bytes")

	var buf bytes.Buffer
	err := WriteHeader(&buf, Header{Salt: salt}, payload)
	require.NoError(t, err)

	data := buf.Bytes()
	assert.True(t, bytes.HasPrefix(data, []byte("BOBB\x01\x10")))

	h, remaining, err := ReadHeader(data)
	require.NoError(t, err)
	assert.Equal(t, HeaderVersion1, h.Version)
	assert.Equal(t, salt, h.Salt)
	assert.Equal(t, payload, remaining)
}

func TestHeaderInvalidArtifacts(t *testing.T) {
	// Too short
	_, _, err := ReadHeader([]byte("BOB"))
	assert.ErrorContains(t, err, "too short")

	// Bad magic
	_, _, err = ReadHeader([]byte("BSKB\x01\x101234567890123456payload"))
	assert.ErrorContains(t, err, "bad magic")

	// Unsupported version
	_, _, err = ReadHeader([]byte("BOBB\x02\x101234567890123456payload"))
	assert.ErrorContains(t, err, "unsupported header version")

	// Truncated salt
	_, _, err = ReadHeader([]byte("BOBB\x01\x101234"))
	assert.ErrorContains(t, err, "truncated salt")
}
