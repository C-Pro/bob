package tools

import (
	"errors"
	"net"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type mockRetryableErr struct {
	msg       string
	retryable bool
}

func (m *mockRetryableErr) Error() string {
	return m.msg
}

func (m *mockRetryableErr) IsRetryable() bool {
	return m.retryable
}

type mockNetTimeoutErr struct{}

func (m *mockNetTimeoutErr) Error() string   { return "network timeout" }
func (m *mockNetTimeoutErr) Timeout() bool   { return true }
func (m *mockNetTimeoutErr) Temporary() bool { return true }

func TestToolError(t *testing.T) {
	underlying := errors.New("base error")
	te := NewToolError("test_tool", underlying, true)

	assert.Equal(t, "test_tool: base error", te.Error())
	assert.True(t, te.IsRetryable())
	assert.Equal(t, underlying, te.Unwrap())

	var retryable RetryableError
	require.True(t, errors.As(te, &retryable))
	assert.True(t, retryable.IsRetryable())

	teNoName := NewToolError("", underlying, false)
	assert.Equal(t, "base error", teNoName.Error())
	assert.False(t, teNoName.IsRetryable())
}

func TestClassifyWebSearchError(t *testing.T) {
	assert.Nil(t, ClassifyWebSearchError(nil))

	// Non-retryable error
	err1 := errors.New("invalid parameter")
	classified1 := ClassifyWebSearchError(err1)
	require.NotNil(t, classified1)
	var te1 *ToolError
	require.True(t, errors.As(classified1, &te1))
	assert.Equal(t, "web_search", te1.ToolName)
	assert.False(t, te1.IsRetryable())

	// Already a ToolError
	assert.Equal(t, te1, ClassifyWebSearchError(te1))

	// Retryable error via interface
	err2 := &mockRetryableErr{msg: "rate limit", retryable: true}
	classified2 := ClassifyWebSearchError(err2)
	require.NotNil(t, classified2)
	var te2 *ToolError
	require.True(t, errors.As(classified2, &te2))
	assert.True(t, te2.IsRetryable())

	// Network timeout error
	err3 := &mockNetTimeoutErr{}
	classified3 := ClassifyWebSearchError(err3)
	require.NotNil(t, classified3)
	var te3 *ToolError
	require.True(t, errors.As(classified3, &te3))
	assert.True(t, te3.IsRetryable())
}

func TestClassifyWebFetchError(t *testing.T) {
	assert.Nil(t, ClassifyWebFetchError(nil))

	// Non-retryable error
	err1 := errors.New("url cannot be empty")
	classified1 := ClassifyWebFetchError(err1)
	require.NotNil(t, classified1)
	var te1 *ToolError
	require.True(t, errors.As(classified1, &te1))
	assert.Equal(t, "web_fetch", te1.ToolName)
	assert.False(t, te1.IsRetryable())

	// Already a ToolError
	assert.Equal(t, te1, ClassifyWebFetchError(te1))

	// Retryable error via interface
	err2 := &mockRetryableErr{msg: "service unavailable", retryable: true}
	classified2 := ClassifyWebFetchError(err2)
	require.NotNil(t, classified2)
	var te2 *ToolError
	require.True(t, errors.As(classified2, &te2))
	assert.True(t, te2.IsRetryable())

	// Network timeout error
	var netErr net.Error = &mockNetTimeoutErr{}
	classified3 := ClassifyWebFetchError(netErr)
	require.NotNil(t, classified3)
	var te3 *ToolError
	require.True(t, errors.As(classified3, &te3))
	assert.True(t, te3.IsRetryable())
}
