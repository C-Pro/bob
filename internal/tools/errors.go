package tools

import (
	"errors"
	"fmt"
	"net"
)

// RetryableError is an interface implemented by errors that indicate whether an operation can be retried.
type RetryableError interface {
	error
	IsRetryable() bool
}

// ToolError represents a classified error returned by a tool invocation.
type ToolError struct {
	ToolName  string
	Err       error
	Retryable bool
}

// Error returns the formatted error string including the tool name if set.
func (e *ToolError) Error() string {
	if e.ToolName != "" {
		return fmt.Sprintf("%s: %v", e.ToolName, e.Err)
	}
	return e.Err.Error()
}

// Unwrap returns the underlying error.
func (e *ToolError) Unwrap() error {
	return e.Err
}

// IsRetryable reports whether the tool error is transient and eligible for retry.
func (e *ToolError) IsRetryable() bool {
	return e.Retryable
}

// NewToolError creates a new classified ToolError.
func NewToolError(toolName string, err error, retryable bool) *ToolError {
	return &ToolError{
		ToolName:  toolName,
		Err:       err,
		Retryable: retryable,
	}
}

// ClassifyWebSearchError wraps an error from web_search into a *ToolError with retryability classification.
func ClassifyWebSearchError(err error) error {
	if err == nil {
		return nil
	}
	var toolErr *ToolError
	if errors.As(err, &toolErr) {
		return toolErr
	}
	var retryable RetryableError
	if errors.As(err, &retryable) {
		return NewToolError("web_search", err, retryable.IsRetryable())
	}
	var netErr net.Error
	if errors.As(err, &netErr) && netErr.Timeout() {
		return NewToolError("web_search", err, true)
	}
	return NewToolError("web_search", err, false)
}

// ClassifyWebFetchError wraps an error from web_fetch into a *ToolError with retryability classification.
func ClassifyWebFetchError(err error) error {
	if err == nil {
		return nil
	}
	var toolErr *ToolError
	if errors.As(err, &toolErr) {
		return toolErr
	}
	var retryable RetryableError
	if errors.As(err, &retryable) {
		return NewToolError("web_fetch", err, retryable.IsRetryable())
	}
	var netErr net.Error
	if errors.As(err, &netErr) && netErr.Timeout() {
		return NewToolError("web_fetch", err, true)
	}
	return NewToolError("web_fetch", err, false)
}
