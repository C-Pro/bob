package fsm

import (
	"encoding/json"
	"fmt"
	"strings"

	"bob/internal/agentapi"
)

// StoredToolResultVersion is the current version tag for stored FSM step result envelopes.
const StoredToolResultVersion = 1

type storedToolResult struct {
	Version int                 `json:"bob_tool_result_version"`
	Result  agentapi.ToolResult `json:"result"`
}

// EncodeStoredToolResult packs an agentapi.ToolResult into a versioned JSON envelope.
// If the content exceeds MaxStepResultBytes, it is truncated before wrapping so that
// the envelope is always valid, well-formed JSON.
func EncodeStoredToolResult(res agentapi.ToolResult) (string, error) {
	if len(res.Content) > MaxStepResultBytes {
		res.Content = res.Content[:MaxStepResultBytes] + StepTruncationNotice
	}
	env := storedToolResult{
		Version: StoredToolResultVersion,
		Result:  res,
	}
	b, err := json.Marshal(env)
	if err != nil {
		return "", fmt.Errorf("failed to marshal stored tool result envelope: %w", err)
	}
	return string(b), nil
}

// DecodeStoredToolResult unpacks an agentapi.ToolResult from a stored string.
// If the string contains a valid versioned envelope, it is unmarshaled into ToolResult.
// If it is a legacy raw string or non-envelope JSON, it is treated as raw ToolResult.Content.
func DecodeStoredToolResult(raw string) (agentapi.ToolResult, error) {
	if raw == "" {
		return agentapi.ToolResult{}, nil
	}
	trimmed := strings.TrimSpace(raw)
	if strings.HasPrefix(trimmed, "{") && strings.Contains(trimmed, `"bob_tool_result_version"`) {
		var env storedToolResult
		if err := json.Unmarshal([]byte(trimmed), &env); err == nil && env.Version > 0 {
			return env.Result, nil
		}
	}
	return agentapi.ToolResult{Content: raw}, nil
}
