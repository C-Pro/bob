package fsm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"

	"bob/internal/llm"

	openai "github.com/sashabaranov/go-openai"
)

const (
	systemOneLoopGuardHistoryWindow = 12
	maxSystemOneGoalBytes           = 2048
)

var systemOneSanitizedStateFields = []string{
	"original_objective",
	"immutable_constraints",
	"verified_findings",
	"decisions_and_rejected",
	"current_subgoal",
	"next_intended_actions",
	"unresolved_blockers",
	"produced_artifacts",
	"active_skill_versions",
	"active_memory_versions",
}

// SystemOneStateRenderResult holds the go-probe-sanitized-v1 prompt text and truncation flag.
type SystemOneStateRenderResult struct {
	Text      string
	Truncated bool
}

// SystemOneDecisionClient abstracts the System-1 sidecar probe calls used by FSM control plane.
type SystemOneDecisionClient interface {
	Model() string
	DecideCompaction(ctx context.Context, state string, maxSeqLen int) (*llm.DecisionResultEntry, *llm.DecisionResponse, error)
	DecideLoopGuard(ctx context.Context, state string, images []string) (*llm.DecisionResultEntry, *llm.DecisionResponse, error)
}

// RenderSystemOneCompactionState renders a compaction probe state using go-probe-sanitized-v1
// semantics matching eval/harness/build_jev_dataset.py.
func RenderSystemOneCompactionState(input CompactionProbeInput) SystemOneStateRenderResult {
	execIter := 0
	if input.Run != nil {
		execIter = input.Run.Iteration
	}
	goal := strings.TrimSpace(input.OriginalUserGoal)
	if input.PreviousState != nil && input.PreviousState.OriginalObjective != "" {
		goal = input.PreviousState.OriginalObjective
	}
	return renderSystemOneProbeState(
		input.TargetIter,
		execIter,
		goal,
		input.PreviousState,
		input.StepsToCompact,
		input.TailSteps,
		nil,
	)
}

// RenderSystemOneLoopGuardState renders a loop-guard probe state using go-probe-sanitized-v1
// with a trailing window of up to 12 steps and candidate tool calls.
func RenderSystemOneLoopGuardState(run *FSMRun, goal string, history []FSMStep, candidateCalls []openai.ToolCall) SystemOneStateRenderResult {
	window := history
	if len(window) > systemOneLoopGuardHistoryWindow {
		window = window[len(window)-systemOneLoopGuardHistoryWindow:]
	}

	cutoffIter := 0
	for _, s := range window {
		if s.Iteration > cutoffIter {
			cutoffIter = s.Iteration
		}
	}

	execIter := 0
	if run != nil {
		execIter = run.Iteration
	}

	return renderSystemOneProbeState(
		cutoffIter,
		execIter,
		goal,
		nil,
		window,
		nil,
		candidateCalls,
	)
}

func renderSystemOneProbeState(
	cutoffIter int,
	execIter int,
	goal string,
	previous *CompactedTaskState,
	steps []FSMStep,
	tail []FSMStep,
	candidates []openai.ToolCall,
) SystemOneStateRenderResult {
	truncated := false
	clip := func(text string, budget, reserve int, suffix string) string {
		clipped, didClip := clipUTF8ForSystemOne(text, budget, reserve, suffix)
		if didClip {
			truncated = true
		}
		return clipped
	}

	var b strings.Builder
	fmt.Fprintf(&b, "Candidate Compaction Boundary: Iterations 1..%d (Current Run Iteration: %d)\n", cutoffIter, execIter)
	clippedGoal := clip(goal, maxSystemOneGoalBytes, 15, "...[truncated]")
	fmt.Fprintf(&b, "Original Objective: %s\n\n", clippedGoal)

	prevUpper := 0
	if previous != nil {
		prevUpper = previous.CompactedUpToIter
		sanitized := sanitizePreviousStateCompactJSON(previous)
		clippedPrev := clip(sanitized, maxProbePrevStateBytes, 15, "...[truncated]")
		fmt.Fprintf(&b, "Previous Compacted State (v%d, up to iter %d):\n%s\n\n", previous.Version, previous.CompactedUpToIter, clippedPrev)
	}

	b.WriteString("Candidate Steps to Compact:\n")
	for _, s := range steps {
		if s.Iteration <= prevUpper || s.Iteration > cutoffIter {
			continue
		}
		args := clip(strings.TrimSpace(canonicalArgsText(s.ArgsJSON)), maxProbeStepArgsBytes, 23, "...[truncated for probe]")
		obs := strings.TrimSpace(storedResultForProbe(s.ResultJSON))
		if obs == "" {
			obs = s.ErrorText
		}
		obs = clip(obs, maxProbeStepOutputBytes, 23, "...[truncated for probe]")
		fmt.Fprintf(&b, "- Iter=%d Tool=%s Status=%s Args=%s\n  Output: %s\n", s.Iteration, s.ToolName, s.Status, args, obs)
		if b.Len() >= maxProbePromptBytes {
			b.WriteString("...[additional candidate steps truncated for probe budget]\n")
			truncated = true
			break
		}
	}

	if len(tail) > 0 && b.Len() < maxProbePromptBytes {
		b.WriteString("\nRetained Raw Tail Steps (not compacted in this pass):\n")
		for _, s := range tail {
			if s.Iteration > cutoffIter && s.Iteration <= execIter {
				fmt.Fprintf(&b, "- Iter=%d Tool=%s Status=%s\n", s.Iteration, s.ToolName, s.Status)
			}
		}
	}

	if len(candidates) > 0 {
		b.WriteString("\nCandidate Tool Calls (not yet executed):\n")
		for _, tc := range candidates {
			args := clip(tc.Function.Arguments, maxProbeStepArgsBytes, 23, "...[truncated for probe]")
			fmt.Fprintf(&b, "- Tool=%s Args=%s\n", tc.Function.Name, args)
		}
	}

	finalText := clip(b.String(), maxProbePromptBytes, 16, "\n...[truncated]")
	return SystemOneStateRenderResult{
		Text:      finalText,
		Truncated: truncated,
	}
}

func clipUTF8ForSystemOne(text string, budget, reserve int, suffix string) (string, bool) {
	if len(text) <= budget {
		return text, false
	}
	limit := budget - reserve
	if limit < 0 {
		limit = 0
	}
	if limit > len(text) {
		limit = len(text)
	}
	return strings.ToValidUTF8(text[:limit], "") + suffix, true
}

func compactJSONString(v any) (string, error) {
	buf, err := appendCompactPythonJSON(nil, v)
	if err != nil {
		return "", err
	}
	return string(buf), nil
}

func appendCompactPythonJSON(dst []byte, v any) ([]byte, error) {
	switch val := v.(type) {
	case nil:
		return append(dst, "null"...), nil
	case bool:
		if val {
			return append(dst, "true"...), nil
		}
		return append(dst, "false"...), nil
	case string:
		return appendPythonJSONString(dst, val), nil
	case json.Number:
		return appendPythonJSONNumber(dst, val)
	case float64:
		return append(dst, formatPythonFloat(val)...), nil
	case int:
		return strconv.AppendInt(dst, int64(val), 10), nil
	case int64:
		return strconv.AppendInt(dst, val, 10), nil
	case []string:
		dst = append(dst, '[')
		for i, item := range val {
			if i > 0 {
				dst = append(dst, ',')
			}
			dst = appendPythonJSONString(dst, item)
		}
		return append(dst, ']'), nil
	case []map[string]string:
		dst = append(dst, '[')
		for i, item := range val {
			if i > 0 {
				dst = append(dst, ',')
			}
			var err error
			dst, err = appendCompactPythonJSON(dst, item)
			if err != nil {
				return nil, err
			}
		}
		return append(dst, ']'), nil
	case []any:
		dst = append(dst, '[')
		for i, item := range val {
			if i > 0 {
				dst = append(dst, ',')
			}
			var err error
			dst, err = appendCompactPythonJSON(dst, item)
			if err != nil {
				return nil, err
			}
		}
		return append(dst, ']'), nil
	case map[string]string:
		keys := make([]string, 0, len(val))
		for k := range val {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		dst = append(dst, '{')
		for i, k := range keys {
			if i > 0 {
				dst = append(dst, ',')
			}
			dst = appendPythonJSONString(dst, k)
			dst = append(dst, ':')
			dst = appendPythonJSONString(dst, val[k])
		}
		return append(dst, '}'), nil
	case map[string]any:
		keys := make([]string, 0, len(val))
		for k := range val {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		dst = append(dst, '{')
		for i, k := range keys {
			if i > 0 {
				dst = append(dst, ',')
			}
			dst = appendPythonJSONString(dst, k)
			dst = append(dst, ':')
			var err error
			dst, err = appendCompactPythonJSON(dst, val[k])
			if err != nil {
				return nil, err
			}
		}
		return append(dst, '}'), nil
	default:
		// Fallback via encoding/json roundtrip into map/slice primitives.
		raw, err := json.Marshal(v)
		if err != nil {
			return nil, err
		}
		dec := json.NewDecoder(bytes.NewReader(raw))
		dec.UseNumber()
		var normalized any
		if err := dec.Decode(&normalized); err != nil {
			return nil, err
		}
		return appendCompactPythonJSON(dst, normalized)
	}
}

func appendPythonJSONString(dst []byte, s string) []byte {
	const hexDigits = "0123456789abcdef"
	dst = append(dst, '"')
	for i := 0; i < len(s); {
		r, width := utf8.DecodeRuneInString(s[i:])
		switch {
		case r == '"':
			dst = append(dst, '\\', '"')
		case r == '\\':
			dst = append(dst, '\\', '\\')
		case r == '\b':
			dst = append(dst, '\\', 'b')
		case r == '\f':
			dst = append(dst, '\\', 'f')
		case r == '\n':
			dst = append(dst, '\\', 'n')
		case r == '\r':
			dst = append(dst, '\\', 'r')
		case r == '\t':
			dst = append(dst, '\\', 't')
		case r < 0x20:
			dst = append(dst, '\\', 'u', '0', '0', hexDigits[r>>4], hexDigits[r&0xF])
		default:
			dst = append(dst, s[i:i+width]...)
		}
		i += width
	}
	return append(dst, '"')
}

func appendPythonJSONNumber(dst []byte, num json.Number) ([]byte, error) {
	raw := num.String()
	if !strings.ContainsAny(raw, ".eE") {
		return append(dst, raw...), nil
	}
	f, err := strconv.ParseFloat(raw, 64)
	if err != nil {
		return nil, err
	}
	return append(dst, formatPythonFloat(f)...), nil
}

func formatPythonFloat(f float64) string {
	if math.IsNaN(f) || math.IsInf(f, 0) {
		return "null"
	}
	abs := math.Abs(f)
	if abs >= 1e16 || (f != 0 && abs < 1e-4) {
		return strconv.FormatFloat(f, 'g', -1, 64)
	}
	s := strconv.FormatFloat(f, 'f', -1, 64)
	if !strings.ContainsRune(s, '.') {
		s += ".0"
	}
	return s
}

func decodeJSONBounded(raw string) (any, bool) {
	var current any = raw
	decodedAny := false
	for i := 0; i < 8; i++ {
		str, ok := current.(string)
		if !ok {
			break
		}
		dec := json.NewDecoder(strings.NewReader(str))
		dec.UseNumber()
		var next any
		if err := dec.Decode(&next); err != nil {
			break
		}
		// Require exact EOF so malformed inputs like `{"x":1} garbage` stay raw strings like Python json.loads.
		var trailing any
		if err := dec.Decode(&trailing); !errors.Is(err, io.EOF) {
			break
		}
		if nextStr, ok := next.(string); ok && nextStr == str {
			break
		}
		current = next
		decodedAny = true
	}
	return current, decodedAny
}

func asProbeText(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	encoded, err := compactJSONString(v)
	if err != nil {
		return ""
	}
	return encoded
}

func canonicalArgsText(rawArgs string) string {
	decoded, ok := decodeJSONBounded(rawArgs)
	if !ok {
		return strings.TrimSpace(rawArgs)
	}
	if s, isStr := decoded.(string); isStr {
		return strings.TrimSpace(s)
	}
	return asProbeText(decoded)
}

func storedResultForProbe(rawResult string) string {
	decoded, ok := decodeJSONBounded(rawResult)
	if ok {
		if m, isMap := decoded.(map[string]any); isMap {
			if verNum, hasVer := m["bob_tool_result_version"]; hasVer && jsonNumberPositive(verNum) {
				inner := m["result"]
				if innerStr, ok := inner.(string); ok {
					if d, ok := decodeJSONBounded(innerStr); ok {
						inner = d
					}
				}
				if innerMap, ok := inner.(map[string]any); ok {
					content, exists := innerMap["content"]
					if !exists || content == nil {
						return ""
					}
					return asProbeText(content)
				}
			}
		}
		if s, isStr := decoded.(string); isStr {
			return s
		}
	}
	return rawResult
}

func jsonNumberPositive(v any) bool {
	switch n := v.(type) {
	case json.Number:
		if f, err := n.Float64(); err == nil {
			return f > 0
		}
	case float64:
		return n > 0
	case int:
		return n > 0
	}
	return false
}

func sanitizePreviousStateCompactJSON(prev *CompactedTaskState) string {
	if prev == nil {
		return "{}"
	}
	rawBytes, err := json.Marshal(prev)
	if err != nil {
		return "{}"
	}
	dec := json.NewDecoder(bytes.NewReader(rawBytes))
	dec.UseNumber()
	var parsed map[string]any
	if err := dec.Decode(&parsed); err != nil {
		return "{}"
	}

	safe := make(map[string]any, len(systemOneSanitizedStateFields))
	for _, field := range systemOneSanitizedStateFields {
		val, exists := parsed[field]
		if !exists {
			continue
		}
		if field == "verified_findings" {
			if list, ok := val.([]any); ok {
				cleaned := make([]any, 0, len(list))
				for _, item := range list {
					if obj, isObj := item.(map[string]any); isObj {
						summary, _ := obj["summary"].(string)
						cleaned = append(cleaned, map[string]string{"summary": summary})
					} else {
						cleaned = append(cleaned, item)
					}
				}
				val = cleaned
			}
		}
		safe[field] = val
	}

	encoded, err := compactJSONString(safe)
	if err != nil {
		return "{}"
	}
	return encoded
}
