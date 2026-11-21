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
	maxProbeCandidateSectionBytes   = 4096
	maxProbeTailSectionBytes        = 2048
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

// SystemOneStateRenderResult holds rendered probe text and truncation evidence.
type SystemOneStateRenderResult struct {
	Text                     string
	Truncated                bool
	RetainedSteps            int
	OmittedSteps             int
	RetainedStepRange        []int
	OmittedSections          []string
	EssentialEvidencePresent bool
}

// SystemOneDecisionClient abstracts the System-1 sidecar probe calls used by FSM control plane.
type SystemOneDecisionClient interface {
	Model() string
	DecideCompaction(ctx context.Context, state string, maxSeqLen int) (*llm.DecisionResultEntry, *llm.DecisionResponse, error)
	DecideLoopGuard(ctx context.Context, state string, images []string) (*llm.DecisionResultEntry, *llm.DecisionResponse, error)
}

// RenderSystemOneCompactionState renders a compaction probe state using go-probe-sanitized-v2
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

// RenderSystemOneLoopGuardState renders a loop-guard probe state using go-probe-sanitized-v2
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
	result := SystemOneStateRenderResult{EssentialEvidencePresent: true}
	clip := func(text string, budget, reserve int, suffix string) string {
		clipped, didClip := clipUTF8ForSystemOne(text, budget, reserve, suffix)
		result.Truncated = result.Truncated || didClip
		return clipped
	}

	var prefix strings.Builder
	fmt.Fprintf(&prefix, "Candidate Compaction Boundary: Iterations 1..%d (Current Run Iteration: %d)\n", cutoffIter, execIter)
	fmt.Fprintf(&prefix, "Original Objective: %s\n\n", clip(goal, maxSystemOneGoalBytes, 15, "...[truncated]"))
	prevUpper := 0
	if previous != nil {
		prevUpper = previous.CompactedUpToIter
		prevText, clipped := sanitizePreviousStateCompactJSONBounded(previous, maxProbePrevStateBytes)
		if clipped {
			result.Truncated = true
			result.OmittedSections = append(result.OmittedSections, "previous_state_fields")
		}
		fmt.Fprintf(&prefix, "Previous Compacted State (v%d, up to iter %d):\n%s\n\n", previous.Version, previous.CompactedUpToIter, prevText)
	}
	prefix.WriteString("Candidate Steps to Compact:\n")

	var selected []FSMStep
	var stepLines []string
	for _, step := range steps {
		if step.Iteration <= prevUpper || step.Iteration > cutoffIter {
			continue
		}
		selected = append(selected, step)
		args := clip(strings.TrimSpace(canonicalArgsText(step.ArgsJSON)), maxProbeStepArgsBytes, 23, "...[truncated for probe]")
		output := strings.TrimSpace(storedResultForProbe(step.ResultJSON))
		if output == "" {
			output = step.ErrorText
		}
		output, clipped := clipMiddleUTF8ForSystemOne(output, maxProbeStepOutputBytes)
		result.Truncated = result.Truncated || clipped
		stepLines = append(stepLines, fmt.Sprintf("- Iter=%d Tool=%s Status=%s Args=%s\n  Output: %s\n", step.Iteration, step.ToolName, step.Status, args, output))
	}

	tailLines := make([]string, 0, len(tail))
	for _, step := range tail {
		if step.Iteration > cutoffIter && step.Iteration <= execIter {
			tailLines = append(tailLines, fmt.Sprintf("- Iter=%d Tool=%s Status=%s\n", step.Iteration, step.ToolName, step.Status))
		}
	}
	tailSection := ""
	tailKept := []int(nil)
	if len(tail) > 0 {
		header := "\nRetained Raw Tail Steps (not compacted in this pass):\n"
		body, kept, omitted := fitProbeLines(tailLines, maxProbeTailSectionBytes-len(header), "intermediate raw tail steps")
		tailSection, tailKept = header+body, kept
		if omitted > 0 {
			result.Truncated = true
			result.OmittedSections = append(result.OmittedSections, "raw_tail_steps")
		}
	}

	candidateSection := ""
	candidateKept := []int(nil)
	if len(candidates) > 0 {
		candidateLines := make([]string, 0, len(candidates))
		for _, call := range candidates {
			args := clip(call.Function.Arguments, maxProbeStepArgsBytes, 23, "...[truncated for probe]")
			candidateLines = append(candidateLines, fmt.Sprintf("- Tool=%s Args=%s\n", call.Function.Name, args))
		}
		header := "\nCandidate Tool Calls (not yet executed):\n"
		body, kept, omitted := fitProbeLines(candidateLines, maxProbeCandidateSectionBytes-len(header), "intermediate candidate tool calls")
		candidateSection, candidateKept = header+body, kept
		if omitted > 0 {
			result.Truncated = true
			result.OmittedSections = append(result.OmittedSections, "candidate_tool_calls")
		}
	}

	stepBudget := maxProbePromptBytes - prefix.Len() - len(tailSection) - len(candidateSection)
	stepText, kept, omitted := fitProbeLines(stepLines, stepBudget, "intermediate candidate steps")
	result.RetainedSteps = len(kept)
	result.OmittedSteps = omitted
	if omitted > 0 {
		result.Truncated = true
		result.OmittedSections = append(result.OmittedSections, "candidate_steps")
	}
	if len(kept) > 0 {
		result.RetainedStepRange = []int{selected[kept[0]].Iteration, selected[kept[len(kept)-1]].Iteration}
	}
	result.EssentialEvidencePresent = (len(selected) == 0 || len(kept) > 0 && kept[len(kept)-1] == len(selected)-1) &&
		(len(tailLines) == 0 || len(tailKept) > 0 && tailKept[len(tailKept)-1] == len(tailLines)-1) &&
		(len(candidates) == 0 || len(candidateKept) > 0 && candidateKept[len(candidateKept)-1] == len(candidates)-1)
	result.Text = prefix.String() + stepText + tailSection + candidateSection
	return result
}

func clipMiddleUTF8ForSystemOne(text string, budget int) (string, bool) {
	if len(text) <= budget {
		return text, false
	}
	const marker = "...[middle truncated for probe]..."
	available := budget - len(marker)
	headBytes := available / 2
	tailBytes := available - headBytes
	head := strings.ToValidUTF8(text[:headBytes], "")
	tail := strings.ToValidUTF8(text[len(text)-tailBytes:], "")
	return head + marker + tail, true
}

func fitProbeLines(lines []string, budget int, section string) (string, []int, int) {
	var total int
	for _, line := range lines {
		total += len(line)
	}
	if total <= budget {
		indices := make([]int, len(lines))
		for i := range lines {
			indices[i] = i
		}
		return strings.Join(lines, ""), indices, 0
	}
	if budget <= 0 {
		return "", nil, len(lines)
	}
	markerReserve := 96
	headBudget := max(0, (budget-markerReserve)/3)
	tailBudget := max(0, budget-markerReserve-headBudget)
	headEnd, headUsed := 0, 0
	for headEnd < len(lines) && headUsed+len(lines[headEnd]) <= headBudget {
		headUsed += len(lines[headEnd])
		headEnd++
	}
	tailStart, tailUsed := len(lines), 0
	for tailStart > headEnd && tailUsed+len(lines[tailStart-1]) <= tailBudget {
		tailStart--
		tailUsed += len(lines[tailStart])
	}
	if tailStart == len(lines) && len(lines) > 0 && len(lines[len(lines)-1]) <= budget-markerReserve {
		tailStart = len(lines) - 1
		tailUsed = len(lines[tailStart])
	}
	omitted := tailStart - headEnd
	marker := fmt.Sprintf("...[%d %s omitted for probe budget]...\n", omitted, section)
	for headUsed+tailUsed+len(marker) > budget && headEnd > 0 {
		headEnd--
		headUsed -= len(lines[headEnd])
		omitted++
		marker = fmt.Sprintf("...[%d %s omitted for probe budget]...\n", omitted, section)
	}
	for headUsed+tailUsed+len(marker) > budget && tailStart < len(lines) {
		tailUsed -= len(lines[tailStart])
		tailStart++
		omitted++
		marker = fmt.Sprintf("...[%d %s omitted for probe budget]...\n", omitted, section)
	}
	var text strings.Builder
	indices := make([]int, 0, len(lines)-omitted)
	for i := 0; i < headEnd; i++ {
		text.WriteString(lines[i])
		indices = append(indices, i)
	}
	text.WriteString(marker)
	for i := tailStart; i < len(lines); i++ {
		text.WriteString(lines[i])
		indices = append(indices, i)
	}
	return text.String(), indices, omitted
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

func sanitizePreviousStateCompactJSONBounded(prev *CompactedTaskState, budget int) (string, bool) {
	raw := sanitizePreviousStateCompactJSON(prev)
	if len(raw) <= budget {
		return raw, false
	}
	var parsed map[string]any
	if err := json.Unmarshal([]byte(raw), &parsed); err != nil {
		return "{}", true
	}
	for _, limit := range []struct{ stringBytes, arrayItems int }{{512, 16}, {256, 8}, {128, 4}, {64, 2}} {
		bounded := make(map[string]any, len(parsed))
		for key, value := range parsed {
			bounded[key] = boundPreviousStateValue(value, limit.stringBytes, limit.arrayItems)
		}
		encoded, err := compactJSONString(bounded)
		if err == nil && len(encoded) <= budget {
			return encoded, true
		}
	}
	bounded := make(map[string]any, len(parsed))
	for key, value := range parsed {
		bounded[key] = boundPreviousStateValue(value, 64, 2)
	}
	for i := len(systemOneSanitizedStateFields) - 1; i >= 0; i-- {
		delete(bounded, systemOneSanitizedStateFields[i])
		encoded, err := compactJSONString(bounded)
		if err == nil && len(encoded) <= budget {
			return encoded, true
		}
	}
	return "{}", true
}

func boundPreviousStateValue(value any, stringBytes, arrayItems int) any {
	switch typed := value.(type) {
	case string:
		bounded, _ := clipUTF8ForSystemOne(typed, stringBytes, 15, "...[truncated]")
		return bounded
	case []any:
		limit := min(len(typed), arrayItems)
		out := make([]any, 0, limit+1)
		for _, item := range typed[:limit] {
			out = append(out, boundPreviousStateValue(item, stringBytes, arrayItems))
		}
		if limit < len(typed) {
			out = append(out, fmt.Sprintf("...[%d items omitted]...", len(typed)-limit))
		}
		return out
	case map[string]any:
		out := make(map[string]any, len(typed))
		for key, item := range typed {
			out[key] = boundPreviousStateValue(item, stringBytes, arrayItems)
		}
		return out
	default:
		return value
	}
}
