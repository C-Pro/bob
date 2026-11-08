package trajectory

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"time"

	openai "github.com/sashabaranov/go-openai"
)

// newEventID generates a secure hex identifier for outbox events.
func newEventID() string {
	b := make([]byte, 16)
	if _, err := rand.Read(b); err != nil {
		return fmt.Sprintf("ev-%d", time.Now().UnixNano())
	}
	return hex.EncodeToString(b)
}

// MakeTurnStartEvent constructs an immutable turn_start outbox event.
func MakeTurnStartEvent(trajID, runID string, gen int, payload TurnStartPayload) (*Event, error) {
	if gen <= 0 {
		gen = 1
	}
	payload.Generation = gen
	if payload.RunID == "" {
		payload.RunID = runID
	}
	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal turn_start payload: %w", err)
	}
	return &Event{
		ID:           newEventID(),
		TrajectoryID: trajID,
		RunID:        runID,
		Generation:   gen,
		Iteration:    0,
		Phase:        "start",
		StepID:       "",
		Attempt:      1,
		EventType:    "turn_start",
		Payload:      payloadBytes,
		CreatedAt:    time.Now().UTC(),
	}, nil
}

// MakeToolAttemptEvent constructs an immutable tool_attempt outbox event.
func MakeToolAttemptEvent(trajID, runID string, gen, iteration int, phase, stepID string, attempt int, step Step) (*Event, error) {
	if gen <= 0 {
		gen = 1
	}
	if phase == "" {
		phase = "loop"
	}
	if attempt <= 0 {
		attempt = 1
	}
	payloadBytes, err := json.Marshal(step)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal tool_attempt step: %w", err)
	}
	return &Event{
		ID:           newEventID(),
		TrajectoryID: trajID,
		RunID:        runID,
		Generation:   gen,
		Iteration:    iteration,
		Phase:        phase,
		StepID:       stepID,
		Attempt:      attempt,
		EventType:    "tool_attempt",
		Payload:      payloadBytes,
		CreatedAt:    time.Now().UTC(),
	}, nil
}

// LLMResponsePayload stores assistant generation output and reasoning traces.
type LLMResponsePayload struct {
	Content          string            `json:"content"`
	ReasoningContent string            `json:"reasoning_content,omitempty"`
	ToolCalls        []openai.ToolCall `json:"tool_calls,omitempty"`
}

// MakeLLMResponseEvent constructs an immutable llm_response outbox event.
func MakeLLMResponseEvent(trajID, runID string, gen, iteration int, phase string, content, reasoning string, toolCalls []openai.ToolCall) (*Event, error) {
	if gen <= 0 {
		gen = 1
	}
	if phase == "" {
		phase = "loop"
	}
	payload := LLMResponsePayload{
		Content:          content,
		ReasoningContent: reasoning,
		ToolCalls:        toolCalls,
	}
	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal llm_response payload: %w", err)
	}
	return &Event{
		ID:           newEventID(),
		TrajectoryID: trajID,
		RunID:        runID,
		Generation:   gen,
		Iteration:    iteration,
		Phase:        phase,
		StepID:       "",
		Attempt:      1,
		EventType:    "llm_response",
		Payload:      payloadBytes,
		CreatedAt:    time.Now().UTC(),
	}, nil
}

// MakeTurnTerminalEvent constructs an immutable turn_terminal outbox event.
func MakeTurnTerminalEvent(trajID, runID string, gen int, payload TurnTerminalPayload, phase ...string) (*Event, error) {
	if gen <= 0 {
		gen = 1
	}
	p := "terminal"
	if len(phase) > 0 && phase[0] != "" {
		p = phase[0]
	}
	payload.Generation = gen
	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal turn_terminal payload: %w", err)
	}
	return &Event{
		ID:           newEventID(),
		TrajectoryID: trajID,
		RunID:        runID,
		Generation:   gen,
		Iteration:    0,
		Phase:        p,
		StepID:       "",
		Attempt:      1,
		EventType:    "turn_terminal",
		Payload:      payloadBytes,
		CreatedAt:    time.Now().UTC(),
	}, nil
}

// MakeCompactionEvent constructs an immutable compaction outbox event.
func MakeCompactionEvent(trajID, runID string, gen, iteration int, stepID string, payload CompactionEventPayload) (*Event, error) {
	if gen <= 0 {
		gen = 1
	}
	if len(payload.StateJSON) > 0 && !json.Valid(payload.StateJSON) {
		return nil, fmt.Errorf("invalid compaction state_json payload")
	}
	if stepID == "" {
		if !payload.ShouldCompact || payload.RubricVerdict == "probed_skip" {
			stepID = fmt.Sprintf("probe_skip_i%d", iteration)
		} else if payload.Version > 0 {
			stepID = fmt.Sprintf("compact_v%d_i%d", payload.Version, iteration)
		} else {
			stepID = fmt.Sprintf("probe_i%d", iteration)
		}
	}
	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal compaction payload: %w", err)
	}
	return &Event{
		ID:           newEventID(),
		TrajectoryID: trajID,
		RunID:        runID,
		Generation:   gen,
		Iteration:    iteration,
		Phase:        "compaction",
		StepID:       stepID,
		Attempt:      1,
		EventType:    "compaction",
		Payload:      payloadBytes,
		CreatedAt:    time.Now().UTC(),
	}, nil
}

// ParseExitCode scans tool output text for structured "Exit Code: <int>" lines.
func ParseExitCode(content string) *int {
	lines := strings.Split(content, "\n")
	for _, l := range lines {
		trimmed := strings.TrimSpace(l)
		if strings.HasPrefix(trimmed, "Exit Code:") {
			valStr := strings.TrimSpace(strings.TrimPrefix(trimmed, "Exit Code:"))
			if code, err := strconv.Atoi(valStr); err == nil {
				return &code
			}
		}
	}
	return nil
}

// DerivePrivacyLevel assigns privacy classification and export approval.
// Townhall chats default to public/approved unless private memories or secrets are indicated.
// Direct Messages default to private/unapproved.
func DerivePrivacyLevel(isDM bool, userInput, taskState string, activeSkills []string) (string, bool) {
	if isDM {
		return "private", false
	}
	// Check for sensitive markers in input or task state
	combined := strings.ToLower(userInput + " " + taskState)
	for _, kw := range []string{"api_key", "password", "secret", "token", "credential", "private"} {
		if strings.Contains(combined, kw) {
			return "private", false
		}
	}
	return "public", true
}

// BuildTrajectoryProjection constructs a canonical Trajectory record from turn start, terminal payload, and steps.
func BuildTrajectoryProjection(start TurnStartPayload, terminal TurnTerminalPayload, steps []Step, finalOutput string, delay time.Duration) *Trajectory {
	now := time.Now().UTC()
	var succCount, failCount int
	for _, s := range steps {
		if s.ExitCode == nil || *s.ExitCode == 0 {
			succCount++
		} else {
			failCount++
		}
	}

	budgetExhausted := terminal.Reason == "budget_exhausted"
	breakdown := EvalBreakdown{
		TotalToolCalls:      len(steps),
		SuccessfulToolCalls: succCount,
		FailedToolCalls:     failCount,
		BudgetExhausted:     budgetExhausted,
		FSMFailed:           terminal.Failed,
		TerminationReason:   terminal.Reason,
		RewardVersion:       RewardVersionV1,
	}
	if len(steps) > 0 {
		breakdown.ToolSuccessRate = float64(succCount) / float64(len(steps))
	}

	rEval := CalculateReward(breakdown)

	privacy := start.PrivacyLevel
	exportApproved := start.ExportApproved
	if privacy == "" {
		privacy = "private"
		exportApproved = false
	}

	gen := terminal.Generation
	if gen <= 0 {
		gen = start.Generation
	}
	if gen <= 0 {
		gen = 1
	}

	output := finalOutput
	if output == "" && terminal.FinalReply != "" {
		output = terminal.FinalReply
	}

	stepSkills, stepMemories := ExtractStepKnowledgeVersions(steps)
	activeSkills := MergeVersionIDs(start.ActiveSkills, stepSkills, terminal.ActiveSkills)
	activeMemories := MergeVersionIDs(start.ActiveMemories, stepMemories, terminal.ActiveMemories)
	taskState := FormatTaskStateWithProvenance(start.TaskState, start.ContextManifest, activeMemories, activeSkills)

	return &Trajectory{
		ID:             start.RunID,
		RunID:          start.RunID,
		FrontendID:     start.FrontendID,
		SessionID:      start.SessionID,
		ScopeID:        start.ScopeID,
		ActorID:        start.ActorID,
		Model:          start.Model,
		ExecutionKind:  start.ExecutionKind,
		PrivacyLevel:   privacy,
		ExportApproved: exportApproved,
		Generation:     gen,
		CreatedAt:      now,
		CompletedAt:    now,
		UserInput:      start.Input,
		TaskState:      taskState,
		ActiveSkills:   activeSkills,
		ActiveMemories: activeMemories,
		Steps:          steps,
		FinalOutput:    output,
		Reward:         rEval,
		Breakdown:      breakdown,
		Revision:       1,
	}
}

// MergeVersionIDs deduplicates and trims version identifiers while preserving first-seen order.
func MergeVersionIDs(lists ...[]string) []string {
	seen := make(map[string]bool)
	var out []string
	for _, list := range lists {
		for _, v := range list {
			trimmed := strings.TrimSpace(v)
			if trimmed == "" || seen[trimmed] {
				continue
			}
			seen[trimmed] = true
			out = append(out, trimmed)
		}
	}
	return out
}

const (
	// ProvenanceManifestHeader marks the start of the structured provenance block embedded in Trajectory.TaskState.
	ProvenanceManifestHeader = "[Provenance Manifest]"
	// ProvenanceManifestFooter marks the end of the structured provenance block embedded in Trajectory.TaskState.
	ProvenanceManifestFooter = "[/Provenance Manifest]"
)

// isValidKnowledgeVersionID verifies that verID is a resolved version identifier "<item>@<rev>"
// matching the expected prefix ("skill_" or "mem_") and a positive integer revision.
func isValidKnowledgeVersionID(verID, expectedPrefix string) bool {
	if !strings.HasPrefix(verID, expectedPrefix) {
		return false
	}
	atIdx := strings.LastIndex(verID, "@")
	if atIdx <= len(expectedPrefix) || atIdx >= len(verID)-1 {
		return false
	}
	rev, err := strconv.Atoi(verID[atIdx+1:])
	return err == nil && rev >= 1
}

// ExtractStepKnowledgeVersions inspects completed tool steps and extracts loaded skill version IDs
// ("skill_<id>@<rev>") and memory version IDs ("mem_<id>@<rev>") from load_skill and load_memory invocations.
// Only steps with a successful observation containing a resolved version_id and no error field are recorded.
func ExtractStepKnowledgeVersions(steps []Step) (skills []string, memories []string) {
	seenSkills := make(map[string]bool)
	seenMemories := make(map[string]bool)

	for _, s := range steps {
		if s.ExitCode != nil && *s.ExitCode != 0 {
			continue
		}
		action := strings.TrimSpace(s.Action)
		if action != "load_skill" && action != "load_memory" {
			continue
		}

		var obsPayload struct {
			VersionID string `json:"version_id"`
			Error     any    `json:"error"`
		}
		if err := json.Unmarshal([]byte(s.Observation), &obsPayload); err != nil || obsPayload.Error != nil {
			continue
		}
		verID := strings.TrimSpace(obsPayload.VersionID)
		if verID == "" {
			continue
		}

		switch action {
		case "load_skill":
			if isValidKnowledgeVersionID(verID, "skill_") && !seenSkills[verID] {
				seenSkills[verID] = true
				skills = append(skills, verID)
			}
		case "load_memory":
			if isValidKnowledgeVersionID(verID, "mem_") && !seenMemories[verID] {
				seenMemories[verID] = true
				memories = append(memories, verID)
			}
		}
	}
	return skills, memories
}

// ParseTaskStateProvenance separates a TaskState string into its base system prompt and any embedded
// provenance fields (active_memories, active_skills, context_manifest) from a validated trailing
// [Provenance Manifest] ... [/Provenance Manifest] block. If no valid trailing provenance block is
// present, taskState is returned unchanged as basePrompt.
func ParseTaskStateProvenance(taskState string) (basePrompt string, manifestJSON json.RawMessage, memories, skills []string) {
	trimmedRight := strings.TrimRight(taskState, " \t\r\n")
	if !strings.HasSuffix(trimmedRight, ProvenanceManifestFooter) {
		return taskState, nil, nil, nil
	}

	withoutFooter := trimmedRight[:len(trimmedRight)-len(ProvenanceManifestFooter)]
	idx := strings.LastIndex(withoutFooter, ProvenanceManifestHeader)
	if idx < 0 {
		return taskState, nil, nil, nil
	}
	if idx > 0 && withoutFooter[idx-1] != '\n' {
		return taskState, nil, nil, nil
	}

	block := withoutFooter[idx+len(ProvenanceManifestHeader):]
	var parsedMemories, parsedSkills []string
	var parsedManifest json.RawMessage
	hasField := false

	for _, line := range strings.Split(block, "\n") {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			continue
		}
		switch {
		case strings.HasPrefix(trimmed, "active_memories:"):
			raw := strings.TrimSpace(strings.TrimPrefix(trimmed, "active_memories:"))
			if raw == "" {
				return taskState, nil, nil, nil
			}
			parsedMemories = MergeVersionIDs(parsedMemories, strings.Split(raw, ","))
			hasField = true
		case strings.HasPrefix(trimmed, "active_skills:"):
			raw := strings.TrimSpace(strings.TrimPrefix(trimmed, "active_skills:"))
			if raw == "" {
				return taskState, nil, nil, nil
			}
			parsedSkills = MergeVersionIDs(parsedSkills, strings.Split(raw, ","))
			hasField = true
		case strings.HasPrefix(trimmed, "context_manifest:"):
			raw := strings.TrimSpace(strings.TrimPrefix(trimmed, "context_manifest:"))
			if raw == "" || !json.Valid([]byte(raw)) {
				return taskState, nil, nil, nil
			}
			parsedManifest = json.RawMessage(raw)
			hasField = true
		default:
			// Unrecognized line inside candidate provenance block -> treat entire taskState as literal prompt text.
			return taskState, nil, nil, nil
		}
	}

	if !hasField {
		return taskState, nil, nil, nil
	}

	prefix := withoutFooter[:idx]
	switch {
	case strings.HasSuffix(prefix, "\r\n\r\n"):
		basePrompt = prefix[:len(prefix)-4]
	case strings.HasSuffix(prefix, "\n\n"):
		basePrompt = prefix[:len(prefix)-2]
	case strings.HasSuffix(prefix, "\r\n"):
		basePrompt = prefix[:len(prefix)-2]
	case strings.HasSuffix(prefix, "\n"):
		basePrompt = prefix[:len(prefix)-1]
	default:
		basePrompt = prefix
	}
	return basePrompt, parsedManifest, parsedMemories, parsedSkills
}

// FormatTaskStateWithProvenance combines the base system prompt with a deterministic
// [Provenance Manifest] ... [/Provenance Manifest] trailing block whenever active memories,
// active skills, or an assembly manifest are present.
// If none are present, baseTaskState is returned unchanged.
func FormatTaskStateWithProvenance(baseTaskState string, manifestJSON json.RawMessage, activeMemories, activeSkills []string) string {
	cleanBase, existingManifest, existingMem, existingSkills := ParseTaskStateProvenance(baseTaskState)
	mergedMemories := MergeVersionIDs(existingMem, activeMemories)
	mergedSkills := MergeVersionIDs(existingSkills, activeSkills)

	effectiveManifest := manifestJSON
	if len(effectiveManifest) == 0 {
		effectiveManifest = existingManifest
	}
	if len(effectiveManifest) > 0 && !json.Valid(effectiveManifest) {
		effectiveManifest = nil
	}

	if len(mergedMemories) == 0 && len(mergedSkills) == 0 && len(effectiveManifest) == 0 {
		return cleanBase
	}

	var sb strings.Builder
	if cleanBase != "" {
		sb.WriteString(cleanBase)
		sb.WriteString("\n\n")
	}
	sb.WriteString(ProvenanceManifestHeader)
	if len(mergedMemories) > 0 {
		sb.WriteString("\nactive_memories: ")
		sb.WriteString(strings.Join(mergedMemories, ", "))
	}
	if len(mergedSkills) > 0 {
		sb.WriteString("\nactive_skills: ")
		sb.WriteString(strings.Join(mergedSkills, ", "))
	}
	if len(effectiveManifest) > 0 {
		var compact json.RawMessage
		if b, err := json.Marshal(effectiveManifest); err == nil {
			compact = b
		} else {
			compact = effectiveManifest
		}
		sb.WriteString("\ncontext_manifest: ")
		sb.Write(compact)
	}
	sb.WriteString("\n")
	sb.WriteString(ProvenanceManifestFooter)
	return sb.String()
}
