package fsm

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
	"time"

	"bob/internal/trajectory"

	openai "github.com/sashabaranov/go-openai"
)

const (
	compactionProbeTimeout  = 30 * time.Second
	maxProbeStepArgsBytes   = 512
	maxProbeStepOutputBytes = 1024
	maxProbePrevStateBytes  = 8192
	maxProbePromptBytes     = 32768
	maxProbeCompletionToks  = 8192
	maxCompactedStateBytes  = 32768
)

// CompactionConfig configures adaptive task trajectory compaction (SelfCompact) for the FSM tool loop.
type CompactionConfig struct {
	Enabled            bool
	MinIterations      int
	IntervalIterations int
	TailIterations     int
	SoftThresholdBytes int
	HardThresholdBytes int
}

// DefaultCompactionConfig returns the standard SelfCompact configuration matching internal/config defaults.
func DefaultCompactionConfig() CompactionConfig {
	return CompactionConfig{
		Enabled:            true,
		MinIterations:      3,
		IntervalIterations: 3,
		TailIterations:     1,
		SoftThresholdBytes: 65536,
		HardThresholdBytes: 262144,
	}
}

func (c CompactionConfig) normalized() CompactionConfig {
	out := c
	if out.MinIterations <= 0 {
		out.MinIterations = 3
	}
	if out.IntervalIterations <= 0 {
		out.IntervalIterations = 3
	}
	if out.TailIterations < 0 {
		out.TailIterations = 1
	}
	if out.SoftThresholdBytes <= 0 {
		out.SoftThresholdBytes = 65536
	}
	if out.HardThresholdBytes <= 0 {
		out.HardThresholdBytes = 262144
	}
	if out.HardThresholdBytes < out.SoftThresholdBytes {
		out.HardThresholdBytes = out.SoftThresholdBytes
	}
	return out
}

// CompactionProbeInput encapsulates the run state and step history passed to a CompactionDecider.
type CompactionProbeInput struct {
	Run              *FSMRun
	Model            string
	TargetIter       int
	PreviousState    *CompactedTaskState
	StepsToCompact   []FSMStep
	TailSteps        []FSMStep
	OriginalUserGoal string
	ActiveBytes      int
	ActiveSkills     []string
	ActiveMemories   []string
}

// CompactionDecider evaluates the SelfCompact rubric and optionally produces a CompactedTaskState.
type CompactionDecider interface {
	DecideCompaction(ctx context.Context, input CompactionProbeInput) (*CompactionProbeResult, error)
}

// CompactionDeciderFunc adapts a function to the CompactionDecider interface.
type CompactionDeciderFunc func(ctx context.Context, input CompactionProbeInput) (*CompactionProbeResult, error)

// DecideCompaction calls f(ctx, input).
func (f CompactionDeciderFunc) DecideCompaction(ctx context.Context, input CompactionProbeInput) (*CompactionProbeResult, error) {
	return f(ctx, input)
}

type compactionTriggerMode int

const (
	compactionTriggerNone compactionTriggerMode = iota
	compactionTriggerProbeRubric
	compactionTriggerForceBackstop
)

// RenderCompactedStateMessage formats a CompactedTaskState as a structured system context message
// that replaces raw tool-call and tool-result messages for iterations 1..CompactedUpToIter.
func RenderCompactedStateMessage(state *CompactedTaskState) openai.ChatCompletionMessage {
	if state == nil {
		return openai.ChatCompletionMessage{
			Role: openai.ChatMessageRoleSystem,
		}
	}

	var b strings.Builder
	fmt.Fprintf(&b, "[Compacted Task State v%d | Iterations 1..%d]\n", state.Version, state.CompactedUpToIter)
	if state.OriginalObjective != "" {
		fmt.Fprintf(&b, "Original Objective: %s\n", state.OriginalObjective)
	}
	if len(state.ImmutableConstraints) > 0 {
		b.WriteString("Immutable Constraints:\n")
		for _, c := range state.ImmutableConstraints {
			fmt.Fprintf(&b, "- %s\n", c)
		}
	}
	if len(state.VerifiedFindings) > 0 {
		b.WriteString("Verified Findings:\n")
		for _, f := range state.VerifiedFindings {
			if len(f.SourceStepIDs) > 0 {
				fmt.Fprintf(&b, "- %s [source_steps: %s]\n", f.Summary, strings.Join(f.SourceStepIDs, ", "))
			} else {
				fmt.Fprintf(&b, "- %s\n", f.Summary)
			}
		}
	}
	if len(state.DecisionsAndRejected) > 0 {
		b.WriteString("Decisions & Rejected Approaches:\n")
		for _, d := range state.DecisionsAndRejected {
			fmt.Fprintf(&b, "- %s\n", d)
		}
	}
	if state.CurrentSubgoal != "" {
		fmt.Fprintf(&b, "Current Subgoal: %s\n", state.CurrentSubgoal)
	}
	if len(state.NextIntendedActions) > 0 {
		b.WriteString("Next Intended Actions:\n")
		for _, a := range state.NextIntendedActions {
			fmt.Fprintf(&b, "- %s\n", a)
		}
	}
	if len(state.UnresolvedBlockers) > 0 {
		b.WriteString("Unresolved Blockers:\n")
		for _, u := range state.UnresolvedBlockers {
			fmt.Fprintf(&b, "- %s\n", u)
		}
	}
	if len(state.ProducedArtifacts) > 0 {
		b.WriteString("Produced Artifacts:\n")
		for _, p := range state.ProducedArtifacts {
			fmt.Fprintf(&b, "- %s\n", p)
		}
	}
	if len(state.ActiveSkillVersions) > 0 {
		fmt.Fprintf(&b, "Active Skills: %s\n", strings.Join(state.ActiveSkillVersions, ", "))
	}
	if len(state.ActiveMemoryVersions) > 0 {
		fmt.Fprintf(&b, "Active Memories: %s\n", strings.Join(state.ActiveMemoryVersions, ", "))
	}
	if len(state.CompactedStepIDs) > 0 {
		fmt.Fprintf(&b, "Compacted Steps: %s\n", strings.Join(state.CompactedStepIDs, ", "))
	}
	fmt.Fprintf(&b, "Note: Raw tool calls and observations for iterations 1..%d have been compacted into this structured state. Use recall_task_history if you need exact raw tool outputs from compacted steps.", state.CompactedUpToIter)

	return openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleSystem,
		Content: b.String(),
	}
}

func isStepFailure(s FSMStep) bool {
	if s.Status == StepStatusFailed || s.Status == StepStatusTimedOut || strings.TrimSpace(s.ErrorText) != "" {
		return true
	}
	decoded, _ := DecodeStoredToolResult(s.ResultJSON)
	content := strings.TrimSpace(decoded.Content)
	if content == "" {
		return false
	}
	if code := trajectory.ParseExitCode(content); code != nil && *code != 0 {
		return true
	}
	if strings.HasPrefix(content, "Error:") || strings.HasPrefix(content, "error:") {
		return true
	}
	var errObj struct {
		Error string `json:"error"`
	}
	if strings.HasPrefix(content, "{") && json.Unmarshal([]byte(content), &errObj) == nil && strings.TrimSpace(errObj.Error) != "" {
		return true
	}
	return false
}

// isStuckRepeatingFailure reports whether the last two iterations (currentIter-1 and currentIter)
// executed identical (ToolName, ArgsJSON) tool call sequences where at least one matching call failed in both iterations.
func isStuckRepeatingFailure(steps []FSMStep, currentIter int) bool {
	if currentIter < 2 {
		return false
	}
	var prevSteps, currSteps []FSMStep
	for _, s := range steps {
		switch s.Iteration {
		case currentIter - 1:
			prevSteps = append(prevSteps, s)
		case currentIter:
			currSteps = append(currSteps, s)
		}
	}
	if len(prevSteps) == 0 || len(prevSteps) != len(currSteps) {
		return false
	}

	matchedFailure := false
	for i := range prevSteps {
		if prevSteps[i].ToolName != currSteps[i].ToolName {
			return false
		}
		if strings.TrimSpace(prevSteps[i].ArgsJSON) != strings.TrimSpace(currSteps[i].ArgsJSON) {
			return false
		}
		if isStepFailure(prevSteps[i]) && isStepFailure(currSteps[i]) {
			matchedFailure = true
		}
	}
	return matchedFailure
}

func evaluateCompactionTrigger(
	cfg CompactionConfig,
	currentIter int,
	prevCompactedIter int,
	activeBytes int,
	allSteps []FSMStep,
) (compactionTriggerMode, int) {
	if !cfg.Enabled {
		return compactionTriggerNone, 0
	}
	cfg = cfg.normalized()

	if currentIter < cfg.MinIterations {
		return compactionTriggerNone, 0
	}

	for _, s := range allSteps {
		if s.Iteration <= currentIter && !s.Status.IsTerminal() {
			return compactionTriggerNone, 0
		}
	}

	targetIter := currentIter - cfg.TailIterations
	if targetIter <= 0 || targetIter <= prevCompactedIter {
		return compactionTriggerNone, 0
	}

	hasCompactableSteps := false
	for _, s := range allSteps {
		if s.Iteration > prevCompactedIter && s.Iteration <= targetIter {
			hasCompactableSteps = true
			break
		}
	}
	if !hasCompactableSteps {
		return compactionTriggerNone, 0
	}

	if isStuckRepeatingFailure(allSteps, currentIter) {
		return compactionTriggerNone, 0
	}

	if activeBytes >= cfg.HardThresholdBytes {
		return compactionTriggerForceBackstop, targetIter
	}

	if (currentIter-prevCompactedIter) >= cfg.IntervalIterations || activeBytes >= cfg.SoftThresholdBytes {
		return compactionTriggerProbeRubric, targetIter
	}

	return compactionTriggerNone, 0
}

func extractOriginalObjective(messages []openai.ChatCompletionMessage) string {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == openai.ChatMessageRoleUser && strings.TrimSpace(messages[i].Content) != "" {
			return strings.TrimSpace(messages[i].Content)
		}
	}
	return ""
}

func extractActiveKnowledgeVersions(steps []FSMStep, prev *CompactedTaskState) (skills []string, memories []string) {
	seenSkills := make(map[string]bool)
	seenMemories := make(map[string]bool)
	if prev != nil {
		for _, v := range prev.ActiveSkillVersions {
			if v != "" && !seenSkills[v] {
				seenSkills[v] = true
				skills = append(skills, v)
			}
		}
		for _, v := range prev.ActiveMemoryVersions {
			if v != "" && !seenMemories[v] {
				seenMemories[v] = true
				memories = append(memories, v)
			}
		}
	}

	for _, s := range steps {
		if s.Status != StepStatusCompleted {
			continue
		}
		if s.ToolName != "load_skill" && s.ToolName != "load_memory" {
			continue
		}
		decoded, _ := DecodeStoredToolResult(s.ResultJSON)
		var payload struct {
			VersionID string `json:"version_id"`
		}
		if json.Unmarshal([]byte(decoded.Content), &payload) != nil || payload.VersionID == "" {
			continue
		}
		if s.ToolName == "load_skill" && !seenSkills[payload.VersionID] {
			seenSkills[payload.VersionID] = true
			skills = append(skills, payload.VersionID)
		}
		if s.ToolName == "load_memory" && !seenMemories[payload.VersionID] {
			seenMemories[payload.VersionID] = true
			memories = append(memories, payload.VersionID)
		}
	}
	return skills, memories
}

func collectCompactedStepIDs(prev *CompactedTaskState, stepsToCompact []FSMStep) []string {
	seen := make(map[string]bool)
	var out []string
	if prev != nil {
		for _, id := range prev.CompactedStepIDs {
			if id != "" && !seen[id] {
				seen[id] = true
				out = append(out, id)
			}
		}
	}
	for _, s := range stepsToCompact {
		if s.ID != "" && !seen[s.ID] {
			seen[s.ID] = true
			out = append(out, s.ID)
		}
	}
	return out
}

func summarizeStepForState(s FSMStep) string {
	decoded, _ := DecodeStoredToolResult(s.ResultJSON)
	snippet := strings.TrimSpace(decoded.Content)
	if snippet == "" && s.ErrorText != "" {
		snippet = s.ErrorText
	}
	snippet = strings.Join(strings.Fields(snippet), " ")
	if len(snippet) > 240 {
		snippet = snippet[:240] + "..."
	}
	if snippet == "" {
		return fmt.Sprintf("Iter %d %s (%s)", s.Iteration, s.ToolName, s.Status)
	}
	return fmt.Sprintf("Iter %d %s (%s): %s", s.Iteration, s.ToolName, s.Status, snippet)
}

func buildDeterministicCompactedState(input CompactionProbeInput, verdict string) *CompactedTaskState {
	prevVer := 0
	var constraints, decisions, nextActions, blockers, artifacts []string
	var findings []VerifiedFinding
	objective := input.OriginalUserGoal
	subgoal := ""

	if input.PreviousState != nil {
		prevVer = input.PreviousState.Version
		if objective == "" {
			objective = input.PreviousState.OriginalObjective
		}
		subgoal = strings.TrimSpace(input.PreviousState.CurrentSubgoal)
		constraints = append(constraints, input.PreviousState.ImmutableConstraints...)
		findings = append(findings, input.PreviousState.VerifiedFindings...)
		decisions = append(decisions, input.PreviousState.DecisionsAndRejected...)
		nextActions = append(nextActions, input.PreviousState.NextIntendedActions...)
		blockers = append(blockers, input.PreviousState.UnresolvedBlockers...)
		artifacts = append(artifacts, input.PreviousState.ProducedArtifacts...)
	}

	if len(input.StepsToCompact) <= 8 {
		for _, s := range input.StepsToCompact {
			findings = append(findings, VerifiedFinding{
				Summary:       summarizeStepForState(s),
				SourceStepIDs: []string{s.ID},
			})
		}
	} else {
		var batchIDs []string
		firstIter := input.StepsToCompact[0].Iteration
		lastIter := input.StepsToCompact[len(input.StepsToCompact)-1].Iteration
		for _, s := range input.StepsToCompact {
			if s.ID != "" {
				batchIDs = append(batchIDs, s.ID)
			}
		}
		findings = append(findings, VerifiedFinding{
			Summary:       fmt.Sprintf("Consolidated %d tool steps from iterations %d..%d (last: %s)", len(input.StepsToCompact), firstIter, lastIter, summarizeStepForState(input.StepsToCompact[len(input.StepsToCompact)-1])),
			SourceStepIDs: batchIDs,
		})
	}

	if subgoal == "" {
		subgoal = fmt.Sprintf("Continue task execution after compacting iterations 1..%d", input.TargetIter)
		if len(input.TailSteps) > 0 {
			lastTail := input.TailSteps[len(input.TailSteps)-1]
			subgoal = fmt.Sprintf("Evaluate results from iteration %d (%s) and proceed toward objective", lastTail.Iteration, lastTail.ToolName)
		}
	}

	return &CompactedTaskState{
		Version:              prevVer + 1,
		ParentVersion:        prevVer,
		RunID:                input.Run.ID,
		CompactedUpToIter:    input.TargetIter,
		CompactedStepIDs:     collectCompactedStepIDs(input.PreviousState, input.StepsToCompact),
		OriginalObjective:    objective,
		ImmutableConstraints: constraints,
		VerifiedFindings:     findings,
		DecisionsAndRejected: decisions,
		CurrentSubgoal:       subgoal,
		NextIntendedActions:  nextActions,
		UnresolvedBlockers:   blockers,
		ProducedArtifacts:    artifacts,
		ActiveSkillVersions:  input.ActiveSkills,
		ActiveMemoryVersions: input.ActiveMemories,
		RubricVerdict:        verdict,
		Model:                input.Model,
		InputBytes:           input.ActiveBytes,
	}
}

const selfCompactRubricSystemPrompt = `You are a task trajectory compaction controller (SelfCompact).
Evaluate whether the agent's intermediate tool iterations up to the candidate boundary should be compacted into a Structured Task State.

COMPACT CRITERIA (should_compact = true):
1. A subtask has completed and its result is verified.
2. A search or investigation has converged on durable findings.
3. An earlier plan or failed approach is no longer active.
4. Large or repeated tool outputs can be distilled into a compact working state without losing actionable details.

DO-NOT-COMPACT CRITERIA (should_compact = false):
1. The agent is midway through an unresolved derivation or multi-step verification that requires exact raw intermediate lines.
2. The candidate steps have not yet reached a coherent checkpoint.
3. The agent is repeating identical failing calls.

Respond ONLY with valid JSON matching this schema:
{
  "should_compact": true,
  "reason": "concise justification",
  "state": {
    "original_objective": "string",
    "immutable_constraints": ["string"],
    "verified_findings": [{"summary": "string", "source_step_ids": ["step_id"]}],
    "decisions_and_rejected": ["string"],
    "current_subgoal": "string",
    "next_intended_actions": ["string"],
    "unresolved_blockers": ["string"],
    "produced_artifacts": ["string"]
  }
}`

type llmCompactionDecider struct {
	client LLMClient
}

func (d *llmCompactionDecider) DecideCompaction(ctx context.Context, input CompactionProbeInput) (*CompactionProbeResult, error) {
	if d == nil || d.client == nil {
		return nil, fmt.Errorf("llm client is not configured for compaction decider")
	}

	probeCtx, cancel := context.WithTimeout(ctx, compactionProbeTimeout)
	defer cancel()

	var b strings.Builder
	fmt.Fprintf(&b, "Run ID: %s\n", input.Run.ID)
	fmt.Fprintf(&b, "Candidate Compaction Boundary: Iterations 1..%d (Current Run Iteration: %d)\n", input.TargetIter, input.Run.Iteration)
	goal := input.OriginalUserGoal
	if len(goal) > 2048 {
		goal = goal[:2048] + "...[truncated]"
	}
	fmt.Fprintf(&b, "Original Objective: %s\n\n", goal)

	if input.PreviousState != nil {
		prevJSON, _ := json.Marshal(input.PreviousState)
		prevStr := string(prevJSON)
		if len(prevStr) > maxProbePrevStateBytes {
			prevStr = prevStr[:maxProbePrevStateBytes] + "...[truncated]"
		}
		fmt.Fprintf(&b, "Previous Compacted State (v%d, up to iter %d):\n%s\n\n", input.PreviousState.Version, input.PreviousState.CompactedUpToIter, prevStr)
	}

	b.WriteString("Candidate Steps to Compact:\n")
	for _, s := range input.StepsToCompact {
		args := strings.TrimSpace(s.ArgsJSON)
		if len(args) > maxProbeStepArgsBytes {
			args = args[:maxProbeStepArgsBytes] + "...[truncated for probe]"
		}
		decoded, _ := DecodeStoredToolResult(s.ResultJSON)
		obs := strings.TrimSpace(decoded.Content)
		if obs == "" && s.ErrorText != "" {
			obs = s.ErrorText
		}
		if len(obs) > maxProbeStepOutputBytes {
			obs = obs[:maxProbeStepOutputBytes] + "...[truncated for probe]"
		}
		fmt.Fprintf(&b, "- StepID=%s Iter=%d Tool=%s Status=%s Args=%s\n  Output: %s\n", s.ID, s.Iteration, s.ToolName, s.Status, args, obs)
		if b.Len() >= maxProbePromptBytes {
			b.WriteString("...[additional candidate steps truncated for probe budget]\n")
			break
		}
	}

	if len(input.TailSteps) > 0 && b.Len() < maxProbePromptBytes {
		b.WriteString("\nRetained Raw Tail Steps (not compacted in this pass):\n")
		for _, s := range input.TailSteps {
			fmt.Fprintf(&b, "- StepID=%s Iter=%d Tool=%s Status=%s\n", s.ID, s.Iteration, s.ToolName, s.Status)
		}
	}

	userPrompt := b.String()
	if len(userPrompt) > maxProbePromptBytes {
		userPrompt = userPrompt[:maxProbePromptBytes] + "\n...[truncated]"
	}

	req := openai.ChatCompletionRequest{
		Model:               input.Model,
		MaxCompletionTokens: maxProbeCompletionToks,
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleSystem,
				Content: selfCompactRubricSystemPrompt,
			},
			{
				Role:    openai.ChatMessageRoleUser,
				Content: userPrompt,
			},
		},
	}

	resp, err := d.client.CreateChatCompletion(probeCtx, req)
	if err != nil {
		return nil, err
	}
	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("compaction probe returned empty choices")
	}
	finishReason := resp.Choices[0].FinishReason
	if finishReason == openai.FinishReasonLength {
		slog.Warn("compaction probe output was truncated", "finish_reason", finishReason, "max_completion_tokens", maxProbeCompletionToks)
	}

	raw := strings.TrimSpace(resp.Choices[0].Message.Content)
	if len(raw) > maxCompactedStateBytes*2 {
		return nil, fmt.Errorf("compaction probe response (%d bytes) exceeds maximum allowed size", len(raw))
	}
	raw = stripMarkdownJSONFence(raw)

	var result CompactionProbeResult
	if err := json.Unmarshal([]byte(raw), &result); err != nil {
		return nil, fmt.Errorf("failed to parse compaction probe JSON (finish_reason=%s, raw_preview=%.512q): %w", finishReason, raw, err)
	}
	return &result, nil
}

func stripMarkdownJSONFence(s string) string {
	trimmed := strings.TrimSpace(s)
	if strings.HasPrefix(trimmed, "```") {
		if nl := strings.IndexByte(trimmed, '\n'); nl != -1 {
			trimmed = trimmed[nl+1:]
		}
		if end := strings.LastIndex(trimmed, "```"); end != -1 {
			trimmed = trimmed[:end]
		}
	}
	return strings.TrimSpace(trimmed)
}

func recordProbeSkipEvent(ctx context.Context, store *Store, runID string, iteration, prevVersion, prevCompactedIter int, model string, activeBytes int, reason string) error {
	ev, err := trajectory.MakeCompactionEvent(runID, runID, 1, iteration, "", trajectory.CompactionEventPayload{
		Version:           0,
		ParentVersion:     prevVersion,
		CompactedUpToIter: prevCompactedIter,
		ShouldCompact:     false,
		RubricVerdict:     CompactionVerdictProbedSkip,
		Reason:            reason,
		Model:             model,
		InputBytes:        activeBytes,
	})
	if err != nil {
		return fmt.Errorf("failed to build probe skip event: %w", err)
	}
	if err := store.RecordEvent(ctx, ev); err != nil {
		return fmt.Errorf("failed to record probe skip event: %w", err)
	}
	return nil
}

func validateCandidateCompaction(
	state *CompactedTaskState,
	baseMsgs []openai.ChatCompletionMessage,
	allSteps []FSMStep,
	activeBytes int,
) (skipReason string, err error) {
	stateBytes, err := NormalizeCompactionState(state)
	if err != nil {
		return "", err
	}
	renderedMsg := RenderCompactedStateMessage(state)
	if len(stateBytes) > maxCompactedStateBytes || len(renderedMsg.Content) > maxCompactedStateBytes {
		return "probe_oversized_state", nil
	}

	candidateMsgs := reconstructChatMessages(baseMsgs, allSteps, state)
	encodedCandidate, err := EncodeMessages(candidateMsgs)
	if err != nil {
		return "", fmt.Errorf("failed to encode candidate compacted messages: %w", err)
	}
	if len(encodedCandidate) >= activeBytes {
		return "compaction_did_not_shrink_context", nil
	}
	return "", nil
}

func (r *ToolLoopRunner) maybeCompact(ctx context.Context, run *FSMRun, store *Store, model string) error {
	r.mu.RLock()
	cfg := r.compactionCfg
	decider := r.compactionDecider
	r.mu.RUnlock()

	if !cfg.Enabled || store == nil || run == nil {
		return nil
	}
	cfg = cfg.normalized()

	alreadyRecorded, err := store.HasCompactionEventForIteration(ctx, run.ID, run.Iteration)
	if err != nil {
		return err
	}
	if alreadyRecorded {
		return nil
	}

	allSteps, err := store.ListStepsByRun(ctx, run.ID)
	if err != nil {
		return fmt.Errorf("failed to list steps for compaction check: %w", err)
	}

	latestCompaction, err := store.GetLatestCompaction(ctx, run.ID)
	if err != nil {
		return fmt.Errorf("failed to load latest compaction for run %s: %w", run.ID, err)
	}

	prevCompactedIter := 0
	prevVersion := 0
	if latestCompaction != nil {
		prevCompactedIter = latestCompaction.CompactedUpToIter
		prevVersion = latestCompaction.Version
	}

	baseMsgs, err := DecodeMessages(run.ContextJSON)
	if err != nil {
		return fmt.Errorf("failed to decode base messages for compaction check: %w", err)
	}
	activeMsgs := reconstructChatMessages(baseMsgs, allSteps, latestCompaction)
	encodedActive, err := EncodeMessages(activeMsgs)
	if err != nil {
		return fmt.Errorf("failed to encode active messages for compaction check: %w", err)
	}
	activeBytes := len(encodedActive)

	mode, targetIter := evaluateCompactionTrigger(cfg, run.Iteration, prevCompactedIter, activeBytes, allSteps)
	if mode == compactionTriggerNone {
		return nil
	}

	var stepsToCompact, tailSteps []FSMStep
	for _, s := range allSteps {
		if s.Iteration > prevCompactedIter && s.Iteration <= targetIter {
			stepsToCompact = append(stepsToCompact, s)
		} else if s.Iteration > targetIter && s.Iteration <= run.Iteration {
			tailSteps = append(tailSteps, s)
		}
	}

	originalGoal := extractOriginalObjective(baseMsgs)
	if latestCompaction != nil && latestCompaction.OriginalObjective != "" {
		originalGoal = latestCompaction.OriginalObjective
	}
	activeSkills, activeMemories := extractActiveKnowledgeVersions(allSteps, latestCompaction)

	input := CompactionProbeInput{
		Run:              run,
		Model:            model,
		TargetIter:       targetIter,
		PreviousState:    latestCompaction,
		StepsToCompact:   stepsToCompact,
		TailSteps:        tailSteps,
		OriginalUserGoal: originalGoal,
		ActiveBytes:      activeBytes,
		ActiveSkills:     activeSkills,
		ActiveMemories:   activeMemories,
	}

	if mode == compactionTriggerForceBackstop {
		state := buildDeterministicCompactedState(input, CompactionVerdictBackstopForced)
		skipReason, err := validateCandidateCompaction(state, baseMsgs, allSteps, activeBytes)
		if err != nil {
			return err
		}
		if skipReason != "" {
			slog.Warn("fsm forced backstop compaction skipped because candidate state failed validation",
				"run_id", run.ID,
				"iteration", run.Iteration,
				"reason", skipReason,
			)
			return recordProbeSkipEvent(ctx, store, run.ID, run.Iteration, prevVersion, prevCompactedIter, model, activeBytes, skipReason)
		}
		ev, err := MakeSnapshotCompactionEvent(state, run.ID, 1, run.Iteration, "active context exceeded hard byte threshold")
		if err != nil {
			return fmt.Errorf("failed to build forced compaction event: %w", err)
		}
		if err := store.SaveCompactionWithEvent(ctx, state, ev); err != nil {
			return fmt.Errorf("failed to save forced compaction state: %w", err)
		}
		slog.Info("fsm trajectory compacted via hard byte backstop",
			"run_id", run.ID,
			"version", state.Version,
			"compacted_up_to_iter", state.CompactedUpToIter,
			"input_bytes", state.InputBytes,
			"output_bytes", state.OutputBytes,
		)
		return nil
	}

	if decider == nil {
		decider = &llmCompactionDecider{client: r.llmClient}
	}

	probeRes, err := decider.DecideCompaction(ctx, input)
	if err != nil {
		slog.Warn("fsm compaction rubric probe failed, recording probe skip event", "run_id", run.ID, "iteration", run.Iteration, "error", err)
		return recordProbeSkipEvent(ctx, store, run.ID, run.Iteration, prevVersion, prevCompactedIter, model, activeBytes, fmt.Sprintf("probe_error: %v", err))
	}
	if probeRes == nil || !probeRes.ShouldCompact {
		reason := ""
		if probeRes != nil {
			reason = probeRes.Reason
		}
		return recordProbeSkipEvent(ctx, store, run.ID, run.Iteration, prevVersion, prevCompactedIter, model, activeBytes, reason)
	}

	state := probeRes.State
	if state == nil {
		state = buildDeterministicCompactedState(input, CompactionVerdictProbedCompact)
	} else {
		state.ID = ""
		state.Version = prevVersion + 1
		state.ParentVersion = prevVersion
		state.RunID = run.ID
		state.CompactedUpToIter = targetIter
		state.CompactedStepIDs = collectCompactedStepIDs(latestCompaction, stepsToCompact)
		if strings.TrimSpace(state.OriginalObjective) == "" {
			state.OriginalObjective = originalGoal
		}
		if strings.TrimSpace(state.CurrentSubgoal) == "" {
			if latestCompaction != nil && strings.TrimSpace(latestCompaction.CurrentSubgoal) != "" {
				state.CurrentSubgoal = latestCompaction.CurrentSubgoal
			} else {
				state.CurrentSubgoal = fmt.Sprintf("Continue execution from iteration %d", targetIter+1)
			}
		}
		if len(state.NextIntendedActions) == 0 && latestCompaction != nil {
			state.NextIntendedActions = append([]string(nil), latestCompaction.NextIntendedActions...)
		}
		if len(state.ActiveSkillVersions) == 0 {
			state.ActiveSkillVersions = activeSkills
		}
		if len(state.ActiveMemoryVersions) == 0 {
			state.ActiveMemoryVersions = activeMemories
		}
		state.RubricVerdict = CompactionVerdictProbedCompact
		state.Model = model
		state.InputBytes = activeBytes
	}

	skipReason, err := validateCandidateCompaction(state, baseMsgs, allSteps, activeBytes)
	if err != nil {
		return err
	}
	if skipReason != "" {
		slog.Warn("fsm compaction probe candidate rejected by size/reduction check",
			"run_id", run.ID,
			"iteration", run.Iteration,
			"reason", skipReason,
		)
		return recordProbeSkipEvent(ctx, store, run.ID, run.Iteration, prevVersion, prevCompactedIter, model, activeBytes, skipReason)
	}

	ev, err := MakeSnapshotCompactionEvent(state, run.ID, 1, run.Iteration, probeRes.Reason)
	if err != nil {
		return fmt.Errorf("failed to build probed compaction event: %w", err)
	}
	if err := store.SaveCompactionWithEvent(ctx, state, ev); err != nil {
		return fmt.Errorf("failed to save probed compaction state: %w", err)
	}
	slog.Info("fsm trajectory compacted via rubric probe",
		"run_id", run.ID,
		"version", state.Version,
		"compacted_up_to_iter", state.CompactedUpToIter,
		"input_bytes", state.InputBytes,
		"output_bytes", state.OutputBytes,
	)
	return nil
}
