package fsm

import (
	"container/list"
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"log/slog"
	"reflect"
	"strings"
	"sync"
	"time"

	"bob/internal/llm"
	"bob/internal/trajectory"

	openai "github.com/sashabaranov/go-openai"
)

const loopGuardHybridBackstopCeiling = 5
const loopGuardDecisionCacheSize = 128

type loopDecisionCacheKey struct {
	client SystemOneDecisionClient
	model  string
	mode   string
	state  [32]byte
}

type loopDecisionCacheValue struct {
	key   loopDecisionCacheKey
	entry *llm.DecisionResultEntry
	resp  *llm.DecisionResponse
}

var loopDecisionCache = struct {
	sync.Mutex
	entries map[loopDecisionCacheKey]*list.Element
	order   *list.List
}{entries: make(map[loopDecisionCacheKey]*list.Element), order: list.New()}

func cachedLoopDecision(key loopDecisionCacheKey) (*llm.DecisionResultEntry, *llm.DecisionResponse, bool) {
	loopDecisionCache.Lock()
	defer loopDecisionCache.Unlock()
	element := loopDecisionCache.entries[key]
	if element == nil {
		return nil, nil, false
	}
	loopDecisionCache.order.MoveToFront(element)
	value := element.Value.(loopDecisionCacheValue)
	return value.entry, value.resp, true
}

func cacheLoopDecision(key loopDecisionCacheKey, entry *llm.DecisionResultEntry, resp *llm.DecisionResponse) {
	loopDecisionCache.Lock()
	defer loopDecisionCache.Unlock()
	if element := loopDecisionCache.entries[key]; element != nil {
		element.Value = loopDecisionCacheValue{key: key, entry: entry, resp: resp}
		loopDecisionCache.order.MoveToFront(element)
		return
	}
	element := loopDecisionCache.order.PushFront(loopDecisionCacheValue{key: key, entry: entry, resp: resp})
	loopDecisionCache.entries[key] = element
	if loopDecisionCache.order.Len() > loopGuardDecisionCacheSize {
		last := loopDecisionCache.order.Back()
		delete(loopDecisionCache.entries, last.Value.(loopDecisionCacheValue).key)
		loopDecisionCache.order.Remove(last)
	}
}

// LoopGuardNoticePrefix is the prefix used for step observations intercepted by the loop guard.
const LoopGuardNoticePrefix = "Loop guard notice:"

// LoopGuardBlockedErrorText is the error text recorded on steps intercepted by the loop guard.
const LoopGuardBlockedErrorText = "loop_guard_blocked"

// LoopGuardConfig configures loop detection and prevention in the tool loop FSM.
type LoopGuardConfig struct {
	Enabled             bool
	MaxConsecutiveCalls int // Number of consecutive identical calls before interception (default: 3)
	MaxCycleRepetitions int // Number of identical cycle repetitions before interception (default: 2)
	MaxCycleLength      int // Maximum cycle period to inspect (default: 5)
}

// DefaultLoopGuardConfig returns standard production defaults for the loop guard.
func DefaultLoopGuardConfig() LoopGuardConfig {
	return LoopGuardConfig{
		Enabled:             true,
		MaxConsecutiveCalls: 3,
		MaxCycleRepetitions: 2,
		MaxCycleLength:      5,
	}
}

// LoopVerdictAction indicates the action recommended by the loop guard.
type LoopVerdictAction string

const (
	LoopActionNone           LoopVerdictAction = "none"
	LoopActionIntercept      LoopVerdictAction = "intercept"       // Strike 1: Intercept execution and return notice
	LoopActionForceSynthesis LoopVerdictAction = "force_synthesis" // Strike 2: Terminate tool loop and force synthesis
)

// LoopVerdict contains the outcome of loop guard analysis.
type LoopVerdict struct {
	Action       LoopVerdictAction
	Reason       string
	NoticeByCall map[int]string // candidate tool call index -> notice message
}

type toolCallSig struct {
	name string
	args string // normalized
}

func normalizeToolArgs(args string) string {
	trimmed := strings.TrimSpace(args)
	if trimmed == "" {
		return ""
	}
	var v any
	if err := json.Unmarshal([]byte(trimmed), &v); err == nil {
		if b, err := json.Marshal(v); err == nil {
			return string(b)
		}
	}
	return trimmed
}

func makeSig(name, args string) toolCallSig {
	return toolCallSig{
		name: strings.TrimSpace(name),
		args: normalizeToolArgs(args),
	}
}

// FormatLoopGuardNotice creates the user/model-visible observation returned when a step is intercepted.
func FormatLoopGuardNotice(toolName, argsJSON string) string {
	return fmt.Sprintf("%s Tool %q with identical arguments was already executed with unchanged output. Redundant execution was blocked to prevent an infinite loop. Do not repeat this command. Proceed immediately to take action (e.g. create or edit files) or provide your final response.", LoopGuardNoticePrefix, toolName)
}

// DetectLoop analyzes previous execution steps and candidate tool calls to identify stuck repetitive loops.
func DetectLoop(history []FSMStep, candidateCalls []openai.ToolCall, cfg LoopGuardConfig) LoopVerdict {
	if !cfg.Enabled || len(candidateCalls) == 0 {
		return LoopVerdict{Action: LoopActionNone}
	}
	if cfg.MaxConsecutiveCalls <= 0 {
		cfg.MaxConsecutiveCalls = 3
	}
	if cfg.MaxCycleRepetitions < 2 {
		cfg.MaxCycleRepetitions = 2
	}
	if cfg.MaxCycleLength <= 0 {
		cfg.MaxCycleLength = 5
	}

	evidence := inspectLoopHistory(history, candidateCalls, cfg)
	if len(evidence.notices) == 0 {
		return LoopVerdict{Action: LoopActionNone}
	}
	if evidence.struck && len(evidence.notices) == len(candidateCalls) {
		return LoopVerdict{Action: LoopActionForceSynthesis, Reason: evidence.reason}
	}
	return LoopVerdict{Action: LoopActionIntercept, Reason: evidence.reason, NoticeByCall: evidence.notices}
}

func isBlockedLoopStep(s FSMStep) bool {
	if s.ErrorText == LoopGuardBlockedErrorText || strings.HasPrefix(s.ResultJSON, LoopGuardNoticePrefix) {
		return true
	}
	var marker struct {
		Blocked bool `json:"loop_guard_blocked"`
	}
	return json.Unmarshal([]byte(s.ResultJSON), &marker) == nil && marker.Blocked
}

type loopHistoryEvidence struct {
	notices map[int]string
	reason  string
	struck  bool
}

func loopObservation(s FSMStep) [32]byte {
	content := strings.TrimSpace(storedResultForProbe(s.ResultJSON))
	var plain struct {
		Content string `json:"content"`
	}
	if json.Unmarshal([]byte(content), &plain) == nil && plain.Content != "" {
		content = plain.Content
	}
	return sha256.Sum256([]byte(content + "\x00" + s.ErrorText))
}

func isStateMutatingStep(s FSMStep) bool {
	if (s.Status != StepStatusCompleted && s.Status != StepStatusFailed && s.Status != StepStatusTimedOut) || isBlockedLoopStep(s) {
		return false
	}
	name := strings.ToLower(s.ToolName)
	for _, marker := range []string{"write", "edit", "patch", "delete", "create_file", "save_file"} {
		if strings.Contains(name, marker) {
			return true
		}
	}
	args := strings.ToLower(s.ArgsJSON)
	for _, marker := range []string{"apply_patch", "sed -i", " >", "> ", ">>", ".write_text(", ".write_bytes(",
		"git checkout", "git restore", "git commit", "git add", "git apply", "mkdir ", "touch ", "tee "} {
		if strings.Contains(args, marker) {
			return true
		}
	}
	return false
}

func executedLoopSteps(history []FSMStep) []FSMStep {
	var executed []FSMStep
	for _, step := range history {
		if isStateMutatingStep(step) {
			executed = nil
			continue
		}
		if isBlockedLoopStep(step) || (step.Status != StepStatusCompleted && step.Status != StepStatusFailed && step.Status != StepStatusTimedOut) {
			continue
		}
		executed = append(executed, step)
	}
	return executed
}

func candidateLoopEvidence(executed []FSMStep, candidateCalls []openai.ToolCall, cfg LoopGuardConfig) (map[int]string, string) {
	if cfg.MaxConsecutiveCalls <= 0 {
		cfg.MaxConsecutiveCalls = 3
	}
	if cfg.MaxCycleRepetitions < 2 {
		cfg.MaxCycleRepetitions = 2
	}
	if cfg.MaxCycleLength <= 0 {
		cfg.MaxCycleLength = 5
	}
	notices := make(map[int]string)
	reason := ""
	for index, call := range candidateCalls {
		sig := makeSig(call.Function.Name, call.Function.Arguments)
		if isStateMutatingStep(FSMStep{ToolName: call.Function.Name, ArgsJSON: call.Function.Arguments, Status: StepStatusCompleted}) {
			continue
		}
		needed := max(2, cfg.MaxConsecutiveCalls-1)
		if len(executed) >= needed {
			last := executed[len(executed)-1]
			matching := true
			for i := len(executed) - needed; i < len(executed); i++ {
				if makeSig(executed[i].ToolName, executed[i].ArgsJSON) != sig || loopObservation(executed[i]) != loopObservation(last) {
					matching = false
					break
				}
			}
			if matching {
				notices[index] = FormatLoopGuardNotice(call.Function.Name, call.Function.Arguments)
				reason = fmt.Sprintf("tool %q repeated %d consecutive times without state progress", sig.name, needed+1)
				continue
			}
		}
		for period := 2; period <= cfg.MaxCycleLength && period*cfg.MaxCycleRepetitions <= len(executed); period++ {
			start := len(executed) - period*cfg.MaxCycleRepetitions
			matching := true
			for i := start + period; i < len(executed); i++ {
				if makeSig(executed[i].ToolName, executed[i].ArgsJSON) != makeSig(executed[i-period].ToolName, executed[i-period].ArgsJSON) || loopObservation(executed[i]) != loopObservation(executed[i-period]) {
					matching = false
					break
				}
			}
			if matching && sig == makeSig(executed[len(executed)-period+index%period].ToolName, executed[len(executed)-period+index%period].ArgsJSON) {
				notices[index] = FormatLoopGuardNotice(call.Function.Name, call.Function.Arguments)
				reason = fmt.Sprintf("cyclic sequence of %d tool calls repeated %d times", period, cfg.MaxCycleRepetitions)
				break
			}
		}
	}
	return notices, reason
}

func inspectLoopHistory(history []FSMStep, candidateCalls []openai.ToolCall, cfg LoopGuardConfig) loopHistoryEvidence {
	executed := executedLoopSteps(history)
	notices, reason := candidateLoopEvidence(executed, candidateCalls, cfg)
	evidence := loopHistoryEvidence{notices: notices, reason: reason}
	if len(notices) == 0 {
		return evidence
	}
	for i := len(history) - 1; i >= 0; i-- {
		step := history[i]
		if isStateMutatingStep(step) {
			break
		}
		if !isBlockedLoopStep(step) {
			if step.Status == StepStatusCompleted || step.Status == StepStatusFailed || step.Status == StepStatusTimedOut {
				break
			}
			continue
		}
		prior := executedLoopSteps(history[:i])
		blockedCall := openai.ToolCall{Function: openai.FunctionCall{Name: step.ToolName, Arguments: step.ArgsJSON}}
		priorNotices, _ := candidateLoopEvidence(prior, []openai.ToolCall{blockedCall}, cfg)
		if len(priorNotices) == 0 {
			continue
		}
		for index := range notices {
			if makeSig(candidateCalls[index].Function.Name, candidateCalls[index].Function.Arguments) == makeSig(step.ToolName, step.ArgsJSON) {
				evidence.struck = true
				evidence.reason = fmt.Sprintf("tool %q repeated after being blocked by loop guard", step.ToolName)
				return evidence
			}
		}
	}
	return evidence
}

func trailingUnchangedExecuted(history []FSMStep, candidate openai.ToolCall) int {
	executed := executedLoopSteps(history)
	if len(executed) == 0 {
		return 0
	}
	sig := makeSig(candidate.Function.Name, candidate.Function.Arguments)
	last := loopObservation(executed[len(executed)-1])
	count := 0
	for i := len(executed) - 1; i >= 0; i-- {
		if makeSig(executed[i].ToolName, executed[i].ArgsJSON) != sig || loopObservation(executed[i]) != last {
			break
		}
		count++
	}
	return count
}

type loopProgressLedger struct {
	targets             map[string]bool
	observations        map[[32]byte]int
	repeatedObservation bool
	recentFailure       bool
	consecutiveFailures int
	stateMutatingEdits  int
}

func loopTargetKeys(name, args string) []string {
	name = strings.ToLower(strings.TrimSpace(name))
	keys := []string{name + "|args:" + normalizeToolArgs(args)}
	decoded, ok := decodeJSONBounded(args)
	if !ok {
		return keys
	}
	obj, ok := decoded.(map[string]any)
	if !ok {
		return keys
	}
	for _, field := range []string{"path", "file", "url", "query", "resource", "command"} {
		value, ok := obj[field].(string)
		if !ok {
			continue
		}
		normalized := strings.Join(strings.Fields(strings.ToLower(value)), " ")
		if normalized == "" {
			continue
		}
		keys = append(keys, name+"|"+field+":"+normalized)
		if field == "command" {
			words := strings.Fields(normalized)
			if len(words) > 2 {
				words = words[:2]
			}
			keys = append(keys, name+"|command_stem:"+strings.Join(words, " "))
		}
	}
	return keys
}

func buildLoopProgressLedger(history []FSMStep) loopProgressLedger {
	ledger := loopProgressLedger{targets: make(map[string]bool), observations: make(map[[32]byte]int)}
	var sinceEdit []FSMStep
	for _, step := range history {
		if isStateMutatingStep(step) {
			ledger.targets = make(map[string]bool)
			sinceEdit = nil
			ledger.stateMutatingEdits++
			continue
		}
		if isBlockedLoopStep(step) || (step.Status != StepStatusCompleted && step.Status != StepStatusFailed && step.Status != StepStatusTimedOut) {
			continue
		}
		for _, key := range loopTargetKeys(step.ToolName, step.ArgsJSON) {
			ledger.targets[key] = true
		}
		sinceEdit = append(sinceEdit, step)
	}
	recent := sinceEdit
	if len(recent) > systemOneLoopGuardHistoryWindow {
		recent = recent[len(recent)-systemOneLoopGuardHistoryWindow:]
	}
	for _, step := range recent {
		fingerprint := loopObservation(step)
		ledger.observations[fingerprint]++
		if ledger.observations[fingerprint] >= 2 {
			ledger.repeatedObservation = true
		}
	}
	failureWindow := recent
	if len(failureWindow) > 4 {
		failureWindow = failureWindow[len(failureWindow)-4:]
	}
	for _, step := range failureWindow {
		if isStepFailure(step) {
			ledger.recentFailure = true
			ledger.consecutiveFailures++
		} else {
			ledger.consecutiveFailures = 0
		}
	}
	return ledger
}

func loopGuardProbeTrigger(run *FSMRun, history []FSMStep, candidates []openai.ToolCall, heuristic LoopVerdict, evidence loopHistoryEvidence, cfg SystemOneControlConfig) string {
	if cfg.LoopGuardProbeAlways {
		return "forced"
	}
	if heuristic.Action != LoopActionNone || len(evidence.notices) > 0 {
		return "executed_stall"
	}
	ledger := buildLoopProgressLedger(history)
	for _, call := range candidates {
		for _, key := range loopTargetKeys(call.Function.Name, call.Function.Arguments) {
			if ledger.targets[key] {
				return "target_revisit"
			}
		}
	}
	if ledger.repeatedObservation {
		return "repeated_observation"
	}
	if ledger.recentFailure {
		return "recent_failure"
	}
	interval := cfg.LoopGuardAuditInterval
	if interval <= 0 {
		interval = 4
	}
	if run != nil && run.Iteration > 0 && run.Iteration%interval == 0 {
		return "periodic_audit"
	}
	return ""
}

// EvaluateLoopGuard runs deterministic and optional System-1 semantic loop detection
// according to off/shadow/hybrid policies.
func EvaluateLoopGuard(
	ctx context.Context,
	run *FSMRun,
	goal string,
	history []FSMStep,
	candidateCalls []openai.ToolCall,
	cfg LoopGuardConfig,
	s1Client SystemOneDecisionClient,
	s1Cfg SystemOneControlConfig,
) (LoopVerdict, *trajectory.SystemOneDecisionPayload, error) {
	if !cfg.Enabled || len(candidateCalls) == 0 {
		return LoopVerdict{Action: LoopActionNone}, nil, nil
	}

	evidence := inspectLoopHistory(history, candidateCalls, cfg)
	heuristicVerdict := DetectLoop(history, candidateCalls, cfg)
	mode := strings.ToLower(strings.TrimSpace(s1Cfg.LoopGuardMode))
	if s1Client == nil || mode == "" || mode == "off" {
		return heuristicVerdict, nil, nil
	}
	trigger := loopGuardProbeTrigger(run, history, candidateCalls, heuristicVerdict, evidence, s1Cfg)
	if trigger == "" {
		return heuristicVerdict, nil, nil
	}

	rendered := RenderSystemOneLoopGuardState(run, goal, history, candidateCalls)
	probeStart := time.Now()
	var entry *llm.DecisionResultEntry
	var resp *llm.DecisionResponse
	var s1Err error
	var cacheHit bool
	if reflect.TypeOf(s1Client).Comparable() {
		key := loopDecisionCacheKey{client: s1Client, model: s1Client.Model(), mode: mode, state: sha256.Sum256([]byte(rendered.Text))}
		entry, resp, cacheHit = cachedLoopDecision(key)
		if !cacheHit {
			entry, resp, s1Err = s1Client.DecideLoopGuard(ctx, rendered.Text, nil)
			if s1Err == nil && entry != nil {
				cacheLoopDecision(key, entry, resp)
			}
		}
	} else {
		entry, resp, s1Err = s1Client.DecideLoopGuard(ctx, rendered.Text, nil)
	}
	elapsedMs := time.Since(probeStart).Milliseconds()
	if s1Err != nil {
		latencyMs := elapsedMs
		if resp != nil && resp.Latency > 0 {
			latencyMs = resp.Latency.Milliseconds()
		}
		telemetry := &trajectory.SystemOneDecisionPayload{
			Task:            llm.SystemOneTaskLoopGuard,
			Truncated:       rendered.Truncated,
			ProbeTrigger:    trigger,
			LatencyMs:       latencyMs,
			AppliedAction:   string(heuristicVerdict.Action),
			HeuristicAction: string(heuristicVerdict.Action),
			FallbackUsed:    mode == "hybrid" && s1Cfg.FallbackOnError,
			ShadowMode:      mode == "shadow",
			Error:           s1Err.Error(),
		}
		applySystemOneRenderTelemetry(telemetry, rendered)
		if heuristicVerdict.Action == LoopActionForceSynthesis {
			return heuristicVerdict, telemetry, nil
		}
		if mode == "shadow" {
			// Shadow mode never alters deterministic execution; strict policy still logs/surfaces error explicitly.
			if !s1Cfg.FallbackOnError {
				slog.Error("system1 loop_guard shadow probe failed under strict fallback_on_error=false",
					"error", s1Err,
					"latency_ms", latencyMs,
				)
			}
			return heuristicVerdict, telemetry, nil
		}
		if !s1Cfg.FallbackOnError {
			telemetry.AppliedAction = "error"
			return LoopVerdict{}, telemetry, fmt.Errorf("system1 loop_guard probe failed: %w", s1Err)
		}
		return heuristicVerdict, telemetry, nil
	}

	if cacheHit {
		resp = nil
		elapsedMs = 0
	}
	telemetry := buildSystemOneDecisionTelemetry(llm.SystemOneTaskLoopGuard, entry, resp, elapsedMs, rendered)
	telemetry.HeuristicAction = string(heuristicVerdict.Action)
	telemetry.ShadowMode = mode == "shadow"
	telemetry.ProbeTrigger = trigger
	telemetry.CacheHit = cacheHit
	if heuristicVerdict.Action == LoopActionForceSynthesis {
		telemetry.AppliedAction = string(LoopActionForceSynthesis)
		return heuristicVerdict, telemetry, nil
	}

	if mode == "shadow" {
		telemetry.AppliedAction = string(heuristicVerdict.Action)
		return heuristicVerdict, telemetry, nil
	}

	if mode != "hybrid" {
		err := fmt.Errorf("unsupported system1 loop_guard_mode %q", s1Cfg.LoopGuardMode)
		telemetry.Error = err.Error()
		if !s1Cfg.FallbackOnError {
			telemetry.AppliedAction = "error"
			return LoopVerdict{}, telemetry, err
		}
		telemetry.AppliedAction = string(heuristicVerdict.Action)
		telemetry.FallbackUsed = true
		return heuristicVerdict, telemetry, nil
	}

	choice := ""
	if entry != nil {
		choice = entry.Choice
	}

	switch choice {
	case string(LoopActionNone):
		// Enforce deterministic hard backstop ceiling (>= 5 consecutive unchanged calls) against false negatives.
		for index := range evidence.notices {
			consecutive := trailingUnchangedExecuted(history, candidateCalls[index]) + 1
			if consecutive < loopGuardHybridBackstopCeiling {
				continue
			}
			reason := fmt.Sprintf("hard backstop triggered after %d consecutive unchanged calls to %q", consecutive, candidateCalls[index].Function.Name)
			if evidence.struck && len(evidence.notices) == len(candidateCalls) {
				telemetry.AppliedAction = string(LoopActionForceSynthesis)
				return LoopVerdict{Action: LoopActionForceSynthesis, Reason: reason}, telemetry, nil
			}
			telemetry.AppliedAction = string(LoopActionIntercept)
			return LoopVerdict{
				Action:       LoopActionIntercept,
				Reason:       reason,
				NoticeByCall: evidence.notices,
			}, telemetry, nil
		}
		telemetry.AppliedAction = string(LoopActionNone)
		return LoopVerdict{Action: LoopActionNone}, telemetry, nil

	case string(LoopActionIntercept):
		if len(evidence.notices) == 0 {
			telemetry.AppliedAction = "advisory"
			return LoopVerdict{Action: LoopActionNone}, telemetry, nil
		}
		telemetry.AppliedAction = string(LoopActionIntercept)
		return LoopVerdict{
			Action:       LoopActionIntercept,
			Reason:       "bob-s1 loop_guard intercepted repetitive tool execution",
			NoticeByCall: evidence.notices,
		}, telemetry, nil

	case string(LoopActionForceSynthesis):
		if len(evidence.notices) == 0 {
			telemetry.AppliedAction = "advisory"
			return LoopVerdict{Action: LoopActionNone}, telemetry, nil
		}
		if !evidence.struck || len(evidence.notices) != len(candidateCalls) {
			telemetry.AppliedAction = string(LoopActionIntercept)
			return LoopVerdict{
				Action:       LoopActionIntercept,
				Reason:       "bob-s1 loop_guard downgraded force_synthesis to intercept before strike 1",
				NoticeByCall: evidence.notices,
			}, telemetry, nil
		}
		telemetry.AppliedAction = string(LoopActionForceSynthesis)
		return LoopVerdict{
			Action: LoopActionForceSynthesis,
			Reason: "bob-s1 loop_guard confirmed repeated blocked loop; forcing synthesis",
		}, telemetry, nil

	default:
		err := fmt.Errorf("unexpected bob-s1 loop_guard choice %q", choice)
		telemetry.Error = err.Error()
		if !s1Cfg.FallbackOnError {
			telemetry.AppliedAction = "error"
			return LoopVerdict{}, telemetry, err
		}
		telemetry.AppliedAction = string(heuristicVerdict.Action)
		telemetry.FallbackUsed = true
		return heuristicVerdict, telemetry, nil
	}
}
