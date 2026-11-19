package fsm

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
	"time"

	"bob/internal/llm"
	"bob/internal/trajectory"

	openai "github.com/sashabaranov/go-openai"
)

const loopGuardHybridBackstopCeiling = 5

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
	if cfg.MaxCycleRepetitions <= 0 {
		cfg.MaxCycleRepetitions = 2
	}
	if cfg.MaxCycleLength <= 0 {
		cfg.MaxCycleLength = 5
	}

	hasPriorStrike, blockedReason, historySigs, candidateSigs := inspectLoopHistory(history, candidateCalls)
	if blockedReason != "" {
		return LoopVerdict{
			Action: LoopActionForceSynthesis,
			Reason: blockedReason,
		}
	}

	allSigs := append(historySigs, candidateSigs...)
	totalCalls := len(allSigs)

	// 1. Check for consecutive identical calls (period p = 1)
	consecutive, lastSig := trailingConsecutiveCalls(allSigs)

	consecutiveDetected := false
	if consecutive >= cfg.MaxConsecutiveCalls {
		consecutiveDetected = true
	} else if consecutive >= 2 && len(history) >= 2 && hasMatchingUnchangedOutputs(history, historySigs, candidateCalls, lastSig, consecutive) {
		consecutiveDetected = true
	}

	if consecutiveDetected {
		reason := fmt.Sprintf("tool %q repeated %d consecutive times without state progress", lastSig.name, consecutive)
		if hasPriorStrike {
			return LoopVerdict{
				Action: LoopActionForceSynthesis,
				Reason: reason,
			}
		}
		notices := make(map[int]string)
		for i := range candidateCalls {
			if candidateSigs[i] == lastSig {
				notices[i] = FormatLoopGuardNotice(candidateCalls[i].Function.Name, candidateCalls[i].Function.Arguments)
			}
		}
		return LoopVerdict{
			Action:       LoopActionIntercept,
			Reason:       reason,
			NoticeByCall: notices,
		}
	}

	// 2. Check for cyclic sequence repetitions (period p in [2, cfg.MaxCycleLength])
	for p := 2; p <= cfg.MaxCycleLength && p*cfg.MaxCycleRepetitions <= totalCalls; p++ {
		// Check if the last p*MaxCycleRepetitions signatures form an identical repeating sequence
		repeats := true
		targetBlock := allSigs[totalCalls-p : totalCalls]
		for r := 1; r < cfg.MaxCycleRepetitions; r++ {
			blockStart := totalCalls - (r+1)*p
			for i := 0; i < p; i++ {
				if allSigs[blockStart+i] != targetBlock[i] {
					repeats = false
					break
				}
			}
			if !repeats {
				break
			}
		}

		if repeats {
			reason := fmt.Sprintf("cyclic sequence of %d tool calls repeated %d times", p, cfg.MaxCycleRepetitions)
			if hasPriorStrike {
				return LoopVerdict{
					Action: LoopActionForceSynthesis,
					Reason: reason,
				}
			}
			notices := make(map[int]string)
			for i := range candidateCalls {
				notices[i] = FormatLoopGuardNotice(candidateCalls[i].Function.Name, candidateCalls[i].Function.Arguments)
			}
			return LoopVerdict{
				Action:       LoopActionIntercept,
				Reason:       reason,
				NoticeByCall: notices,
			}
		}
	}

	return LoopVerdict{Action: LoopActionNone}
}

func isBlockedLoopStep(s FSMStep) bool {
	return s.ErrorText == LoopGuardBlockedErrorText || strings.HasPrefix(s.ResultJSON, LoopGuardNoticePrefix)
}

func inspectLoopHistory(history []FSMStep, candidateCalls []openai.ToolCall) (hasPriorStrike bool, blockedRepeatReason string, historySigs []toolCallSig, candidateSigs []toolCallSig) {
	historySigs = make([]toolCallSig, len(history))
	for i, s := range history {
		historySigs[i] = makeSig(s.ToolName, s.ArgsJSON)
		if isBlockedLoopStep(s) {
			hasPriorStrike = true
		}
	}

	candidateSigs = make([]toolCallSig, len(candidateCalls))
	for i, tc := range candidateCalls {
		candidateSigs[i] = makeSig(tc.Function.Name, tc.Function.Arguments)
	}

	if hasPriorStrike {
		for i, s := range history {
			if !isBlockedLoopStep(s) {
				continue
			}
			blockedSig := historySigs[i]
			for _, cSig := range candidateSigs {
				if cSig == blockedSig {
					return true, fmt.Sprintf("tool %q repeated after being blocked by loop guard", cSig.name), historySigs, candidateSigs
				}
			}
		}
	}
	return hasPriorStrike, "", historySigs, candidateSigs
}

func trailingConsecutiveCalls(allSigs []toolCallSig) (int, toolCallSig) {
	if len(allSigs) == 0 {
		return 0, toolCallSig{}
	}
	lastSig := allSigs[len(allSigs)-1]
	consecutive := 0
	for i := len(allSigs) - 1; i >= 0; i-- {
		if allSigs[i] == lastSig {
			consecutive++
		} else {
			break
		}
	}
	return consecutive, lastSig
}

func hasMatchingUnchangedOutputs(history []FSMStep, historySigs []toolCallSig, candidateCalls []openai.ToolCall, targetSig toolCallSig, consecutive int) bool {
	trailingCount := consecutive - len(candidateCalls)
	if trailingCount < 2 || len(history) < trailingCount {
		return false
	}
	var firstOut string
	firstSet := false
	start := len(history) - trailingCount
	for i := start; i < len(history); i++ {
		if historySigs[i] != targetSig {
			return false
		}
		if !firstSet {
			firstOut = history[i].ResultJSON
			firstSet = true
		} else if history[i].ResultJSON != firstOut {
			return false
		}
	}
	return firstSet
}

func buildAllCandidateNotices(candidateCalls []openai.ToolCall) map[int]string {
	notices := make(map[int]string, len(candidateCalls))
	for i, tc := range candidateCalls {
		notices[i] = FormatLoopGuardNotice(tc.Function.Name, tc.Function.Arguments)
	}
	return notices
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

	hasPriorStrike, blockedReason, historySigs, candidateSigs := inspectLoopHistory(history, candidateCalls)
	if blockedReason != "" {
		return LoopVerdict{
			Action: LoopActionForceSynthesis,
			Reason: blockedReason,
		}, nil, nil
	}

	heuristicVerdict := DetectLoop(history, candidateCalls, cfg)
	mode := strings.ToLower(strings.TrimSpace(s1Cfg.LoopGuardMode))
	if s1Client == nil || mode == "" || mode == "off" || len(history) < 2 {
		return heuristicVerdict, nil, nil
	}

	rendered := RenderSystemOneLoopGuardState(run, goal, history, candidateCalls)
	probeStart := time.Now()
	entry, resp, s1Err := s1Client.DecideLoopGuard(ctx, rendered.Text, nil)
	elapsedMs := time.Since(probeStart).Milliseconds()
	if s1Err != nil {
		latencyMs := elapsedMs
		if resp != nil && resp.Latency > 0 {
			latencyMs = resp.Latency.Milliseconds()
		}
		telemetry := &trajectory.SystemOneDecisionPayload{
			Task:            llm.SystemOneTaskLoopGuard,
			Truncated:       rendered.Truncated,
			LatencyMs:       latencyMs,
			AppliedAction:   string(heuristicVerdict.Action),
			HeuristicAction: string(heuristicVerdict.Action),
			FallbackUsed:    mode == "hybrid" && s1Cfg.FallbackOnError,
			ShadowMode:      mode == "shadow",
			Error:           s1Err.Error(),
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

	telemetry := buildSystemOneDecisionTelemetry(llm.SystemOneTaskLoopGuard, entry, resp, elapsedMs, rendered.Truncated)
	telemetry.HeuristicAction = string(heuristicVerdict.Action)
	telemetry.ShadowMode = mode == "shadow"

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
		allSigs := append(append([]toolCallSig(nil), historySigs...), candidateSigs...)
		consecutive, lastSig := trailingConsecutiveCalls(allSigs)
		if consecutive >= loopGuardHybridBackstopCeiling &&
			len(history) >= 2 &&
			hasMatchingUnchangedOutputs(history, historySigs, candidateCalls, lastSig, consecutive) {
			reason := fmt.Sprintf("hard backstop triggered after %d consecutive unchanged calls to %q", consecutive, lastSig.name)
			if hasPriorStrike {
				telemetry.AppliedAction = string(LoopActionForceSynthesis)
				return LoopVerdict{Action: LoopActionForceSynthesis, Reason: reason}, telemetry, nil
			}
			telemetry.AppliedAction = string(LoopActionIntercept)
			return LoopVerdict{
				Action:       LoopActionIntercept,
				Reason:       reason,
				NoticeByCall: buildAllCandidateNotices(candidateCalls),
			}, telemetry, nil
		}
		telemetry.AppliedAction = string(LoopActionNone)
		return LoopVerdict{Action: LoopActionNone}, telemetry, nil

	case string(LoopActionIntercept):
		telemetry.AppliedAction = string(LoopActionIntercept)
		return LoopVerdict{
			Action:       LoopActionIntercept,
			Reason:       "bob-s1 loop_guard intercepted repetitive tool execution",
			NoticeByCall: buildAllCandidateNotices(candidateCalls),
		}, telemetry, nil

	case string(LoopActionForceSynthesis):
		if !hasPriorStrike {
			telemetry.AppliedAction = string(LoopActionIntercept)
			return LoopVerdict{
				Action:       LoopActionIntercept,
				Reason:       "bob-s1 loop_guard downgraded force_synthesis to intercept before strike 1",
				NoticeByCall: buildAllCandidateNotices(candidateCalls),
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
