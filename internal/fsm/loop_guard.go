package fsm

import (
	"encoding/json"
	"fmt"
	"strings"

	openai "github.com/sashabaranov/go-openai"
)

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
	Action      LoopVerdictAction
	Reason      string
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

	// Check if a previous strike 1 interception has already occurred in this run
	hasPriorStrike := false
	for _, s := range history {
		if s.ErrorText == LoopGuardBlockedErrorText || strings.HasPrefix(s.ResultJSON, LoopGuardNoticePrefix) {
			hasPriorStrike = true
			break
		}
	}

	historySigs := make([]toolCallSig, len(history))
	for i, s := range history {
		historySigs[i] = makeSig(s.ToolName, s.ArgsJSON)
	}

	candidateSigs := make([]toolCallSig, len(candidateCalls))
	for i, tc := range candidateCalls {
		candidateSigs[i] = makeSig(tc.Function.Name, tc.Function.Arguments)
	}

	// If a candidate call matches a tool call that was already blocked by the loop guard, immediately force synthesis
	if hasPriorStrike {
		for _, s := range history {
			if s.ErrorText == LoopGuardBlockedErrorText || strings.HasPrefix(s.ResultJSON, LoopGuardNoticePrefix) {
				blockedSig := makeSig(s.ToolName, s.ArgsJSON)
				for _, cSig := range candidateSigs {
					if cSig == blockedSig {
						return LoopVerdict{
							Action: LoopActionForceSynthesis,
							Reason: fmt.Sprintf("tool %q repeated after being blocked by loop guard", cSig.name),
						}
					}
				}
			}
		}
	}

	allSigs := append(historySigs, candidateSigs...)
	totalCalls := len(allSigs)

	// 1. Check for consecutive identical calls (period p = 1)
	lastSig := allSigs[totalCalls-1]
	consecutive := 0
	for i := totalCalls - 1; i >= 0; i-- {
		if allSigs[i] == lastSig {
			consecutive++
		} else {
			break
		}
	}

	consecutiveDetected := false
	if consecutive >= cfg.MaxConsecutiveCalls {
		consecutiveDetected = true
	} else if consecutive >= 2 && len(history) >= 2 {
		// If called consecutively with unchanged output from previous executions
		identicalOutputs := true
		countMatchingHistory := 0
		for i := len(history) - 1; i >= 0 && len(history)-1-i < consecutive-len(candidateCalls); i-- {
			if historySigs[i] == lastSig {
				countMatchingHistory++
			}
		}
		if countMatchingHistory >= 2 {
			var firstOut string
			firstSet := false
			for i := len(history) - 1; i >= 0; i-- {
				if historySigs[i] == lastSig {
					if !firstSet {
						firstOut = history[i].ResultJSON
						firstSet = true
					} else if history[i].ResultJSON != firstOut {
						identicalOutputs = false
						break
					}
				}
			}
			if identicalOutputs && firstSet {
				consecutiveDetected = true
			}
		}
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
			Action:      LoopActionIntercept,
			Reason:      reason,
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
				Action:      LoopActionIntercept,
				Reason:      reason,
				NoticeByCall: notices,
			}
		}
	}

	return LoopVerdict{Action: LoopActionNone}
}
