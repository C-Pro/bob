package fsm

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"bob/internal/agentapi"
	"bob/internal/llm"
	"bob/internal/trajectory"

	openai "github.com/sashabaranov/go-openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type mockSystemOneDecisionClient struct {
	model        string
	compactionFn func(ctx context.Context, state string, maxSeqLen int) (*llm.DecisionResultEntry, *llm.DecisionResponse, error)
	loopGuardFn  func(ctx context.Context, state string, images []string) (*llm.DecisionResultEntry, *llm.DecisionResponse, error)
	compCalls    atomic.Int32
	lgCalls      atomic.Int32
}

func (m *mockSystemOneDecisionClient) Model() string {
	if m.model == "" {
		return "bob-s1-0.8b"
	}
	return m.model
}

func (m *mockSystemOneDecisionClient) DecideCompaction(ctx context.Context, state string, maxSeqLen int) (*llm.DecisionResultEntry, *llm.DecisionResponse, error) {
	m.compCalls.Add(1)
	return m.compactionFn(ctx, state, maxSeqLen)
}

func (m *mockSystemOneDecisionClient) DecideLoopGuard(ctx context.Context, state string, images []string) (*llm.DecisionResultEntry, *llm.DecisionResponse, error) {
	m.lgCalls.Add(1)
	return m.loopGuardFn(ctx, state, images)
}

type stubFSMLLMClient struct {
	fn    func(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error)
	calls atomic.Int32
}

func (s *stubFSMLLMClient) CreateChatCompletion(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
	s.calls.Add(1)
	return s.fn(ctx, req)
}

type recordedEventRow struct {
	EventType string
	Attempt   int
	Payload   []byte
}

func listRecordedEvents(t *testing.T, db *sql.DB, runID string) []recordedEventRow {
	t.Helper()
	rows, err := db.QueryContext(context.Background(),
		`SELECT event_type, attempt, payload FROM agent_trajectory_events WHERE run_id = ? ORDER BY attempt ASC`,
		runID,
	)
	require.NoError(t, err)
	defer func() { _ = rows.Close() }()

	var out []recordedEventRow
	for rows.Next() {
		var r recordedEventRow
		require.NoError(t, rows.Scan(&r.EventType, &r.Attempt, &r.Payload))
		out = append(out, r)
	}
	require.NoError(t, rows.Err())
	return out
}

func makeCalibratedCompactionResult(choice string, pSkip, pCompact float64) (*llm.DecisionResultEntry, *llm.DecisionResponse) {
	profile := "cal_v2_4k"
	riskThresh := 0.8393841862136965
	entry := &llm.DecisionResultEntry{
		ID:              "compaction",
		ScoringMode:     "joint",
		Choice:          choice,
		ChoiceBasis:     "calibrated_argmax",
		ProbabilityKind: "calibrated_affine",
		Scores: []llm.DecisionScoreEntry{
			{ID: "skip", Logit: -0.5, Probability: pSkip},
			{ID: "compact", Logit: 0.5, Probability: pCompact},
		},
		Calibration: llm.DecisionCalibrationMeta{
			Profile:           &profile,
			Applied:           true,
			RiskThreshold5Pct: &riskThresh,
		},
		PrefixTokens: 250,
		SuffixTokens: 24,
	}
	resp := &llm.DecisionResponse{
		Protocol:  llm.SystemOneProtocolV1,
		Decisions: []llm.DecisionResultEntry{*entry},
		Latency:   18 * time.Millisecond,
	}
	return entry, resp
}

func newCompactionTestStoreAndRun(t *testing.T, runID string, iter int, stepCount int) (*sql.DB, *Store, *FSMRun) {
	t.Helper()
	db := setupTestDB(t)
	store := NewStore(db)

	ctx := context.Background()
	baseMsgs := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleSystem, Content: "You are helpful."},
		{Role: openai.ChatMessageRoleUser, Content: "Investigate test suite and summarize findings"},
	}
	steps := make([]FSMStep, 0, stepCount)
	bigOut := strings.Repeat("verbose test log output line\n", 30)
	for i := 1; i <= stepCount; i++ {
		callID := fmt.Sprintf("call_%s_%d", runID, i)
		out := fmt.Sprintf("Exit Code: 0\nPASS all packages cleanly without issues %d\n%s", i, bigOut)
		baseMsgs = append(baseMsgs,
			openai.ChatCompletionMessage{
				Role: openai.ChatMessageRoleAssistant,
				ToolCalls: []openai.ToolCall{{
					ID:       callID,
					Type:     openai.ToolTypeFunction,
					Function: openai.FunctionCall{Name: "run_command", Arguments: `{"command":"go test ./..."}`},
				}},
			},
			openai.ChatCompletionMessage{
				Role:       openai.ChatMessageRoleTool,
				ToolCallID: callID,
				Content:    out,
			},
		)
		steps = append(steps, FSMStep{
			ID:         fmt.Sprintf("%s_s_%d", runID, i),
			RunID:      runID,
			Iteration:  i,
			StepIndex:  0,
			ToolCallID: callID,
			ToolName:   "run_command",
			ArgsJSON:   `{"command":"go test ./..."}`,
			Status:     StepStatusCompleted,
			ResultJSON: out,
		})
	}
	ctxJSON, err := EncodeMessages(baseMsgs)
	require.NoError(t, err)

	run := &FSMRun{
		ID:           runID,
		FSMType:      FSMTypeToolLoop,
		ChatID:       "townhall",
		Status:       RunStatusRunning,
		CurrentState: StateLLMRequest,
		Iteration:    iter,
		ContextJSON:  ctxJSON,
	}
	require.NoError(t, store.CreateRun(ctx, run))
	require.NoError(t, store.CreateSteps(ctx, steps))
	return db, store, run
}

func TestSystemOneCompactionGate_PoliciesAndInvariants(t *testing.T) {
	ctx := context.Background()
	compCfg := CompactionConfig{
		Enabled:            true,
		MinIterations:      2,
		IntervalIterations: 2,
		TailIterations:     1,
		SoftThresholdBytes: 10, // force rubric probe on small context
		HardThresholdBytes: 1024 * 1024,
	}

	t.Run("confident skip bypasses System-2 and logs telemetry", func(t *testing.T) {
		db, store, run := newCompactionTestStoreAndRun(t, "run_s1_skip", 4, 3)
		llmStub := &stubFSMLLMClient{
			fn: func(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
				return nil, errors.New("System-2 should not be called on confident skip")
			},
		}
		s1 := &mockSystemOneDecisionClient{
			compactionFn: func(ctx context.Context, state string, maxSeqLen int) (*llm.DecisionResultEntry, *llm.DecisionResponse, error) {
				assert.Equal(t, llm.SystemOneCalibratedSeqLen, maxSeqLen)
				assert.NotContains(t, state, "Run ID:")
				e, r := makeCalibratedCompactionResult("skip", 0.91, 0.09)
				return e, r, nil
			},
		}

		runner := NewToolLoopRunner(llmStub, nil, "qwen3.6")
		runner.SetCompactionConfig(compCfg)
		runner.SetSystemOneClient(s1)
		runner.SetSystemOneConfig(SystemOneControlConfig{
			CompactionEnabled:        true,
			CompactionDeferThreshold: 0, // use risk_threshold_5pct
			FallbackOnError:          false,
		})

		require.NoError(t, runner.maybeCompact(ctx, run, store, "qwen3.6"))
		assert.Equal(t, int32(1), s1.compCalls.Load())
		assert.Equal(t, int32(0), llmStub.calls.Load())

		latest, err := store.GetLatestCompaction(ctx, run.ID)
		require.NoError(t, err)
		assert.Nil(t, latest)

		events := listRecordedEvents(t, db, run.ID)
		var sawS1, sawCompSkip bool
		for _, ev := range events {
			if ev.EventType == "system1_decision" {
				sawS1 = true
				var p trajectory.SystemOneDecisionPayload
				require.NoError(t, json.Unmarshal(ev.Payload, &p))
				assert.Equal(t, "skip", p.Choice)
				assert.Equal(t, "skip", p.AppliedAction)
				assert.True(t, p.CalibrationApplied)
				assert.False(t, p.FallbackUsed)
			}
			if ev.EventType == "compaction" {
				sawCompSkip = true
				var p trajectory.CompactionEventPayload
				require.NoError(t, json.Unmarshal(ev.Payload, &p))
				assert.False(t, p.ShouldCompact)
				assert.Equal(t, "system1_skip", p.DecisionSource)
			}
		}
		assert.True(t, sawS1)
		assert.True(t, sawCompSkip)
	})

	t.Run("uncertain probability defers to full System-2 decider", func(t *testing.T) {
		db, store, run := newCompactionTestStoreAndRun(t, "run_s1_defer", 4, 3)
		llmStub := &stubFSMLLMClient{
			fn: func(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
				assert.Contains(t, req.Messages[0].Content, "task trajectory compaction controller (SelfCompact)")
				return &openai.ChatCompletionResponse{
					Choices: []openai.ChatCompletionChoice{{
						Message: openai.ChatCompletionMessage{
							Content: `{"should_compact":false,"reason":"system2 says wait"}`,
						},
					}},
				}, nil
			},
		}
		s1 := &mockSystemOneDecisionClient{
			compactionFn: func(ctx context.Context, state string, maxSeqLen int) (*llm.DecisionResultEntry, *llm.DecisionResponse, error) {
				e, r := makeCalibratedCompactionResult("compact", 0.35, 0.65) // < 0.8394
				return e, r, nil
			},
		}

		runner := NewToolLoopRunner(llmStub, nil, "qwen3.6")
		runner.SetCompactionConfig(compCfg)
		runner.SetSystemOneClient(s1)
		runner.SetSystemOneConfig(SystemOneControlConfig{CompactionEnabled: true})

		require.NoError(t, runner.maybeCompact(ctx, run, store, "qwen3.6"))
		assert.Equal(t, int32(1), llmStub.calls.Load())

		events := listRecordedEvents(t, db, run.ID)
		var p trajectory.SystemOneDecisionPayload
		found := false
		for _, ev := range events {
			if ev.EventType == "system1_decision" {
				found = true
				require.NoError(t, json.Unmarshal(ev.Payload, &p))
			}
		}
		require.True(t, found)
		assert.Equal(t, "defer_to_system2", p.AppliedAction)
		assert.Contains(t, p.DeferReason, "0.6500")
		assert.False(t, p.FallbackUsed)
	})

	t.Run("confident compact runs Stage-2 summarize-only prompt and persists snapshot", func(t *testing.T) {
		_, store, run := newCompactionTestStoreAndRun(t, "run_s1_compact", 4, 3)
		llmStub := &stubFSMLLMClient{
			fn: func(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
				assert.Contains(t, req.Messages[0].Content, "Stage 2 of the SelfCompact trajectory summarizer")
				return &openai.ChatCompletionResponse{
					Choices: []openai.ChatCompletionChoice{{
						Message: openai.ChatCompletionMessage{
							Content: `{"should_compact":true,"reason":"verified chunk","state":{"original_objective":"goal","verified_findings":[{"summary":"tests pass"}],"current_subgoal":"done"}}`,
						},
					}},
				}, nil
			},
		}
		s1 := &mockSystemOneDecisionClient{
			compactionFn: func(ctx context.Context, state string, maxSeqLen int) (*llm.DecisionResultEntry, *llm.DecisionResponse, error) {
				e, r := makeCalibratedCompactionResult("compact", 0.05, 0.95)
				return e, r, nil
			},
		}

		runner := NewToolLoopRunner(llmStub, nil, "qwen3.6")
		runner.SetCompactionConfig(compCfg)
		runner.SetSystemOneClient(s1)
		runner.SetSystemOneConfig(SystemOneControlConfig{CompactionEnabled: true})

		require.NoError(t, runner.maybeCompact(ctx, run, store, "qwen3.6"))
		assert.Equal(t, int32(1), llmStub.calls.Load())

		latest, err := store.GetLatestCompaction(ctx, run.ID)
		require.NoError(t, err)
		require.NotNil(t, latest)
		assert.Equal(t, 1, latest.Version)
		assert.Equal(t, CompactionVerdictProbedCompact, latest.RubricVerdict)
	})

	t.Run("Stage-2 nil state NEVER triggers buildDeterministicCompactedState", func(t *testing.T) {
		db, store, run := newCompactionTestStoreAndRun(t, "run_s1_nil_stage2", 4, 3)
		llmStub := &stubFSMLLMClient{
			fn: func(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
				return &openai.ChatCompletionResponse{
					Choices: []openai.ChatCompletionChoice{{
						Message: openai.ChatCompletionMessage{
							Content: `{"should_compact":true,"reason":"missing state object","state":null}`,
						},
					}},
				}, nil
			},
		}
		s1 := &mockSystemOneDecisionClient{
			compactionFn: func(ctx context.Context, state string, maxSeqLen int) (*llm.DecisionResultEntry, *llm.DecisionResponse, error) {
				e, r := makeCalibratedCompactionResult("compact", 0.04, 0.96)
				return e, r, nil
			},
		}

		runner := NewToolLoopRunner(llmStub, nil, "qwen3.6")
		runner.SetCompactionConfig(compCfg)
		runner.SetSystemOneClient(s1)
		runner.SetSystemOneConfig(SystemOneControlConfig{
			CompactionEnabled: true,
			FallbackOnError:   true, // even with fallback_on_error=true, Stage-2 nil state must fail safely without deterministic synthesis
		})

		require.NoError(t, runner.maybeCompact(ctx, run, store, "qwen3.6"))

		latest, err := store.GetLatestCompaction(ctx, run.ID)
		require.NoError(t, err)
		assert.Nil(t, latest, "must never synthesize deterministic compacted state when Stage-2 summarizer returns nil state")

		events := listRecordedEvents(t, db, run.ID)
		var compEv trajectory.CompactionEventPayload
		var s1Ev trajectory.SystemOneDecisionPayload
		for _, ev := range events {
			switch ev.EventType {
			case "compaction":
				require.NoError(t, json.Unmarshal(ev.Payload, &compEv))
			case "system1_decision":
				require.NoError(t, json.Unmarshal(ev.Payload, &s1Ev))
			}
		}
		assert.False(t, compEv.ShouldCompact)
		assert.Contains(t, compEv.Reason, "probe_error")
		assert.Contains(t, compEv.Reason, "nil state")
		assert.Equal(t, "compact", s1Ev.Choice)
		assert.Equal(t, "probe_error", s1Ev.AppliedAction)
	})

	t.Run("size/reduction rejection finalizes applied_action as skip while preserving choice", func(t *testing.T) {
		db, store, run := newCompactionTestStoreAndRun(t, "run_s1_noshrink", 4, 3)
		llmStub := &stubFSMLLMClient{
			fn: func(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
				// Oversized/uncompacted state won't shrink small active context when verbose findings exceed active bytes.
				bigSummary := strings.Repeat("verbose finding detail ", 250)
				return &openai.ChatCompletionResponse{
					Choices: []openai.ChatCompletionChoice{{
						Message: openai.ChatCompletionMessage{
							Content: fmt.Sprintf(`{"should_compact":true,"reason":"expanded","state":{"original_objective":"goal","verified_findings":[{"summary":%q}],"current_subgoal":"next"}}`, bigSummary),
						},
					}},
				}, nil
			},
		}
		s1 := &mockSystemOneDecisionClient{
			compactionFn: func(ctx context.Context, state string, maxSeqLen int) (*llm.DecisionResultEntry, *llm.DecisionResponse, error) {
				e, r := makeCalibratedCompactionResult("compact", 0.04, 0.96)
				return e, r, nil
			},
		}

		runner := NewToolLoopRunner(llmStub, nil, "qwen3.6")
		runner.SetCompactionConfig(compCfg)
		runner.SetSystemOneClient(s1)
		runner.SetSystemOneConfig(SystemOneControlConfig{CompactionEnabled: true})

		require.NoError(t, runner.maybeCompact(ctx, run, store, "qwen3.6"))

		events := listRecordedEvents(t, db, run.ID)
		var s1Ev trajectory.SystemOneDecisionPayload
		for _, ev := range events {
			if ev.EventType == "system1_decision" {
				require.NoError(t, json.Unmarshal(ev.Payload, &s1Ev))
			}
		}
		assert.Equal(t, "compact", s1Ev.Choice)
		assert.Equal(t, "skip:compaction_did_not_shrink_context", s1Ev.AppliedAction)
	})

	t.Run("delayed failing compaction probe records non-zero latency", func(t *testing.T) {
		db, store, run := newCompactionTestStoreAndRun(t, "run_s1_delay_err", 4, 3)
		s1 := &mockSystemOneDecisionClient{
			compactionFn: func(ctx context.Context, state string, maxSeqLen int) (*llm.DecisionResultEntry, *llm.DecisionResponse, error) {
				time.Sleep(15 * time.Millisecond)
				return nil, nil, errors.New("connection reset")
			},
		}

		runner := NewToolLoopRunner(&stubFSMLLMClient{}, nil, "qwen3.6")
		runner.SetCompactionConfig(compCfg)
		runner.SetSystemOneClient(s1)
		runner.SetSystemOneConfig(SystemOneControlConfig{CompactionEnabled: true, FallbackOnError: false})

		require.NoError(t, runner.maybeCompact(ctx, run, store, "qwen3.6"))

		events := listRecordedEvents(t, db, run.ID)
		var s1Ev trajectory.SystemOneDecisionPayload
		for _, ev := range events {
			if ev.EventType == "system1_decision" {
				require.NoError(t, json.Unmarshal(ev.Payload, &s1Ev))
			}
		}
		assert.GreaterOrEqual(t, s1Ev.LatencyMs, int64(10))
		assert.Equal(t, "probe_error", s1Ev.AppliedAction)
	})

	t.Run("calibration not applied + fallback_on_error=false skips System-2 cleanly", func(t *testing.T) {
		db, store, run := newCompactionTestStoreAndRun(t, "run_s1_nocal_strict", 4, 3)
		llmStub := &stubFSMLLMClient{
			fn: func(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
				return nil, errors.New("System-2 must NOT run when fallback_on_error=false")
			},
		}
		s1 := &mockSystemOneDecisionClient{
			compactionFn: func(ctx context.Context, state string, maxSeqLen int) (*llm.DecisionResultEntry, *llm.DecisionResponse, error) {
				e, r := makeCalibratedCompactionResult("skip", 0.9, 0.1)
				e.Calibration.Applied = false
				reason := "max_seq_len_mismatch"
				e.Calibration.MismatchReason = &reason
				return e, r, nil
			},
		}

		runner := NewToolLoopRunner(llmStub, nil, "qwen3.6")
		runner.SetCompactionConfig(compCfg)
		runner.SetSystemOneClient(s1)
		runner.SetSystemOneConfig(SystemOneControlConfig{
			CompactionEnabled: true,
			FallbackOnError:   false,
		})

		require.NoError(t, runner.maybeCompact(ctx, run, store, "qwen3.6"))
		assert.Equal(t, int32(0), llmStub.calls.Load())

		events := listRecordedEvents(t, db, run.ID)
		var compEv trajectory.CompactionEventPayload
		for _, ev := range events {
			if ev.EventType == "compaction" {
				require.NoError(t, json.Unmarshal(ev.Payload, &compEv))
			}
		}
		assert.Contains(t, compEv.Reason, "calibration was not applied")
	})
}

func TestEvaluateLoopGuard_SemanticModesAndTelemetry(t *testing.T) {
	ctx := context.Background()
	run := &FSMRun{ID: "run_lg", Iteration: 6}
	cfg := DefaultLoopGuardConfig()
	callSame := openai.ToolCall{
		ID:       "call_1",
		Type:     openai.ToolTypeFunction,
		Function: openai.FunctionCall{Name: "run_command", Arguments: `{"command":"pytest -q"}`},
	}

	makeSteps := func(count int, out string) []FSMStep {
		outSlice := make([]FSMStep, count)
		for i := range count {
			outSlice[i] = FSMStep{
				ID:         fmt.Sprintf("s_%d", i),
				RunID:      run.ID,
				Iteration:  i + 1,
				ToolName:   "run_command",
				ArgsJSON:   `{"command":"pytest -q"}`,
				Status:     StepStatusCompleted,
				ResultJSON: out,
			}
		}
		return outSlice
	}

	t.Run("unexecuted blocked call cannot establish a strike", func(t *testing.T) {
		hist := []FSMStep{
			{Iteration: 1, ToolName: "run_command", ArgsJSON: `{"command":"pytest -q"}`, ErrorText: LoopGuardBlockedErrorText, Status: StepStatusSkipped},
			{Iteration: 2, ToolName: "read_file", ArgsJSON: `{"path":"a.go"}`, ResultJSON: "ok", Status: StepStatusCompleted},
		}
		s1 := &mockSystemOneDecisionClient{
			loopGuardFn: func(ctx context.Context, state string, images []string) (*llm.DecisionResultEntry, *llm.DecisionResponse, error) {
				return &llm.DecisionResultEntry{ID: "loop_guard", Choice: "force_synthesis"}, &llm.DecisionResponse{}, nil
			},
		}
		v, tel, err := EvaluateLoopGuard(ctx, run, "goal", hist, []openai.ToolCall{callSame}, cfg, s1, SystemOneControlConfig{LoopGuardMode: "hybrid"})
		require.NoError(t, err)
		require.NotNil(t, tel)
		assert.Equal(t, LoopActionNone, v.Action)
		assert.Equal(t, "advisory", tel.AppliedAction)
		assert.Equal(t, int32(1), s1.lgCalls.Load())
	})

	t.Run("len(history) < 2 skips sidecar", func(t *testing.T) {
		s1 := &mockSystemOneDecisionClient{
			loopGuardFn: func(ctx context.Context, state string, images []string) (*llm.DecisionResultEntry, *llm.DecisionResponse, error) {
				t.Fatal("sidecar must not be called with < 2 history steps")
				return nil, nil, nil
			},
		}
		v, tel, err := EvaluateLoopGuard(ctx, run, "goal", makeSteps(1, "Exit Code: 1"), []openai.ToolCall{callSame}, cfg, s1, SystemOneControlConfig{LoopGuardMode: "hybrid"})
		require.NoError(t, err)
		assert.Nil(t, tel)
		assert.Equal(t, LoopActionNone, v.Action)
	})

	t.Run("shadow mode preserves heuristic verdict and records telemetry", func(t *testing.T) {
		hist := makeSteps(2, "Exit Code: 1\nsame failure") // heuristic DetectLoop -> intercept
		s1 := &mockSystemOneDecisionClient{
			loopGuardFn: func(ctx context.Context, state string, images []string) (*llm.DecisionResultEntry, *llm.DecisionResponse, error) {
				return &llm.DecisionResultEntry{ID: "loop_guard", Choice: "none", PrefixTokens: 80, SuffixTokens: 12}, &llm.DecisionResponse{Latency: 9 * time.Millisecond}, nil
			},
		}
		v, tel, err := EvaluateLoopGuard(ctx, run, "goal", hist, []openai.ToolCall{callSame}, cfg, s1, SystemOneControlConfig{LoopGuardMode: "shadow", FallbackOnError: true})
		require.NoError(t, err)
		assert.Equal(t, LoopActionIntercept, v.Action)
		require.NotNil(t, tel)
		assert.True(t, tel.ShadowMode)
		assert.Equal(t, "none", tel.Choice)
		assert.Equal(t, "intercept", tel.AppliedAction)
		assert.Equal(t, "intercept", tel.HeuristicAction)
	})

	t.Run("hybrid none overrides heuristic for constructive non-zero iteration", func(t *testing.T) {
		hist := makeSteps(2, "Exit Code: 1\nsame test failure line")
		s1 := &mockSystemOneDecisionClient{
			loopGuardFn: func(ctx context.Context, state string, images []string) (*llm.DecisionResultEntry, *llm.DecisionResponse, error) {
				return &llm.DecisionResultEntry{ID: "loop_guard", Choice: "none"}, &llm.DecisionResponse{}, nil
			},
		}
		v, tel, err := EvaluateLoopGuard(ctx, run, "goal", hist, []openai.ToolCall{callSame}, cfg, s1, SystemOneControlConfig{LoopGuardMode: "hybrid"})
		require.NoError(t, err)
		assert.Equal(t, LoopActionNone, v.Action)
		assert.Equal(t, "none", tel.AppliedAction)
		assert.Equal(t, "intercept", tel.HeuristicAction)
	})

	t.Run("hybrid hard backstop catches >=5 consecutive unchanged calls even if sidecar says none", func(t *testing.T) {
		hist := makeSteps(4, "Exit Code: 1\nexact same output")
		s1 := &mockSystemOneDecisionClient{
			loopGuardFn: func(ctx context.Context, state string, images []string) (*llm.DecisionResultEntry, *llm.DecisionResponse, error) {
				return &llm.DecisionResultEntry{ID: "loop_guard", Choice: "none"}, &llm.DecisionResponse{}, nil
			},
		}
		v, tel, err := EvaluateLoopGuard(ctx, run, "goal", hist, []openai.ToolCall{callSame}, cfg, s1, SystemOneControlConfig{LoopGuardMode: "hybrid"})
		require.NoError(t, err)
		assert.Equal(t, LoopActionIntercept, v.Action)
		assert.Contains(t, v.Reason, "hard backstop")
		assert.Equal(t, "intercept", tel.AppliedAction)
	})

	t.Run("hybrid hard backstop checks only trailing consecutive run (ignores older different observation)", func(t *testing.T) {
		callOther := FSMStep{ID: "b", RunID: run.ID, Iteration: 2, ToolName: "read_file", ArgsJSON: `{"path":"x"}`, Status: StepStatusCompleted, ResultJSON: "b-out"}
		hist := []FSMStep{
			{ID: "old_a", RunID: run.ID, Iteration: 1, ToolName: "run_command", ArgsJSON: `{"command":"pytest -q"}`, Status: StepStatusCompleted, ResultJSON: "OLD different output"},
			callOther,
			{ID: "new_1", RunID: run.ID, Iteration: 3, ToolName: "run_command", ArgsJSON: `{"command":"pytest -q"}`, Status: StepStatusCompleted, ResultJSON: "Exit Code: 1\nsame trailing output"},
			{ID: "new_2", RunID: run.ID, Iteration: 4, ToolName: "run_command", ArgsJSON: `{"command":"pytest -q"}`, Status: StepStatusCompleted, ResultJSON: "Exit Code: 1\nsame trailing output"},
			{ID: "new_3", RunID: run.ID, Iteration: 5, ToolName: "run_command", ArgsJSON: `{"command":"pytest -q"}`, Status: StepStatusCompleted, ResultJSON: "Exit Code: 1\nsame trailing output"},
			{ID: "new_4", RunID: run.ID, Iteration: 6, ToolName: "run_command", ArgsJSON: `{"command":"pytest -q"}`, Status: StepStatusCompleted, ResultJSON: "Exit Code: 1\nsame trailing output"},
		}
		s1 := &mockSystemOneDecisionClient{
			loopGuardFn: func(ctx context.Context, state string, images []string) (*llm.DecisionResultEntry, *llm.DecisionResponse, error) {
				return &llm.DecisionResultEntry{ID: "loop_guard", Choice: "none"}, &llm.DecisionResponse{}, nil
			},
		}
		v, tel, err := EvaluateLoopGuard(ctx, run, "goal", hist, []openai.ToolCall{callSame}, cfg, s1, SystemOneControlConfig{LoopGuardMode: "hybrid"})
		require.NoError(t, err)
		assert.Equal(t, LoopActionIntercept, v.Action)
		assert.Contains(t, v.Reason, "hard backstop")
		assert.Equal(t, "intercept", tel.AppliedAction)
	})

	t.Run("hybrid downgrades force_synthesis to intercept before strike 1", func(t *testing.T) {
		hist := makeSteps(2, "out")
		s1 := &mockSystemOneDecisionClient{
			loopGuardFn: func(ctx context.Context, state string, images []string) (*llm.DecisionResultEntry, *llm.DecisionResponse, error) {
				return &llm.DecisionResultEntry{ID: "loop_guard", Choice: "force_synthesis"}, &llm.DecisionResponse{}, nil
			},
		}
		v, tel, err := EvaluateLoopGuard(ctx, run, "goal", hist, []openai.ToolCall{callSame}, cfg, s1, SystemOneControlConfig{LoopGuardMode: "hybrid"})
		require.NoError(t, err)
		assert.Equal(t, LoopActionIntercept, v.Action)
		assert.NotEmpty(t, v.NoticeByCall[0])
		assert.Equal(t, "intercept", tel.AppliedAction)

		// An unrelated blocked call is not evidence for this candidate.
		hist[0].ErrorText = LoopGuardBlockedErrorText
		hist[0].ArgsJSON = `{"command":"other"}`
		v2, tel2, err := EvaluateLoopGuard(ctx, run, "goal", hist, []openai.ToolCall{callSame}, cfg, s1, SystemOneControlConfig{LoopGuardMode: "hybrid"})
		require.NoError(t, err)
		assert.Equal(t, LoopActionNone, v2.Action)
		assert.Equal(t, "advisory", tel2.AppliedAction)

		// A blocked repeat after two identical executed observations is a valid strike.
		hist = makeSteps(2, "out")
		hist = append(hist, FSMStep{Iteration: 3, ToolName: "run_command", ArgsJSON: callSame.Function.Arguments,
			ErrorText: LoopGuardBlockedErrorText, Status: StepStatusSkipped})
		v3, tel3, err := EvaluateLoopGuard(ctx, run, "goal", hist, []openai.ToolCall{callSame}, cfg, s1, SystemOneControlConfig{LoopGuardMode: "hybrid"})
		require.NoError(t, err)
		assert.Equal(t, LoopActionForceSynthesis, v3.Action)
		if tel3 != nil {
			assert.Equal(t, "force_synthesis", tel3.AppliedAction)
		}
	})

	t.Run("sidecar error respects fallback_on_error false vs true", func(t *testing.T) {
		hist := makeSteps(2, "Exit Code: 1\nsame")
		s1Err := &mockSystemOneDecisionClient{
			loopGuardFn: func(ctx context.Context, state string, images []string) (*llm.DecisionResultEntry, *llm.DecisionResponse, error) {
				return nil, nil, errors.New("sidecar down")
			},
		}

		_, telStrict, err := EvaluateLoopGuard(ctx, run, "goal", hist, []openai.ToolCall{callSame}, cfg, s1Err, SystemOneControlConfig{LoopGuardMode: "hybrid", FallbackOnError: false})
		require.Error(t, err)
		require.NotNil(t, telStrict)
		assert.False(t, telStrict.FallbackUsed)
		assert.Equal(t, "error", telStrict.AppliedAction)

		vFallback, telFallback, err := EvaluateLoopGuard(ctx, run, "goal", hist, []openai.ToolCall{callSame}, cfg, s1Err, SystemOneControlConfig{LoopGuardMode: "hybrid", FallbackOnError: true})
		require.NoError(t, err)
		assert.Equal(t, LoopActionIntercept, vFallback.Action)
		require.NotNil(t, telFallback)
		assert.True(t, telFallback.FallbackUsed)
	})

	t.Run("shadow strict failure preserves heuristic execution and records non-zero error latency", func(t *testing.T) {
		hist := makeSteps(2, "Exit Code: 1\nsame")
		s1DelayedErr := &mockSystemOneDecisionClient{
			loopGuardFn: func(ctx context.Context, state string, images []string) (*llm.DecisionResultEntry, *llm.DecisionResponse, error) {
				time.Sleep(15 * time.Millisecond)
				return nil, nil, errors.New("timeout reached")
			},
		}

		v, tel, err := EvaluateLoopGuard(ctx, run, "goal", hist, []openai.ToolCall{callSame}, cfg, s1DelayedErr, SystemOneControlConfig{
			LoopGuardMode:   "shadow",
			FallbackOnError: false,
		})
		require.NoError(t, err, "shadow mode must not alter execution even under fallback_on_error=false")
		assert.Equal(t, LoopActionIntercept, v.Action)
		require.NotNil(t, tel)
		assert.True(t, tel.ShadowMode)
		assert.False(t, tel.FallbackUsed)
		assert.Equal(t, "intercept", tel.AppliedAction)
		assert.Contains(t, tel.Error, "timeout reached")
		assert.GreaterOrEqual(t, tel.LatencyMs, int64(10))
	})
}

func TestStore_RecordSystemOneDecision_CrashReplaySafe(t *testing.T) {
	db, store, run := newCompactionTestStoreAndRun(t, "run_replay", 3, 2)
	ctx := context.Background()

	require.NoError(t, store.RecordSystemOneDecision(ctx, run.ID, 3, "compaction", "", trajectory.SystemOneDecisionPayload{
		Task:          "compaction",
		Choice:        "skip",
		LatencyMs:     12,
		AppliedAction: "skip",
	}))
	// Simulated crash-recovery replay at exact same iteration/phase/step with different latency/outcome.
	require.NoError(t, store.RecordSystemOneDecision(ctx, run.ID, 3, "compaction", "", trajectory.SystemOneDecisionPayload{
		Task:          "compaction",
		Choice:        "skip",
		LatencyMs:     19,
		FallbackUsed:  true,
		AppliedAction: "skip",
	}))

	events := listRecordedEvents(t, db, run.ID)
	var attempts []int
	for _, ev := range events {
		if ev.EventType == "system1_decision" {
			attempts = append(attempts, ev.Attempt)
		}
	}
	assert.Equal(t, []int{1, 2}, attempts)
}

func TestToolLoop_HandlePrepareSteps_SystemOneLoopGuard(t *testing.T) {
	ctx := context.Background()
	db, store, run := newCompactionTestStoreAndRun(t, "run_prepare_lg", 3, 2)
	msgs, err := DecodeMessages(run.ContextJSON)
	require.NoError(t, err)
	msgs = append(msgs, openai.ChatCompletionMessage{
		Role: openai.ChatMessageRoleAssistant,
		ToolCalls: []openai.ToolCall{{
			ID:       "call_next",
			Type:     openai.ToolTypeFunction,
			Function: openai.FunctionCall{Name: "run_command", Arguments: `{"command":"go test ./..."}`},
		}},
	})
	updatedCtxJSON, err := EncodeMessages(msgs)
	require.NoError(t, err)
	run.ContextJSON = updatedCtxJSON
	run.CurrentState = StatePrepareSteps
	require.NoError(t, store.UpdateRun(ctx, run))

	s1Err := &mockSystemOneDecisionClient{
		loopGuardFn: func(ctx context.Context, state string, images []string) (*llm.DecisionResultEntry, *llm.DecisionResponse, error) {
			return nil, nil, errors.New("sidecar unreachable")
		},
	}

	toolset := &mockCustomToolset{
		defs: []agentapi.ToolDefinition{{
			Schema: openai.Tool{
				Type:     openai.ToolTypeFunction,
				Function: &openai.FunctionDefinition{Name: "run_command"},
			},
		}},
	}

	runner := NewToolLoopRunner(&stubFSMLLMClient{}, nil, "qwen3.6")
	runner.SetSystemOneClient(s1Err)
	runner.SetSystemOneConfig(SystemOneControlConfig{
		LoopGuardMode:   "hybrid",
		FallbackOnError: false,
	})

	err = runner.handlePrepareSteps(ctx, run, store, toolset)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "sidecar unreachable")

	events := listRecordedEvents(t, db, run.ID)
	var found bool
	for _, ev := range events {
		if ev.EventType == "system1_decision" {
			found = true
			var p trajectory.SystemOneDecisionPayload
			require.NoError(t, json.Unmarshal(ev.Payload, &p))
			assert.Equal(t, "loop_guard", p.Task)
			assert.Equal(t, "error", p.AppliedAction)
			assert.False(t, p.FallbackUsed)
		}
	}
	assert.True(t, found)
}

func TestStore_RecordSystemOneDecision_ConcurrentContention(t *testing.T) {
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "concurrent_s1.db")
	dsn := fmt.Sprintf("file:%s?_pragma=journal_mode(WAL)&_pragma=foreign_keys(ON)&_pragma=busy_timeout(5000)&_pragma=synchronous(NORMAL)", dbPath)
	db, err := sql.Open("sqlite", dsn)
	require.NoError(t, err)
	db.SetMaxOpenConns(8)
	defer func() { _ = db.Close() }()

	ctx := context.Background()
	require.NoError(t, EnsureDBSchema(ctx, db))
	store := NewStore(db)
	require.NoError(t, store.CreateRun(ctx, &FSMRun{
		ID:           "run_concurrent_s1",
		FSMType:      FSMTypeToolLoop,
		ChatID:       "townhall",
		Status:       RunStatusRunning,
		CurrentState: StatePrepareSteps,
		Iteration:    5,
		ContextJSON:  "[]",
	}))

	const workers = 12
	var wg sync.WaitGroup
	errs := make(chan error, workers)
	for i := range workers {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			errs <- store.RecordSystemOneDecision(ctx, "run_concurrent_s1", 5, "loop", "system1_loop_guard_i5", trajectory.SystemOneDecisionPayload{
				Task:          "loop_guard",
				Choice:        "none",
				LatencyMs:     int64(10 + idx),
				AppliedAction: "none",
			})
		}(i)
	}
	wg.Wait()
	close(errs)

	for err := range errs {
		require.NoError(t, err)
	}

	events := listRecordedEvents(t, db, "run_concurrent_s1")
	require.Len(t, events, workers)
	for i, ev := range events {
		assert.Equal(t, i+1, ev.Attempt)
	}
}
