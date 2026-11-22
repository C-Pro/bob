package fsm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"bob/internal/agentapi"
	"bob/internal/llm"

	openai "github.com/sashabaranov/go-openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func testLoopCall(name, args string) openai.ToolCall {
	return openai.ToolCall{Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: name, Arguments: args}}
}

func testLoopStep(id, name, args, output string) FSMStep {
	return FSMStep{ID: id, ToolName: name, ArgsJSON: args, ResultJSON: output, Status: StepStatusCompleted}
}

func TestLoopGuardEvidenceScopeAndDecodedOutput(t *testing.T) {
	cfg := DefaultLoopGuardConfig()
	call := testLoopCall("run_command", `{"command":"go test ./..."}`)
	other := testLoopCall("read_file", `{"path":"other.go"}`)
	first, err := EncodeStoredToolResult(agentapi.ToolResult{Content: "same output"})
	require.NoError(t, err)
	second := `{"result":{"content":"same output"},"bob_tool_result_version":1,"extra":"envelope noise"}`
	history := []FSMStep{
		testLoopStep("s1", call.Function.Name, call.Function.Arguments, first),
		testLoopStep("s2", call.Function.Name, call.Function.Arguments, second),
	}
	verdict := DetectLoop(history, []openai.ToolCall{call, other}, cfg)
	assert.Equal(t, LoopActionIntercept, verdict.Action)
	assert.Contains(t, verdict.NoticeByCall, 0)
	assert.NotContains(t, verdict.NoticeByCall, 1)

	history[1].ResultJSON = `{"bob_tool_result_version":1,"result":{"content":"new output"}}`
	assert.Equal(t, LoopActionNone, DetectLoop(history, []openai.ToolCall{call}, cfg).Action)

	history[1].ResultJSON = second
	history = append(history, testLoopStep("edit", "write_file", `{"path":"x.go"}`, "written"))
	assert.Equal(t, LoopActionNone, DetectLoop(history, []openai.ToolCall{call}, cfg).Action)
}

func TestLoopGuardStrikeRequiresExecutedStallAndMatchingCandidate(t *testing.T) {
	cfg := DefaultLoopGuardConfig()
	a := testLoopCall("run_command", `{"command":"go test ./..."}`)
	b := testLoopCall("run_command", `{"command":"go vet ./..."}`)
	history := []FSMStep{
		testLoopStep("s1", a.Function.Name, a.Function.Arguments, "same"),
		testLoopStep("s2", a.Function.Name, a.Function.Arguments, "same"),
		{ToolName: a.Function.Name, ArgsJSON: a.Function.Arguments, Status: StepStatusSkipped,
			ErrorText: LoopGuardBlockedErrorText, ResultJSON: `{"loop_guard_blocked":true}`},
	}
	assert.Equal(t, LoopActionForceSynthesis, DetectLoop(history, []openai.ToolCall{a}, cfg).Action)
	assert.Equal(t, LoopActionIntercept, DetectLoop(history, []openai.ToolCall{a, b}, cfg).Action)
	assert.Equal(t, LoopActionNone, DetectLoop(history, []openai.ToolCall{b}, cfg).Action)

	withoutExecution := append([]FSMStep(nil), history[2])
	assert.Equal(t, LoopActionNone, DetectLoop(withoutExecution, []openai.ToolCall{a}, cfg).Action)
	history = append(history, testLoopStep("progress", "write_file", `{"path":"fixed.go"}`, "written"))
	assert.Equal(t, LoopActionNone, DetectLoop(history, []openai.ToolCall{a}, cfg).Action)
}

func TestSystemOneLoopGuardUnsupportedActionIsAdvisory(t *testing.T) {
	cfg := DefaultLoopGuardConfig()
	call := testLoopCall("run_command", `{"command":"go test ./..."}`)
	history := []FSMStep{
		testLoopStep("s1", "read_file", `{"path":"a.go"}`, "one"),
		testLoopStep("s2", "read_file", `{"path":"b.go"}`, "two"),
	}
	for _, choice := range []string{"intercept", "force_synthesis"} {
		t.Run(choice, func(t *testing.T) {
			sidecar := &mockSystemOneDecisionClient{loopGuardFn: func(context.Context, string, []string) (*llm.DecisionResultEntry, *llm.DecisionResponse, error) {
				return &llm.DecisionResultEntry{Choice: choice}, &llm.DecisionResponse{}, nil
			}}
			verdict, telemetry, err := EvaluateLoopGuard(context.Background(), &FSMRun{ID: "advisory", Iteration: 3},
				"goal", history, []openai.ToolCall{call}, cfg, sidecar, SystemOneControlConfig{LoopGuardMode: "hybrid", LoopGuardProbeAlways: true})
			require.NoError(t, err)
			assert.Equal(t, LoopActionNone, verdict.Action)
			require.NotNil(t, telemetry)
			assert.Equal(t, "advisory", telemetry.AppliedAction)
		})
	}
}

func TestToolLoopSidecarOnlyForceStillExecutesPendingTool(t *testing.T) {
	ctx := context.Background()
	_, store, run := newCompactionTestStoreAndRun(t, "sidecar_advisory_execution", 3, 2)
	call := testLoopCall("read_file", `{"path":"new.go"}`)
	call.ID = "call_new"
	messages, err := DecodeMessages(run.ContextJSON)
	require.NoError(t, err)
	messages = append(messages, openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant,
		ToolCalls: []openai.ToolCall{call}})
	run.ContextJSON, err = EncodeMessages(messages)
	require.NoError(t, err)
	run.CurrentState = StatePrepareSteps
	require.NoError(t, store.UpdateRun(ctx, run))

	sidecar := &mockSystemOneDecisionClient{loopGuardFn: func(context.Context, string, []string) (*llm.DecisionResultEntry, *llm.DecisionResponse, error) {
		return &llm.DecisionResultEntry{Choice: "force_synthesis"}, &llm.DecisionResponse{}, nil
	}}
	executed := 0
	toolset := &mockCustomToolset{defs: []agentapi.ToolDefinition{{Schema: openai.Tool{Type: openai.ToolTypeFunction,
		Function: &openai.FunctionDefinition{Name: "read_file"}}, ReadOnly: true}},
		execFn: func(context.Context, agentapi.ToolCall) (agentapi.ToolResult, error) {
			executed++
			return agentapi.ToolResult{Content: "new file contents"}, nil
		}}
	runner := NewToolLoopRunner(nil, nil, "model")
	runner.SetCompactionConfig(CompactionConfig{Enabled: false})
	runner.SetSystemOneClient(sidecar)
	runner.SetSystemOneConfig(SystemOneControlConfig{LoopGuardMode: "hybrid", LoopGuardProbeAlways: true})
	require.NoError(t, runner.handlePrepareSteps(ctx, run, store, toolset))
	assert.Equal(t, StateExecuteSteps, run.CurrentState)
	steps, err := store.ListStepsByIteration(ctx, run.ID, run.Iteration)
	require.NoError(t, err)
	require.Len(t, steps, 1)
	assert.Equal(t, StepStatusPending, steps[0].Status)
	_, err = runner.handleExecuteSteps(ctx, run, store, toolset, "model")
	require.NoError(t, err)
	assert.Equal(t, 1, executed)
	steps, err = store.ListStepsByIteration(ctx, run.ID, run.Iteration)
	require.NoError(t, err)
	assert.Equal(t, StepStatusCompleted, steps[0].Status)
}

func TestCompactedStateBareFindingsResolveOnlyWithProvenance(t *testing.T) {
	var state CompactedTaskState
	require.NoError(t, json.Unmarshal([]byte(`{"verified_findings":["unique finding","ambiguous","not observed",{"summary":"explicit","source_step_ids":["s1"]}]}`), &state))
	input := CompactionProbeInput{StepsToCompact: []FSMStep{
		testLoopStep("s1", "read_file", `{}`, "unique finding and ambiguous"),
		testLoopStep("s2", "read_file", `{}`, "ambiguous again"),
	}}
	require.NoError(t, resolveVerifiedFindingProvenance(&state, input))
	assert.Equal(t, []VerifiedFinding{
		{Summary: "unique finding", SourceStepIDs: []string{"s1"}},
		{Summary: "explicit", SourceStepIDs: []string{"s1"}},
	}, state.VerifiedFindings)

	var invalid CompactedTaskState
	require.NoError(t, json.Unmarshal([]byte(`{"verified_findings":[{"summary":"fabricated","source_step_ids":["missing"]}]}`), &invalid))
	assert.ErrorContains(t, resolveVerifiedFindingProvenance(&invalid, input), "invalid source_step_id")

	previous := &CompactedTaskState{CompactedStepIDs: []string{"old"}, VerifiedFindings: []VerifiedFinding{{Summary: "prior finding", SourceStepIDs: []string{"old"}}}}
	var carried CompactedTaskState
	require.NoError(t, json.Unmarshal([]byte(`{"verified_findings":["prior finding"]}`), &carried))
	require.NoError(t, resolveVerifiedFindingProvenance(&carried, CompactionProbeInput{PreviousState: previous}))
	assert.Equal(t, []string{"old"}, carried.VerifiedFindings[0].SourceStepIDs)
}

func TestStage2FindingProvenancePersistsOnlyVerifiedSources(t *testing.T) {
	for _, tc := range []struct {
		name      string
		findings  string
		wantError bool
	}{
		{name: "bare string resolved or dropped", findings: `["PASS all packages cleanly without issues 1","unobserved claim"]`},
		{name: "explicit nonexistent source rejected", findings: `[{"summary":"claim","source_step_ids":["fabricated"]}]`, wantError: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, store, run := newCompactionTestStoreAndRun(t, "finding_"+strings.ReplaceAll(tc.name, " ", "_"), 4, 3)
			entry, response := makeCalibratedCompactionResult("compact", 0.02, 0.98)
			sidecar := &mockSystemOneDecisionClient{compactionFn: func(context.Context, string, int) (*llm.DecisionResultEntry, *llm.DecisionResponse, error) {
				return entry, response, nil
			}}
			client := &stubFSMLLMClient{fn: func(context.Context, openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
				body := fmt.Sprintf(`{"should_compact":true,"state":{"original_objective":"goal","current_subgoal":"continue","verified_findings":%s}}`, tc.findings)
				return &openai.ChatCompletionResponse{Choices: []openai.ChatCompletionChoice{{Message: openai.ChatCompletionMessage{Content: body}}}}, nil
			}}
			runner := NewToolLoopRunner(client, nil, "model")
			runner.SetCompactionConfig(CompactionConfig{Enabled: true, MinIterations: 2, IntervalIterations: 2,
				TailIterations: 1, SoftThresholdBytes: 10, HardThresholdBytes: 1 << 20})
			runner.SetSystemOneClient(sidecar)
			runner.SetSystemOneConfig(SystemOneControlConfig{CompactionEnabled: true})
			require.NoError(t, runner.maybeCompact(context.Background(), run, store, "model"))
			state, err := store.GetLatestCompaction(context.Background(), run.ID)
			require.NoError(t, err)
			if tc.wantError {
				assert.Nil(t, state)
				return
			}
			require.NotNil(t, state)
			require.Len(t, state.VerifiedFindings, 1)
			assert.Equal(t, []string{run.ID + "_s_1"}, state.VerifiedFindings[0].SourceStepIDs)
		})
	}
}

func TestSystemOneStage2MalformedJSONRepairsOnceWithSharedDeadline(t *testing.T) {
	entry, response := makeCalibratedCompactionResult("compact", 0.02, 0.98)
	sidecar := &mockSystemOneDecisionClient{compactionFn: func(context.Context, string, int) (*llm.DecisionResultEntry, *llm.DecisionResponse, error) {
		return entry, response, nil
	}}
	calls := 0
	var firstDeadline time.Time
	client := &stubFSMLLMClient{fn: func(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
		calls++
		deadline, ok := ctx.Deadline()
		require.True(t, ok)
		if calls == 1 {
			firstDeadline = deadline
			return &openai.ChatCompletionResponse{Choices: []openai.ChatCompletionChoice{{Message: openai.ChatCompletionMessage{Content: `{"should_compact":`}}}}, nil
		}
		assert.False(t, deadline.After(firstDeadline))
		assert.Contains(t, req.Messages[0].Content, "Repair it")
		return &openai.ChatCompletionResponse{Choices: []openai.ChatCompletionChoice{{Message: openai.ChatCompletionMessage{Content: `{"should_compact":true,"state":{"current_subgoal":"continue"}}`}}}}, nil
	}}
	decider := &systemOneCompactionDecider{s1Client: sidecar, llmClient: client}
	result, err := decider.DecideCompaction(context.Background(), CompactionProbeInput{Run: &FSMRun{ID: "repair"}, Model: "model", TargetIter: 2})
	require.NoError(t, err)
	require.NotNil(t, result.State)
	assert.Equal(t, 2, calls)

	calls = 0
	client.fn = func(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
		calls++
		return &openai.ChatCompletionResponse{Choices: []openai.ChatCompletionChoice{{Message: openai.ChatCompletionMessage{Content: `invalid`}}}}, nil
	}
	_, err = decider.DecideCompaction(context.Background(), CompactionProbeInput{Run: &FSMRun{ID: "repair"}, Model: "model", TargetIter: 2})
	require.Error(t, err)
	assert.Equal(t, 2, calls)

	calls = 0
	client.fn = func(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
		calls++
		return nil, context.DeadlineExceeded
	}
	_, err = decider.DecideCompaction(context.Background(), CompactionProbeInput{Run: &FSMRun{ID: "repair"}, Model: "model", TargetIter: 2})
	require.Error(t, err)
	assert.True(t, errors.Is(err, context.DeadlineExceeded))
	assert.Equal(t, 1, calls)

	calls = 0
	cancelCtx, cancel := context.WithCancel(context.Background())
	client.fn = func(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
		calls++
		cancel()
		return &openai.ChatCompletionResponse{Choices: []openai.ChatCompletionChoice{{Message: openai.ChatCompletionMessage{Content: `invalid`}}}}, nil
	}
	_, err = decider.DecideCompaction(cancelCtx, CompactionProbeInput{Run: &FSMRun{ID: "repair"}, Model: "model", TargetIter: 2})
	require.Error(t, err)
	assert.Equal(t, 1, calls)
}

func TestCompactionProbeCooldownAndHardBytePrecedence(t *testing.T) {
	_, store, run := newCompactionTestStoreAndRun(t, "cooldown", 3, 3)
	ctx := context.Background()
	config := CompactionConfig{Enabled: true, MinIterations: 2, IntervalIterations: 3, TailIterations: 1,
		SoftThresholdBytes: 1, HardThresholdBytes: 1 << 20}
	calls := 0
	runner := NewToolLoopRunner(nil, nil, "model")
	runner.SetCompactionConfig(config)
	runner.SetCompactionDecider(CompactionDeciderFunc(func(context.Context, CompactionProbeInput) (*CompactionProbeResult, error) {
		calls++
		return &CompactionProbeResult{ShouldCompact: false}, nil
	}))
	require.NoError(t, runner.maybeCompact(ctx, run, store, "model"))
	assert.Equal(t, 1, calls)
	for _, iteration := range []int{4, 5} {
		run.Iteration = iteration
		require.NoError(t, runner.maybeCompact(ctx, run, store, "model"))
		assert.Equal(t, 1, calls)
	}
	run.Iteration = 6
	require.NoError(t, runner.maybeCompact(ctx, run, store, "model"))
	assert.Equal(t, 2, calls)
	last, err := store.GetLastCompactionProbeIteration(ctx, run.ID)
	require.NoError(t, err)
	assert.Equal(t, 6, last)

	runner.SetCompactionDecider(CompactionDeciderFunc(func(context.Context, CompactionProbeInput) (*CompactionProbeResult, error) {
		calls++
		return nil, fmt.Errorf("probe unavailable")
	}))
	run.Iteration = 9
	require.NoError(t, runner.maybeCompact(ctx, run, store, "model"))
	assert.Equal(t, 3, calls)
	run.Iteration = 10
	require.NoError(t, runner.maybeCompact(ctx, run, store, "model"))
	assert.Equal(t, 3, calls)

	stuck := []FSMStep{
		{Iteration: 1, ToolName: "test", ArgsJSON: `{}`, Status: StepStatusFailed, ErrorText: "same"},
		{Iteration: 2, ToolName: "test", ArgsJSON: `{}`, Status: StepStatusFailed, ErrorText: "same"},
	}
	config.HardThresholdBytes = 100
	config.SoftThresholdBytes = 10
	mode, _ := evaluateCompactionTrigger(config, 2, 0, 101, stuck)
	assert.Equal(t, compactionTriggerForceBackstop, mode)
}

func TestStage2CompactionUsesShortStepAliasesAndRejectsInvalidRefs(t *testing.T) {
	ctx := context.Background()
	_, store, run := newCompactionTestStoreAndRun(t, "alias_roundtrip", 5, 4)

	prevID := run.ID + "_prior_i1_s0"
	blockedID := run.ID + "_blocked_i1_s1"
	candID := run.ID + "_s_1"
	tailID := run.ID + "_s_4"

	prevState := &CompactedTaskState{
		ID:                run.ID + "_c_v1",
		Version:           1,
		ParentVersion:     0,
		RunID:             run.ID,
		CompactedUpToIter: 1,
		CompactedStepIDs:  []string{prevID},
		VerifiedFindings: []VerifiedFinding{{
			Summary:       "prior verified fact",
			SourceStepIDs: []string{prevID},
		}},
		CurrentSubgoal: "initial",
	}

	blockedStep := FSMStep{
		ID:         blockedID,
		RunID:      run.ID,
		Iteration:  2,
		StepIndex:  99,
		ToolName:   "bash",
		ArgsJSON:   `{"command":"ls"}`,
		Status:     StepStatusSkipped,
		ResultJSON: `{"loop_guard_blocked":true,"content":"blocked"}`,
	}

	input := CompactionProbeInput{
		Run:              run,
		Model:            "model",
		TargetIter:       3,
		PreviousState:    prevState,
		StepsToCompact:   []FSMStep{testLoopStep(prevID, "read_file", `{}`, "prior"), blockedStep, testLoopStep(candID, "exec", `{}`, "cand")},
		TailSteps:        []FSMStep{testLoopStep(tailID, "exec", `{}`, "tail")},
		OriginalUserGoal: "ship fix",
	}
	input.StepsToCompact[0].Status = StepStatusCompleted
	input.StepsToCompact[2].Status = StepStatusCompleted

	// S1 = prevID (carried + repeated in StepsToCompact -> reused), S2 = candID; blockedID -> ineligible; TailSteps -> TailRef=T1.
	prompt, aliases := renderFullCompactionPromptWithAliases(input)
	assert.Contains(t, prompt, "- StepID=S1 ")
	assert.Contains(t, prompt, "- StepID=S2 ")
	assert.Contains(t, prompt, "- StepID=ineligible ")
	assert.Contains(t, prompt, "- TailRef=T1 ")
	assert.NotContains(t, prompt, prevID)
	assert.NotContains(t, prompt, candID)
	assert.NotContains(t, prompt, blockedID)
	assert.NotContains(t, prompt, tailID)
	assert.NotContains(t, prompt, prevState.ID)
	// PreviousState must not be mutated in place.
	assert.Equal(t, []string{prevID}, prevState.CompactedStepIDs)
	assert.Equal(t, []string{prevID}, prevState.VerifiedFindings[0].SourceStepIDs)
	assert.Equal(t, prevID, aliases.resolve("s1"))
	assert.Equal(t, candID, aliases.resolve("S2"))

	// Full Stage-2 end-to-end test via systemOneCompactionDecider & maybeCompact.
	entry, response := makeCalibratedCompactionResult("compact", 0.02, 0.98)
	sidecar := &mockSystemOneDecisionClient{compactionFn: func(context.Context, string, int) (*llm.DecisionResultEntry, *llm.DecisionResponse, error) {
		return entry, response, nil
	}}

	var seenPrompt string
	client := &stubFSMLLMClient{fn: func(_ context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
		seenPrompt = req.Messages[1].Content
		body := `{"should_compact":true,"reason":"milestone","state":{"original_objective":"goal","current_subgoal":"finish","verified_findings":[{"summary":"f1","source_step_ids":["s1","S1"]},{"summary":"f2","source_step_ids":["S2"]}]}}`
		return &openai.ChatCompletionResponse{Choices: []openai.ChatCompletionChoice{{Message: openai.ChatCompletionMessage{Content: body}}}}, nil
	}}
	runner := NewToolLoopRunner(client, nil, "model")
	runner.SetCompactionConfig(CompactionConfig{Enabled: true, MinIterations: 2, IntervalIterations: 2,
		TailIterations: 1, SoftThresholdBytes: 10, HardThresholdBytes: 1 << 20})
	runner.SetSystemOneClient(sidecar)
	runner.SetSystemOneConfig(SystemOneControlConfig{CompactionEnabled: true})

	require.NoError(t, runner.maybeCompact(ctx, run, store, "model"))
	assert.Contains(t, seenPrompt, "- StepID=S1 ")
	assert.NotContains(t, seenPrompt, run.ID+"_s_1")

	saved, err := store.GetLatestCompaction(ctx, run.ID)
	require.NoError(t, err)
	require.NotNil(t, saved)
	require.Len(t, saved.VerifiedFindings, 2)
	assert.Equal(t, []string{run.ID + "_s_1"}, saved.VerifiedFindings[0].SourceStepIDs)
	assert.Equal(t, []string{run.ID + "_s_2"}, saved.VerifiedFindings[1].SourceStepIDs)

	for _, badRef := range []string{"S999", "T1", "ineligible", run.ID + "_c_v1"} {
		t.Run("rejects_"+badRef, func(t *testing.T) {
			_, badStore, badRun := newCompactionTestStoreAndRun(t, "badref_"+badRef, 4, 3)
			badClient := &stubFSMLLMClient{fn: func(context.Context, openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
				body := fmt.Sprintf(`{"should_compact":true,"state":{"current_subgoal":"x","verified_findings":[{"summary":"claim","source_step_ids":[%q]}]}}`, badRef)
				return &openai.ChatCompletionResponse{Choices: []openai.ChatCompletionChoice{{Message: openai.ChatCompletionMessage{Content: body}}}}, nil
			}}
			badRunner := NewToolLoopRunner(badClient, nil, "model")
			badRunner.SetCompactionConfig(CompactionConfig{Enabled: true, MinIterations: 2, IntervalIterations: 2,
				TailIterations: 1, SoftThresholdBytes: 10, HardThresholdBytes: 1 << 20})
			badRunner.SetSystemOneClient(sidecar)
			badRunner.SetSystemOneConfig(SystemOneControlConfig{CompactionEnabled: true})
			require.NoError(t, badRunner.maybeCompact(ctx, badRun, badStore, "model"))
			latest, err := badStore.GetLatestCompaction(ctx, badRun.ID)
			require.NoError(t, err)
			assert.Nil(t, latest)
		})
	}
}
