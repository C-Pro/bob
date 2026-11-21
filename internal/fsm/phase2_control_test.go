package fsm

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"bob/internal/llm"
	"bob/internal/trajectory"

	openai "github.com/sashabaranov/go-openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLoopGuardEventTriggersAndProgressLedger(t *testing.T) {
	base := []FSMStep{
		testLoopStep("a", "read_file", `{"path":"a.go"}`, "alpha"),
		testLoopStep("b", "read_file", `{"path":"b.go"}`, "beta"),
	}
	probe := func(iter int, history []FSMStep, candidate openai.ToolCall, cfg SystemOneControlConfig) string {
		return loopGuardProbeTrigger(&FSMRun{Iteration: iter}, history, []openai.ToolCall{candidate},
			LoopVerdict{Action: LoopActionNone}, loopHistoryEvidence{}, cfg)
	}
	novel := testLoopCall("read_file", `{"path":"c.go"}`)
	assert.Empty(t, probe(3, base, novel, SystemOneControlConfig{}))
	sidecar := &mockSystemOneDecisionClient{loopGuardFn: func(context.Context, string, []string) (*llm.DecisionResultEntry, *llm.DecisionResponse, error) {
		t.Fatal("clean novel progress must not call the sidecar")
		return nil, nil, nil
	}}
	verdict, telemetry, err := EvaluateLoopGuard(context.Background(), &FSMRun{ID: "novel", Iteration: 3}, "goal",
		base, []openai.ToolCall{novel}, DefaultLoopGuardConfig(), sidecar, SystemOneControlConfig{LoopGuardMode: "hybrid"})
	require.NoError(t, err)
	assert.Equal(t, LoopActionNone, verdict.Action)
	assert.Nil(t, telemetry)
	assert.Equal(t, int32(0), sidecar.lgCalls.Load())
	assert.Equal(t, "target_revisit", probe(3, base, testLoopCall("read_file", `{"path":"a.go"}`), SystemOneControlConfig{}))
	assert.Equal(t, "periodic_audit", probe(4, base, novel, SystemOneControlConfig{}))
	assert.Equal(t, "forced", probe(3, base, novel, SystemOneControlConfig{LoopGuardProbeAlways: true}))
	assert.Equal(t, "periodic_audit", probe(5, base, novel, SystemOneControlConfig{LoopGuardAuditInterval: 5}))

	command := append(append([]FSMStep(nil), base...), testLoopStep("c", "run_command", `{"command":"go test ./..."}`, "pass"))
	assert.Equal(t, "target_revisit", probe(3, command, testLoopCall("run_command", `{"command":"go test ./internal/fsm"}`), SystemOneControlConfig{}))

	repeated := []FSMStep{
		testLoopStep("a", "read_file", `{"path":"a.go"}`, "same"),
		testLoopStep("b", "read_file", `{"path":"b.go"}`, "same"),
	}
	assert.Equal(t, "repeated_observation", probe(3, repeated, novel, SystemOneControlConfig{}))
	failed := []FSMStep{testLoopStep("f", "run_command", `{"command":"go test ./..."}`, "Exit Code: 1\nfailure")}
	assert.Equal(t, "recent_failure", probe(3, failed, novel, SystemOneControlConfig{}))
	assert.Equal(t, 1, buildLoopProgressLedger(failed).consecutiveFailures)
	progress := append(failed, testLoopStep("edit", "write_file", `{"path":"a.go"}`, "written"))
	progress = append(progress, testLoopStep("read", "read_file", `{"path":"b.go"}`, "new output"))
	assert.Empty(t, probe(3, progress, novel, SystemOneControlConfig{}))
	assert.Equal(t, 1, buildLoopProgressLedger(progress).stateMutatingEdits)
}

func TestLoopGuardExactStateDecisionCache(t *testing.T) {
	ctx := context.Background()
	history := []FSMStep{
		testLoopStep("a", "read_file", `{"path":"a.go"}`, "alpha"),
		testLoopStep("b", "read_file", `{"path":"b.go"}`, "beta"),
	}
	history[0].Iteration = 1
	history[1].Iteration = 2
	candidate := testLoopCall("read_file", `{"path":"c.go"}`)
	sidecar := &mockSystemOneDecisionClient{model: "phase2-cache-model", loopGuardFn: func(context.Context, string, []string) (*llm.DecisionResultEntry, *llm.DecisionResponse, error) {
		return &llm.DecisionResultEntry{Choice: "none"}, &llm.DecisionResponse{}, nil
	}}
	cfg := SystemOneControlConfig{LoopGuardMode: "hybrid", LoopGuardProbeAlways: true}
	call := func(goal string, control SystemOneControlConfig) *trajectory.SystemOneDecisionPayload {
		verdict, telemetry, err := EvaluateLoopGuard(ctx, &FSMRun{ID: "cache", Iteration: 3}, goal,
			history, []openai.ToolCall{candidate}, DefaultLoopGuardConfig(), sidecar, control)
		require.NoError(t, err)
		assert.Equal(t, LoopActionNone, verdict.Action)
		require.NotNil(t, telemetry)
		return telemetry
	}
	first := call("goal", cfg)
	assert.False(t, first.CacheHit)
	assert.Equal(t, 2, first.RetainedSteps)
	assert.True(t, first.EssentialEvidencePresent)
	second := call("goal", cfg)
	assert.True(t, second.CacheHit)
	assert.Equal(t, int32(1), sidecar.lgCalls.Load())
	call("changed goal", cfg)
	assert.Equal(t, int32(2), sidecar.lgCalls.Load())
	call("goal", SystemOneControlConfig{LoopGuardMode: "shadow", LoopGuardProbeAlways: true})
	assert.Equal(t, int32(3), sidecar.lgCalls.Load())
	sidecar.model = "phase2-cache-model-2"
	call("goal", cfg)
	assert.Equal(t, int32(4), sidecar.lgCalls.Load())
}

func TestSystemOneRendererReservesNewestEvidenceAndValidPreviousJSON(t *testing.T) {
	previous := &CompactedTaskState{Version: 1, CompactedUpToIter: 0,
		OriginalObjective: "goal", CurrentSubgoal: strings.Repeat("long state text ", 1200),
		VerifiedFindings: []VerifiedFinding{{Summary: strings.Repeat("finding", 1500), SourceStepIDs: []string{"secret_step"}}}}
	steps := make([]FSMStep, 100)
	for i := range steps {
		output := fmt.Sprintf("START_%d ", i+1) + strings.Repeat("x", 2300) + fmt.Sprintf(" END_ERROR_%d", i+1)
		steps[i] = FSMStep{Iteration: i + 1, ToolName: "run_command", ArgsJSON: `{"command":"test"}`,
			ResultJSON: output, Status: StepStatusCompleted}
	}
	tail := []FSMStep{{Iteration: 101, ToolName: "tail_read", Status: StepStatusCompleted}}
	candidates := []openai.ToolCall{testLoopCall("candidate_tool", `{"path":"important.go"}`)}
	rendered := renderSystemOneProbeState(100, 101, "goal", previous, steps, tail, candidates)
	assert.True(t, rendered.Truncated)
	assert.LessOrEqual(t, len(rendered.Text), maxProbePromptBytes)
	assert.Contains(t, rendered.Text, "- Iter=1 Tool=run_command")
	assert.Contains(t, rendered.Text, "- Iter=100 Tool=run_command")
	assert.Contains(t, rendered.Text, "END_ERROR_100")
	assert.Contains(t, rendered.Text, "...[middle truncated for probe]...")
	assert.Contains(t, rendered.Text, "intermediate candidate steps omitted for probe budget")
	assert.Contains(t, rendered.Text, "- Iter=101 Tool=tail_read")
	assert.Contains(t, rendered.Text, "- Tool=candidate_tool Args=")
	assert.Greater(t, rendered.OmittedSteps, 0)
	assert.Equal(t, []int{1, 100}, rendered.RetainedStepRange)
	assert.True(t, rendered.EssentialEvidencePresent)

	start := strings.Index(rendered.Text, "Previous Compacted State (v1, up to iter 0):\n")
	require.NotEqual(t, -1, start)
	jsonText := strings.SplitN(rendered.Text[start:], "\n", 3)[1]
	var parsed map[string]any
	require.NoError(t, json.Unmarshal([]byte(jsonText), &parsed))
	assert.NotContains(t, jsonText, "secret_step")
	assert.LessOrEqual(t, len(jsonText), maxProbePrevStateBytes)

	telemetry := buildSystemOneDecisionTelemetry(llm.SystemOneTaskCompaction, nil, nil, 0, rendered)
	assert.Equal(t, rendered.RetainedSteps, telemetry.RetainedSteps)
	assert.Equal(t, rendered.OmittedSteps, telemetry.OmittedSteps)
	assert.Equal(t, rendered.RetainedStepRange, telemetry.RetainedStepRange)
	assert.True(t, telemetry.EssentialEvidencePresent)
}
