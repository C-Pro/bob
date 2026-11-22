package fsm

import (
	"fmt"
	"strings"
	"testing"

	"bob/internal/agentapi"

	openai "github.com/sashabaranov/go-openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRenderSystemOneCompactionState_SanitizedGoldenParity(t *testing.T) {
	env1, err := EncodeStoredToolResult(agentapi.ToolResult{
		Content: "Line 1: alpha <tag> & café",
	})
	require.NoError(t, err)
	env2, err := EncodeStoredToolResult(agentapi.ToolResult{
		Content: "",
	})
	require.NoError(t, err)

	prev := &CompactedTaskState{
		ID:                "run_123_c_v1",
		Version:           1,
		ParentVersion:     0,
		RunID:             "run_123",
		CompactedUpToIter: 1,
		CompactedStepIDs:  []string{"step_1"},
		OriginalObjective: "Inspect `main.go` & fix UTF-8 bug",
		ImmutableConstraints: []string{
			"No breaking API changes",
		},
		VerifiedFindings: []VerifiedFinding{
			{Summary: "Found bug in parser", SourceStepIDs: []string{"step_1"}},
		},
		CurrentSubgoal: "Verify fix with tests",
		RubricVerdict:  CompactionVerdictProbedCompact,
		Model:          "qwen3.6",
		CreatedAt:      1700000000,
	}

	input := CompactionProbeInput{
		Run:              &FSMRun{ID: "run_123", Iteration: 4},
		TargetIter:       3,
		PreviousState:    prev,
		OriginalUserGoal: "ignored when previous has OriginalObjective",
		StepsToCompact: []FSMStep{
			{
				ID:         "step_2",
				Iteration:  2,
				ToolName:   "run_command",
				Status:     StepStatusCompleted,
				ArgsJSON:   `{ "zeta": 1,  "alpha": "<ok>&тест" }`,
				ResultJSON: env1,
			},
			{
				ID:         "step_3",
				Iteration:  3,
				ToolName:   "read_file",
				Status:     StepStatusFailed,
				ArgsJSON:   `{"b":2,"a":1}`,
				ResultJSON: env2,
				ErrorText:  "permission denied",
			},
		},
		TailSteps: []FSMStep{
			{ID: "step_4", Iteration: 4, ToolName: "edit_file", Status: StepStatusCompleted},
		},
	}

	rendered := RenderSystemOneCompactionState(input)
	assert.False(t, rendered.Truncated)

	expectedPrevJSON := `{"current_subgoal":"Verify fix with tests","immutable_constraints":["No breaking API changes"],"original_objective":"Inspect ` + "`main.go`" + ` & fix UTF-8 bug","verified_findings":[{"summary":"Found bug in parser"}]}`
	expected := strings.Join([]string{
		"Candidate Compaction Boundary: Iterations 1..3 (Current Run Iteration: 4)",
		"Original Objective: Inspect `main.go` & fix UTF-8 bug",
		"",
		"Previous Compacted State (v1, up to iter 1):",
		expectedPrevJSON,
		"",
		"Candidate Steps to Compact:",
		`- Iter=2 Tool=run_command Status=COMPLETED Args={"alpha":"<ok>&тест","zeta":1}`,
		"  Output: Line 1: alpha <tag> & café",
		`- Iter=3 Tool=read_file Status=FAILED Args={"a":1,"b":2}`,
		"  Output: permission denied",
		"",
		"Retained Raw Tail Steps (not compacted in this pass):",
		"- Iter=4 Tool=edit_file Status=COMPLETED",
		"",
	}, "\n")

	assert.Equal(t, expected, rendered.Text)
	assert.NotContains(t, rendered.Text, "Run ID:")
	assert.NotContains(t, rendered.Text, "StepID=")
	assert.NotContains(t, rendered.Text, "source_step_ids")
	assert.NotContains(t, rendered.Text, "rubric_verdict")
	assert.NotContains(t, rendered.Text, `\u003c`)
	assert.NotContains(t, rendered.Text, `\u0026`)
}

func TestRenderSystemOneLoopGuardState_WindowAndCandidates(t *testing.T) {
	history := make([]FSMStep, 15)
	for i := range history {
		history[i] = FSMStep{
			ID:         fmt.Sprintf("step_%d", i+1),
			Iteration:  i + 1,
			ToolName:   "grep",
			Status:     StepStatusCompleted,
			ArgsJSON:   fmt.Sprintf(`{ "n": %d, "a": "x" }`, i+1),
			ResultJSON: fmt.Sprintf(`out-%d`, i+1),
		}
	}
	candidates := []openai.ToolCall{
		{
			ID:   "call_1",
			Type: openai.ToolTypeFunction,
			Function: openai.FunctionCall{
				Name:      "grep",
				Arguments: `{ "z": true, "a": "<val>" }`,
			},
		},
	}

	rendered := RenderSystemOneLoopGuardState(
		&FSMRun{ID: "run_x", Iteration: 16},
		"Find foo",
		history,
		candidates,
	)

	assert.False(t, rendered.Truncated)
	assert.Contains(t, rendered.Text, "Candidate Compaction Boundary: Iterations 1..15 (Current Run Iteration: 16)\n")
	// Trailing window of 12 steps keeps iterations 4..15 and excludes 1..3.
	assert.NotContains(t, rendered.Text, "- Iter=3 ")
	assert.Contains(t, rendered.Text, "- Iter=4 Tool=grep Status=COMPLETED Args={\"a\":\"x\",\"n\":4}\n")
	assert.Contains(t, rendered.Text, "- Iter=15 Tool=grep Status=COMPLETED Args={\"a\":\"x\",\"n\":15}\n")
	// Candidate args preserve raw function argument spelling verbatim per build_jev_dataset.py:299.
	assert.Contains(t, rendered.Text, "\nCandidate Tool Calls (not yet executed):\n- Tool=grep Args={ \"z\": true, \"a\": \"<val>\" }\n")
	assert.NotContains(t, rendered.Text, "Previous Compacted State")
	assert.NotContains(t, rendered.Text, "Run ID:")
}

func TestRenderSystemOneState_PythonExactGoldenParity(t *testing.T) {
	env1, err := EncodeStoredToolResult(agentapi.ToolResult{Content: "res"})
	require.NoError(t, err)

	prev := &CompactedTaskState{
		Version:           1,
		CompactedUpToIter: 1,
		OriginalObjective: "Obj <1>",
		VerifiedFindings: []VerifiedFinding{
			{Summary: "ok \u2028 line", SourceStepIDs: []string{"s1"}},
		},
		CurrentSubgoal: "next",
	}
	steps := []FSMStep{
		{
			Iteration:  2,
			ToolName:   "run_command",
			Status:     StepStatusCompleted,
			ArgsJSON:   `{ "z": 1e0, "a": "<tag>` + "\u2029" + `" }`,
			ResultJSON: env1,
		},
		{
			Iteration: 3,
			ToolName:  "bash",
			Status:    StepStatusFailed,
			ArgsJSON:  `{"x":1} trailing`,
			ErrorText: "bad",
		},
	}
	cands := []openai.ToolCall{
		{
			ID:   "call_1",
			Type: openai.ToolTypeFunction,
			Function: openai.FunctionCall{
				Name:      "grep",
				Arguments: ` { "z": true,  "a": "raw" } `,
			},
		},
	}

	rendered := renderSystemOneProbeState(3, 4, "Obj <1>", prev, steps, nil, cands)
	assert.False(t, rendered.Truncated)

	// Generated directly by eval/harness/build_jev_dataset.py render_state.
	pythonGolden := "Candidate Compaction Boundary: Iterations 1..3 (Current Run Iteration: 4)\n" +
		"Original Objective: Obj <1>\n\n" +
		"Previous Compacted State (v1, up to iter 1):\n" +
		"{\"current_subgoal\":\"next\",\"original_objective\":\"Obj <1>\",\"verified_findings\":[{\"summary\":\"ok \u2028 line\"}]}\n\n" +
		"Candidate Steps to Compact:\n" +
		"- Iter=2 Tool=run_command Status=COMPLETED Args={\"a\":\"<tag>\u2029\",\"z\":1.0}\n" +
		"  Output: res\n" +
		"- Iter=3 Tool=bash Status=FAILED Args={\"x\":1} trailing\n" +
		"  Output: bad\n\n" +
		"Candidate Tool Calls (not yet executed):\n" +
		"- Tool=grep Args= { \"z\": true,  \"a\": \"raw\" } \n"

	assert.Equal(t, pythonGolden, rendered.Text)
}

func TestRenderSystemOneState_UTF8Truncation(t *testing.T) {
	// Cyrillic string > 512 bytes must truncate cleanly on UTF-8 boundary with matching suffix.
	cyrArgs := `{"q":"` + strings.Repeat("тест", 100) + `"}`
	rendered := RenderSystemOneCompactionState(CompactionProbeInput{
		Run:              &FSMRun{Iteration: 2},
		TargetIter:       1,
		OriginalUserGoal: strings.Repeat("Ж", 1500),
		StepsToCompact: []FSMStep{
			{Iteration: 1, ToolName: "bash", Status: StepStatusCompleted, ArgsJSON: cyrArgs, ResultJSON: "ok"},
		},
	})
	assert.True(t, rendered.Truncated)
	assert.Contains(t, rendered.Text, "...[truncated]\n")
	assert.Contains(t, rendered.Text, "...[truncated for probe]\n")
}
