package cli

import (
	"bytes"
	"context"
	"sync"
	"testing"

	"bob/internal/agentapi"
	"bob/internal/fsm"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestProgressRenderer_StepsAndAnswer(t *testing.T) {
	var buf bytes.Buffer
	writer := NewOutputWriter(&buf)
	renderer := NewProgressRenderer(writer, false)

	ctx := context.Background()

	// 1. Tool execution event (success)
	start := int64(1000)
	finish := int64(1250)
	step1 := &fsm.FSMStep{
		ID:          "step-1",
		ToolName:    "web_search",
		ArgsJSON:    `{"query":"golang 1.25 release"}`,
		Status:      fsm.StepStatusCompleted,
		StartedAt:   &start,
		CompletedAt: &finish,
	}
	renderer.OnStepUpdate(ctx, step1)

	// 2. Execute tool event with non-zero exit code
	step2 := &fsm.FSMStep{
		ID:          "step-2",
		ToolName:    "execute",
		ArgsJSON:    `{"command":"make test"}`,
		Status:      fsm.StepStatusCompleted,
		ResultJSON:  "Exit Code: 2\nDuration: 50ms\n[stderr]\ncompile error",
		StartedAt:   &start,
		CompletedAt: &finish,
	}
	renderer.OnStepUpdate(ctx, step2)

	// 3. Write final answer
	err := writer.WriteFinalAnswer("Here is the final answer.")
	require.NoError(t, err)

	out := buf.String()
	expected := "• web_search \"golang 1.25 release\" (250ms)\n" +
		"✗ execute make test (exit 2, 250ms)\n" +
		"\n" +
		"Here is the final answer.\n"

	assert.Equal(t, expected, out)
}

func TestProgressRenderer_Quiet(t *testing.T) {
	var buf bytes.Buffer
	writer := NewOutputWriter(&buf)
	renderer := NewProgressRenderer(writer, true) // quiet = true

	ctx := context.Background()
	step := &fsm.FSMStep{
		ID:       "step-1",
		ToolName: "web_search",
		ArgsJSON: `{"query":"secret"}`,
		Status:   fsm.StepStatusCompleted,
	}
	renderer.OnStepUpdate(ctx, step)

	err := writer.WriteFinalAnswer("Quiet answer only.")
	require.NoError(t, err)

	assert.Equal(t, "Quiet answer only.\n", buf.String(), "quiet mode must output only the answer without leading blank line")
}

func TestProgressRenderer_AgentapiObserver(t *testing.T) {
	var buf bytes.Buffer
	writer := NewOutputWriter(&buf)
	renderer := NewProgressRenderer(writer, false)

	ctx := context.Background()
	ev := agentapi.ProgressEvent{
		RunID: "run-1",
		Type:  agentapi.ProgressStepUpdated,
		Step: &agentapi.StepProgress{
			ID: "step-10",
			Call: agentapi.ToolCall{
				Name:      "web_fetch",
				Arguments: `{"url":"https://example.com"}`,
			},
			Status: agentapi.StepCompleted,
		},
	}
	renderer.Observe(ctx, ev)

	err := writer.WriteFinalAnswer("Fetched page summary.")
	require.NoError(t, err)

	out := buf.String()
	assert.Contains(t, out, "• web_fetch https://example.com")
	assert.Contains(t, out, "\n\nFetched page summary.\n")
}

func TestProgressRenderer_Concurrency(t *testing.T) {
	var buf bytes.Buffer
	writer := NewOutputWriter(&buf)
	renderer := NewProgressRenderer(writer, false)

	ctx := context.Background()
	var wg sync.WaitGroup

	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			step := &fsm.FSMStep{
				ID:       string(rune('A' + idx)),
				ToolName: "web_search",
				ArgsJSON: `{"query":"concurrent"}`,
				Status:   fsm.StepStatusCompleted,
			}
			renderer.OnStepUpdate(ctx, step)
		}(i)
	}

	wg.Wait()
	err := writer.WriteFinalAnswer("Done.")
	require.NoError(t, err)

	lines := bytes.Split(buf.Bytes(), []byte("\n"))
	// 20 steps + 1 blank line + 1 answer + 1 trailing empty
	assert.Equal(t, 23, len(lines))
}
