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
	start := int64(100)
	finish := int64(105)
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

	// 3. Sub-second execution event (render <1s)
	same := int64(100)
	step3 := &fsm.FSMStep{
		ID:          "step-3",
		ToolName:    "web_fetch",
		ArgsJSON:    `{"url":"https://example.com"}`,
		Status:      fsm.StepStatusCompleted,
		StartedAt:   &same,
		CompletedAt: &same,
	}
	renderer.OnStepUpdate(ctx, step3)

	// 4. Missing timestamps (no duration)
	step4 := &fsm.FSMStep{
		ID:       "step-4",
		ToolName: "load_memory",
		ArgsJSON: `{"memory_id":"mem-1"}`,
		Status:   fsm.StepStatusCompleted,
	}
	renderer.OnStepUpdate(ctx, step4)

	// 5. Write final answer
	err := writer.WriteFinalAnswer("Here is the final answer.")
	require.NoError(t, err)

	out := buf.String()
	expected := "• web_search \"golang 1.25 release\" (5s)\n" +
		"✗ execute make test (exit 2, 5s)\n" +
		"• web_fetch https://example.com (<1s)\n" +
		"• load_memory mem-1\n" +
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

func TestSummarizeArgs_KnowledgeTools(t *testing.T) {
	// load_memory with memory_id
	assert.Equal(t, "mem_12345", summarizeArgs("load_memory", `{"memory_id":"mem_12345"}`))
	// load_memory fallback to key or item_id
	assert.Equal(t, "legacy_key", summarizeArgs("load_memory", `{"key":"legacy_key"}`))
	assert.Equal(t, "item_abc", summarizeArgs("load_memory", `{"item_id":"item_abc"}`))

	// load_skill with skill_id
	assert.Equal(t, "skill_deploy", summarizeArgs("load_skill", `{"skill_id":"skill_deploy"}`))
	// load_skill fallback to name or item_id
	assert.Equal(t, "deploy_app", summarizeArgs("load_skill", `{"name":"deploy_app"}`))
	assert.Equal(t, "skill_item_1", summarizeArgs("load_skill", `{"item_id":"skill_item_1"}`))

	// propose_memory
	assert.Equal(t, "User prefers Go over Python", summarizeArgs("propose_memory", `{"type":"preference","content":"User prefers Go over Python"}`))

	// propose_skill
	assert.Equal(t, "docker_build", summarizeArgs("propose_skill", `{"name":"docker_build","description":"builds docker images"}`))
}
