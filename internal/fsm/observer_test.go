package fsm

import (
	"context"
	"sync"
	"testing"
	"time"

	openai "github.com/sashabaranov/go-openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type testMockObserver struct {
	mu           sync.Mutex
	prepared     [][]FSMStep
	stepUpdates  []FSMStep
	finishedRuns []struct {
		status RunStatus
		err    error
	}
}

func (o *testMockObserver) OnStepsPrepared(ctx context.Context, steps []FSMStep) {
	o.mu.Lock()
	defer o.mu.Unlock()
	copied := make([]FSMStep, len(steps))
	copy(copied, steps)
	o.prepared = append(o.prepared, copied)
}

func (o *testMockObserver) OnStepUpdate(ctx context.Context, step *FSMStep) {
	o.mu.Lock()
	defer o.mu.Unlock()
	if step != nil {
		o.stepUpdates = append(o.stepUpdates, *step)
	}
}

func (o *testMockObserver) OnRunFinished(ctx context.Context, status RunStatus, runErr error) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.finishedRuns = append(o.finishedRuns, struct {
		status RunStatus
		err    error
	}{status: status, err: runErr})
}

func TestEngine_RunToolLoop_WithProgressObserver(t *testing.T) {
	db := setupTestDB(t)
	defer func() { _ = db.Close() }()

	store := NewStore(db)
	storeProv := NewStaticStoreProvider(store)

	invoker := ToolInvokerFunc(func(ctx context.Context, name string, argsJSON string) (string, error) {
		return `{"results": "done"}`, nil
	})

	callCount := 0
	mockLLM := &mockLLMClient{
		handler: func(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
			callCount++
			if callCount == 1 {
				return &openai.ChatCompletionResponse{
					Choices: []openai.ChatCompletionChoice{
						{
							Message: openai.ChatCompletionMessage{
								Role: openai.ChatMessageRoleAssistant,
								ToolCalls: []openai.ToolCall{
									{
										ID:   "call_1",
										Type: openai.ToolTypeFunction,
										Function: openai.FunctionCall{
											Name:      "web_search",
											Arguments: `{"query": "golang"}`,
										},
									},
								},
							},
						},
					},
				}, nil
			}
			return &openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    openai.ChatMessageRoleAssistant,
							Content: "Search completed successfully.",
						},
					},
				},
			}, nil
		},
	}

	engine := NewEngine(storeProv, mockLLM, invoker, WithDefaultModel("test-model"))

	obs := &testMockObserver{}
	req := ToolLoopRequest{
		RunID:            "run_obs_test",
		ChatID:           "chat_obs",
		UserID:           "user_1",
		IsDM:             true,
		Model:            "test-model",
		Messages:         []openai.ChatCompletionMessage{{Role: openai.ChatMessageRoleUser, Content: "Search"}},
		Tools:            []openai.Tool{{Type: openai.ToolTypeFunction, Function: &openai.FunctionDefinition{Name: "web_search"}}},
		ProgressObserver: obs,
	}

	res, err := engine.RunToolLoop(context.Background(), req)
	require.NoError(t, err)
	assert.Equal(t, "Search completed successfully.", res.Content)

	obs.mu.Lock()
	defer obs.mu.Unlock()

	// 1. OnStepsPrepared called once with 1 step
	require.Len(t, obs.prepared, 1)
	assert.Equal(t, "web_search", obs.prepared[0][0].ToolName)

	// 2. OnStepUpdate called for RUNNING and COMPLETED
	require.GreaterOrEqual(t, len(obs.stepUpdates), 2)
	assert.Equal(t, StepStatusRunning, obs.stepUpdates[0].Status)
	assert.Equal(t, StepStatusCompleted, obs.stepUpdates[1].Status)

	// 3. OnRunFinished called with RunStatusCompleted
	require.Len(t, obs.finishedRuns, 1)
	assert.Equal(t, RunStatusCompleted, obs.finishedRuns[0].status)
	assert.NoError(t, obs.finishedRuns[0].err)
}

func TestEngine_RunToolLoop_ObserverContextCancellation(t *testing.T) {
	db := setupTestDB(t)
	defer func() { _ = db.Close() }()

	store := NewStore(db)
	storeProv := NewStaticStoreProvider(store)

	invoker := ToolInvokerFunc(func(ctx context.Context, name string, argsJSON string) (string, error) {
		time.Sleep(100 * time.Millisecond)
		return `{"status": "ok"}`, nil
	})

	mockLLM := &mockLLMClient{
		handler: func(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
			return &openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role: openai.ChatMessageRoleAssistant,
							ToolCalls: []openai.ToolCall{
								{
									ID:   "call_cancel",
									Type: openai.ToolTypeFunction,
									Function: openai.FunctionCall{
										Name:      "web_search",
										Arguments: `{"query": "cancel test"}`,
									},
								},
							},
						},
					},
				},
			}, nil
		},
	}

	engine := NewEngine(storeProv, mockLLM, invoker, WithDefaultModel("test-model"))

	obs := &testMockObserver{}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()

	req := ToolLoopRequest{
		RunID:            "run_obs_cancel",
		ChatID:           "chat_cancel",
		UserID:           "user_1",
		IsDM:             true,
		Model:            "test-model",
		Messages:         []openai.ChatCompletionMessage{{Role: openai.ChatMessageRoleUser, Content: "Search"}},
		Tools:            []openai.Tool{{Type: openai.ToolTypeFunction, Function: &openai.FunctionDefinition{Name: "web_search"}}},
		ProgressObserver: obs,
	}

	_, err := engine.RunToolLoop(ctx, req)
	require.Error(t, err)

	obs.mu.Lock()
	defer obs.mu.Unlock()

	require.Len(t, obs.finishedRuns, 1)
	assert.Equal(t, RunStatusFailed, obs.finishedRuns[0].status)
	assert.Error(t, obs.finishedRuns[0].err)
}
