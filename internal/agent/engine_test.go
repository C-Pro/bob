package agent

import (
	"context"
	"errors"
	"sync"
	"testing"

	openai "github.com/sashabaranov/go-openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"bob/internal/agentapi"
	"bob/internal/fsm"
	"bob/internal/llm"
)

type mockLLMClient struct {
	mu           sync.Mutex
	plainCalls   [][]openai.ChatCompletionMessage
	loopCalls    [][]openai.ChatCompletionMessage
	plainReply   string
	plainErr     error
	loopReply    string
	loopErr      error
}

func (m *mockLLMClient) GenerateChatResponse(ctx context.Context, messages []openai.ChatCompletionMessage) (string, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.plainCalls = append(m.plainCalls, messages)
	return m.plainReply, m.plainErr
}

func (m *mockLLMClient) GenerateChatResponseWithToolLoop(
	ctx context.Context,
	messages []openai.ChatCompletionMessage,
	tools []openai.Tool,
	executor llm.ToolExecutor,
	maxIterations int,
) (string, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.loopCalls = append(m.loopCalls, messages)
	return m.loopReply, m.loopErr
}

type mockFSMEngine struct {
	mu       sync.Mutex
	requests []fsm.ToolLoopRequest
	res      *fsm.ToolLoopResult
	err      error
}

func (m *mockFSMEngine) RunToolLoop(ctx context.Context, req fsm.ToolLoopRequest) (*fsm.ToolLoopResult, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.requests = append(m.requests, req)
	if m.err != nil {
		return nil, m.err
	}
	return m.res, nil
}

type mockFrontend struct {
	mu           sync.Mutex
	bindCalls    []agentapi.RunDescriptor
	bindErr      error
	bindings     agentapi.Bindings
	deliverCalls []agentapi.Completion
	deliverErr   error
}

func (f *mockFrontend) Bind(ctx context.Context, desc agentapi.RunDescriptor) (agentapi.Bindings, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.bindCalls = append(f.bindCalls, desc)
	if f.bindErr != nil {
		return agentapi.Bindings{}, f.bindErr
	}
	return f.bindings, nil
}

func (f *mockFrontend) Deliver(ctx context.Context, comp agentapi.Completion) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.deliverCalls = append(f.deliverCalls, comp)
	return f.deliverErr
}

type mockToolset struct {
	tools []agentapi.ToolDefinition
}

func (m *mockToolset) Definitions(ctx context.Context) ([]agentapi.ToolDefinition, error) {
	return m.tools, nil
}

func (m *mockToolset) Execute(ctx context.Context, call agentapi.ToolCall) (agentapi.ToolResult, error) {
	return agentapi.ToolResult{Content: "tool executed: " + call.Name}, nil
}

func TestEngine_Run_NormalFSMExecution(t *testing.T) {
	ctx := context.Background()
	fe := &mockFrontend{
		bindings: agentapi.Bindings{},
	}
	fsmEng := &mockFSMEngine{
		res: &fsm.ToolLoopResult{
			RunID:           "run_123",
			Content:         "fsm reply",
			TotalIterations: 2,
			AgentAttachments: []agentapi.Attachment{
				{ID: "att_1", Name: "report.pdf"},
			},
			Actions: []agentapi.Action{
				{Type: agentapi.ActionSuppressReply},
			},
		},
	}

	eng, err := NewEngine(Dependencies{
		Config: Config{
			DefaultModel: "model-test",
		},
		FSMEngine: fsmEng,
		Frontends: map[string]agentapi.Frontend{
			"cli": fe,
		},
	})
	require.NoError(t, err)

	turn := agentapi.Turn{
		Run: agentapi.RunDescriptor{
			RunID: "run_123",
			Session: agentapi.SessionRef{
				FrontendID: "cli",
				SessionID:  "sess_1",
			},
			Actor: agentapi.Actor{ID: "user_1"},
		},
		SystemPrompt: "You are an assistant.",
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleUser, Content: "Hello"},
		},
	}

	res, err := eng.Run(ctx, turn)
	require.NoError(t, err)
	assert.Equal(t, "run_123", res.RunID)
	assert.Equal(t, "fsm reply", res.Content)
	assert.Equal(t, 2, res.Iterations)
	require.Len(t, res.Attachments, 1)
	assert.Equal(t, "att_1", res.Attachments[0].ID)
	require.Len(t, res.Actions, 1)

	fe.mu.Lock()
	require.Len(t, fe.bindCalls, 1)
	assert.Equal(t, "run_123", fe.bindCalls[0].RunID)
	fe.mu.Unlock()

	fsmEng.mu.Lock()
	require.Len(t, fsmEng.requests, 1)
	assert.Equal(t, "sess_1", fsmEng.requests[0].ChatID)
	assert.Equal(t, "user_1", fsmEng.requests[0].UserID)
	// System prompt prepended
	require.Len(t, fsmEng.requests[0].Messages, 2)
	assert.Equal(t, openai.ChatMessageRoleSystem, fsmEng.requests[0].Messages[0].Role)
	assert.Equal(t, "You are an assistant.", fsmEng.requests[0].Messages[0].Content)
	fsmEng.mu.Unlock()
}

func TestEngine_Run_VolatileFallbackWhenFSMEngineFails(t *testing.T) {
	ctx := context.Background()
	fe := &mockFrontend{
		bindings: agentapi.Bindings{},
	}
	fsmEng := &mockFSMEngine{
		err: errors.New("db disk I/O error"),
	}
	llmClient := &mockLLMClient{
		plainReply: "fallback reply from volatile",
	}

	eng, err := NewEngine(Dependencies{
		FSMEngine: fsmEng,
		LLMClient: llmClient,
		Frontends: map[string]agentapi.Frontend{
			"cli": fe,
		},
	})
	require.NoError(t, err)

	turn := agentapi.Turn{
		Run: agentapi.RunDescriptor{
			RunID: "run_fallback",
			Session: agentapi.SessionRef{
				FrontendID: "cli",
				SessionID:  "sess_fb",
			},
		},
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleUser, Content: "Hello"},
		},
	}

	res, err := eng.Run(ctx, turn)
	require.NoError(t, err)
	assert.Equal(t, "run_fallback", res.RunID)
	assert.Equal(t, "fallback reply from volatile", res.Content)
}

func TestEngine_Run_ContextCancelled_NoFallback(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // already cancelled

	fe := &mockFrontend{
		bindings: agentapi.Bindings{},
	}
	fsmEng := &mockFSMEngine{
		err: context.Canceled,
	}
	llmClient := &mockLLMClient{
		plainReply: "should not be called",
	}

	eng, err := NewEngine(Dependencies{
		FSMEngine: fsmEng,
		LLMClient: llmClient,
		Frontends: map[string]agentapi.Frontend{
			"cli": fe,
		},
	})
	require.NoError(t, err)

	turn := agentapi.Turn{
		Run: agentapi.RunDescriptor{
			RunID: "run_cancel",
			Session: agentapi.SessionRef{
				FrontendID: "cli",
				SessionID:  "sess_cancel",
			},
		},
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleUser, Content: "Hello"},
		},
	}

	_, err = eng.Run(ctx, turn)
	require.Error(t, err)
	assert.ErrorIs(t, err, context.Canceled)

	llmClient.mu.Lock()
	assert.Empty(t, llmClient.plainCalls)
	llmClient.mu.Unlock()
}

func TestEngine_Run_UnregisteredFrontend_Fails(t *testing.T) {
	ctx := context.Background()
	eng, err := NewEngine(Dependencies{})
	require.NoError(t, err)

	turn := agentapi.Turn{
		Run: agentapi.RunDescriptor{
			RunID: "run_unknown",
			Session: agentapi.SessionRef{
				FrontendID: "nonexistent",
				SessionID:  "s1",
			},
		},
	}

	_, err = eng.Run(ctx, turn)
	require.Error(t, err)
	assert.Contains(t, err.Error(), `unregistered frontend "nonexistent"`)
}

func TestEngine_Run_BindFailure_Fails(t *testing.T) {
	ctx := context.Background()
	fe := &mockFrontend{
		bindErr: errors.New("token expired"),
	}
	eng, err := NewEngine(Dependencies{
		Frontends: map[string]agentapi.Frontend{
			"besedka": fe,
		},
	})
	require.NoError(t, err)

	turn := agentapi.Turn{
		Run: agentapi.RunDescriptor{
			RunID: "run_bind_fail",
			Session: agentapi.SessionRef{
				FrontendID: "besedka",
				SessionID:  "s1",
			},
		},
	}

	_, err = eng.Run(ctx, turn)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "token expired")
}

func TestEngine_Run_DefenseInDepth_AssistantMessageContinuation(t *testing.T) {
	ctx := context.Background()
	fe := &mockFrontend{}
	llmClient := &mockLLMClient{
		plainReply: "continued",
	}

	eng, err := NewEngine(Dependencies{
		LLMClient: llmClient,
		Frontends: map[string]agentapi.Frontend{
			"cli": fe,
		},
	})
	require.NoError(t, err)

	origMessages := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleUser, Content: "Part 1"},
		{Role: openai.ChatMessageRoleAssistant, Content: "Response 1"},
	}

	turn := agentapi.Turn{
		Run: agentapi.RunDescriptor{
			RunID: "run_cont",
			Session: agentapi.SessionRef{
				FrontendID: "cli",
				SessionID:  "s1",
			},
		},
		Messages: origMessages,
	}

	_, err = eng.Run(ctx, turn)
	require.NoError(t, err)

	// Original messages slice must not be modified
	assert.Len(t, origMessages, 2)

	llmClient.mu.Lock()
	require.Len(t, llmClient.plainCalls, 1)
	passedMsgs := llmClient.plainCalls[0]
	require.Len(t, passedMsgs, 3)
	assert.Equal(t, openai.ChatMessageRoleUser, passedMsgs[2].Role)
	assert.Equal(t, "Please continue.", passedMsgs[2].Content)
	llmClient.mu.Unlock()
}

func TestEngine_Run_VolatileToolLoop(t *testing.T) {
	ctx := context.Background()
	tset := &mockToolset{
		tools: []agentapi.ToolDefinition{
			{Schema: openai.Tool{Type: openai.ToolTypeFunction, Function: &openai.FunctionDefinition{Name: "test_tool"}}},
		},
	}
	fe := &mockFrontend{
		bindings: agentapi.Bindings{
			Tools: tset,
		},
	}
	llmClient := &mockLLMClient{
		loopReply: "reply with tool loop",
	}

	eng, err := NewEngine(Dependencies{
		LLMClient: llmClient,
		Frontends: map[string]agentapi.Frontend{
			"cli": fe,
		},
	})
	require.NoError(t, err)

	turn := agentapi.Turn{
		Run: agentapi.RunDescriptor{
			RunID: "run_vloop",
			Session: agentapi.SessionRef{
				FrontendID: "cli",
				SessionID:  "s1",
			},
		},
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleUser, Content: "run tool"},
		},
	}

	res, err := eng.Run(ctx, turn)
	require.NoError(t, err)
	assert.Equal(t, "reply with tool loop", res.Content)

	llmClient.mu.Lock()
	assert.Len(t, llmClient.loopCalls, 1)
	llmClient.mu.Unlock()
}

func TestEngine_Close_Idempotent(t *testing.T) {
	eng, err := NewEngine(Dependencies{})
	require.NoError(t, err)

	require.NoError(t, eng.Close())
	require.NoError(t, eng.Close())

	turn := agentapi.Turn{
		Run: agentapi.RunDescriptor{
			Session: agentapi.SessionRef{FrontendID: "cli", SessionID: "s1"},
		},
	}
	_, err = eng.Run(context.Background(), turn)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "engine is closed")
}

func TestEngine_Run_DeliverSuccess(t *testing.T) {
	ctx := context.Background()
	fe := &mockFrontend{}
	llmClient := &mockLLMClient{
		plainReply: "test reply",
	}

	eng, err := NewEngine(Dependencies{
		LLMClient: llmClient,
		Frontends: map[string]agentapi.Frontend{
			"cli": fe,
		},
	})
	require.NoError(t, err)

	turn := agentapi.Turn{
		Run: agentapi.RunDescriptor{
			RunID: "run_deliver_ok",
			Session: agentapi.SessionRef{
				FrontendID: "cli",
				SessionID:  "s1",
			},
		},
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleUser, Content: "hello"},
		},
		Deliver: true,
	}

	res, err := eng.Run(ctx, turn)
	require.NoError(t, err)
	assert.Equal(t, "test reply", res.Content)

	fe.mu.Lock()
	defer fe.mu.Unlock()
	require.Len(t, fe.deliverCalls, 1)
	assert.Equal(t, "run_deliver_ok", fe.deliverCalls[0].Run.RunID)
	assert.Equal(t, "test reply", fe.deliverCalls[0].Result.Content)
	assert.Equal(t, agentapi.RunCompleted, fe.deliverCalls[0].Status)
}

func TestEngine_Run_DeliverFailure_RetainsResult(t *testing.T) {
	ctx := context.Background()
	expectedDeliverErr := errors.New("network delivery timeout")
	fe := &mockFrontend{
		deliverErr: expectedDeliverErr,
	}
	llmClient := &mockLLMClient{
		plainReply: "completed output",
	}

	eng, err := NewEngine(Dependencies{
		LLMClient: llmClient,
		Frontends: map[string]agentapi.Frontend{
			"cli": fe,
		},
	})
	require.NoError(t, err)

	turn := agentapi.Turn{
		Run: agentapi.RunDescriptor{
			RunID: "run_deliver_fail",
			Session: agentapi.SessionRef{
				FrontendID: "cli",
				SessionID:  "s1",
			},
		},
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleUser, Content: "hello"},
		},
		Deliver: true,
	}

	res, err := eng.Run(ctx, turn)
	require.Error(t, err)

	var delivErr *DeliveryError
	require.True(t, errors.As(err, &delivErr), "error should be a DeliveryError")
	assert.Equal(t, "completed output", delivErr.Result.Content)
	assert.Equal(t, expectedDeliverErr, delivErr.Unwrap())
	assert.Equal(t, "completed output", res.Content)

	fe.mu.Lock()
	defer fe.mu.Unlock()
	require.Len(t, fe.deliverCalls, 1)
}

