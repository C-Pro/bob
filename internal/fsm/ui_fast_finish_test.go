package fsm

import (
	"context"
	"encoding/json"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"bob/internal/agentapi"
	"bob/internal/models"
	"bob/internal/sandbox"
	"bob/internal/tools"

	openai "github.com/sashabaranov/go-openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewStepFromToolCall_AskUserMaxAttemptsOne(t *testing.T) {
	askStep := NewStepFromToolCall("run_1", 1, 0, openai.ToolCall{
		ID:       "call_ask",
		Function: openai.FunctionCall{Name: tools.AskUserToolName, Arguments: `{}`},
	}, ExecutionModeSequential)
	assert.Equal(t, 1, askStep.MaxAttempts)

	searchStep := NewStepFromToolCall("run_1", 1, 1, openai.ToolCall{
		ID:       "call_search",
		Function: openai.FunctionCall{Name: "web_search", Arguments: `{}`},
	}, ExecutionModeParallel)
	assert.Equal(t, 3, searchStep.MaxAttempts)
}

func TestToolLoop_AskUserFastFinish_SingleLLMCall(t *testing.T) {
	db := setupTestDB(t)
	store := NewStore(db)
	storeProvider := NewStaticStoreProvider(store)
	ctx := context.Background()

	var llmCalls int32
	llm := &mockLLMClient{
		handler: func(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
			n := atomic.AddInt32(&llmCalls, 1)
			if n > 1 {
				return nil, errors.New("unexpected second LLM call after ask_user fast-finish")
			}
			return &openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role: openai.ChatMessageRoleAssistant,
							ToolCalls: []openai.ToolCall{
								{
									ID:   "call_ask_1",
									Type: openai.ToolTypeFunction,
									Function: openai.FunctionCall{
										Name: tools.AskUserToolName,
										Arguments: `{
											"message": "Which region would you like to deploy to?",
											"elements": [
												{"id": "us", "type": "button", "label": "US", "style": "primary"},
												{"id": "eu", "type": "button", "label": "EU"}
											]
										}`,
									},
								},
							},
						},
					},
				},
			}, nil
		},
	}

	reg := tools.NewRegistry(nil, nil)
	sess := tools.NewChatSessionContext("townhall", "alice", false)
	sess.SupportsInteractiveUI = true

	eng := NewEngine(storeProvider, llm, nil, WithDefaultModel("test-model"))
	res, err := eng.RunToolLoop(ctx, ToolLoopRequest{
		ChatID:        "townhall",
		UserID:        "alice",
		IsDM:          false,
		Model:         "test-model",
		Messages:      []openai.ChatCompletionMessage{{Role: openai.ChatMessageRoleUser, Content: "Deploy app"}},
		Toolset:       reg.SessionToolset(sess),
		MaxIterations: 10,
	})
	require.NoError(t, err)
	assert.Equal(t, int32(1), atomic.LoadInt32(&llmCalls))
	assert.Equal(t, "Which region would you like to deploy to?", res.Content)

	require.Len(t, res.Actions, 1)
	assert.Equal(t, agentapi.ActionInteractivePrompt, res.Actions[0].Type)
	var payload agentapi.InteractivePromptPayload
	require.NoError(t, json.Unmarshal(res.Actions[0].Payload, &payload))
	assert.Equal(t, "Which region would you like to deploy to?", payload.Message)
	require.Len(t, payload.Elements, 2)
}

func TestToolLoop_AskUserFailureAllowsRecoveryWithoutFastFinish(t *testing.T) {
	db := setupTestDB(t)
	store := NewStore(db)
	storeProvider := NewStaticStoreProvider(store)
	ctx := context.Background()

	var llmCalls int32
	llm := &mockLLMClient{
		handler: func(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
			n := atomic.AddInt32(&llmCalls, 1)
			if n == 1 {
				return &openai.ChatCompletionResponse{
					Choices: []openai.ChatCompletionChoice{
						{
							Message: openai.ChatCompletionMessage{
								Role: openai.ChatMessageRoleAssistant,
								ToolCalls: []openai.ToolCall{
									{
										ID:   "call_bad_ask",
										Type: openai.ToolTypeFunction,
										Function: openai.FunctionCall{
											Name:      tools.AskUserToolName,
											Arguments: `{"message":"bad","elements":[{"id":"bob:sandbox:approve:sbx-1","type":"button","label":"Approve"}]}`,
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
							Content: "Please type your preferred option.",
						},
					},
				},
			}, nil
		},
	}

	reg := tools.NewRegistry(nil, nil)
	sess := tools.NewChatSessionContext("townhall", "alice", false)
	sess.SupportsInteractiveUI = true

	eng := NewEngine(storeProvider, llm, nil, WithDefaultModel("test-model"))
	res, err := eng.RunToolLoop(ctx, ToolLoopRequest{
		ChatID:        "townhall",
		UserID:        "alice",
		IsDM:          false,
		Model:         "test-model",
		Messages:      []openai.ChatCompletionMessage{{Role: openai.ChatMessageRoleUser, Content: "Help"}},
		Toolset:       reg.SessionToolset(sess),
		MaxIterations: 10,
	})
	require.NoError(t, err)
	assert.Equal(t, int32(2), atomic.LoadInt32(&llmCalls))
	assert.Equal(t, "Please type your preferred option.", res.Content)
	assert.Empty(t, res.Actions)

	steps, err := store.ListStepsByRun(ctx, res.RunID)
	require.NoError(t, err)
	require.Len(t, steps, 1)
	assert.Equal(t, StepStatusFailed, steps[0].Status)
	assert.Equal(t, 1, steps[0].Attempt, "failed ask_user must not retry beyond 1 attempt")
}

type noopSandboxDriver struct{}

func (d *noopSandboxDriver) Type() sandbox.DriverType          { return sandbox.DriverBwrap }
func (d *noopSandboxDriver) Available(_ context.Context) bool  { return true }
func (d *noopSandboxDriver) Create(_ context.Context, _ *sandbox.UserSandbox, _ string) error {
	return nil
}
func (d *noopSandboxDriver) Exec(_ context.Context, _ *sandbox.UserSandbox, _ []string, _ time.Duration) (*sandbox.ExecResult, error) {
	return &sandbox.ExecResult{ExitCode: 0}, nil
}
func (d *noopSandboxDriver) Destroy(_ context.Context, _ *sandbox.UserSandbox) error { return nil }

func TestToolLoop_MultipleAskUserAndSandboxSuppressOrders(t *testing.T) {
	db := setupTestDB(t)
	store := NewStore(db)
	storeProvider := NewStaticStoreProvider(store)
	ctx := context.Background()

	// 1. Multiple ask_user calls in one batch preserve both actions and fast-finish on first message
	var calls int32
	llm := &mockLLMClient{
		handler: func(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
			if atomic.AddInt32(&calls, 1) > 1 {
				return nil, errors.New("unexpected 2nd LLM call")
			}
			return &openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{{
					Message: openai.ChatCompletionMessage{
						Role: openai.ChatMessageRoleAssistant,
						ToolCalls: []openai.ToolCall{
							{
								ID:   "call_q1",
								Type: openai.ToolTypeFunction,
								Function: openai.FunctionCall{
									Name:      tools.AskUserToolName,
									Arguments: `{"message":"Which region?","elements":[{"id":"us","type":"button","label":"US"}]}`,
								},
							},
							{
								ID:   "call_q2",
								Type: openai.ToolTypeFunction,
								Function: openai.FunctionCall{
									Name:      tools.AskUserToolName,
									Arguments: `{"message":"Which format?","elements":[{"id":"json","type":"button","label":"JSON"}]}`,
								},
							},
						},
					},
				}},
			}, nil
		},
	}

	reg := tools.NewRegistry(nil, nil)
	sess := tools.NewChatSessionContext("townhall", "alice", false)
	sess.SupportsInteractiveUI = true

	eng := NewEngine(storeProvider, llm, nil, WithDefaultModel("test-model"))
	res, err := eng.RunToolLoop(ctx, ToolLoopRequest{
		ChatID:        "townhall",
		UserID:        "alice",
		IsDM:          false,
		Model:         "test-model",
		Messages:      []openai.ChatCompletionMessage{{Role: openai.ChatMessageRoleUser, Content: "Ask both"}},
		Toolset:       reg.SessionToolset(sess),
		MaxIterations: 10,
	})
	require.NoError(t, err)
	assert.Equal(t, int32(1), atomic.LoadInt32(&calls))
	assert.Equal(t, "Which region?", res.Content)
	require.Len(t, res.Actions, 2)
	assert.Equal(t, "Which region?", res.Actions[0].Content)
	assert.Equal(t, "Which format?", res.Actions[1].Content)

	// 2. Both tool orders (ask_user + sandbox_request and sandbox_request + ask_user) preserve both actions
	for _, tc := range []struct {
		name  string
		first bool
	}{
		{name: "ask_user_then_sandbox_request", first: true},
		{name: "sandbox_request_then_ask_user", first: false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			sbxMgr := sandbox.NewManager(sandbox.Config{
				DataDir: t.TempDir(),
				Enabled: true,
				Drivers: []string{"bwrap"},
			}, []sandbox.Driver{&noopSandboxDriver{}})
			defer func() { _ = sbxMgr.Close() }()

			askTC := openai.ToolCall{
				ID:   "call_ask",
				Type: openai.ToolTypeFunction,
				Function: openai.FunctionCall{
					Name:      tools.AskUserToolName,
					Arguments: `{"message":"Select framework while sandbox waits:","elements":[{"id":"go","type":"button","label":"Go"}]}`,
				},
			}
			sbxTC := openai.ToolCall{
				ID:   "call_sbx",
				Type: openai.ToolTypeFunction,
				Function: openai.FunctionCall{
					Name:      "sandbox_request",
					Arguments: `{"driver":"bwrap","network":"none","reason":"Run tests"}`,
				},
			}
			batch := []openai.ToolCall{askTC, sbxTC}
			if !tc.first {
				batch = []openai.ToolCall{sbxTC, askTC}
			}

			var orderCalls int32
			orderLLM := &mockLLMClient{
				handler: func(_ context.Context, _ openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
					if atomic.AddInt32(&orderCalls, 1) > 1 {
						return nil, errors.New("unexpected 2nd LLM call")
					}
					return &openai.ChatCompletionResponse{
						Choices: []openai.ChatCompletionChoice{{
							Message: openai.ChatCompletionMessage{
								Role:      openai.ChatMessageRoleAssistant,
								ToolCalls: batch,
							},
						}},
					}, nil
				},
			}

			dmReg := tools.NewRegistry(nil, nil, sbxMgr)
			var sbxCreated bool
			dmSess := tools.NewChatSessionContext("dm_alice", "alice", true)
			dmSess.SupportsInteractiveUI = true
			dmSess.SandboxRequestCreated = &sbxCreated
			dmSess.UINotifier = func(_ string, _ string, _ *models.UIData) error { return nil }

			orderEng := NewEngine(storeProvider, orderLLM, nil, WithDefaultModel("test-model"))
			orderRes, oErr := orderEng.RunToolLoop(ctx, ToolLoopRequest{
				ChatID:        "dm_alice",
				UserID:        "alice",
				IsDM:          true,
				Model:         "test-model",
				Messages:      []openai.ChatCompletionMessage{{Role: openai.ChatMessageRoleUser, Content: "Do both"}},
				Toolset:       dmReg.SessionToolset(dmSess),
				MaxIterations: 10,
			})
			require.NoError(t, oErr)
			assert.Equal(t, int32(1), atomic.LoadInt32(&orderCalls))
			assert.True(t, sbxCreated)

			var hasPrompt, hasSuppress bool
			for _, a := range orderRes.Actions {
				if a.Type == agentapi.ActionInteractivePrompt && a.Content == "Select framework while sandbox waits:" {
					hasPrompt = true
				}
				if a.Type == agentapi.ActionSuppressReply {
					hasSuppress = true
				}
			}
			assert.True(t, hasPrompt, "expected ActionInteractivePrompt to be preserved")
			assert.True(t, hasSuppress, "expected ActionSuppressReply to be preserved")
		})
	}
}
