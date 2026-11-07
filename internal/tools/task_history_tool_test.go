package tools

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"
	"unicode/utf8"

	openai "github.com/sashabaranov/go-openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type mockTaskHistoryStore struct {
	stepsByChat map[string][]TaskHistoryStep
}

func (m *mockTaskHistoryStore) ListTaskHistorySteps(_ context.Context, chatID, userID string, isDM bool, runID string) ([]TaskHistoryStep, error) {
	key := fmt.Sprintf("%s|%s|%t", chatID, userID, isDM)
	steps, ok := m.stepsByChat[key]
	if !ok {
		return nil, fmt.Errorf("no steps found for session %s", key)
	}
	var out []TaskHistoryStep
	for _, s := range steps {
		if s.RunID == runID {
			out = append(out, s)
		}
	}
	return out, nil
}

type mockTaskHistoryProvider struct {
	store TaskHistoryStore
}

func (p *mockTaskHistoryProvider) GetTaskHistoryStore(_ context.Context, _ string, _ bool) (TaskHistoryStore, error) {
	return p.store, nil
}

func TestRecallTaskHistory_LookupByIterationStepIDToolCallIDAndQuery(t *testing.T) {
	store := &mockTaskHistoryStore{
		stepsByChat: map[string][]TaskHistoryStep{
			"dm_alice|alice|true": {
				{
					ID:         "step_r1_1_0",
					RunID:      "run_1",
					Iteration:  1,
					StepIndex:  0,
					ToolName:   "sandbox_exec",
					ToolCallID: "call_1",
					ArgsJSON:   `{"command":"pytest -q"}`,
					Status:     "COMPLETED",
					Content:    "3 passed, 1 failed: AssertionError in test_auth.py",
				},
				{
					ID:         "step_r1_2_0",
					RunID:      "run_1",
					Iteration:  2,
					StepIndex:  0,
					ToolName:   "web_fetch",
					ToolCallID: "call_2",
					ArgsJSON:   `{"url":"https://example.com/docs"}`,
					Status:     "FAILED",
					ErrorText:  "HTTP 503 Service Unavailable",
				},
				{
					ID:         "step_r2_1_0",
					RunID:      "run_2_other",
					Iteration:  1,
					StepIndex:  0,
					ToolName:   "sandbox_exec",
					ToolCallID: "call_other",
					ArgsJSON:   `{"command":"cat /etc/shadow"}`,
					Status:     "COMPLETED",
					Content:    "SENSITIVE_OTHER_RUN_DATA",
				},
			},
		},
	}

	reg := NewRegistry(nil, nil)
	reg.SetTaskHistoryProvider(&mockTaskHistoryProvider{store: store})

	sess := NewChatSessionContext("dm_alice", "alice", true)
	sess.FSMRunID = "run_1"
	ctx := WithChatSession(context.Background(), sess)

	// 1. Verify tool definition is included when FSMRunID is set
	defs := reg.ToolDefinitionsForSession(sess)
	var foundDef bool
	for _, d := range defs {
		if d.Function != nil && d.Function.Name == RecallTaskHistoryToolName {
			foundDef = true
			break
		}
	}
	assert.True(t, foundDef, "recall_task_history must be advertised when FSMRunID and provider are set")

	// 2. Lookup by iteration
	rawIter1, err := reg.Execute(ctx, RecallTaskHistoryToolName, `{"iteration":1}`)
	require.NoError(t, err)
	var respIter1 struct {
		RunID        string                `json:"run_id"`
		MatchedCount int                   `json:"matched_count"`
		Steps        []TaskHistoryStepView `json:"steps"`
	}
	require.NoError(t, json.Unmarshal([]byte(rawIter1), &respIter1))
	assert.Equal(t, "run_1", respIter1.RunID)
	require.Len(t, respIter1.Steps, 1)
	assert.Equal(t, "step_r1_1_0", respIter1.Steps[0].ID)
	assert.Contains(t, respIter1.Steps[0].Output, "AssertionError in test_auth.py")

	// 3. Lookup by step_id
	rawStep2, err := reg.Execute(ctx, RecallTaskHistoryToolName, `{"step_id":"step_r1_2_0"}`)
	require.NoError(t, err)
	var respStep2 struct {
		Steps []TaskHistoryStepView `json:"steps"`
	}
	require.NoError(t, json.Unmarshal([]byte(rawStep2), &respStep2))
	require.Len(t, respStep2.Steps, 1)
	assert.Equal(t, "call_2", respStep2.Steps[0].ToolCallID)
	assert.Equal(t, "HTTP 503 Service Unavailable", respStep2.Steps[0].ErrorText)

	// 4. Lookup by tool_call_id
	rawCall1, err := reg.Execute(ctx, RecallTaskHistoryToolName, `{"tool_call_id":"call_1"}`)
	require.NoError(t, err)
	var respCall1 struct {
		Steps []TaskHistoryStepView `json:"steps"`
	}
	require.NoError(t, json.Unmarshal([]byte(rawCall1), &respCall1))
	require.Len(t, respCall1.Steps, 1)
	assert.Equal(t, "step_r1_1_0", respCall1.Steps[0].ID)

	// 5. Lookup by keyword query (case-insensitive)
	rawQuery, err := reg.Execute(ctx, RecallTaskHistoryToolName, `{"query":"assertionerror"}`)
	require.NoError(t, err)
	var respQuery struct {
		Steps []TaskHistoryStepView `json:"steps"`
	}
	require.NoError(t, json.Unmarshal([]byte(rawQuery), &respQuery))
	require.Len(t, respQuery.Steps, 1)
	assert.Equal(t, "step_r1_1_0", respQuery.Steps[0].ID)
}

func TestRecallTaskHistory_ByteOffsetPaginationAndUTF8Safety(t *testing.T) {
	// Construct a ~40 KiB output containing multi-byte UTF-8 characters across chunk boundaries
	unit := "Line-🔍-αβγ-中文-0123456789\n"
	largeOutput := strings.Repeat(unit, 1500) // ~45 KiB
	require.Greater(t, len(largeOutput), 2*DefaultTaskHistoryMaxBytes)

	store := &mockTaskHistoryStore{
		stepsByChat: map[string][]TaskHistoryStep{
			"townhall||false": {
				{
					ID:         "step_big_1",
					RunID:      "run_big",
					Iteration:  1,
					StepIndex:  0,
					ToolName:   "web_fetch",
					ToolCallID: "call_big",
					ArgsJSON:   `{"url":"https://example.com/large"}`,
					Status:     "COMPLETED",
					Content:    largeOutput,
				},
			},
		},
	}

	var reconstructed strings.Builder
	offset := 0
	pages := 0
	for {
		pages++
		require.LessOrEqual(t, pages, 10, "pagination should terminate in a few pages")

		argsJSON := fmt.Sprintf(`{"step_id":"step_big_1","offset_bytes":%d,"max_bytes":16384}`, offset)
		rawResp, err := ExecuteRecallTaskHistoryWithStore(context.Background(), store, "townhall", "", false, "run_big", argsJSON)
		require.NoError(t, err)

		var parsed struct {
			Steps []TaskHistoryStepView `json:"steps"`
		}
		assert.LessOrEqual(t, len(rawResp), MaxTaskHistoryMaxBytes, "serialized response must never exceed 16 KiB context ceiling")
		require.NoError(t, json.Unmarshal([]byte(rawResp), &parsed))
		require.Len(t, parsed.Steps, 1)

		view := parsed.Steps[0]
		assert.Equal(t, len(largeOutput), view.TotalOutputBytes)
		assert.True(t, utf8.ValidString(view.Output), "every paginated slice must be valid UTF-8")
		reconstructed.WriteString(view.Output)

		if !view.HasMore {
			assert.Nil(t, view.NextOffsetBytes)
			break
		}
		require.NotNil(t, view.NextOffsetBytes)
		assert.Greater(t, *view.NextOffsetBytes, offset)
		offset = *view.NextOffsetBytes
	}

	assert.GreaterOrEqual(t, pages, 3)
	assert.Equal(t, largeOutput, reconstructed.String(), "concatenated pages must losslessly match full raw output")
}

func TestRecallTaskHistory_CrossRunAndCrossChatIsolation(t *testing.T) {
	store := &mockTaskHistoryStore{
		stepsByChat: map[string][]TaskHistoryStep{
			"dm_alice|alice|true": {
				{
					ID:         "step_alice_r1",
					RunID:      "run_alice_1",
					Iteration:  1,
					ToolName:   "sandbox_exec",
					ToolCallID: "call_alice_1",
					Content:    "ALICE_RUN_1_OUTPUT",
				},
				{
					ID:         "step_alice_r2",
					RunID:      "run_alice_2",
					Iteration:  1,
					ToolName:   "sandbox_exec",
					ToolCallID: "call_alice_2",
					Content:    "ALICE_RUN_2_SECRET",
				},
			},
			"dm_bob|bob|true": {
				{
					ID:         "step_bob_r1",
					RunID:      "run_bob_1",
					Iteration:  1,
					ToolName:   "sandbox_exec",
					ToolCallID: "call_bob_1",
					Content:    "BOB_SECRET_OUTPUT",
				},
			},
		},
	}

	reg := NewRegistry(nil, nil)
	reg.SetTaskHistoryProvider(&mockTaskHistoryProvider{store: store})
	reg.SetDMAuthorizer(func(_ context.Context, chatID, userID string) error {
		if chatID == "dm_"+userID {
			return nil
		}
		return errors.New("chat not owned by user")
	})

	aliceSess := NewChatSessionContext("dm_alice", "alice", true)
	aliceSess.FSMRunID = "run_alice_1"
	aliceCtx := WithChatSession(context.Background(), aliceSess)

	// 1. Attempting to query step_id from another run of the same user returns 0 results
	rawCrossRun, err := reg.Execute(aliceCtx, RecallTaskHistoryToolName, `{"step_id":"step_alice_r2"}`)
	require.NoError(t, err)
	assert.NotContains(t, rawCrossRun, "ALICE_RUN_2_SECRET")
	assert.Contains(t, rawCrossRun, `"matched_count":0`)

	// 2. Attempting to pass run_id or chat_id in arguments is rejected
	_, err = reg.Execute(aliceCtx, RecallTaskHistoryToolName, `{"run_id":"run_alice_2"}`)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "unknown field")

	// 3. Attempting to spoof another user's DM chat is blocked by DMAuthorizer
	spoofSess := NewChatSessionContext("dm_bob", "alice", true)
	spoofSess.FSMRunID = "run_bob_1"
	spoofCtx := WithChatSession(context.Background(), spoofSess)
	_, err = reg.Execute(spoofCtx, RecallTaskHistoryToolName, `{}`)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "unauthorized DM session")

	// 4. Calling outside an active FSM run (empty FSMRunID) fails closed
	noRunSess := NewChatSessionContext("dm_alice", "alice", true)
	noRunCtx := WithChatSession(context.Background(), noRunSess)
	_, err = reg.Execute(noRunCtx, RecallTaskHistoryToolName, `{}`)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "only available within an active FSM task run")
}

func TestRecallTaskHistory_ReadOnlyAndProgressFormatting(t *testing.T) {
	assert.True(t, IsReadOnlyTool(RecallTaskHistoryToolName))

	title, desc := FormatProgressStep(openai.ToolCall{
		Function: openai.FunctionCall{
			Name:      RecallTaskHistoryToolName,
			Arguments: `{"step_id":"step_123"}`,
		},
	})
	assert.Equal(t, "Recall Task History", title)
	assert.Equal(t, "Inspecting step: step_123", desc)
}

func TestRecallTaskHistory_SmallBudgetsHeavilyEscapedAndFieldPagination(t *testing.T) {
	// 1. Budgets 1..3 are rejected; budget 4 progresses rune-by-rune on 4-byte UTF-8 emojis ("😀🎉🚀")
	emojiStore := &mockTaskHistoryStore{
		stepsByChat: map[string][]TaskHistoryStep{
			"townhall||false": {
				{
					ID:         "step_emoji",
					RunID:      "run_emoji",
					Iteration:  1,
					ToolName:   "t1",
					ToolCallID: "c1",
					ArgsJSON:   `{}`,
					Status:     "COMPLETED",
					Content:    "😀🎉🚀",
				},
			},
		},
	}

	for _, invalidMax := range []int{1, 2, 3, -1} {
		_, err := ExecuteRecallTaskHistoryWithStore(
			context.Background(),
			emojiStore,
			"townhall", "", false, "run_emoji",
			fmt.Sprintf(`{"step_id":"step_emoji","max_bytes":%d}`, invalidMax),
		)
		require.Error(t, err, "max_bytes=%d must be rejected", invalidMax)
		assert.Contains(t, err.Error(), "max_bytes must be at least 4 bytes")
	}

	var emojiRebuilt strings.Builder
	offset := 0
	for i := 0; i < 3; i++ {
		raw, err := ExecuteRecallTaskHistoryWithStore(
			context.Background(),
			emojiStore,
			"townhall", "", false, "run_emoji",
			fmt.Sprintf(`{"step_id":"step_emoji","offset_bytes":%d,"max_bytes":4}`, offset),
		)
		require.NoError(t, err)
		var parsed struct {
			Steps []TaskHistoryStepView `json:"steps"`
		}
		require.NoError(t, json.Unmarshal([]byte(raw), &parsed))
		require.Len(t, parsed.Steps, 1)
		assert.True(t, utf8.ValidString(parsed.Steps[0].Output))
		assert.Equal(t, 4, parsed.Steps[0].ReturnedBytes)
		emojiRebuilt.WriteString(parsed.Steps[0].Output)
		if !parsed.Steps[0].HasMore {
			break
		}
		require.NotNil(t, parsed.Steps[0].NextOffsetBytes)
		assert.Greater(t, *parsed.Steps[0].NextOffsetBytes, offset)
		offset = *parsed.Steps[0].NextOffsetBytes
	}
	assert.Equal(t, "😀🎉🚀", emojiRebuilt.String())

	// 2. Heavily JSON-escaped content, >16 KiB ArgsJSON, >16 KiB ErrorText, and >16 KiB ToolCallID stay <= MaxTaskHistoryMaxBytes (16 KiB) and paginate losslessly
	escapedOutput := strings.Repeat("\n\t\"\\<>", 4000) // ~24 KiB raw, >60 KiB when JSON-escaped
	hugeArgs := `{"script":"` + strings.Repeat("echo hello_world; ", 1200) + `"}`
	hugeError := "Stderr failure:\n" + strings.Repeat("stackframe_line_error_trace_0123456789\n", 600)
	hugeCallID := "call_" + strings.Repeat("C", 20000)
	require.Greater(t, len(hugeArgs), MaxTaskHistoryMaxBytes)
	require.Greater(t, len(hugeError), MaxTaskHistoryMaxBytes)
	require.Greater(t, len(hugeCallID), MaxTaskHistoryMaxBytes)

	heavyStore := &mockTaskHistoryStore{
		stepsByChat: map[string][]TaskHistoryStep{
			"townhall||false": {
				{
					ID:         "step_heavy",
					RunID:      "run_heavy",
					Iteration:  1,
					ToolName:   "sandbox_exec",
					ToolCallID: hugeCallID,
					ArgsJSON:   hugeArgs,
					Status:     "FAILED",
					Content:    escapedOutput,
					ErrorText:  hugeError,
				},
			},
		},
	}

	// Paginate heavily escaped output and verify every JSON response is <= 16 KiB even with huge ToolCallID, ArgsJSON, and ErrorText
	var rebuiltEscaped strings.Builder
	offOut := 0
	for pages := 0; pages < 15; pages++ {
		raw, err := ExecuteRecallTaskHistoryWithStore(
			context.Background(),
			heavyStore,
			"townhall", "", false, "run_heavy",
			fmt.Sprintf(`{"step_id":"step_heavy","offset_bytes":%d}`, offOut),
		)
		require.NoError(t, err)
		assert.LessOrEqual(t, len(raw), MaxTaskHistoryMaxBytes, "serialized response with heavy escaping and huge metadata/args/error must fit in 16 KiB")

		var parsed struct {
			Steps []TaskHistoryStepView `json:"steps"`
		}
		require.NoError(t, json.Unmarshal([]byte(raw), &parsed))
		require.Len(t, parsed.Steps, 1)
		assert.True(t, parsed.Steps[0].ArgsTruncated, "inline args preview must indicate truncation when args > 512B")
		assert.True(t, parsed.Steps[0].ErrorTruncated, "inline error preview must indicate truncation when error > 512B")
		assert.Equal(t, len(hugeArgs), parsed.Steps[0].TotalArgsBytes)
		assert.Equal(t, len(hugeError), parsed.Steps[0].TotalErrorBytes)
		rebuiltEscaped.WriteString(parsed.Steps[0].Output)
		if !parsed.Steps[0].HasMore {
			break
		}
		require.NotNil(t, parsed.Steps[0].NextOffsetBytes)
		offOut = *parsed.Steps[0].NextOffsetBytes
	}
	assert.Equal(t, escapedOutput, rebuiltEscaped.String())

	// Paginate hugeArgs via field="args"
	var rebuiltArgs strings.Builder
	offArgs := 0
	for pages := 0; pages < 10; pages++ {
		raw, err := ExecuteRecallTaskHistoryWithStore(
			context.Background(),
			heavyStore,
			"townhall", "", false, "run_heavy",
			fmt.Sprintf(`{"step_id":"step_heavy","field":"args","offset_bytes":%d}`, offArgs),
		)
		require.NoError(t, err)
		assert.LessOrEqual(t, len(raw), MaxTaskHistoryMaxBytes)

		var parsed struct {
			Steps []TaskHistoryStepView `json:"steps"`
		}
		require.NoError(t, json.Unmarshal([]byte(raw), &parsed))
		require.Len(t, parsed.Steps, 1)
		rebuiltArgs.WriteString(parsed.Steps[0].ArgsJSON)
		if !parsed.Steps[0].HasMore {
			break
		}
		require.NotNil(t, parsed.Steps[0].NextOffsetBytes)
		offArgs = *parsed.Steps[0].NextOffsetBytes
	}
	assert.Equal(t, hugeArgs, rebuiltArgs.String())

	// Paginate hugeError via field="error"
	var rebuiltError strings.Builder
	offErr := 0
	for pages := 0; pages < 10; pages++ {
		raw, err := ExecuteRecallTaskHistoryWithStore(
			context.Background(),
			heavyStore,
			"townhall", "", false, "run_heavy",
			fmt.Sprintf(`{"step_id":"step_heavy","field":"error","offset_bytes":%d}`, offErr),
		)
		require.NoError(t, err)
		assert.LessOrEqual(t, len(raw), MaxTaskHistoryMaxBytes)

		var parsed struct {
			Steps []TaskHistoryStepView `json:"steps"`
		}
		require.NoError(t, json.Unmarshal([]byte(raw), &parsed))
		require.Len(t, parsed.Steps, 1)
		rebuiltError.WriteString(parsed.Steps[0].ErrorText)
		if !parsed.Steps[0].HasMore {
			break
		}
		require.NotNil(t, parsed.Steps[0].NextOffsetBytes)
		offErr = *parsed.Steps[0].NextOffsetBytes
	}
	assert.Equal(t, hugeError, rebuiltError.String())
}

func TestRecallTaskHistory_StepOffsetPaginationOver25Steps(t *testing.T) {
	var allSteps []TaskHistoryStep
	for i := 0; i < 25; i++ {
		allSteps = append(allSteps, TaskHistoryStep{
			ID:         fmt.Sprintf("step_many_%02d", i),
			RunID:      "run_many",
			Iteration:  1,
			StepIndex:  i,
			ToolName:   "web_search",
			ToolCallID: fmt.Sprintf("call_%02d", i),
			ArgsJSON:   fmt.Sprintf(`{"q":"item_%02d"}`, i),
			Status:     "COMPLETED",
			Content:    fmt.Sprintf("Result for item_%02d", i),
		})
	}

	store := &mockTaskHistoryStore{
		stepsByChat: map[string][]TaskHistoryStep{
			"townhall||false": allSteps,
		},
	}

	var seenIDs []string
	stepOffset := 0
	for pages := 0; pages < 10; pages++ {
		raw, err := ExecuteRecallTaskHistoryWithStore(
			context.Background(),
			store,
			"townhall", "", false, "run_many",
			fmt.Sprintf(`{"iteration":1,"step_offset":%d}`, stepOffset),
		)
		require.NoError(t, err)
		assert.LessOrEqual(t, len(raw), MaxTaskHistoryMaxBytes)

		var parsed struct {
			MatchedCount   int                   `json:"matched_count"`
			StepOffset     int                   `json:"step_offset"`
			ReturnedCount  int                   `json:"returned_count"`
			HasMoreSteps   bool                  `json:"has_more_steps"`
			NextStepOffset *int                  `json:"next_step_offset"`
			Steps          []TaskHistoryStepView `json:"steps"`
		}
		require.NoError(t, json.Unmarshal([]byte(raw), &parsed))
		assert.Equal(t, 25, parsed.MatchedCount)
		assert.Equal(t, stepOffset, parsed.StepOffset)
		require.NotEmpty(t, parsed.Steps)

		for _, s := range parsed.Steps {
			seenIDs = append(seenIDs, s.ID)
		}

		if !parsed.HasMoreSteps {
			assert.Nil(t, parsed.NextStepOffset)
			break
		}
		require.NotNil(t, parsed.NextStepOffset)
		assert.Greater(t, *parsed.NextStepOffset, stepOffset)
		stepOffset = *parsed.NextStepOffset
	}

	require.Len(t, seenIDs, 25)
	assert.Equal(t, "step_many_00", seenIDs[0])
	assert.Equal(t, "step_many_24", seenIDs[24])
}
