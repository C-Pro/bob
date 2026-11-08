package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"unicode/utf8"

	openai "github.com/sashabaranov/go-openai"
)

const (
	// RecallTaskHistoryToolName is the function name for the task history recovery tool.
	RecallTaskHistoryToolName = "recall_task_history"
	// MinTaskHistoryMaxBytes is the minimum allowed byte budget for content slicing (1 UTF-8 code point = up to 4 bytes).
	MinTaskHistoryMaxBytes = 4
	// DefaultTaskHistoryMaxBytes is the default byte budget returned per recall_task_history call (16 KiB).
	DefaultTaskHistoryMaxBytes = 16 * 1024
	// MaxTaskHistoryMaxBytes is the maximum serialized JSON response size returned per recall_task_history call (16 KiB, matching MaxToolResultSizeInContext).
	MaxTaskHistoryMaxBytes = 16 * 1024
	// MaxTaskHistoryMatchedSteps caps the number of matched steps returned in a single page.
	MaxTaskHistoryMatchedSteps = 20
	// MaxInlinePreviewBytes caps non-targeted fields (e.g. args_json or error_text when field="output") so they cannot overflow the 16 KiB response envelope.
	MaxInlinePreviewBytes = 512
)

// TaskHistoryStep represents a single durable FSM tool step and its raw output.
type TaskHistoryStep struct {
	ID          string `json:"id"`
	RunID       string `json:"run_id"`
	Iteration   int    `json:"iteration"`
	StepIndex   int    `json:"step_index"`
	ToolName    string `json:"tool_name"`
	ToolCallID  string `json:"tool_call_id"`
	ArgsJSON    string `json:"args_json"`
	Status      string `json:"status"`
	Content     string `json:"content,omitempty"`
	ErrorText   string `json:"error_text,omitempty"`
	StartedAt   *int64 `json:"started_at,omitempty"`
	CompletedAt *int64 `json:"completed_at,omitempty"`
}

// TaskHistoryStore retrieves durable task steps strictly scoped to a specific chat, user, and FSM run.
type TaskHistoryStore interface {
	ListTaskHistorySteps(ctx context.Context, chatID, userID string, isDM bool, runID string) ([]TaskHistoryStep, error)
}

// TaskHistoryProvider resolves a TaskHistoryStore for a given chat context.
type TaskHistoryProvider interface {
	GetTaskHistoryStore(ctx context.Context, chatID string, isDM bool) (TaskHistoryStore, error)
}

// RecallTaskHistoryArgs defines the input arguments for the recall_task_history tool.
type RecallTaskHistoryArgs struct {
	Iteration   *int   `json:"iteration,omitempty"`
	StepID      string `json:"step_id,omitempty"`
	ToolCallID  string `json:"tool_call_id,omitempty"`
	Query       string `json:"query,omitempty"`
	Field       string `json:"field,omitempty"`
	StepOffset  int    `json:"step_offset,omitempty"`
	OffsetBytes int    `json:"offset_bytes,omitempty"`
	MaxBytes    int    `json:"max_bytes,omitempty"`
}

// TaskHistoryStepView represents a sliced step observation returned by recall_task_history.
type TaskHistoryStepView struct {
	ID               string `json:"id"`
	Iteration        int    `json:"iteration"`
	StepIndex        int    `json:"step_index"`
	ToolName         string `json:"tool_name"`
	ToolCallID       string `json:"tool_call_id"`
	Field            string `json:"field,omitempty"`
	ArgsJSON         string `json:"args_json,omitempty"`
	TotalArgsBytes   int    `json:"total_args_bytes"`
	ArgsTruncated    bool   `json:"args_truncated,omitempty"`
	Status           string `json:"status"`
	Output           string `json:"output,omitempty"`
	ErrorText        string `json:"error_text,omitempty"`
	TotalErrorBytes  int    `json:"total_error_bytes,omitempty"`
	ErrorTruncated   bool   `json:"error_truncated,omitempty"`
	TotalOutputBytes int    `json:"total_output_bytes"`
	OffsetBytes      int    `json:"offset_bytes"`
	ReturnedBytes    int    `json:"returned_bytes"`
	HasMore          bool   `json:"has_more"`
	NextOffsetBytes  *int   `json:"next_offset_bytes,omitempty"`
}

var recallTaskHistorySchema = map[string]interface{}{
	"type": "object",
	"properties": map[string]interface{}{
		"iteration": map[string]interface{}{
			"type":        "integer",
			"description": "Optional 1-based iteration number to retrieve all tool steps and observations from that iteration.",
		},
		"step_id": map[string]interface{}{
			"type":        "string",
			"description": "Optional step ID (e.g. from VerifiedFindings source_step_ids or compacted_step_ids) to retrieve the raw step arguments and output.",
		},
		"tool_call_id": map[string]interface{}{
			"type":        "string",
			"description": "Optional tool_call_id to retrieve the raw step arguments and output for a specific tool call.",
		},
		"query": map[string]interface{}{
			"type":        "string",
			"description": "Optional case-insensitive keyword/substring filter across step tool names, arguments, outputs, and errors in the current task run.",
		},
		"field": map[string]interface{}{
			"type":        "string",
			"enum":        []string{"output", "args", "error"},
			"description": "Optional target field to paginate with offset_bytes: 'output' (default), 'args' (for large tool arguments), or 'error' (for large error messages).",
		},
		"step_offset": map[string]interface{}{
			"type":        "integer",
			"description": "Optional 0-based step index offset when paginating across more than 20 matching steps (default 0).",
		},
		"offset_bytes": map[string]interface{}{
			"type":        "integer",
			"description": "Optional 0-based byte offset into the step's target field for paginating large outputs/args/errors (default 0).",
		},
		"max_bytes": map[string]interface{}{
			"type":        "integer",
			"description": "Optional maximum byte budget to return (min 4, default 16384, max 16384).",
		},
	},
}

// RecallTaskHistoryToolDefinition returns the OpenAI tool schema for recall_task_history.
func RecallTaskHistoryToolDefinition() openai.Tool {
	return openai.Tool{
		Type: openai.ToolTypeFunction,
		Function: &openai.FunctionDefinition{
			Name:        RecallTaskHistoryToolName,
			Description: "Retrieve raw tool arguments, status, and full or byte-paginated outputs from earlier steps in the current FSM task run (including steps summarized by task state compaction or outputs larger than 16KB).",
			Parameters:  recallTaskHistorySchema,
		},
	}
}

// SetTaskHistoryProvider configures the TaskHistoryProvider on the Registry.
func (r *Registry) SetTaskHistoryProvider(p TaskHistoryProvider) {
	r.taskHistoryProvider = p
}

func (r *Registry) executeRecallTaskHistory(ctx context.Context, argsJSON string) (string, error) {
	if r.taskHistoryProvider == nil {
		return "", errors.New("task history provider is not configured")
	}

	session, ok := ChatSessionFromContext(ctx)
	if !ok {
		return "", errors.New("chat session context missing")
	}

	chatID := strings.TrimSpace(session.ChatID)
	if chatID == "" {
		return "", errors.New("chat_id is required in session context")
	}
	runID := strings.TrimSpace(session.FSMRunID)
	if runID == "" {
		return "", errors.New("recall_task_history is only available within an active FSM task run")
	}

	isDM := session.IsDM
	if chatID == "townhall" {
		isDM = false
	}
	if isDM && r.dmAuthorizer != nil {
		if err := r.dmAuthorizer(ctx, chatID, session.UserID); err != nil {
			return "", fmt.Errorf("unauthorized DM session: %w", err)
		}
	}

	store, err := r.taskHistoryProvider.GetTaskHistoryStore(ctx, chatID, isDM)
	if err != nil {
		return "", fmt.Errorf("failed to resolve task history store: %w", err)
	}
	if store == nil {
		return "", errors.New("task history store is unavailable")
	}

	return ExecuteRecallTaskHistoryWithStore(ctx, store, chatID, session.UserID, isDM, runID, argsJSON)
}

// ExecuteRecallTaskHistoryWithStore executes a recall_task_history query against a TaskHistoryStore
// strictly scoped to the provided (chatID, userID, isDM, runID).
func ExecuteRecallTaskHistoryWithStore(
	ctx context.Context,
	store TaskHistoryStore,
	chatID, userID string,
	isDM bool,
	runID, argsJSON string,
) (string, error) {
	if store == nil {
		return "", errors.New("task history store is not configured")
	}
	chatID = strings.TrimSpace(chatID)
	runID = strings.TrimSpace(runID)
	if chatID == "" || runID == "" {
		return "", errors.New("recall_task_history requires both chat_id and active run_id")
	}

	var args RecallTaskHistoryArgs
	trimmedArgs := strings.TrimSpace(argsJSON)
	if trimmedArgs != "" && trimmedArgs != "{}" {
		dec := json.NewDecoder(bytes.NewReader([]byte(trimmedArgs)))
		dec.DisallowUnknownFields()
		if err := dec.Decode(&args); err != nil {
			return "", fmt.Errorf("failed to parse arguments: %w", err)
		}
	}

	if args.Iteration != nil && *args.Iteration <= 0 {
		return "", errors.New("iteration must be a positive integer")
	}
	if args.StepOffset < 0 {
		return "", errors.New("step_offset cannot be negative")
	}
	if args.OffsetBytes < 0 {
		return "", errors.New("offset_bytes cannot be negative")
	}
	if args.MaxBytes < 0 || (args.MaxBytes > 0 && args.MaxBytes < MinTaskHistoryMaxBytes) {
		return "", fmt.Errorf("max_bytes must be at least %d bytes", MinTaskHistoryMaxBytes)
	}

	targetField := strings.ToLower(strings.TrimSpace(args.Field))
	if targetField == "" {
		targetField = "output"
	}
	switch targetField {
	case "output", "args", "error":
	default:
		return "", fmt.Errorf("invalid field %q: supported values are 'output', 'args', and 'error'", args.Field)
	}

	maxBytes := args.MaxBytes
	if maxBytes == 0 {
		maxBytes = DefaultTaskHistoryMaxBytes
	}
	if maxBytes > MaxTaskHistoryMaxBytes {
		maxBytes = MaxTaskHistoryMaxBytes
	}

	args.StepID = strings.TrimSpace(args.StepID)
	args.ToolCallID = strings.TrimSpace(args.ToolCallID)
	args.Query = strings.TrimSpace(args.Query)

	steps, err := store.ListTaskHistorySteps(ctx, chatID, userID, isDM, runID)
	if err != nil {
		return "", fmt.Errorf("failed to query task history: %w", err)
	}

	var matched []TaskHistoryStep
	queryLower := strings.ToLower(args.Query)
	for _, s := range steps {
		if s.RunID != runID {
			continue
		}
		if args.Iteration != nil && s.Iteration != *args.Iteration {
			continue
		}
		if args.StepID != "" && s.ID != args.StepID {
			continue
		}
		if args.ToolCallID != "" && s.ToolCallID != args.ToolCallID {
			continue
		}
		if queryLower != "" && !matchesTaskHistoryStep(s, queryLower) {
			continue
		}
		matched = append(matched, s)
	}

	totalMatched := len(matched)
	var pageMatched []TaskHistoryStep
	if args.StepOffset < totalMatched {
		endIdx := args.StepOffset + MaxTaskHistoryMatchedSteps
		if endIdx > totalMatched {
			endIdx = totalMatched
		}
		pageMatched = matched[args.StepOffset:endIdx]
	}

	return marshalBoundedTaskHistoryResponse(runID, targetField, args.StepOffset, args.OffsetBytes, maxBytes, totalMatched, pageMatched)
}

func marshalBoundedTaskHistoryResponse(
	runID, targetField string,
	stepOffset, offsetBytes, maxBytes, totalMatched int,
	pageMatched []TaskHistoryStep,
) (string, error) {
	currentSteps := pageMatched
	contentBudget := maxBytes

	for {
		views := buildTaskHistoryViews(currentSteps, targetField, offsetBytes, contentBudget)
		hasMoreSteps := stepOffset+len(views) < totalMatched
		var nextStepOffset *int
		if hasMoreSteps {
			next := stepOffset + len(views)
			nextStepOffset = &next
		}

		payload := map[string]interface{}{
			"run_id":          runID,
			"matched_count":   totalMatched,
			"step_offset":     stepOffset,
			"returned_count":  len(views),
			"has_more_steps":  hasMoreSteps,
			"truncated_steps": hasMoreSteps,
			"steps":           views,
		}
		if nextStepOffset != nil {
			payload["next_step_offset"] = *nextStepOffset
		}
		if len(views) == 0 {
			payload["message"] = "No matching task history steps found in the current run."
		}

		respBytes, err := json.Marshal(payload)
		if err != nil {
			return "", fmt.Errorf("failed to format recall_task_history response: %w", err)
		}

		if len(respBytes) <= MaxTaskHistoryMaxBytes {
			return string(respBytes), nil
		}

		// Response exceeds MaxTaskHistoryMaxBytes (16 KiB). Shrink contentBudget proportionally or page step count
		// so downstream formatToolStepResult never truncates the JSON envelope.
		if len(currentSteps) == 1 && contentBudget > MinTaskHistoryMaxBytes {
			targetBytes := MaxTaskHistoryMaxBytes - 256
			if targetBytes < 1024 {
				targetBytes = 1024
			}
			scaled := (contentBudget * targetBytes) / len(respBytes)
			if scaled >= contentBudget {
				scaled = contentBudget - (len(respBytes)-MaxTaskHistoryMaxBytes)/2 - 64
			}
			if scaled < MinTaskHistoryMaxBytes {
				scaled = MinTaskHistoryMaxBytes
			}
			contentBudget = scaled
			continue
		}

		if len(currentSteps) > 1 {
			if contentBudget > 512 {
				contentBudget /= 2
				if contentBudget < 512 {
					contentBudget = 512
				}
				continue
			}
			currentSteps = currentSteps[:len(currentSteps)-1]
			continue
		}

		return "", fmt.Errorf("step metadata exceeds maximum response size (%d bytes)", MaxTaskHistoryMaxBytes)
	}
}

func buildTaskHistoryViews(steps []TaskHistoryStep, targetField string, offsetBytes, maxBytes int) []TaskHistoryStepView {
	if len(steps) == 0 {
		return []TaskHistoryStepView{}
	}

	remainingBudget := maxBytes
	perStepCap := maxBytes
	if len(steps) > 1 {
		perStepCap = maxBytes / len(steps)
		if perStepCap < MinTaskHistoryMaxBytes {
			perStepCap = MinTaskHistoryMaxBytes
		}
	}

	views := make([]TaskHistoryStepView, 0, len(steps))
	for i, s := range steps {
		if i > 0 && remainingBudget < MinTaskHistoryMaxBytes {
			break
		}
		stepBudget := perStepCap
		if len(steps) == 1 {
			stepBudget = remainingBudget
		} else if stepBudget > remainingBudget {
			stepBudget = remainingBudget
		}
		if stepBudget < 0 {
			stepBudget = 0
		}

		var targetRaw string
		switch targetField {
		case "args":
			targetRaw = s.ArgsJSON
		case "error":
			targetRaw = s.ErrorText
		default:
			targetRaw = s.Content
		}

		slicedVal, actualOffset, returnedBytes, hasMore, nextOffset := sliceTaskHistoryOutput(targetRaw, offsetBytes, stepBudget)
		remainingBudget -= returnedBytes

		argsPreview, argsTrunc := truncateInlinePreview(s.ArgsJSON, MaxInlinePreviewBytes)
		errPreview, errTrunc := truncateInlinePreview(s.ErrorText, MaxInlinePreviewBytes)
		idPreview, _ := truncateInlinePreview(s.ID, MaxInlinePreviewBytes)
		toolNamePreview, _ := truncateInlinePreview(s.ToolName, MaxInlinePreviewBytes)
		callIDPreview, _ := truncateInlinePreview(s.ToolCallID, MaxInlinePreviewBytes)

		view := TaskHistoryStepView{
			ID:               idPreview,
			Iteration:        s.Iteration,
			StepIndex:        s.StepIndex,
			ToolName:         toolNamePreview,
			ToolCallID:       callIDPreview,
			Field:            targetField,
			ArgsJSON:         argsPreview,
			TotalArgsBytes:   len(s.ArgsJSON),
			ArgsTruncated:    argsTrunc,
			Status:           s.Status,
			ErrorText:        errPreview,
			TotalErrorBytes:  len(s.ErrorText),
			ErrorTruncated:   errTrunc,
			TotalOutputBytes: len(targetRaw),
			OffsetBytes:      actualOffset,
			ReturnedBytes:    returnedBytes,
			HasMore:          hasMore,
			NextOffsetBytes:  nextOffset,
		}

		switch targetField {
		case "args":
			view.ArgsJSON = slicedVal
			view.ArgsTruncated = hasMore
		case "error":
			view.ErrorText = slicedVal
			view.ErrorTruncated = hasMore
		default:
			view.Output = slicedVal
		}

		views = append(views, view)
	}
	return views
}

func truncateInlinePreview(s string, maxBytes int) (string, bool) {
	if len(s) <= maxBytes {
		return s, false
	}
	end := maxBytes
	for end > 0 && !utf8.RuneStart(s[end]) {
		end--
	}
	return s[:end] + "...[truncated]", true
}

func matchesTaskHistoryStep(s TaskHistoryStep, queryLower string) bool {
	if strings.Contains(strings.ToLower(s.ToolName), queryLower) {
		return true
	}
	if strings.Contains(strings.ToLower(s.ArgsJSON), queryLower) {
		return true
	}
	if strings.Contains(strings.ToLower(s.Content), queryLower) {
		return true
	}
	if strings.Contains(strings.ToLower(s.ErrorText), queryLower) {
		return true
	}
	if strings.Contains(strings.ToLower(s.ID), queryLower) {
		return true
	}
	if strings.Contains(strings.ToLower(s.ToolCallID), queryLower) {
		return true
	}
	return false
}

// sliceTaskHistoryOutput slices raw text starting at offsetBytes for up to maxBytes,
// adjusting UTF-8 boundaries so multi-byte runes are never split and NextOffsetBytes is lossless.
func sliceTaskHistoryOutput(raw string, offsetBytes, maxBytes int) (slice string, actualOffset, returnedBytes int, hasMore bool, nextOffset *int) {
	total := len(raw)
	if offsetBytes >= total || total == 0 {
		return "", offsetBytes, 0, false, nil
	}

	start := offsetBytes
	for start > 0 && !utf8.RuneStart(raw[start]) {
		start--
	}

	if maxBytes <= 0 {
		return "", start, 0, false, nil
	}

	end := total
	if maxBytes <= total-start {
		end = start + maxBytes
	}
	if end >= total {
		return raw[start:total], start, total - start, false, nil
	}

	// Back up end if it falls in the middle of a multi-byte UTF-8 rune.
	for end > start && !utf8.RuneStart(raw[end]) {
		end--
	}
	if end == start {
		// A positive budget must advance past at least one complete rune.
		r, rSize := utf8.DecodeRuneInString(raw[start:])
		if r == utf8.RuneError && rSize <= 1 {
			end = start + 1
		} else {
			end = start + rSize
		}
	}

	if end >= total {
		return raw[start:total], start, total - start, false, nil
	}
	next := end
	return raw[start:end], start, end - start, true, &next
}
