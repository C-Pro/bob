package tools

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"

	"bob/internal/scheduler"

	openai "github.com/sashabaranov/go-openai"
)

// SchedulerStoreProvider resolves a scheduler.Store for a chat context.
type SchedulerStoreProvider interface {
	GetSchedulerStore(ctx context.Context, chatID string, isDM bool) (*scheduler.Store, error)
}

// ScheduleTaskArgs defines arguments for schedule_task tool.
type ScheduleTaskArgs struct {
	ScheduleName      string          `json:"schedule_name"`
	Type              string          `json:"type"`
	ScheduleSpec      string          `json:"schedule_spec"`
	Instruction       string          `json:"instruction"`
	RunTimeoutSeconds int             `json:"run_timeout_seconds"`
	MaxTurns          int             `json:"max_turns"`
	MaxRuns           int             `json:"max_runs,omitempty"`
	ExpiresAt         string          `json:"expires_at,omitempty"`
	Permissions       json.RawMessage `json:"permissions,omitempty"`
}

// ListSchedulesArgs defines arguments for list_schedules tool.
type ListSchedulesArgs struct {
	ActiveOnly *bool `json:"active_only,omitempty"`
}

// CancelScheduleArgs defines arguments for cancel_schedule tool.
type CancelScheduleArgs struct {
	ScheduleName string `json:"schedule_name"`
}

var (
	scheduleTaskToolSchema = map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"schedule_name": map[string]interface{}{
				"type":        "string",
				"description": "Unique identifier for this schedule within the chat (lowercase alphanumeric words separated by underscores, max 4 words, e.g. 'daily_brief', 'stock_watcher').",
			},
			"type": map[string]interface{}{
				"type":        "string",
				"enum":        []string{"once", "cron", "interval"},
				"description": "Trigger mechanism: 'once' (delayed single run), 'cron' (recurring 5-field cron syntax), or 'interval' (recurring fixed duration interval).",
			},
			"schedule_spec": map[string]interface{}{
				"type":        "string",
				"description": "Specification for schedule execution: duration/timestamp for 'once' (e.g. '+30m', '3600', or RFC3339 timestamp), 5-field cron expression for 'cron' (e.g. '0 9 * * *'), or interval seconds for 'interval' (minimum 300, e.g. '600').",
			},
			"instruction": map[string]interface{}{
				"type":        "string",
				"description": "Detailed task instructions for the agent to execute when triggered.",
			},
			"run_timeout_seconds": map[string]interface{}{
				"type":        "integer",
				"description": "Mandatory maximum duration in seconds allocated for each execution (e.g. 300 for 5m). Bound by server limits. No default.",
			},
			"max_turns": map[string]interface{}{
				"type":        "integer",
				"description": "Mandatory maximum conversation turns allowed for each execution (e.g. 15). Bound by server limits. No default.",
			},
			"max_runs": map[string]interface{}{
				"type":        "integer",
				"description": "Optional maximum number of times the task will execute before completing automatically.",
			},
			"expires_at": map[string]interface{}{
				"type":        "string",
				"description": "Optional expiration timestamp (RFC3339 or duration like '30d') after which the schedule terminates (max 30 days).",
			},
			"permissions": map[string]interface{}{
				"type":        "object",
				"description": "Optional elevated execution capabilities requested for scheduled runs (e.g. sandbox environment). If requested, requires explicit user approval via /schedule approve <name>.",
				"properties": map[string]interface{}{
					"sandbox": map[string]interface{}{
						"type":        "object",
						"description": "Isolated sandbox configuration requested for execution.",
						"properties": map[string]interface{}{
							"driver": map[string]interface{}{
								"type":        "string",
								"enum":        []string{"bwrap", "docker"},
								"description": "Sandbox driver: 'bwrap' or 'docker'.",
							},
							"image": map[string]interface{}{
								"type":        "string",
								"description": "Container image name (for docker driver).",
							},
							"network": map[string]interface{}{
								"type":        "string",
								"enum":        []string{"none", "restricted"},
								"description": "Network access mode: 'none' (offline) or 'restricted' (whitelisted domains).",
							},
							"domains": map[string]interface{}{
								"type":        "array",
								"items":       map[string]interface{}{"type": "string"},
								"description": "Allowed hostnames when network is 'restricted'.",
							},
							"mounts": map[string]interface{}{
								"type":        "array",
								"items": map[string]interface{}{
									"type": "object",
									"properties": map[string]interface{}{
										"path":      map[string]interface{}{"type": "string"},
										"read_only": map[string]interface{}{"type": "boolean"},
									},
									"required": []string{"path"},
								},
								"description": "Workspace paths to mount inside sandbox.",
							},
							"timeout_seconds": map[string]interface{}{
								"type":        "integer",
								"description": "Maximum per-command execution timeout inside the sandbox.",
							},
						},
						"required": []string{"driver"},
					},
				},
			},
		},
		"required": []string{"schedule_name", "type", "schedule_spec", "instruction", "run_timeout_seconds", "max_turns"},
	}

	listSchedulesToolSchema = map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"active_only": map[string]interface{}{
				"type":        "boolean",
				"description": "If true (default), returns only active, paused, or pending schedules. If false, returns all historical schedules.",
			},
		},
	}

	cancelScheduleToolSchema = map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"schedule_name": map[string]interface{}{
				"type":        "string",
				"description": "The name of the schedule to cancel.",
			},
		},
		"required": []string{"schedule_name"},
	}
)

func scheduleToolDefinitions() []openai.Tool {
	return []openai.Tool{
		{
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{
				Name:        "schedule_task",
				Description: "Schedule a recurring or delayed autonomous task execution. All newly created schedules enter PENDING_APPROVAL and require explicit confirmation from the human user via /schedule approve <name> before they can execute. The agent cannot self-activate schedules or self-grant permissions. Available only in 1-on-1 direct messages.",
				Parameters:  scheduleTaskToolSchema,
			},
		},
		{
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{
				Name:        "list_schedules",
				Description: "List active or historical scheduled tasks in the current chat.",
				Parameters:  listSchedulesToolSchema,
			},
		},
		{
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{
				Name:        "cancel_schedule",
				Description: "Cancel an active or pending task schedule in the current chat by its schedule name.",
				Parameters:  cancelScheduleToolSchema,
			},
		},
	}
}

func (r *Registry) executeScheduleTask(ctx context.Context, argsJSON string) (string, error) {
	session, ok := ChatSessionFromContext(ctx)
	if !ok || session.ChatID == "" {
		return "", errors.New("chat session context missing")
	}
	if !session.IsDM {
		return "", errors.New("scheduling tools are strictly prohibited in Townhall and can only be used in Direct Messages")
	}
	if r.schedulerStoreProvider == nil {
		return "", errors.New("scheduler store provider is not configured")
	}

	store, err := r.schedulerStoreProvider.GetSchedulerStore(ctx, session.ChatID, session.IsDM)
	if err != nil {
		return "", fmt.Errorf("failed to get scheduler store: %w", err)
	}

	var args ScheduleTaskArgs
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		return "", fmt.Errorf("invalid arguments: %w", err)
	}

	// 1. Name validation
	args.ScheduleName = strings.TrimSpace(args.ScheduleName)
	if err := scheduler.ValidateScheduleName(args.ScheduleName); err != nil {
		return "", err
	}

	// 2. Bounds validation
	minTimeout := r.schedulerMinRunTimeout
	if minTimeout == 0 {
		minTimeout = 1 * time.Minute
	}
	maxTimeout := r.schedulerMaxRunTimeout
	if maxTimeout == 0 {
		maxTimeout = 1 * time.Hour
	}
	if args.RunTimeoutSeconds <= 0 || time.Duration(args.RunTimeoutSeconds)*time.Second < minTimeout || time.Duration(args.RunTimeoutSeconds)*time.Second > maxTimeout {
		return "", fmt.Errorf("run_timeout_seconds (%d) must be between %d and %d seconds", args.RunTimeoutSeconds, int(minTimeout.Seconds()), int(maxTimeout.Seconds()))
	}

	minTurns := r.schedulerMinMaxTurns
	if minTurns == 0 {
		minTurns = 10
	}
	maxTurns := r.schedulerMaxMaxTurns
	if maxTurns == 0 {
		maxTurns = 100
	}
	if args.MaxTurns < minTurns || args.MaxTurns > maxTurns {
		return "", fmt.Errorf("max_turns (%d) must be between %d and %d", args.MaxTurns, minTurns, maxTurns)
	}

	instruction := strings.TrimSpace(args.Instruction)
	if instruction == "" {
		return "", errors.New("instruction cannot be empty")
	}
	if len(instruction) > 4096 {
		return "", errors.New("instruction exceeds maximum allowed length of 4096 characters")
	}

	// 3. Recurrence parsing
	now := time.Now()
	schedType := scheduler.ScheduleType(strings.ToUpper(strings.TrimSpace(args.Type)))
	var nextRunAt int64
	var intervalSec int
	var cronExpr string

	switch schedType {
	case scheduler.ScheduleTypeOnce:
		t, err := parseOnceSpec(args.ScheduleSpec, now)
		if err != nil {
			return "", err
		}
		nextRunAt = t
		args.MaxRuns = 1

	case scheduler.ScheduleTypeInterval:
		sec, err := parseIntervalSpec(args.ScheduleSpec)
		if err != nil {
			return "", err
		}
		intervalSec = sec
		nextRunAt = now.Unix() + int64(sec)

	case scheduler.ScheduleTypeCron:
		expr := strings.TrimSpace(args.ScheduleSpec)
		cronSched, err := scheduler.ParseCronSchedule(expr)
		if err != nil {
			return "", fmt.Errorf("invalid cron expression %q: %w", expr, err)
		}
		next := cronSched.Next(now)
		if next.IsZero() {
			return "", errors.New("cron schedule does not have any upcoming execution times")
		}
		next2 := cronSched.Next(next)
		if !next2.IsZero() && next2.Sub(next) < 5*time.Minute {
			return "", errors.New("minimum cron cadence is 5 minutes (300 seconds)")
		}
		cronExpr = expr
		nextRunAt = next.Unix()

	default:
		return "", fmt.Errorf("invalid schedule type %q: must be 'once', 'cron', or 'interval'", args.Type)
	}

	// 4. Expiration parsing
	expiresAtUnix, err := parseExpiresAt(args.ExpiresAt, now)
	if err != nil {
		return "", err
	}

	// 5. Permissions extraction & validation
	var grant *scheduler.ScheduleGrant
	status := scheduler.ScheduleStatusPendingApproval

	rawPerms := strings.TrimSpace(string(args.Permissions))
	if rawPerms != "" && rawPerms != "null" && rawPerms != "{}" {
		var env scheduler.PermissionEnvelope
		if err := json.Unmarshal([]byte(rawPerms), &env); err != nil {
			return "", fmt.Errorf("invalid permissions object: %w", err)
		}

		if env.Sandbox != nil {
			driver := strings.ToLower(strings.TrimSpace(env.Sandbox.Driver))
			if driver != "bwrap" && driver != "docker" {
				return "", fmt.Errorf("unsupported sandbox driver %q: must be 'bwrap' or 'docker'", driver)
			}

			paramsHash := scheduler.ComputeRawJSONHash(rawPerms)

			validUntil := now.Add(30 * 24 * time.Hour).Unix()
			if expiresAtUnix != nil {
				validUntil = *expiresAtUnix
			}

			grantID := fmt.Sprintf("grant_%d", time.Now().UnixNano())
			grant = &scheduler.ScheduleGrant{
				ID:                    grantID,
				PermissionRequestJSON: rawPerms,
				ParamsHash:            paramsHash,
				ValidUntil:            validUntil,
			}
		}
	}

	schedID := fmt.Sprintf("sched_%d", time.Now().UnixNano())
	sched := &scheduler.Schedule{
		ID:                schedID,
		Name:              args.ScheduleName,
		ChatID:            session.ChatID,
		UserID:            session.UserID,
		ScheduleType:      schedType,
		CronExpr:          cronExpr,
		IntervalSeconds:   intervalSec,
		Instruction:       instruction,
		Status:            status,
		NextRunAt:         nextRunAt,
		ExpiresAt:         expiresAtUnix,
		MaxRuns:           args.MaxRuns,
		RunTimeoutSeconds: args.RunTimeoutSeconds,
		MaxTurns:          args.MaxTurns,
	}

	if err := store.CreateSchedule(ctx, sched, grant); err != nil {
		if strings.Contains(err.Error(), "UNIQUE constraint failed") || strings.Contains(err.Error(), "uq_schedules_chat_name") {
			return "", fmt.Errorf("a schedule named %q already exists in this chat. Please use a different name or cancel the existing one", args.ScheduleName)
		}
		return "", fmt.Errorf("failed to create schedule: %w", err)
	}

	var reason string
	if grant != nil {
		reason = fmt.Sprintf("Elevated sandbox permissions (%s) were requested", grant.PermissionRequestJSON)
	} else {
		reason = "All schedules require human user confirmation before execution"
	}

	return fmt.Sprintf("Task schedule %q created and is awaiting user approval (PENDING_APPROVAL).\n"+
		"%s; the user must approve it before execution will start.\n"+
		"Notify the user to run `/schedule approve %s` to activate this schedule.", args.ScheduleName, reason, args.ScheduleName), nil
}

func (r *Registry) executeListSchedules(ctx context.Context, argsJSON string) (string, error) {
	session, ok := ChatSessionFromContext(ctx)
	if !ok || session.ChatID == "" {
		return "", errors.New("chat session context missing")
	}
	if !session.IsDM {
		return "", errors.New("scheduling tools are strictly prohibited in Townhall and can only be used in Direct Messages")
	}
	if r.schedulerStoreProvider == nil {
		return "", errors.New("scheduler store provider is not configured")
	}

	store, err := r.schedulerStoreProvider.GetSchedulerStore(ctx, session.ChatID, session.IsDM)
	if err != nil {
		return "", fmt.Errorf("failed to get scheduler store: %w", err)
	}

	var args ListSchedulesArgs
	activeOnly := true
	if argsJSON != "" {
		_ = json.Unmarshal([]byte(argsJSON), &args)
		if args.ActiveOnly != nil {
			activeOnly = *args.ActiveOnly
		}
	}

	schedules, err := store.ListSchedules(ctx, session.ChatID, activeOnly)
	if err != nil {
		return "", fmt.Errorf("failed to list schedules: %w", err)
	}

	if len(schedules) == 0 {
		return "No schedules found for this chat.", nil
	}

	type schedItem struct {
		Name              string `json:"name"`
		Type              string `json:"type"`
		Spec              string `json:"spec,omitempty"`
		Status            string `json:"status"`
		NextRun           string `json:"next_run,omitempty"`
		RunCount          int    `json:"run_count"`
		RunTimeoutSeconds int    `json:"run_timeout_seconds"`
		MaxTurns          int    `json:"max_turns"`
		Instruction       string `json:"instruction"`
	}

	items := make([]schedItem, 0, len(schedules))
	for _, s := range schedules {
		spec := ""
		switch s.ScheduleType {
		case scheduler.ScheduleTypeCron:
			spec = s.CronExpr
		case scheduler.ScheduleTypeInterval:
			spec = fmt.Sprintf("%ds", s.IntervalSeconds)
		}
		nextStr := ""
		if s.NextRunAt > 0 && !s.Status.IsTerminal() {
			nextStr = time.Unix(s.NextRunAt, 0).UTC().Format("2006-01-02 15:04:05 MST")
		}
		items = append(items, schedItem{
			Name:              s.Name,
			Type:              string(s.ScheduleType),
			Spec:              spec,
			Status:            string(s.Status),
			NextRun:           nextStr,
			RunCount:          s.RunCount,
			RunTimeoutSeconds: s.RunTimeoutSeconds,
			MaxTurns:          s.MaxTurns,
			Instruction:       s.Instruction,
		})
	}

	b, err := json.MarshalIndent(items, "", "  ")
	if err != nil {
		return "", fmt.Errorf("failed to encode schedules: %w", err)
	}
	return string(b), nil
}

func (r *Registry) executeCancelSchedule(ctx context.Context, argsJSON string) (string, error) {
	session, ok := ChatSessionFromContext(ctx)
	if !ok || session.ChatID == "" {
		return "", errors.New("chat session context missing")
	}
	if !session.IsDM {
		return "", errors.New("scheduling tools are strictly prohibited in Townhall and can only be used in Direct Messages")
	}
	if r.schedulerStoreProvider == nil {
		return "", errors.New("scheduler store provider is not configured")
	}

	store, err := r.schedulerStoreProvider.GetSchedulerStore(ctx, session.ChatID, session.IsDM)
	if err != nil {
		return "", fmt.Errorf("failed to get scheduler store: %w", err)
	}

	var args CancelScheduleArgs
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		return "", fmt.Errorf("invalid arguments: %w", err)
	}

	name := strings.TrimSpace(args.ScheduleName)
	if name == "" {
		return "", errors.New("schedule_name cannot be empty")
	}

	sched, err := store.GetScheduleByName(ctx, session.ChatID, name)
	if err != nil {
		if errors.Is(err, scheduler.ErrScheduleNotFound) {
			return "", fmt.Errorf("schedule %q not found", name)
		}
		return "", fmt.Errorf("failed to get schedule: %w", err)
	}

	if session.UserID != "" && sched.UserID != session.UserID {
		return "", errors.New("permission denied: you can only cancel your own schedules")
	}

	cancelled, err := store.CancelSchedule(ctx, session.ChatID, name)
	if err != nil {
		return "", fmt.Errorf("failed to cancel schedule: %w", err)
	}

	return fmt.Sprintf("Schedule %q has been successfully cancelled and any permissions revoked.", cancelled.Name), nil
}

func parseOnceSpec(spec string, now time.Time) (int64, error) {
	spec = strings.TrimSpace(spec)
	if spec == "" {
		return 0, errors.New("schedule_spec cannot be empty")
	}

	// 1. Try RFC3339
	if t, err := time.Parse(time.RFC3339, spec); err == nil {
		if t.Unix() <= now.Unix() {
			return 0, errors.New("scheduled execution time must be in the future")
		}
		return t.Unix(), nil
	}

	// 2. Try integer seconds
	if sec, err := strconv.Atoi(spec); err == nil {
		if sec <= 0 {
			return 0, errors.New("seconds offset must be greater than 0")
		}
		return now.Add(time.Duration(sec) * time.Second).Unix(), nil
	}

	// 3. Try duration with optional leading '+'
	cleanSpec := strings.TrimPrefix(spec, "+")
	if d, err := time.ParseDuration(cleanSpec); err == nil {
		if d <= 0 {
			return 0, errors.New("duration must be greater than 0")
		}
		return now.Add(d).Unix(), nil
	}

	return 0, fmt.Errorf("invalid 'once' schedule_spec %q: must be RFC3339 timestamp, seconds offset, or duration (e.g. '30m', '+1h')", spec)
}

func parseIntervalSpec(spec string) (int, error) {
	spec = strings.TrimSpace(spec)
	if spec == "" {
		return 0, errors.New("schedule_spec cannot be empty")
	}

	// Try integer seconds
	if sec, err := strconv.Atoi(spec); err == nil {
		if sec < 300 {
			return 0, errors.New("interval must be at least 300 seconds (5 minutes)")
		}
		return sec, nil
	}

	// Try duration
	cleanSpec := strings.TrimPrefix(spec, "+")
	if d, err := time.ParseDuration(cleanSpec); err == nil {
		sec := int(d.Seconds())
		if sec < 300 {
			return 0, errors.New("interval must be at least 300 seconds (5 minutes)")
		}
		return sec, nil
	}

	return 0, fmt.Errorf("invalid interval schedule_spec %q: must be integer seconds or duration (e.g. '600', '10m', '1h')", spec)
}

func parseExpiresAt(spec string, now time.Time) (*int64, error) {
	spec = strings.TrimSpace(spec)
	if spec == "" {
		return nil, nil
	}

	var expTime time.Time
	// Try RFC3339
	if t, err := time.Parse(time.RFC3339, spec); err == nil {
		expTime = t
	} else {
		cleanSpec := strings.TrimPrefix(spec, "+")
		if strings.HasSuffix(cleanSpec, "d") || strings.HasSuffix(cleanSpec, "D") {
			daysStr := cleanSpec[:len(cleanSpec)-1]
			if days, err := strconv.Atoi(daysStr); err == nil && days > 0 {
				expTime = now.Add(time.Duration(days) * 24 * time.Hour)
			} else {
				return nil, fmt.Errorf("invalid expires_at %q: must be RFC3339 timestamp or duration (e.g. '7d', '24h')", spec)
			}
		} else if d, err := time.ParseDuration(cleanSpec); err == nil {
			expTime = now.Add(d)
		} else if sec, err := strconv.Atoi(spec); err == nil {
			expTime = now.Add(time.Duration(sec) * time.Second)
		} else {
			return nil, fmt.Errorf("invalid expires_at %q: must be RFC3339 timestamp or duration (e.g. '7d', '24h')", spec)
		}
	}

	if expTime.Unix() <= now.Unix() {
		return nil, errors.New("expires_at must be in the future")
	}

	maxExp := now.Add(30 * 24 * time.Hour)
	if expTime.After(maxExp) {
		return nil, errors.New("schedule expiration cannot exceed 30 days")
	}

	v := expTime.Unix()
	return &v, nil
}
