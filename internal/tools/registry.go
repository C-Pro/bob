package tools

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"time"
	"unicode/utf8"

	"bob/internal/memory"
	"bob/internal/models"
	"bob/internal/sandbox"
	"bob/internal/tools/tavily"
	"bob/internal/tools/webfetch"

	openai "github.com/sashabaranov/go-openai"
)

type chatContextKey struct{}

// ChatSessionContext holds request-scoped chat context for tool execution.
type ChatSessionContext struct {
	ChatID                string
	UserID                string
	IsDM                  bool
	IsScheduled           bool
	SourceMessageSeq      *int64
	FSMRunID              string
	Notifier              func(chatID, text string) error
	SandboxRequestCreated *bool
	StagedAttachments     *StagedAttachmentCollector
	KnowledgeBudget       *KnowledgeBudgetTracker
}

// StageAttachment appends an attachment to the staged attachments collector if configured.
func (s ChatSessionContext) StageAttachment(att models.Attachment) error {
	if s.StagedAttachments == nil {
		return errors.New("attachment staging collector is not configured")
	}
	return s.StagedAttachments.Add(att)
}

// GetStagedAttachments returns all attachments currently staged in the session context.
func (s ChatSessionContext) GetStagedAttachments() []models.Attachment {
	if s.StagedAttachments == nil {
		return nil
	}
	return s.StagedAttachments.All()
}

// NewChatSessionContext creates a ChatSessionContext with initialized collectors.
func NewChatSessionContext(chatID, userID string, isDM bool, limits ...KnowledgeBudgetLimits) ChatSessionContext {
	lim := DefaultKnowledgeBudgetLimits()
	if len(limits) > 0 {
		lim = limits[0]
	}
	return ChatSessionContext{
		ChatID:            chatID,
		UserID:            userID,
		IsDM:              isDM,
		StagedAttachments: NewStagedAttachmentCollector(),
		KnowledgeBudget:   NewKnowledgeBudgetTracker(lim),
	}
}

// WithChatSession returns a new context with the given ChatSessionContext attached.
func WithChatSession(ctx context.Context, session ChatSessionContext) context.Context {
	return context.WithValue(ctx, chatContextKey{}, session)
}

// ChatSessionFromContext retrieves the ChatSessionContext from context if present.
func ChatSessionFromContext(ctx context.Context) (ChatSessionContext, bool) {
	s, ok := ctx.Value(chatContextKey{}).(ChatSessionContext)
	return s, ok
}

// Registry manages available LLM tool definitions and executes tool calls.
type Registry struct {
	tavilyClient                   *tavily.Client
	memoryManager                  *memory.Manager
	sandboxManager                 *sandbox.Manager
	attachmentClient               AttachmentClient
	maxAttachmentSizeBytes         int64
	schedulerStoreProvider         SchedulerStoreProvider
	schedulerMinRunTimeout         time.Duration
	schedulerMaxRunTimeout         time.Duration
	schedulerMinMaxTurns           int
	schedulerMaxMaxTurns           int
	knowledgeStoreProvider         KnowledgeStoreProvider
	knowledgeSearcherProvider      KnowledgeSearcherProvider
	knowledgeLimits                KnowledgeBudgetLimits
	knowledgeDefaultDiscoveryLimit int
	knowledgeMaxDiscoveryLimit     int
	taskHistoryProvider            TaskHistoryProvider
	dmAuthorizer                   DMAuthorizer
	toolDefinitions                []openai.Tool
	dmToolDefinitions              []openai.Tool
	knowledgeToolDefs              []openai.Tool
}

// DMAuthorizer validates that chatID is an authorized 1-on-1 DM owned by userID.
type DMAuthorizer func(ctx context.Context, chatID, userID string) error

// SetDMAuthorizer sets the authoritative DM validator function for knowledge tools.
func (r *Registry) SetDMAuthorizer(authorizer DMAuthorizer) {
	r.dmAuthorizer = authorizer
}

// NewRegistry creates a new tool registry and initializes static tool definitions once.
func NewRegistry(tavilyClient *tavily.Client, memoryManager *memory.Manager, sandboxManager ...*sandbox.Manager) *Registry {
	var sm *sandbox.Manager
	if len(sandboxManager) > 0 {
		sm = sandboxManager[0]
	}
	r := &Registry{
		tavilyClient:                   tavilyClient,
		memoryManager:                  memoryManager,
		sandboxManager:                 sm,
		knowledgeLimits:                DefaultKnowledgeBudgetLimits(),
		knowledgeDefaultDiscoveryLimit: KnowledgeDefaultDiscoveryLimit,
		knowledgeMaxDiscoveryLimit:     KnowledgeMaxDiscoveryLimit,
	}
	r.initToolDefinitions()
	return r
}

// SetMemoryManager updates the memory manager.
func (r *Registry) SetMemoryManager(m *memory.Manager) {
	r.memoryManager = m
}

// SetSandboxManager updates the sandbox manager.
func (r *Registry) SetSandboxManager(sm *sandbox.Manager) {
	r.sandboxManager = sm
	r.initToolDefinitions()
}

// SetSchedulerStoreProvider configures the scheduler store provider for the registry.
func (r *Registry) SetSchedulerStoreProvider(p SchedulerStoreProvider) {
	r.schedulerStoreProvider = p
}

// SetSchedulerLimits configures validation boundaries for scheduled tasks.
func (r *Registry) SetSchedulerLimits(minTimeout, maxTimeout time.Duration, minTurns, maxTurns int) {
	r.schedulerMinRunTimeout = minTimeout
	r.schedulerMaxRunTimeout = maxTimeout
	r.schedulerMinMaxTurns = minTurns
	r.schedulerMaxMaxTurns = maxTurns
}

// SetAttachmentClient configures the client used for attachment downloads and uploads.
func (r *Registry) SetAttachmentClient(c AttachmentClient) {
	r.attachmentClient = c
}

// SetMaxAttachmentSize sets the maximum attachment size boundary in bytes.
func (r *Registry) SetMaxAttachmentSize(bytes int64) {
	r.maxAttachmentSizeBytes = bytes
	r.initToolDefinitions()
}

// SetKnowledgeStoreProvider configures the knowledge store provider for the registry.
func (r *Registry) SetKnowledgeStoreProvider(p KnowledgeStoreProvider) {
	r.knowledgeStoreProvider = p
}

// SetKnowledgeSearcherProvider configures the knowledge searcher provider for the registry.
func (r *Registry) SetKnowledgeSearcherProvider(p KnowledgeSearcherProvider) {
	r.knowledgeSearcherProvider = p
}

// SetKnowledgeLimits configures budget and discovery boundaries for knowledge tools.
func (r *Registry) SetKnowledgeLimits(limits KnowledgeBudgetLimits, defaultDiscoveryLimit, maxDiscoveryLimit int) {
	if limits.MaxLoadedMemories > 0 {
		r.knowledgeLimits.MaxLoadedMemories = limits.MaxLoadedMemories
	}
	if limits.MaxLoadedMemoryBytes > 0 {
		r.knowledgeLimits.MaxLoadedMemoryBytes = limits.MaxLoadedMemoryBytes
	}
	if limits.MaxLoadedSkills > 0 {
		r.knowledgeLimits.MaxLoadedSkills = limits.MaxLoadedSkills
	}
	if limits.MaxLoadedSkillBytes > 0 {
		r.knowledgeLimits.MaxLoadedSkillBytes = limits.MaxLoadedSkillBytes
	}
	if defaultDiscoveryLimit > 0 {
		r.knowledgeDefaultDiscoveryLimit = defaultDiscoveryLimit
	}
	if maxDiscoveryLimit > 0 {
		r.knowledgeMaxDiscoveryLimit = maxDiscoveryLimit
	}
}

// KnowledgeLimits returns the currently configured knowledge budget limits.
func (r *Registry) KnowledgeLimits() KnowledgeBudgetLimits {
	return r.knowledgeLimits
}

func (r *Registry) initToolDefinitions() {
	webSearchSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"query": map[string]interface{}{
				"type":        "string",
				"description": "The search query to look up on the live web.",
			},
			"search_depth": map[string]interface{}{
				"type":        "string",
				"enum":        []string{"basic", "advanced"},
				"description": "Search depth. 'basic' is faster; 'advanced' performs deeper search.",
			},
			"max_results": map[string]interface{}{
				"type":        "integer",
				"description": "Maximum number of search results to return (1-5, default 5).",
			},
		},
		"required": []string{"query"},
	}

	webFetchSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"url": map[string]interface{}{
				"type":        "string",
				"description": "The target HTTP or HTTPS URL to fetch content from.",
			},
			"mode": map[string]interface{}{
				"type":        "string",
				"enum":        []string{"auto", "raw", "extract"},
				"description": "Operational fetching mode. 'auto' (default) parses HTML into readable text with dynamic fallback; 'raw' returns the direct body without HTML parsing (recommended for code, scripts, configs, JSON); 'extract' queries Tavily Extract directly for heavy dynamic SPAs or if 'auto' failed to capture needed content). All modes truncate output to 16KB.",
			},
		},
		"required": []string{"url"},
	}

	recallMemorySchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"query": map[string]interface{}{
				"type":        "string",
				"description": "Search query to recall past conversations, topics, facts, episodes, or discussions from long-term memory. Optional when chunk_id, start_seq, or end_seq is provided.",
			},
			"chunk_id": map[string]interface{}{
				"type":        "string",
				"description": "Optional exact chunk_id (e.g. 'chunk_townhall_1_33') or episode_id (e.g. 'ep_townhall_1_33') to retrieve raw conversation chunks directly.",
			},
			"start_seq": map[string]interface{}{
				"type":        "integer",
				"description": "Optional starting message sequence number to retrieve raw conversation chunks overlapping [start_seq, end_seq].",
			},
			"end_seq": map[string]interface{}{
				"type":        "integer",
				"description": "Optional ending message sequence number to retrieve raw conversation chunks overlapping [start_seq, end_seq].",
			},
			"limit": map[string]interface{}{
				"type":        "integer",
				"description": "Maximum number of memory passages to retrieve (1-10, default 5).",
			},
		},
	}

	allowedNetworkModes := []string{"none", "restricted"}
	if r.sandboxManager != nil {
		allowedNetworkModes = r.sandboxManager.AllowedNetworkModes()
	}

	var networkDescriptions []string
	for _, m := range allowedNetworkModes {
		switch strings.ToLower(m) {
		case "none":
			networkDescriptions = append(networkDescriptions, "'none' (default, completely offline)")
		case "restricted":
			networkDescriptions = append(networkDescriptions, "'restricted' (access limited to whitelisted domains)")
		case "full":
			networkDescriptions = append(networkDescriptions, "'full' (unrestricted internet access)")
		}
	}
	networkDesc := "Network access level. Permitted modes on this server: " + strings.Join(networkDescriptions, ", ") + ". Apply least privilege."

	sandboxRequestSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"driver": map[string]interface{}{
				"type":        "string",
				"enum":        []string{"bwrap", "docker"},
				"description": "Sandbox driver: 'bwrap' (fast local isolation via Bubblewrap) or 'docker' (container isolation).",
			},
			"image": map[string]interface{}{
				"type":        "string",
				"description": "Container image name when using 'docker' (e.g. 'alpine:latest', 'golang:alpine', 'python:3.11-slim', 'node:20-slim').",
			},
			"network": map[string]interface{}{
				"type":        "string",
				"enum":        allowedNetworkModes,
				"description": networkDesc,
			},
			"domains": map[string]interface{}{
				"type": "array",
				"items": map[string]interface{}{
					"type": "string",
				},
				"description": "List of domains to allow when network is 'restricted' (e.g. ['pypi.org', 'files.pythonhosted.org']).",
			},
			"mounts": map[string]interface{}{
				"type": "array",
				"items": map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"path": map[string]interface{}{
							"type":        "string",
							"description": "Relative directory path inside your private workspace (e.g. '.' for whole workspace, or 'project1' for a specific subdirectory).",
						},
						"read_only": map[string]interface{}{
							"type":        "boolean",
							"description": "Whether the mount is read-only (default false).",
						},
					},
					"required": []string{"path"},
				},
				"description": "Subdirectories or whole workspace ('.') in your user workspace to mount into the sandbox.",
			},
			"lifetime_minutes": map[string]interface{}{
				"type":        "integer",
				"description": "Requested sandbox lifetime in minutes (1-30, default 30).",
			},
			"reason": map[string]interface{}{
				"type":        "string",
				"description": "Plain explanation to the user describing why the sandbox and each requested permission are needed.",
			},
		},
		"required": []string{"driver", "reason"},
	}

	sandboxExecSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"command": map[string]interface{}{
				"type":        "string",
				"description": "Shell command line to execute inside your active sandbox. IMPORTANT: Use one-shot, non-interactive commands only (do NOT use TUI tools like nano, vim, top, or interactive prompts).",
			},
			"description": map[string]interface{}{
				"type":        "string",
				"description": "Short one-line description of what this command runs or accomplishes in the sandbox (e.g. 'Running test suite', 'Calculating factorial of 6', 'Installing dependencies'). Used to report real-time progress to the user.",
			},
			"timeout_seconds": map[string]interface{}{
				"type":        "integer",
				"description": "Command execution timeout in seconds (default 30, min 5, max 600). Specify a higher limit for long-running commands like tests or builds.",
			},
		},
		"required": []string{"command", "description"},
	}

	sandboxDestroySchema := map[string]interface{}{
		"type":       "object",
		"properties": map[string]interface{}{},
	}

	r.toolDefinitions = []openai.Tool{
		{
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{
				Name:        "web_search",
				Description: "Search the live web for current information, news, documentation, and facts using Tavily search. When using results from this tool in your response, always cite sources and include the original markdown links [Title](URL) provided in the search results.",
				Parameters:  webSearchSchema,
			},
		},
		{
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{
				Name:        "web_fetch",
				Description: "Fetch and extract text content from a web page URL or raw text/code file. Supports 3 modes: 'auto' (default: parses HTML into clean readable text with dynamic fallback), 'raw' (recommended for raw code, scripts, JSON, or configs without HTML parsing), and 'extract' (uses Tavily Extract directly for heavy dynamic SPAs or if 'auto' failed to capture needed content). All outputs are capped at 16KB.",
				Parameters:  webFetchSchema,
			},
		},
		{
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{
				Name:        "recall_memory",
				Description: "Recall relevant past conversation history, discussions, topics, and episode navigation summaries from long-term memory. Provide 'query' for search, or 'chunk_id' / 'start_seq' / 'end_seq' to retrieve exact raw chunks. Episode summaries labeled [Model-Derived Episode Summary | seq <start>-<end>] are navigation indexes only; only when a summary is relevant and you need exact details, inspect its raw chunks (skip irrelevant summaries).",
				Parameters:  recallMemorySchema,
			},
		},
	}

	schedTools := scheduleToolDefinitions()
	sandboxTools := []openai.Tool{
		{
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{
				Name:        "sandbox_request",
				Description: "Request creation of a secure isolated execution sandbox. You MUST apply the principle of least privilege, requesting only the minimal permissions required for the task. The human user must explicitly approve your request before the sandbox is created. Available only in 1-on-1 direct messages.",
				Parameters:  sandboxRequestSchema,
			},
		},
		{
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{
				Name:        "sandbox_exec",
				Description: fmt.Sprintf("Execute a shell command inside your active sandbox. Output (stdout, stderr, exit code) is returned upon completion, truncated to a maximum of %dKB per stream. If expected output can be more than %dKB, pipe it to a file when a volume is mounted. Use non-interactive, one-shot commands only. Provide a short one-line description of what is being run to report progress to the user.", sandbox.DefaultMaxOutputBytes/1024, sandbox.DefaultMaxOutputBytes/1024),
				Parameters:  sandboxExecSchema,
			},
		},
		{
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{
				Name:        "sandbox_destroy",
				Description: "Explicitly terminate and tear down your active sandbox when you have finished your tasks or before requesting a different sandbox environment, freeing system resources.",
				Parameters:  sandboxDestroySchema,
			},
		},
	}

	dmTools := make([]openai.Tool, 0, len(r.toolDefinitions)+len(schedTools)+len(sandboxTools))
	dmTools = append(dmTools, r.toolDefinitions...)
	dmTools = append(dmTools, schedTools...)
	if r.sandboxManager != nil {
		dmTools = append(dmTools, sandboxTools...)
	}
	r.dmToolDefinitions = dmTools
	r.knowledgeToolDefs = r.knowledgeToolDefinitions()
}

// ToolDefinitions returns the cached slice of OpenAI tool definitions (for Townhall / public chats).
func (r *Registry) ToolDefinitions() []openai.Tool {
	return r.toolDefinitions
}

// ToolDefinitionsForSession returns tool definitions tailored to the session.
// In DM chats, scheduler tools and optional sandbox tools are included.
// Knowledge tools (propose/discover/load for memories and skills) are only included in
// interactive DM chats and excluded from scheduled background runs.
// When the user already has an active running sandbox, sandbox_request is excluded
// so the agent interacts with the active sandbox rather than requesting a redundant one.
// In Townhall, only base tools (search, fetch, recall) are provided.
func (r *Registry) ToolDefinitionsForSession(session ChatSessionContext) []openai.Tool {
	if session.IsDM {
		var tools []openai.Tool
		if r.sandboxManager != nil && session.UserID != "" {
			if sbx, ok := r.sandboxManager.GetStatus(session.UserID); ok && sbx.Status == sandbox.StatusRunning {
				tools = make([]openai.Tool, 0, len(r.dmToolDefinitions)+len(r.knowledgeToolDefs)+2)
				for _, t := range r.dmToolDefinitions {
					if t.Function != nil && t.Function.Name == "sandbox_request" {
						continue
					}
					tools = append(tools, t)
				}
				tools = append(tools, r.attachmentToolDefinitions()...)
			}
		}
		if tools == nil {
			tools = make([]openai.Tool, 0, len(r.dmToolDefinitions)+len(r.knowledgeToolDefs))
			tools = append(tools, r.dmToolDefinitions...)
		}
		if !session.IsScheduled {
			tools = append(tools, r.knowledgeToolDefs...)
		}
		if r.taskHistoryProvider != nil && strings.TrimSpace(session.FSMRunID) != "" {
			tools = append(tools, RecallTaskHistoryToolDefinition())
		}
		return tools
	}
	if r.taskHistoryProvider != nil && strings.TrimSpace(session.FSMRunID) != "" {
		tools := make([]openai.Tool, 0, len(r.toolDefinitions)+1)
		tools = append(tools, r.toolDefinitions...)
		tools = append(tools, RecallTaskHistoryToolDefinition())
		return tools
	}
	return r.toolDefinitions
}

// ToolDefinitionsForChat returns tool definitions for a given chat session context.
func (r *Registry) ToolDefinitionsForChat(ctx context.Context, chatID string, isDM bool) []openai.Tool {
	session, ok := ChatSessionFromContext(ctx)
	if !ok {
		session = NewChatSessionContext(chatID, "", isDM, r.knowledgeLimits)
	} else {
		if session.ChatID == "" {
			session.ChatID = chatID
		}
		session.IsDM = isDM
		if session.KnowledgeBudget == nil {
			session.KnowledgeBudget = NewKnowledgeBudgetTracker(r.knowledgeLimits)
		}
	}
	return r.ToolDefinitionsForSession(session)
}

// WebSearchArgs defines arguments for the web_search tool.
type WebSearchArgs struct {
	Query       string `json:"query"`
	SearchDepth string `json:"search_depth,omitempty"`
	MaxResults  int    `json:"max_results,omitempty"`
}

// WebFetchArgs defines arguments for the web_fetch tool.
type WebFetchArgs struct {
	URL  string `json:"url"`
	Mode string `json:"mode,omitempty"`
}

// RecallMemoryArgs defines arguments for the recall_memory tool.
type RecallMemoryArgs struct {
	Query    string `json:"query"`
	ChunkID  string `json:"chunk_id,omitempty"`
	StartSeq *int64 `json:"start_seq,omitempty"`
	EndSeq   *int64 `json:"end_seq,omitempty"`
	Limit    int    `json:"limit,omitempty"`
}

// Execute dispatches a tool execution by function name and arguments JSON.
func (r *Registry) Execute(ctx context.Context, name string, argsJSON string) (string, error) {
	switch name {
	case "web_search":
		return r.executeWebSearch(ctx, argsJSON)
	case "web_fetch":
		return r.executeWebFetch(ctx, argsJSON)
	case "recall_memory":
		return r.executeRecallMemory(ctx, argsJSON)
	case "sandbox_request":
		return r.executeSandboxRequest(ctx, argsJSON)
	case "sandbox_exec":
		return r.executeSandboxExec(ctx, argsJSON)
	case "sandbox_destroy":
		return r.executeSandboxDestroy(ctx, argsJSON)
	case "sandbox_download_attachment":
		return r.executeSandboxDownloadAttachment(ctx, argsJSON)
	case "sandbox_upload_attachment":
		return r.executeSandboxUploadAttachment(ctx, argsJSON)
	case "schedule_task":
		return r.executeScheduleTask(ctx, argsJSON)
	case "list_schedules":
		return r.executeListSchedules(ctx, argsJSON)
	case "cancel_schedule":
		return r.executeCancelSchedule(ctx, argsJSON)
	case "propose_memory":
		return r.executeProposeMemory(ctx, argsJSON)
	case "discover_memories":
		return r.executeDiscoverMemories(ctx, argsJSON)
	case "load_memory":
		return r.executeLoadMemory(ctx, argsJSON)
	case "propose_skill":
		return r.executeProposeSkill(ctx, argsJSON)
	case "discover_skills":
		return r.executeDiscoverSkills(ctx, argsJSON)
	case "load_skill":
		return r.executeLoadSkill(ctx, argsJSON)
	case RecallTaskHistoryToolName:
		return r.executeRecallTaskHistory(ctx, argsJSON)
	default:
		return "", fmt.Errorf("unknown tool: %s", name)
	}
}

// MaxRecallMemoryResponseBytes caps the total serialized JSON response of recall_memory to 16 KiB.
const MaxRecallMemoryResponseBytes = 16 * 1024

func (r *Registry) executeRecallMemory(ctx context.Context, argsJSON string) (string, error) {
	if r.memoryManager == nil {
		return "", errors.New("memory manager is not configured")
	}

	var args RecallMemoryArgs
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		return "", fmt.Errorf("failed to parse arguments: %w", err)
	}

	args.Query = strings.TrimSpace(args.Query)
	args.ChunkID = strings.TrimSpace(args.ChunkID)
	isRawLookup := args.ChunkID != "" || args.StartSeq != nil || args.EndSeq != nil
	if args.Query == "" && !isRawLookup {
		return "", errors.New("query cannot be empty (or provide chunk_id, start_seq, or end_seq)")
	}

	limit := args.Limit
	if limit <= 0 {
		limit = 5
	}
	if limit > 10 {
		limit = 10
	}

	session, _ := ChatSessionFromContext(ctx)
	chatID := session.ChatID
	if chatID == "" {
		chatID = "townhall"
	}
	isDM := session.IsDM
	if chatID == "townhall" {
		isDM = false
	}

	var (
		items []memory.MemoryItem
		err   error
	)

	if isRawLookup {
		items, err = r.memoryManager.GetRawChunks(ctx, chatID, isDM, memory.RawChunkQuery{
			ChunkID:  args.ChunkID,
			StartSeq: args.StartSeq,
			EndSeq:   args.EndSeq,
			Limit:    limit,
		})
		if err != nil {
			return "", fmt.Errorf("memory chunk lookup error: %w", err)
		}
		slog.Info("executed recall_memory raw chunk lookup", "chunk_id", args.ChunkID, "start_seq", args.StartSeq, "end_seq", args.EndSeq, "hits", len(items))
	} else {
		items, err = r.memoryManager.Search(ctx, args.Query, chatID, isDM, limit)
		if err != nil {
			return "", fmt.Errorf("memory search error: %w", err)
		}
		slog.Info("executed recall_memory", "query", args.Query, "hits", len(items))
	}

	items = deduplicateRecallMemoryItems(items)

	return marshalBoundedRecallMemoryResponse(args.Query, args.ChunkID, items)
}

func deduplicateRecallMemoryItems(items []memory.MemoryItem) []memory.MemoryItem {
	if len(items) <= 1 {
		return items
	}
	seenIDs := make(map[string]struct{}, len(items))
	seenEpisodeItems := make(map[string]struct{}, len(items))
	out := make([]memory.MemoryItem, 0, len(items))
	for _, it := range items {
		if _, ok := seenIDs[it.ID]; ok {
			continue
		}
		if it.Kind == "episode" && it.EpisodeID != "" {
			if _, ok := seenEpisodeItems[it.EpisodeID]; ok {
				continue
			}
			seenEpisodeItems[it.EpisodeID] = struct{}{}
		}
		seenIDs[it.ID] = struct{}{}
		out = append(out, it)
	}
	return out
}

func formatRecallMemoryResults(items []memory.MemoryItem) string {
	var sb strings.Builder
	var chunkItems, episodeItems []memory.MemoryItem
	for _, item := range items {
		if item.Kind == "episode" {
			episodeItems = append(episodeItems, item)
		} else {
			chunkItems = append(chunkItems, item)
		}
	}

	if len(chunkItems) > 0 {
		sb.WriteString("### Recalled Conversation Chunks:\n")
		for _, item := range chunkItems {
			var timeRange string
			if item.StartTime > 0 && item.EndTime > 0 {
				startT := time.Unix(item.StartTime, 0).UTC().Format("2006-01-02 15:04:05")
				endT := time.Unix(item.EndTime, 0).UTC().Format("2006-01-02 15:04:05")
				timeRange = fmt.Sprintf(" (%s to %s UTC)", startT, endT)
			} else if item.StartTime > 0 {
				timeRange = fmt.Sprintf(" (%s UTC)", time.Unix(item.StartTime, 0).UTC().Format("2006-01-02 15:04:05"))
			}

			seqRange := fmt.Sprintf("seq %d-%d", item.StartSeq, item.EndSeq)
			if item.StartSeq == item.EndSeq {
				seqRange = fmt.Sprintf("seq %d", item.StartSeq)
			}

			fmt.Fprintf(&sb, "\n--- %s [%s]%s ---\n", item.Source, seqRange, timeRange)
			if item.EpisodeID != "" {
				fmt.Fprintf(&sb, "[Raw Conversation Chunk | chunk_id: %s | episode_id: %s]\n", item.ID, item.EpisodeID)
			} else {
				fmt.Fprintf(&sb, "[Raw Conversation Chunk | chunk_id: %s]\n", item.ID)
			}
			fmt.Fprintf(&sb, "%s\n", strings.TrimSpace(item.Content))
		}
	}

	if len(episodeItems) > 0 {
		if sb.Len() > 0 {
			sb.WriteString("\n")
		}
		sb.WriteString("### Matching Episode Navigation Summaries:\n")
		for _, item := range episodeItems {
			fmt.Fprintf(&sb, "\n--- %s [seq %d-%d] ---\n%s\n", item.Source, item.StartSeq, item.EndSeq, strings.TrimSpace(item.Content))
		}
	}

	return strings.TrimSpace(sb.String())
}

func sanitizeBoundedMemoryItemMetadata(it memory.MemoryItem, maxSliceItems, maxElemBytes int) (memory.MemoryItem, bool) {
	// Never truncate lookup identifiers (ID, EpisodeID, ChatID, SourceChunkIDs) with "...";
	// if identifier pointers alone are pathologically large, drop the item rather than emitting broken pointers.
	idBytes := len(it.ID) + len(it.EpisodeID) + len(it.ChatID)
	for _, cid := range it.SourceChunkIDs {
		idBytes += len(cid)
	}
	if idBytes > 2048 {
		return memory.MemoryItem{}, false
	}
	if len(it.SourceChunkIDs) > 0 {
		copiedChunks := make([]string, len(it.SourceChunkIDs))
		copy(copiedChunks, it.SourceChunkIDs)
		it.SourceChunkIDs = copiedChunks
	}
	it.Source = truncateUTF8String(it.Source, 64)
	it.TaskStatus = truncateUTF8String(it.TaskStatus, 64)
	it.Participants = boundStringElements(it.Participants, maxSliceItems, maxElemBytes)
	it.TopicTags = boundStringElements(it.TopicTags, maxSliceItems, maxElemBytes)
	return it, true
}

func boundStringElements(in []string, maxItems, maxElemBytes int) []string {
	if len(in) == 0 {
		return nil
	}
	n := len(in)
	if n > maxItems {
		n = maxItems
	}
	out := make([]string, n)
	for i := 0; i < n; i++ {
		out[i] = truncateUTF8String(in[i], maxElemBytes)
	}
	return out
}

func formatBoundedEpisodeContent(item memory.MemoryItem, maxBytes int) (string, bool) {
	header := fmt.Sprintf("[Model-Derived Episode Summary | seq %d-%d]", item.StartSeq, item.EndSeq)
	chunksLine := ""
	if len(item.SourceChunkIDs) > 0 {
		chunksLine = fmt.Sprintf("Source Chunks: %s\n", strings.Join(item.SourceChunkIDs, ", "))
	}
	note := "(Note: Navigation summary only — if this episode is relevant and you need exact details, inspect its raw chunks via recall_memory with start_seq/end_seq or chunk_id.)"
	mandatoryBytes := len(header) + 1 + len(chunksLine) + len(note)
	if maxBytes < mandatoryBytes {
		return "", false
	}

	if len(item.Content) <= maxBytes && strings.HasPrefix(item.Content, header) {
		return item.Content, true
	}

	remaining := maxBytes - mandatoryBytes
	var b strings.Builder
	b.WriteString(header)
	b.WriteString("\n")
	b.WriteString(chunksLine)
	if remaining > 16 {
		body := strings.TrimSpace(item.Content)
		if strings.HasPrefix(body, header) {
			body = strings.TrimSpace(strings.TrimPrefix(body, header))
		}
		if idx := strings.Index(body, "(Note: Navigation summary only"); idx >= 0 {
			body = strings.TrimSpace(body[:idx])
		}
		if body != "" {
			b.WriteString(truncateUTF8String(body, remaining-1))
			b.WriteString("\n")
		}
	}
	b.WriteString(note)
	return b.String(), true
}

func marshalBoundedRecallMemoryResponse(query, chunkID string, items []memory.MemoryItem) (string, error) {
	boundedQuery := truncateUTF8String(query, 256)
	boundedChunkID := truncateUTF8String(chunkID, 512)

	if len(items) == 0 {
		payload := map[string]interface{}{
			"query":    boundedQuery,
			"count":    0,
			"memories": []interface{}{},
			"message":  "No relevant memories found in long-term memory for the given query.",
		}
		if boundedChunkID != "" {
			payload["chunk_id"] = boundedChunkID
		}
		respBytes, err := json.Marshal(payload)
		if err != nil {
			return "", fmt.Errorf("failed to format memory response: %w", err)
		}
		return string(respBytes), nil
	}

	// Sanitize display metadata on all items upfront while preserving exact lookup identifiers
	sanitized := make([]memory.MemoryItem, 0, len(items))
	for _, it := range items {
		if cleaned, ok := sanitizeBoundedMemoryItemMetadata(it, 8, 64); ok {
			sanitized = append(sanitized, cleaned)
		}
	}

	if len(sanitized) == 0 {
		fallback := map[string]interface{}{
			"query":     boundedQuery,
			"count":     0,
			"memories":  []interface{}{},
			"truncated": true,
			"message":   "Memory results exceeded the 16 KiB response budget; retry with a narrower query, chunk_id, or start_seq/end_seq.",
		}
		respBytes, _ := json.Marshal(fallback)
		return string(respBytes), nil
	}

	buildPayload := func(cur []memory.MemoryItem) ([]byte, error) {
		payload := map[string]interface{}{
			"query":             boundedQuery,
			"count":             len(cur),
			"memories":          cur,
			"formatted_results": formatRecallMemoryResults(cur),
		}
		if boundedChunkID != "" {
			payload["chunk_id"] = boundedChunkID
		}
		return json.Marshal(payload)
	}

	respBytes, err := buildPayload(sanitized)
	if err != nil {
		return "", fmt.Errorf("failed to format memory response: %w", err)
	}
	if len(respBytes) <= MaxRecallMemoryResponseBytes {
		return string(respBytes), nil
	}

	// Iteratively shrink item contents (preserving raw chunks ahead of episode summaries, and dropping episodes if their mandatory header cannot fit)
	perItemBudget := (MaxRecallMemoryResponseBytes / (len(sanitized) * 2)) - 256
	if perItemBudget < 256 {
		perItemBudget = 256
	}

	var working []memory.MemoryItem
	for attempt := 0; attempt < 6; attempt++ {
		working = make([]memory.MemoryItem, 0, len(sanitized))
		for _, orig := range sanitized {
			it, ok := sanitizeBoundedMemoryItemMetadata(orig, 4, 48)
			if !ok {
				continue
			}
			if it.Kind == "episode" {
				epBudget := perItemBudget
				if epBudget > 512 {
					epBudget = 512
				}
				formattedEp, ok := formatBoundedEpisodeContent(it, epBudget)
				if !ok {
					// Omit the episode rather than truncating its mandatory trust header
					continue
				}
				it.Content = formattedEp
			} else {
				it.Content = truncateUTF8String(it.Content, perItemBudget)
			}
			working = append(working, it)
		}

		respBytes, err = buildPayload(working)
		if err != nil {
			return "", fmt.Errorf("failed to format memory response: %w", err)
		}
		if len(respBytes) <= MaxRecallMemoryResponseBytes {
			return string(respBytes), nil
		}
		perItemBudget /= 2
		if perItemBudget < 64 {
			perItemBudget = 64
		}
	}

	// Drop trailing items (dropping episode items before raw chunk items) until <= 16 KiB
	for len(working) > 1 && len(respBytes) > MaxRecallMemoryResponseBytes {
		droppedEp := false
		for i := len(working) - 1; i >= 0; i-- {
			if working[i].Kind == "episode" {
				working = append(working[:i], working[i+1:]...)
				droppedEp = true
				break
			}
		}
		if !droppedEp {
			working = working[:len(working)-1]
		}
		respBytes, err = buildPayload(working)
		if err != nil {
			return "", fmt.Errorf("failed to format memory response: %w", err)
		}
	}

	if len(respBytes) > MaxRecallMemoryResponseBytes {
		fallback := map[string]interface{}{
			"query":     boundedQuery,
			"count":     0,
			"memories":  []interface{}{},
			"truncated": true,
			"message":   "Memory results exceeded the 16 KiB response budget; retry with a narrower query, chunk_id, or start_seq/end_seq.",
		}
		respBytes, _ = json.Marshal(fallback)
	}
	return string(respBytes), nil
}

func truncateUTF8String(s string, maxBytes int) string {
	if maxBytes <= 0 {
		return ""
	}
	if len(s) <= maxBytes {
		return s
	}
	cut := maxBytes
	for cut > 0 && !utf8.RuneStart(s[cut]) {
		cut--
	}
	return s[:cut] + "..."
}

func (r *Registry) executeWebSearch(ctx context.Context, argsJSON string) (string, error) {
	if r.tavilyClient == nil {
		return "", ClassifyWebSearchError(errors.New("tavily client is not configured"))
	}

	var args WebSearchArgs
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		return "", ClassifyWebSearchError(fmt.Errorf("failed to parse arguments: %w", err))
	}

	args.Query = strings.TrimSpace(args.Query)
	if args.Query == "" {
		return "", ClassifyWebSearchError(errors.New("query cannot be empty"))
	}

	req := tavily.SearchRequest{
		Query:         args.Query,
		SearchDepth:   args.SearchDepth,
		MaxResults:    args.MaxResults,
		IncludeAnswer: true,
	}

	resp, err := r.tavilyClient.Search(ctx, req)
	if err != nil {
		return "", ClassifyWebSearchError(fmt.Errorf("search error: %w", err))
	}

	payload := map[string]interface{}{
		"query":       resp.Query,
		"answer":      resp.Answer,
		"results":     resp.Results,
		"instruction": "When presenting these search findings to the user, cite your sources with original markdown links [Title](URL) from the results.",
	}

	respBytes, err := json.Marshal(payload)
	if err != nil {
		return "", ClassifyWebSearchError(fmt.Errorf("failed to format search response: %w", err))
	}

	return string(respBytes), nil
}

func (r *Registry) executeWebFetch(ctx context.Context, argsJSON string) (string, error) {
	var args WebFetchArgs
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		return "", ClassifyWebFetchError(fmt.Errorf("failed to parse arguments: %w", err))
	}

	args.URL = strings.TrimSpace(args.URL)
	if args.URL == "" {
		return "", ClassifyWebFetchError(errors.New("url cannot be empty"))
	}

	mode := strings.ToLower(strings.TrimSpace(args.Mode))
	if mode == "" {
		mode = "auto"
	}

	switch mode {
	case "extract":
		return r.executeFetchExtractMode(ctx, args.URL)
	case "raw":
		return r.executeFetchRawMode(ctx, args.URL)
	case "auto":
		return r.executeFetchAutoMode(ctx, args.URL)
	default:
		return "", ClassifyWebFetchError(fmt.Errorf("invalid mode %q: supported modes are 'auto', 'raw', and 'extract'", mode))
	}
}

func (r *Registry) executeFetchExtractMode(ctx context.Context, targetURL string) (string, error) {
	if r.tavilyClient == nil {
		return "", ClassifyWebFetchError(errors.New("tavily client is not configured for extract mode"))
	}

	resp, err := r.tavilyClient.Extract(ctx, targetURL)
	if err != nil {
		return "", ClassifyWebFetchError(fmt.Errorf("extract error: %w", err))
	}

	if len(resp.Results) == 0 {
		if len(resp.FailedResults) > 0 {
			return "", ClassifyWebFetchError(fmt.Errorf("extract failed for url %s: %s", resp.FailedResults[0].URL, resp.FailedResults[0].Error))
		}
		return "", ClassifyWebFetchError(fmt.Errorf("no content extracted from %s", targetURL))
	}

	res := resp.Results[0]
	truncatedContent := webfetch.TruncateText(res.RawContent, len(res.RawContent) > webfetch.MaxContentSize)

	payload := map[string]interface{}{
		"url":     res.URL,
		"mode":    "extract",
		"content": truncatedContent,
	}

	respBytes, err := json.Marshal(payload)
	if err != nil {
		return "", ClassifyWebFetchError(fmt.Errorf("failed to format extract response: %w", err))
	}
	return string(respBytes), nil
}

func (r *Registry) executeFetchRawMode(ctx context.Context, targetURL string) (string, error) {
	fetchRes, err := webfetch.Fetch(ctx, targetURL, nil)
	if err != nil {
		return "", ClassifyWebFetchError(fmt.Errorf("fetch error: %w", err))
	}

	truncatedContent := webfetch.TruncateText(string(fetchRes.RawBody), fetchRes.Truncated)

	payload := map[string]interface{}{
		"url":          fetchRes.URL,
		"mode":         "raw",
		"content_type": fetchRes.ContentType,
		"status_code":  fetchRes.StatusCode,
		"content":      truncatedContent,
	}

	respBytes, err := json.Marshal(payload)
	if err != nil {
		return "", ClassifyWebFetchError(fmt.Errorf("failed to format raw fetch response: %w", err))
	}
	return string(respBytes), nil
}

func (r *Registry) executeFetchAutoMode(ctx context.Context, targetURL string) (string, error) {
	fetchRes, fetchErr := webfetch.Fetch(ctx, targetURL, nil)
	if fetchErr != nil {
		// If direct fetch fails and Tavily is available, try dynamic extraction
		if r.tavilyClient != nil {
			extractPayload, extractErr := r.executeFetchExtractMode(ctx, targetURL)
			if extractErr == nil {
				return extractPayload, nil
			}
		}
		return "", ClassifyWebFetchError(fmt.Errorf("fetch error: %w", fetchErr))
	}

	if webfetch.IsBinary(fetchRes.ContentType) {
		payload := map[string]interface{}{
			"url":          fetchRes.URL,
			"mode":         "auto",
			"content_type": fetchRes.ContentType,
			"status_code":  fetchRes.StatusCode,
			"content":      fmt.Sprintf("[Binary file of type %s (%d bytes). Direct display not supported; file download support will be added in a future update.]", fetchRes.ContentType, len(fetchRes.RawBody)),
		}
		respBytes, err := json.Marshal(payload)
		if err != nil {
			return "", fmt.Errorf("failed to format response: %w", err)
		}
		return string(respBytes), nil
	}

	if !webfetch.IsHTML(fetchRes.ContentType) {
		truncatedContent := webfetch.TruncateText(string(fetchRes.RawBody), fetchRes.Truncated)
		payload := map[string]interface{}{
			"url":          fetchRes.URL,
			"mode":         "auto",
			"content_type": fetchRes.ContentType,
			"status_code":  fetchRes.StatusCode,
			"content":      truncatedContent,
		}
		respBytes, err := json.Marshal(payload)
		if err != nil {
			return "", fmt.Errorf("failed to format response: %w", err)
		}
		return string(respBytes), nil
	}

	// HTML handling with readability and heuristics
	rResult, rErr := webfetch.ParseReadability(fetchRes.RawBody, fetchRes.URL)
	needsFallback, fallbackReason := webfetch.NeedsDynamicFallback(rResult, rErr, string(fetchRes.RawBody))

	if needsFallback {
		if r.tavilyClient != nil {
			extractResp, extractErr := r.tavilyClient.Extract(ctx, targetURL)
			if extractErr == nil && len(extractResp.Results) > 0 && strings.TrimSpace(extractResp.Results[0].RawContent) != "" {
				res := extractResp.Results[0]
				payload := map[string]interface{}{
					"url":             res.URL,
					"mode":            "auto",
					"source":          "tavily_extract",
					"fallback_reason": fallbackReason,
					"content":         webfetch.TruncateText(res.RawContent, len(res.RawContent) > webfetch.MaxContentSize),
				}
				respBytes, err := json.Marshal(payload)
				if err != nil {
					return "", fmt.Errorf("failed to format extract response: %w", err)
				}
				return string(respBytes), nil
			}
		}

		// If Tavily is unconfigured or failed, return best-effort direct content with note
		var content string
		if rResult != nil && strings.TrimSpace(rResult.TextContent) != "" {
			content = rResult.TextContent + "\n\n[Note: Dynamic rendering fallback could not be executed: " + fallbackReason + "]"
		} else {
			content = string(fetchRes.RawBody) + "\n\n[Note: Dynamic rendering fallback could not be executed: " + fallbackReason + "]"
		}

		payload := map[string]interface{}{
			"url":             fetchRes.URL,
			"mode":            "auto",
			"source":          "direct_html_fallback",
			"fallback_reason": fallbackReason,
			"content":         webfetch.TruncateText(content, fetchRes.Truncated),
		}
		respBytes, err := json.Marshal(payload)
		if err != nil {
			return "", fmt.Errorf("failed to format response: %w", err)
		}
		return string(respBytes), nil
	}

	// Direct Readability Succeeded
	payload := map[string]interface{}{
		"url":     fetchRes.URL,
		"mode":    "auto",
		"title":   rResult.Title,
		"byline":  rResult.Byline,
		"excerpt": rResult.Excerpt,
		"content": webfetch.TruncateText(rResult.TextContent, fetchRes.Truncated),
	}

	respBytes, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("failed to format readability response: %w", err)
	}
	return string(respBytes), nil
}

// SandboxRequestArgs defines arguments for the sandbox_request tool.
type SandboxRequestArgs struct {
	Driver  string   `json:"driver"`
	Image   string   `json:"image,omitempty"`
	Network string   `json:"network,omitempty"`
	Domains []string `json:"domains,omitempty"`
	Mounts  []struct {
		Path     string `json:"path"`
		ReadOnly bool   `json:"read_only,omitempty"`
	} `json:"mounts,omitempty"`
	LifetimeMinutes int    `json:"lifetime_minutes,omitempty"`
	Reason          string `json:"reason"`
}

// SandboxExecArgs defines arguments for the sandbox_exec tool.
type SandboxExecArgs struct {
	Command        string `json:"command"`
	Description    string `json:"description,omitempty"`
	TimeoutSeconds int    `json:"timeout_seconds,omitempty"`
}

func (r *Registry) executeSandboxRequest(ctx context.Context, argsJSON string) (string, error) {
	if r.sandboxManager == nil {
		return "", errors.New("sandbox execution is not configured on this server")
	}
	session, ok := ChatSessionFromContext(ctx)
	if !ok || !session.IsDM {
		return "", errors.New("sandbox tools are strictly available in 1-on-1 direct messages")
	}
	if session.UserID == "" {
		return "", errors.New("cannot determine user ID from chat session")
	}

	var args SandboxRequestArgs
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		return "", fmt.Errorf("failed to parse sandbox_request arguments: %w", err)
	}

	var mounts []sandbox.UserMount
	for _, m := range args.Mounts {
		mounts = append(mounts, sandbox.UserMount{
			RelativePath: m.Path,
			ReadOnly:     m.ReadOnly,
		})
	}

	netMode := sandbox.NetworkMode(strings.ToLower(strings.TrimSpace(args.Network)))
	if netMode == "" {
		netMode = sandbox.NetworkNone
	}

	params := sandbox.RequestParams{
		Driver:          sandbox.DriverType(strings.ToLower(strings.TrimSpace(args.Driver))),
		DockerImage:     strings.TrimSpace(args.Image),
		NetworkMode:     netMode,
		AllowedDomains:  args.Domains,
		Mounts:          mounts,
		LifetimeMinutes: args.LifetimeMinutes,
		Reason:          strings.TrimSpace(args.Reason),
	}

	sbx, err := r.sandboxManager.RequestSandbox(ctx, session.UserID, session.ChatID, params)
	if err != nil {
		return "", fmt.Errorf("sandbox request rejected: %w", err)
	}

	var card strings.Builder
	card.WriteString("🔒 **Sandbox Approval Requested**\n")
	fmt.Fprintf(&card, "• **Driver:** `%s`\n", sbx.Driver)
	if sbx.Driver == sandbox.DriverDocker && sbx.DockerImage != "" {
		fmt.Fprintf(&card, "• **Docker Image:** `%s`\n", sbx.DockerImage)
	}
	fmt.Fprintf(&card, "• **Network:** `%s`\n", sbx.Network.Mode)
	if len(sbx.Network.AllowedHosts) > 0 {
		fmt.Fprintf(&card, "• **Allowed Domains:** `%s`\n", strings.Join(sbx.Network.AllowedHosts, ", "))
	}
	if len(sbx.Mounts) > 0 {
		card.WriteString("• **Mounts:**\n")
		for _, m := range sbx.Mounts {
			ro := "read-write"
			if m.ReadOnly {
				ro = "read-only"
			}
			fmt.Fprintf(&card, "  - `%s` (%s)\n", m.RelativePath, ro)
		}
	} else {
		card.WriteString("• **Workspace Mount:** none (isolated scratch space)\n")
	}
	cleanReason := strings.ReplaceAll(strings.ReplaceAll(sbx.Reason, "\r", ""), "\n", " ")
	cleanReason = strings.TrimSpace(cleanReason)
	if cleanReason == "" {
		cleanReason = "No reason provided"
	}
	fmt.Fprintf(&card, "• **Reason:** %s\n\n", cleanReason)
	card.WriteString("Reply `/sandbox approve` to approve or `/sandbox deny` to reject.")

	if session.Notifier != nil {
		if err := session.Notifier(session.ChatID, card.String()); err != nil {
			_ = r.sandboxManager.DenySandbox(session.UserID)
			return "", fmt.Errorf("failed to send approval card: %w", err)
		}
	}
	if session.SandboxRequestCreated != nil {
		*session.SandboxRequestCreated = true
	}

	var b strings.Builder
	b.WriteString("Sandbox request created and approval card has already been sent to the user in chat.\n")
	b.WriteString("DO NOT generate any conversational message or ask for approval again, as it duplicates the card already displayed.\n")
	b.WriteString("Conclude your turn now.")
	return b.String(), nil
}

func (r *Registry) executeSandboxExec(ctx context.Context, argsJSON string) (string, error) {
	if r.sandboxManager == nil {
		return "", errors.New("sandbox execution is not configured on this server")
	}
	session, ok := ChatSessionFromContext(ctx)
	if !ok || !session.IsDM {
		return "", errors.New("sandbox tools are strictly available in 1-on-1 direct messages")
	}
	if session.UserID == "" {
		return "", errors.New("cannot determine user ID from chat session")
	}

	var args SandboxExecArgs
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		return "", fmt.Errorf("failed to parse sandbox_exec arguments: %w", err)
	}

	cmdStr := strings.TrimSpace(args.Command)
	if cmdStr == "" {
		return "", errors.New("command cannot be empty")
	}

	var timeout time.Duration
	if args.TimeoutSeconds > 0 {
		secs := args.TimeoutSeconds
		if secs > 1800 {
			secs = 1800
		}
		timeout = time.Duration(secs) * time.Second
	}

	ctx = sandbox.WithRunContext(ctx, sandbox.RunContext{
		RunID:  session.FSMRunID,
		ChatID: session.ChatID,
		IsDM:   session.IsDM,
	})

	res, err := r.sandboxManager.Exec(ctx, session.UserID, []string{"sh", "-c", cmdStr}, timeout)
	if err != nil {
		return "", fmt.Errorf("execution failed: %w", err)
	}

	return FormatExecResult(res), nil
}

func (r *Registry) executeSandboxDestroy(ctx context.Context, argsJSON string) (string, error) {
	if r.sandboxManager == nil {
		return "", errors.New("sandbox execution is not configured on this server")
	}
	session, ok := ChatSessionFromContext(ctx)
	if !ok || !session.IsDM {
		return "", errors.New("sandbox tools are strictly available in 1-on-1 direct messages")
	}
	if session.UserID == "" {
		return "", errors.New("cannot determine user ID from chat session")
	}

	if err := r.sandboxManager.Destroy(ctx, session.UserID); err != nil {
		return "", fmt.Errorf("failed to destroy sandbox: %w", err)
	}

	return "Sandbox successfully destroyed and resources released.", nil
}
