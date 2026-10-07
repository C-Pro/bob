package gateway

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"html"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"bob/internal/agent"
	"bob/internal/agentapi"
	"bob/internal/chatcontext"
	"bob/internal/commands"
	"bob/internal/config"
	"bob/internal/fsm"
	"bob/internal/llm"
	"bob/internal/memory"
	"bob/internal/models"
	"bob/internal/prompt"
	"bob/internal/sandbox"
	"bob/internal/sandbox/bwrap"
	"bob/internal/sandbox/docker"
	"bob/internal/scheduler"
	"bob/internal/tools"
	"bob/internal/tools/tavily"

	"github.com/c-pro/geche"
	"github.com/fasthttp/websocket"
	"github.com/liliang-cn/cortexdb/v2/pkg/cortexdb"
	openai "github.com/sashabaranov/go-openai"
)

var _ agentapi.Frontend = (*Gateway)(nil)

// Gateway handles Besedka ingress (listening for mentions/messages) and egress (posting AI responses).
type Gateway struct {
	cfg                  *config.Config
	llmClient            *llm.Client
	httpClient           *http.Client
	toolsRegistry        *tools.Registry
	fsmEngine            *fsm.Engine
	memoryManager        *memory.Manager
	sandboxManager       *sandbox.Manager
	storeProvider        *MemoryStoreProvider
	conn                 *websocket.Conn
	mu                   sync.Mutex
	running              bool
	botUser              models.User
	botUserID            string
	userCache            *UserCache
	contextManager       *chatcontext.Manager
	startTime            time.Time
	location             *models.Location
	locationInterval     time.Duration
	initialLocationDelay time.Duration
	indexingWg           sync.WaitGroup
	userContextInjected  *geche.MapCache[string, bool]
	schedulerEngine      *scheduler.Engine
	schedulerInvoker     *ScheduleInvoker
	knowledgeWorker      *KnowledgeWorker
	lifecycleCtx         context.Context
	lifecycleID          uint64
	chatLocker           *ChatLocker
	chatCache            *ChatCache
	runner               agentapi.TurnRunner
}

// NewGateway creates a new Besedka Gateway instance.
func NewGateway(cfg *config.Config, llmClient *llm.Client) *Gateway {
	httpClient := &http.Client{Timeout: 10 * time.Second}

	var embedder cortexdb.Embedder
	modelSetting := strings.ToLower(strings.TrimSpace(cfg.EmbeddingModel))
	if modelSetting != "" && modelSetting != "none" && modelSetting != "disabled" && modelSetting != "off" && llmClient != nil {
		embedder = llm.NewEmbedder(llmClient, cfg.EmbeddingModel)
	}
	memoryManager := memory.NewManager(cfg, embedder)

	var tavilyClient *tavily.Client
	if cfg.TavilyAPIKey != "" {
		tavilyClient = tavily.NewClient(cfg.TavilyAPIKey, cfg.TavilyBaseURL, httpClient)
	}

	var sandboxManager *sandbox.Manager
	if cfg.SandboxEnabled {
		sbxCfg := cfg.SandboxConfig()
		bwrapDriver := bwrap.NewDriverWithConfig(bwrap.Config{
			CPULimit:          sbxCfg.CPULimit,
			MemoryLimitMB:     sbxCfg.MemoryLimitMB,
			DataDir:           sbxCfg.DataDir,
			ProxyFwdPath:      sbxCfg.ProxyFwdPath,
			AllowRuntimeBuild: sbxCfg.AllowRuntimeBuild,
		})
		dockerDriver := docker.NewDriver(docker.Config{
			SocketPath:        sbxCfg.DockerSocket,
			AllowedImages:     sbxCfg.AllowedImages,
			CPULimit:          sbxCfg.CPULimit,
			MemoryLimitMB:     sbxCfg.MemoryLimitMB,
			DataDir:           sbxCfg.DataDir,
			HostDataDir:       sbxCfg.HostDataDir,
			ProxyFwdPath:      sbxCfg.ProxyFwdPath,
			AllowRuntimeBuild: sbxCfg.AllowRuntimeBuild,
		})
		sandboxManager = sandbox.NewManager(sbxCfg, []sandbox.Driver{bwrapDriver, dockerDriver})
	}

	storeProvider := NewMemoryStoreProvider(memoryManager, cfg.DataDir)
	toolsRegistry := tools.NewRegistry(tavilyClient, memoryManager, sandboxManager)
	toolsRegistry.SetSchedulerStoreProvider(storeProvider)
	toolsRegistry.SetKnowledgeStoreProvider(storeProvider)
	toolsRegistry.SetKnowledgeSearcherProvider(storeProvider)
	toolsRegistry.SetSchedulerLimits(
		cfg.SchedulerMinRunTimeout,
		cfg.SchedulerMaxRunTimeout,
		cfg.SchedulerMinMaxTurns,
		cfg.SchedulerMaxMaxTurns,
	)

	var kw *KnowledgeWorker
	if storeProvider != nil {
		workerCfg := DefaultKnowledgeWorkerConfig()
		if cfg.KnowledgeReconcileInterval > 0 {
			workerCfg.Interval = cfg.KnowledgeReconcileInterval
		}
		kw = NewKnowledgeWorker(storeProvider, workerCfg)
	}

	chatLocker := NewChatLocker()
	gw := &Gateway{
		cfg:                  cfg,
		llmClient:            llmClient,
		httpClient:           httpClient,
		toolsRegistry:        toolsRegistry,
		memoryManager:        memoryManager,
		sandboxManager:       sandboxManager,
		storeProvider:        storeProvider,
		knowledgeWorker:      kw,
		userCache:            NewUserCache(),
		contextManager:       chatcontext.NewManager(cfg.MsgRingBufferSize),
		startTime:            time.Now(),
		locationInterval:     9 * time.Minute,
		initialLocationDelay: 1 * time.Second,
		userContextInjected:  geche.NewMapCache[string, bool](),
		chatLocker:           chatLocker,
		chatCache:            NewChatCache(),
	}

	toolsRegistry.SetAttachmentClient(gw)
	toolsRegistry.SetMaxAttachmentSize(cfg.MaxAttachmentSizeBytes)
	toolsRegistry.SetDMAuthorizer(gw.VerifyDMOwner)
	if storeProvider != nil {
		storeProvider.SetMaxDiscoveryLimit(cfg.KnowledgeMaxDiscoveryLimit)
		toolsRegistry.SetKnowledgeStoreProvider(storeProvider)
		toolsRegistry.SetKnowledgeSearcherProvider(storeProvider)
	}
	toolsRegistry.SetKnowledgeLimits(tools.KnowledgeBudgetLimits{
		MaxLoadedMemories:    cfg.KnowledgeMaxLoadedMemories,
		MaxLoadedMemoryBytes: cfg.KnowledgeMaxLoadedMemoryBytes,
		MaxLoadedSkills:      cfg.KnowledgeMaxLoadedSkills,
		MaxLoadedSkillBytes:  cfg.KnowledgeMaxLoadedSkillBytes,
	}, cfg.KnowledgeDefaultDiscoveryLimit, cfg.KnowledgeMaxDiscoveryLimit)

	invokerCfg := InvokerConfig{
		Tools:               toolsRegistry,
		Sandbox:             sandboxManager,
		Sender:              gw,
		AttachmentProcessor: gw,
		ContextMgr:          gw.contextManager,
		Model:               cfg.OpenAIModel,
		BotID:               "bot",
		BotName:             "Bob",
		SchedulerStoreProv: func(ctx context.Context, chatID string, isDM bool) (*scheduler.Store, error) {
			if gw.storeProvider != nil {
				return gw.storeProvider.GetSchedulerStore(ctx, chatID, isDM)
			}
			return nil, errors.New("store provider not available")
		},
	}
	gw.schedulerInvoker = NewScheduleInvoker(invokerCfg, chatLocker)
	gw.schedulerEngine = scheduler.NewEngine(storeProvider, gw.schedulerInvoker)

	gw.contextManager.SetOnEvict(func(chatID string, evicted []chatcontext.Entry) {
		gw.handleEvictedBatch(chatID, evicted)
	})

	return gw
}

// MemoryManager returns the Gateway's memory Manager.
func (g *Gateway) MemoryManager() *memory.Manager {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.memoryManager
}

// SetMemoryManager sets the memory Manager for the gateway.
func (g *Gateway) SetMemoryManager(m *memory.Manager) {
	g.mu.Lock()
	oldKW := g.knowledgeWorker
	g.memoryManager = m
	var newKW *KnowledgeWorker
	if g.cfg != nil {
		g.storeProvider = NewMemoryStoreProvider(m, g.cfg.DataDir)
		g.storeProvider.SetMaxDiscoveryLimit(g.cfg.KnowledgeMaxDiscoveryLimit)
		if g.toolsRegistry != nil {
			g.toolsRegistry.SetKnowledgeStoreProvider(g.storeProvider)
			g.toolsRegistry.SetKnowledgeSearcherProvider(g.storeProvider)
		}
		workerCfg := DefaultKnowledgeWorkerConfig()
		if g.cfg.KnowledgeReconcileInterval > 0 {
			workerCfg.Interval = g.cfg.KnowledgeReconcileInterval
		}
		newKW = NewKnowledgeWorker(g.storeProvider, workerCfg)
	} else {
		g.storeProvider = nil
	}
	g.knowledgeWorker = newKW
	g.mu.Unlock()

	if oldKW != nil {
		oldKW.Stop()
	}

	if newKW != nil {
		g.mu.Lock()
		if g.running && g.knowledgeWorker == newKW && g.lifecycleCtx != nil && g.lifecycleCtx.Err() == nil {
			newKW.Start(g.lifecycleCtx)
		}
		g.mu.Unlock()
	}
}

// StoreProvider returns the Gateway's MemoryStoreProvider.
func (g *Gateway) StoreProvider() *MemoryStoreProvider {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.storeProvider
}

// SetStoreProvider sets the MemoryStoreProvider for the gateway.
func (g *Gateway) SetStoreProvider(p *MemoryStoreProvider) {
	g.mu.Lock()
	oldKW := g.knowledgeWorker
	g.storeProvider = p
	if g.cfg != nil && p != nil {
		p.SetMaxDiscoveryLimit(g.cfg.KnowledgeMaxDiscoveryLimit)
	}
	if g.schedulerEngine != nil && p != nil {
		g.schedulerEngine = scheduler.NewEngine(p, g.schedulerInvoker)
	}
	if g.toolsRegistry != nil && p != nil {
		g.toolsRegistry.SetKnowledgeStoreProvider(p)
		g.toolsRegistry.SetKnowledgeSearcherProvider(p)
	}
	var newKW *KnowledgeWorker
	if p != nil && g.cfg != nil {
		workerCfg := DefaultKnowledgeWorkerConfig()
		if g.cfg.KnowledgeReconcileInterval > 0 {
			workerCfg.Interval = g.cfg.KnowledgeReconcileInterval
		}
		newKW = NewKnowledgeWorker(p, workerCfg)
	}
	g.knowledgeWorker = newKW
	g.mu.Unlock()

	if oldKW != nil {
		oldKW.Stop()
	}

	if newKW != nil {
		g.mu.Lock()
		if g.running && g.knowledgeWorker == newKW && g.lifecycleCtx != nil && g.lifecycleCtx.Err() == nil {
			newKW.Start(g.lifecycleCtx)
		}
		g.mu.Unlock()
	}
}

// ToolsRegistry returns the Gateway's tools Registry.
func (g *Gateway) ToolsRegistry() *tools.Registry {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.toolsRegistry
}

// SetToolsRegistry sets the tool registry for the gateway.
func (g *Gateway) SetToolsRegistry(r *tools.Registry) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.toolsRegistry = r
	if r != nil {
		r.SetAttachmentClient(g)
		r.SetDMAuthorizer(g.VerifyDMOwner)
		if g.cfg != nil {
			r.SetMaxAttachmentSize(g.cfg.MaxAttachmentSizeBytes)
			r.SetKnowledgeLimits(tools.KnowledgeBudgetLimits{
				MaxLoadedMemories:    g.cfg.KnowledgeMaxLoadedMemories,
				MaxLoadedMemoryBytes: g.cfg.KnowledgeMaxLoadedMemoryBytes,
				MaxLoadedSkills:      g.cfg.KnowledgeMaxLoadedSkills,
				MaxLoadedSkillBytes:  g.cfg.KnowledgeMaxLoadedSkillBytes,
			}, g.cfg.KnowledgeDefaultDiscoveryLimit, g.cfg.KnowledgeMaxDiscoveryLimit)
		}
		if g.storeProvider != nil {
			r.SetKnowledgeStoreProvider(g.storeProvider)
			r.SetKnowledgeSearcherProvider(g.storeProvider)
		}
	}
	if g.fsmEngine != nil {
		g.fsmEngine.SetToolDefinitionProvider(g)
	}
	if g.schedulerInvoker != nil {
		g.schedulerInvoker.SetTools(r)
	}
}

// FSMEngine returns the Gateway's durable FSM Engine.
func (g *Gateway) FSMEngine() *fsm.Engine {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.fsmEngine
}

// SetRunner sets the TurnRunner for the gateway.
func (g *Gateway) SetRunner(r agentapi.TurnRunner) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.runner = r
}

// Runner returns the TurnRunner for the gateway.
func (g *Gateway) Runner() agentapi.TurnRunner {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.runner
}

// SetFSMEngine sets the durable FSM Engine for the gateway and configures it with the gateway ResultSink and ToolDefinitionProvider.
func (g *Gateway) SetFSMEngine(e *fsm.Engine) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.fsmEngine = e
	if e != nil {
		e.RegisterFrontend("besedka", g)
		e.SetResultSink(&fsmResultSink{gateway: g})
		e.SetToolDefinitionProvider(g)
	}
	if g.schedulerInvoker != nil {
		g.schedulerInvoker.SetFSM(e)
	}
}

// ToolDefinitions implements fsm.ToolDefinitionProvider to supply tool definitions for a chat.
func (g *Gateway) ToolDefinitions(ctx context.Context, chatID string, isDM bool) []openai.Tool {
	g.mu.Lock()
	r := g.toolsRegistry
	g.mu.Unlock()
	if r == nil {
		return nil
	}
	session, ok := tools.ChatSessionFromContext(ctx)
	if !ok {
		session = tools.ChatSessionContext{
			ChatID: chatID,
			IsDM:   isDM,
		}
	} else {
		if session.ChatID != "" && chatID != "" && session.ChatID != chatID {
			session.IsDM = false
			session.ChatID = chatID
		} else if session.ChatID == "" {
			session.ChatID = chatID
		}
		session.IsDM = session.IsDM && isDM
	}

	// Authoritative verification for DM tool exposure:
	// If marked as DM, verify DM ownership against authoritative metadata.
	// Fails closed if session.UserID is empty, session.ChatID is townhall, or VerifyDMOwner fails.
	if session.IsDM {
		if session.ChatID == "" || session.ChatID == "townhall" || strings.TrimSpace(session.UserID) == "" || g.VerifyDMOwner(ctx, session.ChatID, session.UserID) != nil {
			session.IsDM = false
		}
	}

	return r.ToolDefinitionsForSession(session)
}

func (g *Gateway) isChatDM(ctx context.Context, chatID string) bool {
	if chatID == "townhall" {
		return false
	}
	if g.chatCache != nil {
		if chat, ok := g.chatCache.Get(chatID); ok {
			return (chat.IsDM || chat.Type == "dm") && chat.Type != "group"
		}
	}
	if g.httpClient != nil {
		chat, err := g.GetChat(ctx, chatID)
		if err == nil {
			return (chat.IsDM || chat.Type == "dm") && chat.Type != "group"
		}
	}
	return strings.HasPrefix(chatID, "dm_")
}

// Bind implements agentapi.Frontend to prepare execution-scoped dependencies for a run.
func (g *Gateway) Bind(ctx context.Context, run agentapi.RunDescriptor) (agentapi.Bindings, error) {
	chatID := run.Session.SessionID
	userID := run.Actor.ID
	isDM := g.isChatDM(ctx, chatID)
	isAuthorizedDMOwner := isDM && g.VerifyDMOwner(ctx, chatID, userID) == nil

	var budgetLimits tools.KnowledgeBudgetLimits
	g.mu.Lock()
	tr := g.toolsRegistry
	g.mu.Unlock()
	if tr != nil {
		budgetLimits = tr.KnowledgeLimits()
	} else if g.cfg != nil {
		budgetLimits = tools.KnowledgeBudgetLimits{
			MaxLoadedMemories:    g.cfg.KnowledgeMaxLoadedMemories,
			MaxLoadedMemoryBytes: g.cfg.KnowledgeMaxLoadedMemoryBytes,
			MaxLoadedSkills:      g.cfg.KnowledgeMaxLoadedSkills,
			MaxLoadedSkillBytes:  g.cfg.KnowledgeMaxLoadedSkillBytes,
		}
	} else {
		budgetLimits = tools.DefaultKnowledgeBudgetLimits()
	}

	sessionCtx := tools.NewChatSessionContext(chatID, userID, isAuthorizedDMOwner, budgetLimits)
	sessionCtx.Notifier = func(targetChatID, text string) error {
		return g.SendMessage(targetChatID, text)
	}

	var toolset agentapi.Toolset
	if tr != nil {
		toolset = tr.SessionToolset(sessionCtx)
	}

	var initialProgressSeq int64
	if run.Metadata != nil {
		if s, ok := run.Metadata["initial_progress_seq"]; ok && s != "" {
			if parsed, err := strconv.ParseInt(s, 10, 64); err == nil && parsed > 0 {
				initialProgressSeq = parsed
			}
		}
	}

	var progressObs agentapi.ProgressObserver
	if initialProgressSeq > 0 {
		progressObs = NewGatewayProgressObserverWithRoot(g, chatID, isDM, initialProgressSeq)
	} else {
		progressObs = NewGatewayProgressObserver(g, chatID, isDM)
	}

	attHandler := NewGatewayAttachmentAdapter(g, chatID)
	sink := &gatewayNotificationSink{gateway: g, chatID: chatID}

	return agentapi.Bindings{
		Tools:         toolset,
		Attachments:   attHandler,
		Progress:      progressObs,
		Notifications: sink,
	}, nil
}

// Deliver implements agentapi.Frontend to deliver completed or failed workflow results to Besedka.
func (g *Gateway) Deliver(ctx context.Context, comp agentapi.Completion) error {
	chatID := comp.Run.Session.SessionID
	if chatID == "" {
		return nil
	}
	if strings.HasPrefix(comp.Run.RunID, "sched_") {
		// Scheduled task runs handle their own notifications inside schedule_invoker;
		// suppress generic failure apologies to the chat on crash recovery.
		return nil
	}

	g.mu.Lock()
	botID := g.botUserID
	botName := g.botUser.GetDisplayName()
	sm := g.sandboxManager
	g.mu.Unlock()
	if botName == "" {
		botName = "Bob"
	}

	// Check for suppress_reply action (e.g. sandbox approval requests)
	for _, act := range comp.Result.Actions {
		if act.Type == agentapi.ActionSuppressReply {
			g.contextManager.Push(chatID, chatcontext.Entry{
				Role:       "assistant",
				SenderID:   botID,
				SenderName: botName,
				Content:    "Sandbox approval requested.",
				Timestamp:  time.Now().Unix(),
			})
			return nil
		}
	}

	var reply string
	switch comp.Status {
	case agentapi.RunCompleted:
		reply = comp.Result.Content
	case agentapi.RunFailed, agentapi.RunTerminated:
		reply = "Sorry, I encountered an issue processing your request. Please try again later."
	default:
		return nil
	}

	if reply == "" {
		return nil
	}

	isDM := g.isChatDM(ctx, chatID)
	if comp.Status == agentapi.RunCompleted && isDM && sm != nil {
		if sbx, ok := sm.GetStatus(comp.Run.Actor.ID); ok && sbx != nil && sbx.Status == sandbox.StatusRunning {
			if !strings.Contains(reply, "/sandbox destroy") {
				rem := time.Until(sbx.ExpiresAt).Round(time.Minute)
				if rem < 0 {
					rem = 0
				}
				reply += fmt.Sprintf("\n\n💡 Would you like to destroy the sandbox (`/sandbox destroy`) or keep it? It will automatically be destroyed in %s.", rem)
			}
		}
	}

	formattedReply := FormatResponse(reply, isDM, g.cfg.TownhallMaxParagraphs, g.cfg.DMMaxParagraphs)

	outgoingAttachments := ToBesedkaAttachments(comp.Result.Attachments)

	if err := g.SendMessageWithAttachments(chatID, formattedReply, outgoingAttachments); err != nil {
		return fmt.Errorf("failed to deliver reply to chat %s: %w", chatID, err)
	}

	pushContent, pushImages := formattedReply, []chatcontext.ImageAttachment(nil)
	if len(outgoingAttachments) > 0 {
		extraText, images := g.processAttachments(ctx, outgoingAttachments)
		pushContent = strings.TrimSpace(formattedReply + extraText)
		pushImages = images
	}

	g.contextManager.Push(chatID, chatcontext.Entry{
		Role:       "assistant",
		SenderID:   botID,
		SenderName: botName,
		Content:    pushContent,
		Images:     pushImages,
		Timestamp:  time.Now().Unix(),
	})

	return nil
}

// DeliverLegacyFSMRun converts an FSMRun to agentapi.Completion and delivers it to Besedka.
func (g *Gateway) DeliverLegacyFSMRun(ctx context.Context, run *fsm.FSMRun) error {
	if run == nil || run.ChatID == "" {
		return nil
	}
	if strings.HasPrefix(run.ID, "sched_") {
		return nil
	}

	var status agentapi.RunStatus
	switch run.Status {
	case fsm.RunStatusCompleted:
		status = agentapi.RunCompleted
	case fsm.RunStatusFailed:
		status = agentapi.RunFailed
	case fsm.RunStatusTerminated:
		status = agentapi.RunTerminated
	default:
		return nil
	}

	var attachments []agentapi.Attachment
	if session, ok := tools.ChatSessionFromContext(ctx); ok {
		attachments = FromBesedkaAttachments(session.GetStagedAttachments())
	}
	fsmEng := g.FSMEngine()
	if len(attachments) == 0 && fsmEng != nil {
		restored, err := fsmEng.RestoreChatSessionContext(ctx, run)
		if err == nil {
			attachments = FromBesedkaAttachments(restored.GetStagedAttachments())
		}
	}

	comp := agentapi.Completion{
		Run: agentapi.RunDescriptor{
			RunID: run.ID,
			Session: agentapi.SessionRef{
				FrontendID: "besedka",
				SessionID:  run.ChatID,
				ScopeID:    run.ScopeID,
			},
			Actor: agentapi.Actor{
				ID: run.UserID,
			},
			Model:         run.Model,
			MaxIterations: run.MaxIterations,
		},
		Result: agentapi.Result{
			RunID:       run.ID,
			Content:     run.ResultJSON,
			Iterations:  run.Iteration,
			Attachments: attachments,
		},
		Status: status,
		Error:  run.ErrorText,
	}

	return g.Deliver(ctx, comp)
}

type fsmResultSink struct {
	gateway *Gateway
}

var _ fsm.ResultSink = (*fsmResultSink)(nil)

func (s *fsmResultSink) Deliver(ctx context.Context, run *fsm.FSMRun) error {
	return s.gateway.DeliverLegacyFSMRun(ctx, run)
}

type gatewayNotificationSink struct {
	gateway *Gateway
	chatID  string
}

func (s *gatewayNotificationSink) Notify(ctx context.Context, notif agentapi.Notification) error {
	return s.gateway.SendMessage(s.chatID, notif.Content)
}

// SandboxManager returns the Gateway's sandbox Manager.
func (g *Gateway) SandboxManager() *sandbox.Manager {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.sandboxManager
}

// SetSandboxManager sets the sandbox Manager for the gateway and updates the tools registry.
func (g *Gateway) SetSandboxManager(sm *sandbox.Manager) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.sandboxManager = sm
	if g.toolsRegistry != nil {
		g.toolsRegistry.SetSandboxManager(sm)
	}
	if g.schedulerInvoker != nil {
		g.schedulerInvoker.SetSandbox(sm)
	}
}

// SchedulerEngine returns the Gateway's scheduler Engine.
func (g *Gateway) SchedulerEngine() *scheduler.Engine {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.schedulerEngine
}

// SetSchedulerEngine sets the scheduler Engine for the gateway.
func (g *Gateway) SetSchedulerEngine(e *scheduler.Engine) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.schedulerEngine = e
}

// SchedulerInvoker returns the Gateway's ScheduleInvoker.
func (g *Gateway) SchedulerInvoker() *ScheduleInvoker {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.schedulerInvoker
}

// ChatLocker returns the Gateway's ChatLocker.
func (g *Gateway) ChatLocker() *ChatLocker {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.chatLocker
}

// KnowledgeWorker returns the Gateway's background knowledge worker.
func (g *Gateway) KnowledgeWorker() *KnowledgeWorker {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.knowledgeWorker
}

// SetKnowledgeWorker sets the background knowledge worker for the gateway.
func (g *Gateway) SetKnowledgeWorker(w *KnowledgeWorker) {
	g.mu.Lock()
	oldKW := g.knowledgeWorker
	g.knowledgeWorker = w
	g.mu.Unlock()

	if oldKW != nil {
		oldKW.Stop()
	}

	if w != nil {
		g.mu.Lock()
		if g.running && g.knowledgeWorker == w && g.lifecycleCtx != nil && g.lifecycleCtx.Err() == nil {
			w.Start(g.lifecycleCtx)
		}
		g.mu.Unlock()
	}
}

var (
	brTagRe        = regexp.MustCompile(`(?i)<br\s*/?>`)
	blockEndTagRe  = regexp.MustCompile(`(?i)</(p|div|li|tr|h[1-6]|blockquote)>`)
	htmlTagRe      = regexp.MustCompile(`<[^>]*>`)
	multiNewlineRe = regexp.MustCompile(`\n{3,}`)
)

// ExtractMessageText extracts clean plain text / markdown from a message.
// It uses RawContent when present, or sanitizes and formats HTML content.
func ExtractMessageText(msg models.Message) string {
	if strings.TrimSpace(msg.RawContent) != "" {
		return strings.TrimSpace(msg.RawContent)
	}
	return StripHTML(msg.Content)
}

// StripHTML converts HTML to clean plaintext with preserved line breaks and unescaped entities.
func StripHTML(input string) string {
	if input == "" {
		return ""
	}
	// 1. Replace <br> tags with newlines
	text := brTagRe.ReplaceAllString(input, "\n")
	// 2. Replace block closing tags with double newlines
	text = blockEndTagRe.ReplaceAllString(text, "\n\n")
	// 3. Strip remaining HTML tags
	text = htmlTagRe.ReplaceAllString(text, "")
	// 4. Unescape HTML entities (&amp;, &lt;, &gt;, &quot;, &#39;, etc.)
	text = html.UnescapeString(text)
	// 5. Normalize consecutive newlines
	text = multiNewlineRe.ReplaceAllString(text, "\n\n")
	return strings.TrimSpace(text)
}

func normalizeForDedup(s string) string {
	return strings.Join(strings.Fields(s), " ")
}

var textExtensions = map[string]bool{
	".txt":  true,
	".md":   true,
	".json": true,
	".yaml": true,
	".yml":  true,
	".go":   true,
	".py":   true,
	".js":   true,
	".ts":   true,
	".jsx":  true,
	".tsx":  true,
	".html": true,
	".css":  true,
	".scss": true,
	".sh":   true,
	".bash": true,
	".csv":  true,
	".sql":  true,
	".xml":  true,
	".toml": true,
	".rs":   true,
	".c":    true,
	".cpp":  true,
	".h":    true,
	".hpp":  true,
	".java": true,
	".env":  true,
	".log":  true,
	".conf": true,
	".ini":  true,
}

func isTextMimeOrExt(name, mimeType string) bool {
	mime := strings.ToLower(strings.TrimSpace(mimeType))
	if strings.HasPrefix(mime, "text/") ||
		mime == "application/json" ||
		mime == "application/xml" ||
		mime == "application/x-yaml" ||
		mime == "application/yaml" ||
		mime == "application/javascript" ||
		mime == "application/x-javascript" ||
		mime == "application/typescript" {
		return true
	}

	ext := strings.ToLower(filepath.Ext(name))
	return textExtensions[ext]
}

func (g *Gateway) processAttachments(ctx context.Context, attachments []models.Attachment) (string, []chatcontext.ImageAttachment) {
	if len(attachments) == 0 {
		return "", nil
	}

	var extraText strings.Builder
	var images []chatcontext.ImageAttachment

	for _, att := range attachments {
		attName := strings.TrimSpace(att.Name)
		if attName == "" {
			attName = "attachment"
		}
		fileID := strings.TrimSpace(att.FileID)

		isImg := att.Type == models.AttachmentTypeImage || strings.HasPrefix(strings.ToLower(att.MimeType), "image/")
		if isImg {
			mimeStr := att.MimeType
			data, mime, err := g.FetchImageThumbnail(ctx, att.FileID)
			if err != nil {
				slog.Warn("failed to fetch image thumbnail", "fileID", att.FileID, "error", err)
				if mimeStr == "" {
					mimeStr = "image"
				}
				fmt.Fprintf(&extraText, "\n\n[Attachment: %s (id: %s, type: %s)] (failed to download thumbnail)", attName, fileID, mimeStr)
				continue
			}
			encoded := base64.StdEncoding.EncodeToString(data)
			images = append(images, chatcontext.ImageAttachment{
				URL: fmt.Sprintf("data:%s;base64,%s", mime, encoded),
			})
			if mimeStr == "" {
				mimeStr = mime
			}
			if mimeStr == "" {
				mimeStr = "image"
			}
			fmt.Fprintf(&extraText, "\n\n[Attachment: %s (id: %s, type: %s)]", attName, fileID, mimeStr)
			continue
		}

		mimeStr := att.MimeType
		if mimeStr == "" {
			mimeStr = "application/octet-stream"
		}

		if isTextMimeOrExt(att.Name, att.MimeType) {
			data, detectedMime, err := g.FetchFileContent(ctx, att.FileID, 16384)
			if err != nil {
				slog.Warn("failed to fetch file attachment", "fileID", att.FileID, "error", err)
				fmt.Fprintf(&extraText, "\n\n[Attachment: %s (id: %s, type: %s)] (failed to download)", attName, fileID, mimeStr)
				continue
			}
			if detectedMime != "" && mimeStr == "application/octet-stream" {
				mimeStr = detectedMime
			}
			if utf8.Valid(data) {
				fmt.Fprintf(&extraText, "\n\n[Attachment: %s (id: %s, type: %s)]:\n```\n%s\n```", attName, fileID, mimeStr, string(data))
			} else {
				fmt.Fprintf(&extraText, "\n\n[Attachment: %s (id: %s, type: %s)] (binary content not displayed)", attName, fileID, mimeStr)
			}
			continue
		}

		// Other binary file
		fmt.Fprintf(&extraText, "\n\n[Attachment: %s (id: %s, type: %s)]", attName, fileID, mimeStr)
	}

	return extraText.String(), images
}

// ProcessAttachments processes attachments to extract text representations and images for context storage.
func (g *Gateway) ProcessAttachments(ctx context.Context, attachments []models.Attachment) (string, []chatcontext.ImageAttachment) {
	return g.processAttachments(ctx, attachments)
}

// IsMentionedOrDM checks if a message should be handled by the bot based on mention or verified DM status.
func IsMentionedOrDM(handle string, isDM bool, content string) (bool, string) {
	plainText := StripHTML(content)
	cleanHandle := strings.TrimPrefix(handle, "@")

	// Match handle case-insensitively
	re := regexp.MustCompile(`(?i)@` + regexp.QuoteMeta(cleanHandle) + `[:,]?`)
	hasMention := re.MatchString(plainText)

	if hasMention {
		promptText := re.ReplaceAllString(plainText, "")
		return true, strings.TrimSpace(promptText)
	}

	if isDM {
		return true, strings.TrimSpace(plainText)
	}

	return false, ""
}

// FormatResponse prepares the LLM reply content for transmission.
// Advisory paragraph limits are guided via system prompts rather than hard truncation.
func FormatResponse(content string, isDM bool, maxTownhallParas, maxDMParas int) string {
	return strings.TrimSpace(content)
}

// DialWebSocket connects to the Besedka chat WebSocket endpoint.
func (g *Gateway) DialWebSocket(ctx context.Context) error {
	u, err := url.Parse(g.cfg.BesedkaURL)
	if err != nil {
		return fmt.Errorf("invalid Besedka URL: %w", err)
	}

	wsScheme := "ws"
	if u.Scheme == "https" {
		wsScheme = "wss"
	}
	wsURL := fmt.Sprintf("%s://%s/api/chat", wsScheme, u.Host)

	header := http.Header{}
	if g.cfg.BesedkaAPIKey != "" {
		header.Set("Authorization", "Bearer "+g.cfg.BesedkaAPIKey)
	}

	dialer := websocket.Dialer{
		HandshakeTimeout: 10 * time.Second,
	}

	conn, resp, err := dialer.DialContext(ctx, wsURL, header)
	if err != nil {
		if resp != nil {
			return fmt.Errorf("failed to dial websocket (status %d): %w", resp.StatusCode, err)
		}
		return fmt.Errorf("failed to dial websocket: %w", err)
	}

	g.mu.Lock()
	g.conn = conn
	g.mu.Unlock()

	return nil
}

// SendMessage sends a response message back to Besedka.
func (g *Gateway) SendMessage(chatID, content string) error {
	return g.SendMessageWithAttachments(chatID, content, nil)
}

// SendMessageWithAttachments sends a response message with optional attachments back to Besedka.
func (g *Gateway) SendMessageWithAttachments(chatID, content string, attachments []models.Attachment) error {
	g.mu.Lock()
	conn := g.conn
	g.mu.Unlock()

	if conn == nil {
		return errors.New("websocket connection is not established")
	}

	clientMsg := models.ClientMessage{
		Type:        models.ClientMessageTypeSend,
		ChatID:      chatID,
		Content:     content,
		Attachments: attachments,
	}

	g.mu.Lock()
	defer g.mu.Unlock()
	return conn.WriteJSON(clientMsg)
}

func isLegacyProgressMessage(content string) bool {
	clean := strings.TrimSpace(content)
	return strings.HasPrefix(clean, "⏳ ") && !strings.Contains(clean, "Sandbox Request")
}

// FormatUserContext formats the user's timezone and preferred language into a system context note.
// FormatUserContextFor formats the user's timezone and preferred language into a system context message,
// optionally attributing it to a specific user tag (e.g. "@alice") in group/townhall chats.
// Returns an empty string if neither field is set.
func FormatUserContextFor(userTag, timeZone, preferredLanguage string) string {
	tz := strings.TrimSpace(timeZone)
	lang := strings.TrimSpace(preferredLanguage)
	if tz == "" && lang == "" {
		return ""
	}
	userTag = strings.TrimSpace(userTag)
	prefix := "[User context"
	if userTag != "" {
		if !strings.HasPrefix(userTag, "@") {
			userTag = "@" + userTag
		}
		prefix = fmt.Sprintf("[User context for %s", userTag)
	}

	if tz != "" && lang != "" {
		return fmt.Sprintf("%s: timezone=%s, language=%s]", prefix, tz, lang)
	}
	if tz != "" {
		return fmt.Sprintf("%s: timezone=%s]", prefix, tz)
	}
	return fmt.Sprintf("%s: language=%s]", prefix, lang)
}

// FormatUserContext formats the user's timezone and preferred language into a system context message.
// Returns an empty string if neither field is set.
func FormatUserContext(timeZone, preferredLanguage string) string {
	return FormatUserContextFor("", timeZone, preferredLanguage)
}

func (g *Gateway) getUserContextInjected() *geche.MapCache[string, bool] {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.userContextInjected == nil {
		g.userContextInjected = geche.NewMapCache[string, bool]()
	}
	return g.userContextInjected
}

func (g *Gateway) injectUserContextOnce(ctx context.Context, chatID, userID string, rb *chatcontext.RingBuffer, isDM bool) {
	if chatID == "" || userID == "" || rb == nil || g.userCache == nil {
		return
	}
	injectedKey := chatID
	if !isDM {
		injectedKey = chatID + ":" + userID
	}
	injectedMap := g.getUserContextInjected()
	if _, err := injectedMap.Get(injectedKey); err == nil {
		return
	}

	user, ok := g.userCache.Get(userID)
	if !ok || (user.TimeZone == "" && user.PreferredLanguage == "") {
		if users, err := g.FetchUsers(ctx); err == nil && len(users) > 0 {
			user, ok = g.userCache.Get(userID)
		}
	}
	if !ok {
		return
	}

	var userTag string
	if !isDM {
		userTag = user.GetUserName()
		if userTag == "" {
			userTag = user.GetDisplayName()
		}
	}

	contextText := FormatUserContextFor(userTag, user.TimeZone, user.PreferredLanguage)
	if contextText == "" {
		_, _ = injectedMap.SetIfAbsent(injectedKey, true)
		return
	}

	if _, stored := injectedMap.SetIfAbsent(injectedKey, true); !stored {
		return
	}

	rb.Push(chatcontext.Entry{
		Role:      "user",
		Content:   contextText,
		Timestamp: time.Now().Unix(),
	})
}

// ResetUserContextSession resets the session injection state for a chat.
func (g *Gateway) ResetUserContextSession(chatID string) {
	injectedMap := g.getUserContextInjected()
	_ = injectedMap.Del(chatID)
	for k := range injectedMap.Snapshot() {
		if strings.HasPrefix(k, chatID+":") {
			_ = injectedMap.Del(k)
		}
	}
}

// SetLocation sets the server location for periodic location reporting.
func (g *Gateway) SetLocation(loc *models.Location) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.location = loc
}

// Location returns the current server location configured on the gateway.
func (g *Gateway) Location() *models.Location {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.location
}

// SetLocationInterval sets the interval for periodic location updates.
func (g *Gateway) SetLocationInterval(d time.Duration) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.locationInterval = d
}

// SetInitialLocationDelay sets the delay before sending the first location frame after connect.
func (g *Gateway) SetInitialLocationDelay(d time.Duration) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.initialLocationDelay = d
}

// SendLocation sends a location update frame to Besedka.
func (g *Gateway) SendLocation(loc *models.Location) error {
	if loc == nil {
		return errors.New("location is nil")
	}

	g.mu.Lock()
	conn := g.conn
	g.mu.Unlock()

	if conn == nil {
		return errors.New("websocket connection is not established")
	}

	clientMsg := models.ClientMessage{
		Type:     models.ClientMessageTypeLocation,
		Location: loc,
	}

	g.mu.Lock()
	defer g.mu.Unlock()
	return conn.WriteJSON(clientMsg)
}

// SendPong sends an application-level pong heartbeat frame to Besedka.
func (g *Gateway) SendPong() error {
	g.mu.Lock()
	conn := g.conn
	g.mu.Unlock()

	if conn == nil {
		return errors.New("websocket connection is not established")
	}

	clientMsg := models.ClientMessage{
		Type: models.ClientMessageTypePong,
	}

	g.mu.Lock()
	defer g.mu.Unlock()
	return conn.WriteJSON(clientMsg)
}

func (g *Gateway) handleEvictedBatch(chatID string, evicted []chatcontext.Entry) {
	g.mu.Lock()
	memMgr := g.memoryManager
	g.mu.Unlock()

	if memMgr == nil || len(evicted) == 0 {
		return
	}
	isDM := chatID != "townhall"
	msgs := make([]memory.MessageToStore, 0, len(evicted))
	for _, e := range evicted {
		if e.Role == "system" || e.Seq <= 0 {
			continue
		}
		msgs = append(msgs, memory.MessageToStore{
			Seq:        e.Seq,
			Timestamp:  e.Timestamp,
			ChatID:     chatID,
			UserID:     e.SenderID,
			SenderName: e.SenderName,
			Role:       e.Role,
			Content:    e.Content,
		})
	}

	g.indexingWg.Go(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
		defer cancel()
		if err := memMgr.IndexMessages(ctx, chatID, isDM, msgs); err != nil {
			slog.Warn("async indexing of evicted batch failed", "chatID", chatID, "error", err)
		}
	})
}

// CatchupChatMemory fetches historical messages from Besedka in batches of up to 100 to catch up missing memory indexes.
func (g *Gateway) CatchupChatMemory(ctx context.Context, chatID string, isDM bool, latestSeq int64) {
	g.mu.Lock()
	memMgr := g.memoryManager
	g.mu.Unlock()

	if memMgr == nil || latestSeq <= 0 {
		return
	}

	lastIndexedSeq, err := memMgr.GetWatermark(ctx, chatID, isDM)
	if err != nil {
		slog.Debug("could not get watermark for catchup", "chatID", chatID, "error", err)
		return
	}

	if latestSeq <= lastIndexedSeq {
		return
	}

	const batchSize int64 = 100
	fromSeq := lastIndexedSeq + 1

	for fromSeq <= latestSeq {
		toSeq := fromSeq + batchSize - 1
		if toSeq > latestSeq {
			toSeq = latestSeq
		}

		fetchedMsgs, err := g.FetchChatMessages(ctx, chatID, fromSeq, toSeq)
		if err != nil {
			slog.Warn("failed to fetch historical messages for memory catchup", "chatID", chatID, "fromSeq", fromSeq, "toSeq", toSeq, "error", err)
			break
		}
		if len(fetchedMsgs) == 0 {
			break
		}

		g.mu.Lock()
		botID := g.botUserID
		botName := g.botUser.GetDisplayName()
		g.mu.Unlock()

		toStore := make([]memory.MessageToStore, 0, len(fetchedMsgs))
		for _, m := range fetchedMsgs {
			if m.Type == models.MessageTypeProgress {
				continue
			}
			cleanContent := ExtractMessageText(m)
			extraText, _ := g.processAttachments(ctx, m.Attachments)
			fullContent := strings.TrimSpace(cleanContent + extraText)
			if fullContent == "" || (botID != "" && m.UserID == botID && isLegacyProgressMessage(fullContent)) {
				continue
			}

			senderName := g.userCache.GetDisplayName(m.UserID)
			role := "user"
			if botID != "" && m.UserID == botID {
				role = "assistant"
				senderName = botName
			}

			toStore = append(toStore, memory.MessageToStore{
				Seq:        m.Seq,
				Timestamp:  m.Timestamp,
				ChatID:     chatID,
				UserID:     m.UserID,
				SenderName: senderName,
				Role:       role,
				Content:    fullContent,
			})
		}

		if len(toStore) > 0 {
			if err := memMgr.IndexMessages(ctx, chatID, isDM, toStore); err != nil {
				slog.Warn("failed to index historical batch during memory catchup", "chatID", chatID, "error", err)
				break
			}
			lastReturnedSeq := fetchedMsgs[len(fetchedMsgs)-1].Seq
			if lastReturnedSeq > toStore[len(toStore)-1].Seq {
				if err := memMgr.SetWatermark(ctx, chatID, isDM, lastReturnedSeq); err != nil {
					slog.Warn("failed to advance watermark for skipped messages during memory catchup", "chatID", chatID, "error", err)
				}
			}
		} else if len(fetchedMsgs) > 0 {
			lastReturnedSeq := fetchedMsgs[len(fetchedMsgs)-1].Seq
			if err := memMgr.SetWatermark(ctx, chatID, isDM, lastReturnedSeq); err != nil {
				slog.Warn("failed to advance watermark during memory catchup", "chatID", chatID, "error", err)
			}
		}

		lastReturnedSeq := fetchedMsgs[len(fetchedMsgs)-1].Seq
		if lastReturnedSeq >= fromSeq {
			fromSeq = lastReturnedSeq + 1
		} else {
			fromSeq = toSeq + 1
		}
	}
}

// WarmupChat loads historical messages for a single chat into its ring buffer and performs memory catchup.
func (g *Gateway) WarmupChat(ctx context.Context, chatID string, lastSeq int64) {
	if chatID == "" {
		return
	}

	isDM := chatID != "townhall"
	g.CatchupChatMemory(ctx, chatID, isDM, lastSeq)

	g.mu.Lock()
	botID := g.botUserID
	botName := g.botUser.GetDisplayName()
	g.mu.Unlock()

	limit := int64(g.cfg.MsgRingBufferSize)
	if limit <= 0 {
		limit = 100
	}

	toSeq := lastSeq
	fromSeq := int64(1)
	if toSeq > 0 {
		fromSeq = max(1, toSeq-limit+1)
	} else {
		toSeq = 1000000
	}

	msgs, err := g.FetchChatMessages(ctx, chatID, fromSeq, toSeq)
	if err != nil {
		slog.Debug("could not fetch messages for chat warmup", "chatID", chatID, "error", err)
		return
	}

	rb := g.contextManager.GetOrCreate(chatID)
	rb.Clear()
	g.ResetUserContextSession(chatID)

	for _, m := range msgs {
		if m.Type == models.MessageTypeProgress {
			continue
		}
		cleanContent := ExtractMessageText(m)
		extraText, images := g.processAttachments(ctx, m.Attachments)
		fullContent := strings.TrimSpace(cleanContent + extraText)
		if (fullContent == "" && len(images) == 0) || (botID != "" && m.UserID == botID && isLegacyProgressMessage(fullContent)) {
			continue
		}

		if botID != "" && m.UserID == botID {
			rb.Push(chatcontext.Entry{
				Seq:        m.Seq,
				Role:       "assistant",
				SenderID:   m.UserID,
				SenderName: botName,
				Content:    fullContent,
				Images:     images,
				Timestamp:  m.Timestamp,
			})
		} else {
			senderName := g.userCache.GetDisplayName(m.UserID)
			rb.Push(chatcontext.Entry{
				Seq:        m.Seq,
				Role:       "user",
				SenderID:   m.UserID,
				SenderName: senderName,
				Content:    fullContent,
				Images:     images,
				Timestamp:  m.Timestamp,
			})
		}
	}
}

// WarmupContext pre-populates metadata and chat history without triggering LLM responses.
func (g *Gateway) WarmupContext(ctx context.Context) error {
	var err error
	for attempt := 1; attempt <= 3; attempt++ {
		if _, err = g.FetchBotUser(ctx); err == nil {
			break
		}
		slog.Warn("retrying bot user fetch during warmup", "attempt", attempt, "error", err)
		time.Sleep(time.Duration(attempt) * 200 * time.Millisecond)
	}

	for attempt := 1; attempt <= 3; attempt++ {
		if _, err = g.FetchUsers(ctx); err == nil {
			break
		}
		slog.Warn("retrying users fetch during warmup", "attempt", attempt, "error", err)
		time.Sleep(time.Duration(attempt) * 200 * time.Millisecond)
	}

	var chats []models.Chat
	for attempt := 1; attempt <= 3; attempt++ {
		if chats, err = g.FetchChats(ctx); err == nil {
			break
		}
		slog.Warn("retrying chats fetch during warmup", "attempt", attempt, "error", err)
		time.Sleep(time.Duration(attempt) * 200 * time.Millisecond)
	}

	hasTownhall := false
	for _, c := range chats {
		if c.ID == "townhall" {
			hasTownhall = true
			break
		}
	}
	if !hasTownhall {
		chats = append(chats, models.Chat{ID: "townhall"})
	}

	for _, chat := range chats {
		if chat.ID != "" {
			g.WarmupChat(ctx, chat.ID, int64(chat.LastSeq))
		}
	}
	g.chatCache.SetAll(chats)

	slog.Info("completed context warmup for active chats", "chatCount", len(chats))
	return nil
}

// ProcessMessage handles a single incoming message from Besedka.
func (g *Gateway) ProcessMessage(ctx context.Context, msg models.Message) error {
	ctx, cancel := context.WithTimeout(ctx, 10*time.Minute)
	defer cancel()

	if msg.ChatID == "" {
		msg.ChatID = "townhall"
	}

	if msg.Type == models.MessageTypeProgress {
		return nil
	}

	cleanContent := ExtractMessageText(msg)
	if cleanContent == "" && len(msg.Attachments) == 0 {
		return nil
	}

	extraText, images := g.processAttachments(ctx, msg.Attachments)
	fullContent := strings.TrimSpace(cleanContent + extraText)
	if fullContent == "" && len(images) == 0 {
		return nil
	}

	g.mu.Lock()
	botID := g.botUserID
	botUser := g.botUser
	startTime := g.startTime
	g.mu.Unlock()

	// Ensure botUserID is populated if missing
	if botID == "" {
		if u, err := g.FetchBotUser(ctx); err == nil && u != nil {
			g.mu.Lock()
			botID = u.ID
			botUser = *u
			g.mu.Unlock()
		}
	}

	// 1. Handle self-messages from the bot itself (Townhall and DM)
	if botID != "" && msg.UserID == botID {
		if isLegacyProgressMessage(fullContent) {
			return nil
		}
		entries := g.contextManager.GetOrCreate(msg.ChatID).Entries()
		// Deduplicate if already appended on outgoing SendMessage
		isDuplicate := false
		if len(entries) > 0 {
			lastEntry := entries[len(entries)-1]
			if lastEntry.Role == "assistant" && normalizeForDedup(lastEntry.Content) == normalizeForDedup(fullContent) {
				isDuplicate = true
			}
		}
		if !isDuplicate {
			g.contextManager.Push(msg.ChatID, chatcontext.Entry{
				Seq:        msg.Seq,
				Role:       "assistant",
				SenderID:   msg.UserID,
				SenderName: botUser.GetDisplayName(),
				Content:    fullContent,
				Images:     images,
				Timestamp:  msg.Timestamp,
			})
		}
		return nil // Never trigger LLM response for self-messages
	}

	// 2. Ignore messages older than bot start time (avoid backfill reprocessing via WS)
	if msg.Timestamp > 0 && msg.Timestamp < startTime.Unix()-5 {
		return nil
	}

	// 3. Resolve user display name (with dynamic cache refresh on miss)
	senderName := g.userCache.GetDisplayName(msg.UserID)
	if senderName == "" {
		if users, err := g.FetchUsers(ctx); err == nil && len(users) > 0 {
			senderName = g.userCache.GetDisplayName(msg.UserID)
		}
	}

	// 4. Determine trigger condition
	// Determine DM status strictly from authoritative metadata before checking IsMentionedOrDM.
	// On lookup or verification failure, fail closed and treat as public/group chat behavior.
	isChatDM := false
	if msg.ChatID != "townhall" && strings.TrimSpace(msg.ChatID) != "" {
		var chat models.Chat
		var ok bool
		if g.chatCache != nil {
			chat, ok = g.chatCache.Get(msg.ChatID)
		}
		if !ok && g.httpClient != nil {
			var err error
			chat, err = g.GetChat(ctx, msg.ChatID)
			if err != nil {
				slog.Debug("could not fetch chat metadata, defaulting to group chat behavior", "chatID", msg.ChatID, "error", err)
			}
		}
		if chat.ID != "" && (chat.IsDM || chat.Type == "dm") && chat.Type != "group" {
			isChatDM = true
		}
	}

	botHandle := "@bot"
	if g.cfg != nil && g.cfg.BotHandle != "" {
		botHandle = g.cfg.BotHandle
	}

	shouldProcess, promptText := IsMentionedOrDM(botHandle, isChatDM, msg.Content)
	if !shouldProcess && msg.ChatID == "townhall" {
		if strings.EqualFold(botHandle, "@bob") {
			shouldProcess, promptText = IsMentionedOrDM("@bot", isChatDM, msg.Content)
		} else if strings.EqualFold(botHandle, "@bot") {
			shouldProcess, promptText = IsMentionedOrDM("@bob", isChatDM, msg.Content)
		}
	}

	cleanText := strings.TrimSpace(fullContent)
	cleanText = strings.Trim(cleanText, "`")
	cleanText = strings.TrimSpace(cleanText)

	fields := strings.Fields(cleanText)
	var rootCommand string
	if len(fields) > 0 {
		rootCommand = fields[0]
	}

	promptFields := strings.Fields(strings.TrimSpace(promptText))
	var promptRootCommand string
	if len(promptFields) > 0 {
		promptRootCommand = promptFields[0]
	}

	// Always trigger on bot slash commands (even in group chats or townhall) so appropriate rejection/guidance is sent
	cmdToCheck := strings.ToLower(rootCommand)
	if cmdToCheck == "" {
		cmdToCheck = strings.ToLower(promptRootCommand)
	}
	if cmdToCheck == "/memory" || cmdToCheck == "/skill" || cmdToCheck == "/schedule" || cmdToCheck == "/sandbox" {
		shouldProcess = true
		if promptText == "" {
			promptText = cleanText
		}
	}

	// 5. Append incoming user message to ring buffer (backfill chat history if buffer was uninitialized)
	rb := g.contextManager.GetOrCreate(msg.ChatID)
	if rb.Len() == 0 && msg.Seq > 1 {
		g.WarmupChat(ctx, msg.ChatID, msg.Seq-1)
	}

	if shouldProcess {
		g.injectUserContextOnce(ctx, msg.ChatID, msg.UserID, rb, isChatDM)
	}

	rb.Push(chatcontext.Entry{
		Seq:        msg.Seq,
		Role:       "user",
		SenderID:   msg.UserID,
		SenderName: senderName,
		Content:    fullContent,
		Images:     images,
		Timestamp:  msg.Timestamp,
	})

	if !shouldProcess {
		return nil
	}

	if isChatDM && strings.EqualFold(rootCommand, "/sandbox") {
		return g.handleSandboxCommand(ctx, msg, cleanText, senderName)
	}

	scheduleCmdText := cleanText
	if strings.EqualFold(promptRootCommand, "/schedule") {
		scheduleCmdText = strings.TrimSpace(promptText)
		rootCommand = promptRootCommand
	}
	if strings.EqualFold(rootCommand, "/schedule") {
		return g.handleScheduleCommand(ctx, msg, scheduleCmdText, senderName, isChatDM)
	}

	memoryCmdText := cleanText
	if strings.EqualFold(promptRootCommand, "/memory") {
		memoryCmdText = strings.TrimSpace(promptText)
		rootCommand = promptRootCommand
	}
	if strings.EqualFold(rootCommand, "/memory") {
		return g.handleMemoryCommand(ctx, msg, memoryCmdText, senderName, isChatDM)
	}

	skillCmdText := cleanText
	if strings.EqualFold(promptRootCommand, "/skill") {
		skillCmdText = strings.TrimSpace(promptText)
		rootCommand = promptRootCommand
	}
	if strings.EqualFold(rootCommand, "/skill") {
		return g.handleSkillCommand(ctx, msg, skillCmdText, senderName, isChatDM)
	}

	return g.generateAndSendAgentReply(ctx, msg, isChatDM, senderName, "", 0)
}

func (g *Gateway) generateAndSendAgentReply(ctx context.Context, msg models.Message, isDM bool, senderName, currentTask string, initialProgressSeq int64) error {
	g.mu.Lock()
	runner := g.runner
	botID := g.botUserID
	botUser := g.botUser
	if botUser.ID == "" && botID != "" {
		if u, ok := g.userCache.Get(botID); ok {
			botUser = u
		}
	}
	toolsRegistry := g.toolsRegistry
	sm := g.sandboxManager
	g.mu.Unlock()

	if runner == nil && g.llmClient == nil {
		if initialProgressSeq > 0 {
			_, _ = g.SendProgressMessage(ctx, msg.ChatID, &models.ProgressData{
				ParentSeq:  initialProgressSeq,
				CardStatus: models.ProgressStatusCompleted,
			})
		}
		return nil
	}

	if runner == nil && g.chatLocker != nil {
		release, err := g.chatLocker.TryAcquire(ctx, msg.ChatID, 30*time.Second)
		if err != nil {
			slog.Warn("chat execution lock busy for incoming message", "chat_id", msg.ChatID, "error", err)
			return err
		}
		defer release()
	}

	var systemPrompt string
	if isDM {
		targetUser, ok := g.userCache.Get(msg.UserID)
		if !ok || targetUser.GetDisplayName() == "" {
			targetUser = models.User{ID: msg.UserID, DisplayName: senderName, UserName: senderName}
		}
		systemPrompt = prompt.RenderDMPrompt(botUser, g.cfg.BotHandle, targetUser, g.cfg.DMMaxParagraphs)
	} else {
		systemPrompt = prompt.RenderTownhallPrompt(botUser, g.cfg.BotHandle, g.cfg.TownhallMaxParagraphs)
	}

	slog.Info("processing bot message request", "chatID", msg.ChatID, "sender", senderName)

	bufferedMsgs := g.contextManager.GetLLMMessages(msg.ChatID)
	convMsgs := make([]openai.ChatCompletionMessage, 0, len(bufferedMsgs)+1)
	convMsgs = append(convMsgs, bufferedMsgs...)

	// Defense-in-depth: Ensure message list never ends with an assistant message (e.g. Gemini 400 constraint)
	if len(convMsgs) > 0 && convMsgs[len(convMsgs)-1].Role == openai.ChatMessageRoleAssistant {
		continuation := "Please continue."
		if currentTask != "" {
			continuation = "Please proceed with: " + currentTask
		}
		convMsgs = append(convMsgs, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: continuation,
		})
	}

	if runner != nil {
		maxIterations := g.cfg.TownhallToolMaxIterations
		if isDM {
			maxIterations = g.cfg.DMToolMaxIterations
		}
		metadata := make(map[string]string)
		if initialProgressSeq > 0 {
			metadata["initial_progress_seq"] = strconv.FormatInt(initialProgressSeq, 10)
		}
		if msg.Seq > 0 {
			metadata["source_message_seq"] = strconv.FormatInt(msg.Seq, 10)
		}

		turn := agentapi.Turn{
			Run: agentapi.RunDescriptor{
				RunID: fmt.Sprintf("run_%s_%d", msg.ChatID, time.Now().UnixNano()),
				Session: agentapi.SessionRef{
					FrontendID: "besedka",
					SessionID:  msg.ChatID,
					ScopeID:    msg.ChatID,
				},
				Actor: agentapi.Actor{
					ID: msg.UserID,
				},
				Kind:          agentapi.Interactive,
				Model:         g.cfg.OpenAIModel,
				MaxIterations: maxIterations,
				Metadata:      metadata,
			},
			SystemPrompt: systemPrompt,
			Messages:     convMsgs,
			Deliver:      true,
		}

		_, err := runner.Run(ctx, turn)
		if err != nil {
			var delivErr *agent.DeliveryError
			if errors.As(err, &delivErr) {
				slog.Error("failed delivering agent reply", "chat_id", msg.ChatID, "error", delivErr.Err)
				return delivErr.Err
			}
			slog.Error("turn runner execution failed", "chat_id", msg.ChatID, "error", err)
			apology := "Sorry, I encountered an issue processing your request. Please try again later."
			_ = g.SendMessage(msg.ChatID, apology)
			return err
		}
		return nil
	}

	llmMsgs := make([]openai.ChatCompletionMessage, 0, len(convMsgs)+1)
	llmMsgs = append(llmMsgs, openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleSystem,
		Content: systemPrompt,
	})
	llmMsgs = append(llmMsgs, convMsgs...)

	var sandboxRequestCreated bool
	var budgetLimits tools.KnowledgeBudgetLimits
	if toolsRegistry != nil {
		budgetLimits = toolsRegistry.KnowledgeLimits()
	} else if g.cfg != nil {
		budgetLimits = tools.KnowledgeBudgetLimits{
			MaxLoadedMemories:    g.cfg.KnowledgeMaxLoadedMemories,
			MaxLoadedMemoryBytes: g.cfg.KnowledgeMaxLoadedMemoryBytes,
			MaxLoadedSkills:      g.cfg.KnowledgeMaxLoadedSkills,
			MaxLoadedSkillBytes:  g.cfg.KnowledgeMaxLoadedSkillBytes,
		}
	} else {
		budgetLimits = tools.DefaultKnowledgeBudgetLimits()
	}
	// Authoritative verification for knowledge and sandbox capabilities:
	// A session is granted DM privileges only if it is an authoritative 1-on-1 DM owned by msg.UserID.
	isAuthorizedDMOwner := isDM && g.VerifyDMOwner(ctx, msg.ChatID, msg.UserID) == nil

	sessionCtx := tools.NewChatSessionContext(msg.ChatID, msg.UserID, isAuthorizedDMOwner, budgetLimits)
	sessionCtx.Notifier = func(chatID, text string) error {
		return g.SendMessage(chatID, text)
	}
	sessionCtx.SandboxRequestCreated = &sandboxRequestCreated

	var toolDefs []openai.Tool
	if toolsRegistry != nil {
		toolDefs = toolsRegistry.ToolDefinitionsForSession(sessionCtx)
	}

	var reply string
	var err error
	var fsmRes *fsm.ToolLoopResult
	if toolsRegistry != nil && len(toolDefs) > 0 {
		toolCtx := tools.WithChatSession(ctx, sessionCtx)
		maxIterations := g.cfg.TownhallToolMaxIterations
		if isDM {
			maxIterations = g.cfg.DMToolMaxIterations
		}

		g.mu.Lock()
		fsmEng := g.fsmEngine
		g.mu.Unlock()

		if fsmEng != nil {
			var seq *int64
			if msg.Seq > 0 {
				s := msg.Seq
				seq = &s
			}
			var progressObs fsm.ProgressObserver
			if initialProgressSeq > 0 {
				progressObs = NewGatewayProgressObserverWithRoot(g, msg.ChatID, isDM, initialProgressSeq)
			} else {
				progressObs = NewGatewayProgressObserver(g, msg.ChatID, isDM)
			}
			fsmReq := fsm.ToolLoopRequest{
				ChatID:           msg.ChatID,
				UserID:           msg.UserID,
				IsDM:             isDM,
				Model:            g.cfg.OpenAIModel,
				Messages:         llmMsgs,
				Toolset:          toolsRegistry.SessionToolset(sessionCtx),
				MaxIterations:    maxIterations,
				SourceMessageSeq: seq,
				ProgressObserver: progressObs,
			}
			fsmRes, err = fsmEng.RunToolLoop(toolCtx, fsmReq)
			if err != nil {
				if toolCtx.Err() != nil {
					// Context was cancelled or timed out; do not fall back
				} else {
					slog.Error("fsm tool loop failed, falling back to volatile loop", "chat_id", msg.ChatID, "error", err)
					reply, err = g.llmClient.GenerateChatResponseWithToolLoop(
						toolCtx,
						llmMsgs,
						toolDefs,
						toolsRegistry,
						maxIterations,
					)
				}
			} else if fsmRes != nil {
				reply = fsmRes.Content
				for _, act := range fsmRes.Actions {
					if act.Type == agentapi.ActionSuppressReply {
						sandboxRequestCreated = true
					}
				}
			}
		} else {
			reply, err = g.llmClient.GenerateChatResponseWithToolLoop(
				toolCtx,
				llmMsgs,
				toolDefs,
				toolsRegistry,
				maxIterations,
			)
			if initialProgressSeq > 0 {
				cardStatus := models.ProgressStatusCompleted
				if err != nil {
					cardStatus = models.ProgressStatusFailed
				}
				_, _ = g.SendProgressMessage(ctx, msg.ChatID, &models.ProgressData{
					ParentSeq:  initialProgressSeq,
					CardStatus: cardStatus,
				})
			}
		}
	} else {
		reply, err = g.llmClient.GenerateChatResponse(ctx, llmMsgs)
		if initialProgressSeq > 0 {
			cardStatus := models.ProgressStatusCompleted
			if err != nil {
				cardStatus = models.ProgressStatusFailed
			}
			_, _ = g.SendProgressMessage(ctx, msg.ChatID, &models.ProgressData{
				ParentSeq:  initialProgressSeq,
				CardStatus: cardStatus,
			})
		}
	}
	if err != nil {
		slog.Error("failed to generate LLM response", "error", err)
		reply = "Sorry, I encountered an issue processing your request. Please try again later."
	}

	if sandboxRequestCreated {
		g.contextManager.Push(msg.ChatID, chatcontext.Entry{
			Role:       "assistant",
			SenderID:   botID,
			SenderName: botUser.GetDisplayName(),
			Content:    "Sandbox approval requested.",
			Timestamp:  time.Now().Unix(),
		})
		return nil
	}

	if err == nil && isDM && sm != nil {
		if sbx, ok := sm.GetStatus(msg.UserID); ok && sbx != nil && sbx.Status == sandbox.StatusRunning {
			if !strings.Contains(reply, "/sandbox destroy") {
				rem := time.Until(sbx.ExpiresAt).Round(time.Minute)
				if rem < 0 {
					rem = 0
				}
				reply += fmt.Sprintf("\n\n💡 Would you like to destroy the sandbox (`/sandbox destroy`) or keep it? It will automatically be destroyed in %s.", rem)
			}
		}
	}

	formattedReply := FormatResponse(reply, isDM, g.cfg.TownhallMaxParagraphs, g.cfg.DMMaxParagraphs)

	var outgoingAttachments []models.Attachment
	if sessionCtx.StagedAttachments != nil {
		outgoingAttachments = sessionCtx.GetStagedAttachments()
	}
	if len(outgoingAttachments) == 0 && fsmRes != nil && len(fsmRes.Attachments) > 0 {
		outgoingAttachments = fsmRes.Attachments
	}

	if err := g.SendMessageWithAttachments(msg.ChatID, formattedReply, outgoingAttachments); err != nil {
		return fmt.Errorf("failed to send reply to chat %s: %w", msg.ChatID, err)
	}

	pushContent, pushImages := formattedReply, []chatcontext.ImageAttachment(nil)
	if len(outgoingAttachments) > 0 {
		extraText, images := g.processAttachments(ctx, outgoingAttachments)
		pushContent = strings.TrimSpace(formattedReply + extraText)
		pushImages = images
	}

	g.contextManager.Push(msg.ChatID, chatcontext.Entry{
		Role:       "assistant",
		SenderID:   botID,
		SenderName: botUser.GetDisplayName(),
		Content:    pushContent,
		Images:     pushImages,
		Timestamp:  time.Now().Unix(),
	})

	return nil
}

// Start listens for incoming WebSocket messages and processes them until context is cancelled.
func (g *Gateway) Start(ctx context.Context) error {
	g.mu.Lock()
	if g.running {
		g.mu.Unlock()
		return errors.New("gateway is already running")
	}
	g.running = true
	g.lifecycleID++
	curGen := g.lifecycleID
	g.lifecycleCtx = ctx
	fsmEng := g.fsmEngine
	schedEng := g.schedulerEngine
	kw := g.knowledgeWorker
	if kw != nil {
		kw.Start(ctx)
	}
	g.mu.Unlock()

	defer func() {
		g.mu.Lock()
		if g.lifecycleID == curGen {
			g.running = false
			g.lifecycleCtx = nil
		}
		g.mu.Unlock()
	}()

	// Start FSM engine background poller and recover interrupted runs
	if fsmEng != nil {
		if err := fsmEng.Start(ctx); err != nil {
			slog.Error("failed to start fsm engine", "error", err)
		}
	}

	// Start scheduler engine background poller
	if schedEng != nil {
		if err := schedEng.Start(ctx); err != nil {
			slog.Error("failed to start scheduler engine", "error", err)
		}
	}

	// Start periodic maintenance ticker (FSM retention pruning and vacuum)
	maintenanceDone := make(chan struct{})
	go g.startMaintenanceLoop(ctx, maintenanceDone)
	defer close(maintenanceDone)

	// Initial context warmup on startup
	if err := g.WarmupContext(ctx); err != nil {
		slog.Warn("context warmup encountered issues on startup", "error", err)
	}

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		if err := g.DialWebSocket(ctx); err != nil {
			slog.Error("websocket dial failed, retrying in 3s", "error", err)
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(3 * time.Second):
				continue
			}
		}

		slog.Info("connected to Besedka websocket gateway")

		// Re-run warmup upon establishing connection to ensure caches & histories are fresh
		go func() {
			if err := g.WarmupContext(ctx); err != nil {
				slog.Warn("warmup after websocket connect encountered issues", "error", err)
			}
		}()

		// WebSocket Ping/Pong keepalive ticker to prevent idle timeout
		pingDone := make(chan struct{})
		g.mu.Lock()
		activeConn := g.conn
		loc := g.location
		locInterval := g.locationInterval
		if locInterval <= 0 {
			locInterval = 9 * time.Minute
		}
		initDelay := g.initialLocationDelay
		if initDelay <= 0 {
			initDelay = 1 * time.Second
		}
		g.mu.Unlock()

		go func(c *websocket.Conn) {
			ticker := time.NewTicker(20 * time.Second)
			defer ticker.Stop()
			for {
				select {
				case <-pingDone:
					return
				case <-ctx.Done():
					return
				case <-ticker.C:
					g.mu.Lock()
					isSame := g.conn == c
					g.mu.Unlock()
					if isSame && c != nil {
						if err := g.SendPong(); err != nil {
							slog.Debug("failed to send periodic keepalive pong", "error", err)
						}
					}
				}
			}
		}(activeConn)

		// Periodic server location update loop (initial update soon after connect, then every 9m)
		locationDone := make(chan struct{})
		if loc != nil {
			go func(targetLoc *models.Location, interval, delay time.Duration) {
				select {
				case <-locationDone:
					return
				case <-ctx.Done():
					return
				case <-time.After(delay):
					if err := g.SendLocation(targetLoc); err != nil {
						slog.Warn("failed to send initial server location frame", "error", err)
					} else {
						slog.Info("sent initial server location frame", "lat", targetLoc.Lat, "lng", targetLoc.Lng)
					}
				}

				ticker := time.NewTicker(interval)
				defer ticker.Stop()

				for {
					select {
					case <-locationDone:
						return
					case <-ctx.Done():
						return
					case <-ticker.C:
						if err := g.SendLocation(targetLoc); err != nil {
							slog.Warn("failed to send periodic server location frame", "error", err)
						} else {
							slog.Debug("sent periodic server location frame", "lat", targetLoc.Lat, "lng", targetLoc.Lng)
						}
					}
				}
			}(loc, locInterval, initDelay)
		}

		for {
			g.mu.Lock()
			conn := g.conn
			g.mu.Unlock()

			if conn == nil {
				break
			}

			_, body, err := conn.ReadMessage()
			if err != nil {
				slog.Warn("websocket read error, reconnecting", "error", err)
				break
			}

			var serverMsg models.ServerMessage
			if err := json.Unmarshal(body, &serverMsg); err != nil {
				slog.Debug("ignored non-JSON websocket frame", "error", err)
				continue
			}

			if serverMsg.Type == models.ServerMessageTypePing {
				if err := g.SendPong(); err != nil {
					slog.Warn("failed to send pong response to ping frame", "error", err)
				}
				continue
			}

			if serverMsg.Type == models.ServerMessageTypeMessages {
				for _, m := range serverMsg.Messages {
					if m.ChatID == "" {
						m.ChatID = serverMsg.ChatID
					}
					go func(msg models.Message) {
						if err := g.ProcessMessage(ctx, msg); err != nil {
							slog.Error("error processing message", "chatID", msg.ChatID, "error", err)
						}
					}(m)
				}
			}
		}

		close(locationDone)
		close(pingDone)

		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(1 * time.Second):
		}
	}
}

// Stop closes the Gateway connection cleanly.
func (g *Gateway) Stop() {
	g.mu.Lock()
	g.running = false
	g.lifecycleCtx = nil
	g.lifecycleID++
	if g.conn != nil {
		if err := g.conn.Close(); err != nil {
			slog.Warn("error closing websocket connection on gateway stop", "error", err)
		}
		g.conn = nil
	}
	fsmEng := g.fsmEngine
	schedEng := g.schedulerEngine
	kw := g.knowledgeWorker
	g.mu.Unlock()

	if kw != nil {
		kw.Stop()
	}

	if schedEng != nil {
		schedEng.Stop()
	}

	if fsmEng != nil {
		fsmEng.Stop()
	}

	g.indexingWg.Wait()

	g.mu.Lock()
	defer g.mu.Unlock()
	if g.memoryManager != nil {
		if err := g.memoryManager.Close(); err != nil {
			slog.Warn("error closing memory manager on gateway stop", "error", err)
		}
	}
	if g.sandboxManager != nil {
		if err := g.sandboxManager.Close(); err != nil {
			slog.Warn("error closing sandbox manager on gateway stop", "error", err)
		}
	}
}

func (g *Gateway) startMaintenanceLoop(ctx context.Context, done chan struct{}) {
	// Run initial maintenance pass after a short startup delay
	select {
	case <-ctx.Done():
		return
	case <-done:
		return
	case <-time.After(30 * time.Second):
		g.RunMaintenance(ctx)
	}

	ticker := time.NewTicker(1 * time.Hour)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-done:
			return
		case <-ticker.C:
			g.RunMaintenance(ctx)
		}
	}
}

// RunMaintenance performs periodic maintenance tasks across all active chat databases,
// including pruning expired terminal FSM runs and reclaiming database pages via incremental vacuum.
func (g *Gateway) RunMaintenance(ctx context.Context) {
	g.mu.Lock()
	memMgr := g.memoryManager
	cfg := g.cfg
	g.mu.Unlock()

	if memMgr == nil || cfg == nil {
		return
	}

	// Discover on-disk chat databases in dataDir to ensure they are loaded in memoryManager
	if cfg.DataDir != "" {
		entries, err := os.ReadDir(cfg.DataDir)
		if err != nil && !os.IsNotExist(err) {
			slog.Warn("failed to read data directory during maintenance", "dataDir", cfg.DataDir, "error", err)
		} else if err == nil {
			for _, entry := range entries {
				if entry.IsDir() {
					continue
				}
				name := entry.Name()
				if name == "townhall.db" {
					if _, err := memMgr.GetDB(ctx, "townhall", false); err != nil {
						slog.Warn("failed to open townhall.db for maintenance", "error", err)
					}
				} else if strings.HasPrefix(name, "dm_") && strings.HasSuffix(name, ".db") {
					chatID := strings.TrimSuffix(strings.TrimPrefix(name, "dm_"), ".db")
					if _, err := memMgr.GetDB(ctx, chatID, true); err != nil {
						slog.Warn("failed to open dm db for maintenance", "chatID", chatID, "error", err)
					}
				}
			}
		}
	}

	activeDBs := memMgr.ActiveDBs()
	retentionDays := cfg.FSMRetentionDays
	if retentionDays <= 0 {
		retentionDays = 7
	}

	cutoffUnix := time.Now().Add(-time.Duration(retentionDays) * 24 * time.Hour).Unix()

	for name, rawDB := range activeDBs {
		if err := fsm.EnsureDBSchema(ctx, rawDB); err != nil {
			slog.Warn("failed to ensure fsm schema during maintenance", "db", name, "error", err)
			continue
		}
		rm := fsm.NewRetentionManager(rawDB, retentionDays)
		pruned, err := rm.PruneAndCompact(ctx)
		if err != nil {
			slog.Warn("fsm retention maintenance failed", "db", name, "error", err)
		} else if pruned > 0 {
			slog.Info("fsm retention maintenance completed", "db", name, "pruned", pruned)
		}

		// Prune terminal schedules
		if err := scheduler.EnsureScheduleSchema(ctx, rawDB); err == nil {
			schedStore := scheduler.NewStore(rawDB)
			if sPruned, sErr := schedStore.PruneTerminalSchedules(ctx, cutoffUnix); sErr == nil && sPruned > 0 {
				slog.Info("scheduler retention maintenance completed", "db", name, "pruned", sPruned)
			}
		}
	}
}

func (g *Gateway) handleSandboxCommand(ctx context.Context, msg models.Message, text, senderName string) error {
	g.mu.Lock()
	botID := g.botUserID
	botUser := g.botUser
	if botUser.ID == "" && botID != "" {
		if u, ok := g.userCache.Get(botID); ok {
			botUser = u
		}
	}
	sm := g.sandboxManager
	g.mu.Unlock()

	parts := strings.Fields(strings.TrimSpace(text))
	subcmd := ""
	var args []string
	if len(parts) > 1 {
		subcmd = parts[1]
		args = parts[2:]
	}

	req := commands.Request{
		Session: agentapi.SessionRef{
			FrontendID: "besedka",
			SessionID:  msg.ChatID,
			ScopeID:    msg.ChatID,
		},
		Actor: agentapi.Actor{
			ID: msg.UserID,
		},
		ActorName:  senderName,
		IsDirect:   true,
		Command:    "sandbox",
		Subcommand: subcmd,
		Args:       args,
		RawText:    text,
	}

	handler := commands.NewSandboxHandler(sm)
	res, err := handler.Handle(ctx, req)
	if err != nil {
		return err
	}

	if res.Continuation != nil {
		userContinuation := "Sandbox is approved. Please proceed with: " + res.Continuation.Reason
		g.contextManager.Push(msg.ChatID, chatcontext.Entry{
			Role:       "user",
			SenderID:   msg.UserID,
			SenderName: senderName,
			Content:    userContinuation,
			Timestamp:  time.Now().Unix(),
		})

		progress := &models.ProgressData{
			CardStatus: models.ProgressStatusRunning,
			Title:      "Working on your request...",
			Steps: []models.ProgressStep{
				{
					ID:          "sandbox_create",
					Title:       "Sandbox created",
					Description: fmt.Sprintf("Proceeding with: %s", res.Continuation.Reason),
					Status:      models.ProgressStatusCompleted,
				},
			},
		}

		var initialProgressSeq int64
		seq, pErr := g.SendProgressMessage(ctx, msg.ChatID, progress)
		if pErr == nil && seq > 0 {
			initialProgressSeq = seq
		} else {
			slog.Warn("could not emit initial progress card for sandbox approval, falling back to chat message", "chat_id", msg.ChatID, "error", pErr)
			ackMsg := fmt.Sprintf("Sandbox created successfully, proceeding with %s...", res.Continuation.Reason)
			if err := g.SendMessage(msg.ChatID, ackMsg); err != nil {
				return fmt.Errorf("failed to send approval message: %w", err)
			}
			g.contextManager.Push(msg.ChatID, chatcontext.Entry{
				Role:       "assistant",
				SenderID:   botID,
				SenderName: botUser.GetDisplayName(),
				Content:    ackMsg,
				Timestamp:  time.Now().Unix(),
			})
		}

		return g.generateAndSendAgentReply(ctx, msg, true, senderName, res.Continuation.Reason, initialProgressSeq)
	}

	if res.RecordAssistantEntry {
		g.contextManager.Push(msg.ChatID, chatcontext.Entry{
			Role:       "assistant",
			SenderID:   botID,
			SenderName: botUser.GetDisplayName(),
			Content:    res.Reply,
			Timestamp:  time.Now().Unix(),
		})
	}

	return g.SendMessage(msg.ChatID, res.Reply)
}

func (g *Gateway) formatSandboxStatus(sbx *sandbox.UserSandbox) string {
	return commands.FormatSandboxStatus(sbx)
}

// GetChat retrieves a chat from the cache, fetching chats from the API if missing.
func (g *Gateway) GetChat(ctx context.Context, chatID string) (models.Chat, error) {
	if ch, ok := g.chatCache.Get(chatID); ok {
		return ch, nil
	}
	chats, err := g.FetchChats(ctx)
	if err != nil {
		return models.Chat{}, err
	}
	g.chatCache.SetAll(chats)
	if ch, ok := g.chatCache.Get(chatID); ok {
		return ch, nil
	}
	return models.Chat{}, fmt.Errorf("chat %q not found", chatID)
}

// VerifyDMOwner checks whether chatID is an authorized 1-on-1 DM owned by userID.
// Returns an error if the chat is not a DM or if the sender is not the DM owner.
func (g *Gateway) VerifyDMOwner(ctx context.Context, chatID, userID string) error {
	if strings.TrimSpace(chatID) == "" || chatID == "townhall" {
		return fmt.Errorf("command is only available in private Direct Messages (DMs)")
	}
	if strings.TrimSpace(userID) == "" {
		return fmt.Errorf("unauthorized: empty user identity")
	}

	chat, ok := g.chatCache.Get(chatID)
	if !ok {
		var err error
		chat, err = g.GetChat(ctx, chatID)
		if err != nil {
			return fmt.Errorf("authorization check failed: unable to verify chat metadata: %w", err)
		}
	}

	// Must be an authoritative 1:1 DM
	if chat.ID != "" && !chat.IsDM && chat.Type != "dm" {
		return fmt.Errorf("command is only available in private Direct Messages (DMs)")
	}

	g.mu.Lock()
	botID := g.botUserID
	g.mu.Unlock()
	if botID == "" {
		if _, err := g.FetchBotUser(ctx); err == nil {
			g.mu.Lock()
			botID = g.botUserID
			g.mu.Unlock()
		}
	}

	var humanOwnerFromUsers string
	if len(chat.UserIDs) > 0 {
		humanCount := 0
		for _, uid := range chat.UserIDs {
			if uid != botID {
				humanOwnerFromUsers = uid
				humanCount++
			}
		}
		if humanCount != 1 {
			return fmt.Errorf("unauthorized: memory/skill management requires a 1-on-1 DM with exactly one human participant")
		}
	} else if botID != "" && strings.HasPrefix(chat.ID, "dm_") {
		if strings.HasPrefix(chat.ID, "dm_"+botID+"_") {
			humanOwnerFromUsers = strings.TrimPrefix(chat.ID, "dm_"+botID+"_")
		} else if strings.HasSuffix(chat.ID, "_"+botID) {
			humanOwnerFromUsers = strings.TrimSuffix(strings.TrimPrefix(chat.ID, "dm_"), "_"+botID)
		}
	}

	ownerID := chat.TargetUserID
	if ownerID == "" {
		ownerID = humanOwnerFromUsers
	} else if humanOwnerFromUsers != "" && ownerID != humanOwnerFromUsers {
		return fmt.Errorf("unauthorized: target user mismatch with chat participant list")
	}

	if ownerID == "" {
		return fmt.Errorf("unauthorized: unable to verify human owner of this DM")
	}

	if userID != ownerID {
		return fmt.Errorf("unauthorized: only the owner of this DM can manage memories and skills")
	}

	if chat.TargetUserID == "" && humanOwnerFromUsers != "" {
		chat.TargetUserID = humanOwnerFromUsers
		if len(chat.UserIDs) == 0 && botID != "" {
			chat.UserIDs = []string{botID, humanOwnerFromUsers}
		}
		g.chatCache.Set(chat)
	}

	return nil
}
