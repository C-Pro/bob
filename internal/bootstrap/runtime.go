package bootstrap

import (
	"errors"
	"flag"
	"fmt"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"bob/internal/agent"
	"bob/internal/agentapi"
	"bob/internal/agentstore"
	"bob/internal/config"
	"bob/internal/fsm"
	"bob/internal/llm"
	"bob/internal/memory"
	"bob/internal/sandbox"
	"bob/internal/sandbox/bwrap"
	"bob/internal/sandbox/docker"
	"bob/internal/tools"
	"bob/internal/tools/tavily"

	"github.com/liliang-cn/cortexdb/v2/pkg/cortexdb"
)

// Options configures runtime initialization and optional subsystem overrides.
type Options struct {
	LLMClient     *llm.Client
	HTTPClient    *http.Client
	SessionLocker *agent.SessionLocker
	EnableSandbox *bool // If set, explicitly overrides cfg.SandboxEnabled
	Frontends     map[string]agentapi.Frontend
	Embedder      cortexdb.Embedder // Optional custom embedder (e.g. for testing)
}

// Runtime contains the shared core agent subsystems instantiated and wired together.
type Runtime struct {
	Config         *config.Config
	HTTPClient     *http.Client
	LLMClient      *llm.Client
	MemoryManager  *memory.Manager
	StoreProvider  *agentstore.MemoryStoreProvider
	ToolsRegistry  *tools.Registry
	SandboxManager *sandbox.Manager
	FSMEngine      *fsm.Engine
	AgentEngine    *agent.Engine
	SessionLocker  *agent.SessionLocker

	closeOnce sync.Once
	closeErr  error
}

// New creates and wires a new Runtime according to the provided configuration and options.
func New(cfg *config.Config, opts Options) (*Runtime, error) {
	if cfg == nil {
		return nil, errors.New("configuration is required")
	}

	if cfg.DataDir != "" {
		if err := os.MkdirAll(cfg.DataDir, 0755); err != nil {
			return nil, fmt.Errorf("failed to create data directory %s: %w", cfg.DataDir, err)
		}
	}

	httpClient := opts.HTTPClient
	if httpClient == nil {
		httpClient = &http.Client{Timeout: 15 * time.Second}
	}

	llmClient := opts.LLMClient
	if llmClient == nil && (strings.TrimSpace(cfg.OpenAIAPIKey) != "" || strings.TrimSpace(cfg.GeminiAPIKey) != "") {
		llmClient = llm.NewClient(cfg, httpClient)
	}

	// Determine embedder
	embedder := opts.Embedder
	if embedder == nil {
		modelSetting := strings.ToLower(strings.TrimSpace(cfg.EmbeddingModel))
		switch {
		case modelSetting == "none" || modelSetting == "disabled" || modelSetting == "off":
			embedder = nil
		case modelSetting != "":
			if llmClient != nil {
				embedder = llm.NewEmbedder(llmClient, cfg.EmbeddingModel)
			}
		default:
			// Default empty string uses lazy local embedder
			if flag.Lookup("test.v") != nil && os.Getenv("BOB_MODELS_DIR") == "" && !memory.HasLocalModelInDir(cfg.DataDir) {
				embedder = nil
			} else {
				embedder = llm.NewLazyLocalEmbedder(cfg)
			}
		}
	}

	memoryManager := memory.NewManager(cfg, embedder)
	storeProvider := agentstore.NewMemoryStoreProvider(memoryManager, cfg.DataDir)
	if cfg.KnowledgeMaxDiscoveryLimit > 0 {
		storeProvider.SetMaxDiscoveryLimit(cfg.KnowledgeMaxDiscoveryLimit)
	}

	// Determine effective sandbox enablement
	enableSandbox := cfg.SandboxEnabled
	if opts.EnableSandbox != nil {
		enableSandbox = *opts.EnableSandbox
	}

	var sandboxManager *sandbox.Manager
	if enableSandbox {
		sbxCfg := cfg.SandboxConfig()
		sbxCfg.Enabled = true

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

	var tavilyClient *tavily.Client
	if cfg.TavilyAPIKey != "" {
		tavilyClient = tavily.NewClient(cfg.TavilyAPIKey, cfg.TavilyBaseURL, httpClient)
	}

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
	toolsRegistry.SetKnowledgeLimits(tools.KnowledgeBudgetLimits{
		MaxLoadedMemories:    cfg.KnowledgeMaxLoadedMemories,
		MaxLoadedMemoryBytes: cfg.KnowledgeMaxLoadedMemoryBytes,
		MaxLoadedSkills:      cfg.KnowledgeMaxLoadedSkills,
		MaxLoadedSkillBytes:  cfg.KnowledgeMaxLoadedSkillBytes,
	}, cfg.KnowledgeDefaultDiscoveryLimit, cfg.KnowledgeMaxDiscoveryLimit)
	toolsRegistry.SetMaxAttachmentSize(cfg.MaxAttachmentSizeBytes)

	sessionLocker := opts.SessionLocker
	if sessionLocker == nil {
		sessionLocker = agent.NewSessionLocker()
	}

	var fsmEngine *fsm.Engine
	var agentEngine *agent.Engine

	if llmClient != nil {
		fsmEngine = fsm.NewEngine(
			storeProvider,
			llmClient,
			toolsRegistry,
			fsm.WithDefaultModel(cfg.OpenAIModel),
			fsm.WithPollInterval(2*time.Second),
			fsm.WithKnowledgeBudgetLimits(tools.KnowledgeBudgetLimits{
				MaxLoadedMemories:    cfg.KnowledgeMaxLoadedMemories,
				MaxLoadedMemoryBytes: cfg.KnowledgeMaxLoadedMemoryBytes,
				MaxLoadedSkills:      cfg.KnowledgeMaxLoadedSkills,
				MaxLoadedSkillBytes:  cfg.KnowledgeMaxLoadedSkillBytes,
			}),
		)

		var err error
		agentEngine, err = agent.NewEngine(agent.Dependencies{
			Config: agent.Config{
				DefaultModel:  cfg.OpenAIModel,
				MaxIterations: cfg.TownhallToolMaxIterations,
				LockTimeout:   30 * time.Second,
			},
			LLMClient:     llmClient,
			FSMEngine:     fsmEngine,
			SessionLocker: sessionLocker,
			Frontends:     opts.Frontends,
		})
		if err != nil {
			if sandboxManager != nil {
				_ = sandboxManager.Close()
			}
			_ = memoryManager.Close()
			return nil, fmt.Errorf("failed to initialize agent engine: %w", err)
		}

		// Register any provided frontends on fsmEngine
		for id, fe := range opts.Frontends {
			fsmEngine.RegisterFrontend(id, fe)
		}
	}

	return &Runtime{
		Config:         cfg,
		HTTPClient:     httpClient,
		LLMClient:      llmClient,
		MemoryManager:  memoryManager,
		StoreProvider:  storeProvider,
		ToolsRegistry:  toolsRegistry,
		SandboxManager: sandboxManager,
		FSMEngine:      fsmEngine,
		AgentEngine:    agentEngine,
		SessionLocker:  sessionLocker,
	}, nil
}

// RegisterFrontend registers a frontend binding with both the AgentEngine and the FSMEngine.
func (r *Runtime) RegisterFrontend(id string, fe agentapi.Frontend) error {
	normID := strings.ToLower(strings.TrimSpace(id))
	if normID == "" {
		return errors.New("frontend ID cannot be empty")
	}
	if fe == nil {
		return errors.New("frontend cannot be nil")
	}

	if r.AgentEngine != nil {
		if err := r.AgentEngine.RegisterFrontend(normID, fe); err != nil {
			return err
		}
	}
	if r.FSMEngine != nil {
		r.FSMEngine.RegisterFrontend(normID, fe)
	}
	return nil
}

// Close gracefully stops runtime components in reverse dependency order:
// 1. Agent Engine (marks closed, rejects new runs)
// 2. FSM Engine (stops background loops)
// 3. Sandbox Manager (destroys active sandboxes)
// 4. Memory Manager (closes embedders and open SQLite databases)
func (r *Runtime) Close() error {
	r.closeOnce.Do(func() {
		var errs []error

		if r.AgentEngine != nil {
			if err := r.AgentEngine.Close(); err != nil {
				errs = append(errs, fmt.Errorf("error closing agent engine: %w", err))
			}
		}

		if r.FSMEngine != nil {
			r.FSMEngine.Stop()
		}

		if r.SandboxManager != nil {
			if err := r.SandboxManager.Close(); err != nil {
				errs = append(errs, fmt.Errorf("error closing sandbox manager: %w", err))
			}
		}

		if r.MemoryManager != nil {
			if err := r.MemoryManager.Close(); err != nil {
				errs = append(errs, fmt.Errorf("error closing memory manager: %w", err))
			}
		}

		if len(errs) > 0 {
			r.closeErr = errors.Join(errs...)
		}
	})
	return r.closeErr
}
