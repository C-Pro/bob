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
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"bob/internal/chatcontext"
	"bob/internal/config"
	"bob/internal/llm"
	"bob/internal/memory"
	"bob/internal/models"
	"bob/internal/prompt"
	"bob/internal/sandbox"
	"bob/internal/sandbox/bwrap"
	"bob/internal/sandbox/docker"
	"bob/internal/tools"
	"bob/internal/tools/tavily"

	"github.com/fasthttp/websocket"
	"github.com/liliang-cn/cortexdb/v2/pkg/cortexdb"
	openai "github.com/sashabaranov/go-openai"
)

// Gateway handles Besedka ingress (listening for mentions/messages) and egress (posting AI responses).
type Gateway struct {
	cfg                  *config.Config
	llmClient            *llm.Client
	httpClient           *http.Client
	toolsRegistry        *tools.Registry
	memoryManager        *memory.Manager
	sandboxManager       *sandbox.Manager
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
}

// NewGateway creates a new Besedka Gateway instance.
func NewGateway(cfg *config.Config, llmClient *llm.Client) *Gateway {
	httpClient := &http.Client{Timeout: 10 * time.Second}

	var embedder cortexdb.Embedder
	if cfg.EmbeddingModel != "" && llmClient != nil {
		embedder = llm.NewEmbedder(llmClient, cfg.EmbeddingModel)
	}
	memoryManager := memory.NewManager(cfg, embedder)

	var tavilyClient *tavily.Client
	if cfg.TavilyAPIKey != "" {
		tavilyClient = tavily.NewClient(cfg.TavilyAPIKey, cfg.TavilyBaseURL, httpClient)
	}

	var sandboxManager *sandbox.Manager
	if cfg.SandboxEnabled {
		bwrapDriver := bwrap.NewDriverWithConfig(bwrap.Config{
			CPULimit:      cfg.SandboxCPULimit,
			MemoryLimitMB: cfg.SandboxMemoryLimitMB,
		})
		dockerDriver := docker.NewDriver(docker.Config{
			SocketPath:    cfg.SandboxDockerSocket,
			AllowedImages: cfg.SandboxAllowedImages,
			CPULimit:      cfg.SandboxCPULimit,
			MemoryLimitMB: cfg.SandboxMemoryLimitMB,
		})
		sandboxManager = sandbox.NewManager(cfg, []sandbox.Driver{bwrapDriver, dockerDriver})
	}

	toolsRegistry := tools.NewRegistry(tavilyClient, memoryManager, sandboxManager)

	gw := &Gateway{
		cfg:                  cfg,
		llmClient:            llmClient,
		httpClient:           httpClient,
		toolsRegistry:        toolsRegistry,
		memoryManager:        memoryManager,
		sandboxManager:       sandboxManager,
		userCache:            NewUserCache(),
		contextManager:       chatcontext.NewManager(cfg.MsgRingBufferSize),
		startTime:            time.Now(),
		locationInterval:     9 * time.Minute,
		initialLocationDelay: 1 * time.Second,
	}

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
	defer g.mu.Unlock()
	g.memoryManager = m
}

// SetToolsRegistry sets the tool registry for the gateway.
func (g *Gateway) SetToolsRegistry(r *tools.Registry) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.toolsRegistry = r
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

		isImg := att.Type == models.AttachmentTypeImage || strings.HasPrefix(strings.ToLower(att.MimeType), "image/")
		if isImg {
			data, mime, err := g.FetchImageThumbnail(ctx, att.FileID)
			if err != nil {
				slog.Warn("failed to fetch image thumbnail", "fileID", att.FileID, "error", err)
				fmt.Fprintf(&extraText, "\n\n[Attachment: %s (failed to download)]", attName)
				continue
			}
			encoded := base64.StdEncoding.EncodeToString(data)
			images = append(images, chatcontext.ImageAttachment{
				URL: fmt.Sprintf("data:%s;base64,%s", mime, encoded),
			})
			continue
		}

		if isTextMimeOrExt(att.Name, att.MimeType) {
			data, _, err := g.FetchFileContent(ctx, att.FileID, 16384)
			if err != nil {
				slog.Warn("failed to fetch file attachment", "fileID", att.FileID, "error", err)
				fmt.Fprintf(&extraText, "\n\n[Attachment: %s (failed to download)]", attName)
				continue
			}
			if utf8.Valid(data) {
				fmt.Fprintf(&extraText, "\n\n[Attachment %s]:\n```\n%s\n```", attName, string(data))
			} else {
				fmt.Fprintf(&extraText, "\n\n[Attachment: %s (binary content not displayed)]", attName)
			}
			continue
		}

		// Other binary file
		mimeStr := att.MimeType
		if mimeStr == "" {
			mimeStr = "binary file"
		}
		fmt.Fprintf(&extraText, "\n\n[Attachment: %s (%s, not displayed)]", attName, mimeStr)
	}

	return extraText.String(), images
}

// IsMentionedOrDM checks if a message should be handled by the bot.
func IsMentionedOrDM(handle, chatID, content string) (bool, string) {
	plainText := StripHTML(content)
	isDM := chatID != "townhall" && (strings.HasPrefix(chatID, "dm_") || chatID != "")
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
	g.mu.Lock()
	conn := g.conn
	g.mu.Unlock()

	if conn == nil {
		return errors.New("websocket connection is not established")
	}

	clientMsg := models.ClientMessage{
		Type:    models.ClientMessageTypeSend,
		ChatID:  chatID,
		Content: content,
	}

	g.mu.Lock()
	defer g.mu.Unlock()
	return conn.WriteJSON(clientMsg)
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
			cleanContent := ExtractMessageText(m)
			extraText, _ := g.processAttachments(ctx, m.Attachments)
			fullContent := strings.TrimSpace(cleanContent + extraText)
			if fullContent == "" {
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

	for _, m := range msgs {
		cleanContent := ExtractMessageText(m)
		extraText, images := g.processAttachments(ctx, m.Attachments)
		fullContent := strings.TrimSpace(cleanContent + extraText)
		if fullContent == "" && len(images) == 0 {
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
	shouldProcess, _ := IsMentionedOrDM(g.cfg.BotHandle, msg.ChatID, msg.Content)

	// 5. Append incoming user message to ring buffer (backfill chat history if buffer was uninitialized)
	rb := g.contextManager.GetOrCreate(msg.ChatID)
	if rb.Len() == 0 && msg.Seq > 1 {
		g.WarmupChat(ctx, msg.ChatID, msg.Seq-1)
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

	cleanText := strings.TrimSpace(fullContent)
	cleanText = strings.Trim(cleanText, "`")
	cleanText = strings.TrimSpace(cleanText)

	isDM := msg.ChatID != "townhall"
	if isDM && strings.HasPrefix(cleanText, "/sandbox") {
		return g.handleSandboxCommand(ctx, msg, cleanText, senderName)
	}

	return g.generateAndSendAgentReply(ctx, msg, isDM, senderName, "")
}

func (g *Gateway) generateAndSendAgentReply(ctx context.Context, msg models.Message, isDM bool, senderName, currentTask string) error {
	g.mu.Lock()
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

	if g.llmClient == nil {
		return nil
	}

	var sandboxActive bool
	var sandboxTTL string
	if sm != nil && isDM {
		if sbx, ok := sm.GetStatus(msg.UserID); ok && sbx != nil && sbx.Status == sandbox.StatusRunning {
			sandboxActive = true
			rem := time.Until(sbx.ExpiresAt).Round(time.Minute)
			if rem < 0 {
				rem = 0
			}
			sandboxTTL = rem.String()
		}
	}

	var systemPrompt string
	if isDM {
		targetUser, ok := g.userCache.Get(msg.UserID)
		if !ok || targetUser.GetDisplayName() == "" {
			targetUser = models.User{ID: msg.UserID, DisplayName: senderName, UserName: senderName}
		}
		if sandboxActive {
			systemPrompt = prompt.RenderDMPromptWithSandbox(botUser, g.cfg.BotHandle, targetUser, g.cfg.DMMaxParagraphs, true, sandboxTTL)
		} else {
			systemPrompt = prompt.RenderDMPrompt(botUser, g.cfg.BotHandle, targetUser, g.cfg.DMMaxParagraphs)
		}
	} else {
		systemPrompt = prompt.RenderTownhallPrompt(botUser, g.cfg.BotHandle, g.cfg.TownhallMaxParagraphs)
	}

	slog.Info("processing bot message request", "chatID", msg.ChatID, "sender", senderName)

	bufferedMsgs := g.contextManager.GetLLMMessages(msg.ChatID)
	llmMsgs := make([]openai.ChatCompletionMessage, 0, len(bufferedMsgs)+2)
	llmMsgs = append(llmMsgs, openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleSystem,
		Content: systemPrompt,
	})
	llmMsgs = append(llmMsgs, bufferedMsgs...)

	// Defense-in-depth: Ensure message list never ends with an assistant message (e.g. Gemini 400 constraint)
	if len(llmMsgs) > 0 && llmMsgs[len(llmMsgs)-1].Role == openai.ChatMessageRoleAssistant {
		continuation := "Please continue."
		if currentTask != "" {
			continuation = "Please proceed with: " + currentTask
		}
		llmMsgs = append(llmMsgs, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: continuation,
		})
	}

	taskDesc := currentTask
	if taskDesc == "" {
		taskDesc = msg.Content
	}
	noProgress := !isDM || userPrefersNoProgress(msg.Content) || (currentTask != "" && userPrefersNoProgress(currentTask))
	progress := tools.NewProgressReporter(msg.ChatID, taskDesc, func(chatID, text string) error {
		return g.SendMessage(chatID, text)
	}, 30*time.Second, noProgress)
	progress.Start()
	defer progress.Stop()

	var sandboxRequestCreated bool
	sessionCtx := tools.ChatSessionContext{
		ChatID: msg.ChatID,
		UserID: msg.UserID,
		IsDM:   isDM,
		Notifier: func(chatID, text string) error {
			return g.SendMessage(chatID, text)
		},
		Progress:              progress,
		SandboxRequestCreated: &sandboxRequestCreated,
	}

	var toolDefs []openai.Tool
	if toolsRegistry != nil {
		toolDefs = toolsRegistry.ToolDefinitionsForSession(sessionCtx)
	}

	var reply string
	var err error
	if toolsRegistry != nil && len(toolDefs) > 0 {
		toolCtx := tools.WithChatSession(ctx, sessionCtx)
		reply, err = g.llmClient.GenerateChatResponseWithToolLoop(
			toolCtx,
			llmMsgs,
			toolDefs,
			toolsRegistry,
			20,
		)
	} else {
		reply, err = g.llmClient.GenerateChatResponse(ctx, llmMsgs)
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
			if !userStatedSandboxPreference(msg.Content) && !strings.Contains(reply, "/sandbox destroy") {
				rem := time.Until(sbx.ExpiresAt).Round(time.Minute)
				if rem < 0 {
					rem = 0
				}
				reply += fmt.Sprintf("\n\n💡 Would you like to destroy the sandbox (`/sandbox destroy`) or keep it? It will automatically be destroyed in %s.", rem)
			}
		}
	}

	formattedReply := FormatResponse(reply, isDM, g.cfg.TownhallMaxParagraphs, g.cfg.DMMaxParagraphs)

	progress.Stop()

	if err := g.SendMessage(msg.ChatID, formattedReply); err != nil {
		return fmt.Errorf("failed to send reply to chat %s: %w", msg.ChatID, err)
	}

	g.contextManager.Push(msg.ChatID, chatcontext.Entry{
		Role:       "assistant",
		SenderID:   botID,
		SenderName: botUser.GetDisplayName(),
		Content:    formattedReply,
		Timestamp:  time.Now().Unix(),
	})

	return nil
}

func userPrefersNoProgress(text string) bool {
	lower := strings.ToLower(text)
	phrases := []string{
		"no progress",
		"without progress",
		"don't report progress",
		"dont report progress",
		"do not report progress",
		"quietly",
		"silently",
		"silent mode",
		"no updates",
		"suppress progress",
		"skip progress",
	}
	for _, p := range phrases {
		if strings.Contains(lower, p) {
			return true
		}
	}
	return false
}

func userStatedSandboxPreference(text string) bool {
	lower := strings.ToLower(text)
	phrases := []string{
		"destroy sandbox",
		"keep sandbox",
		"destroy it",
		"keep it",
		"leave sandbox",
		"leave it running",
		"kill sandbox",
		"terminate sandbox",
		"close sandbox",
		"don't destroy",
		"dont destroy",
		"do not destroy",
	}
	for _, p := range phrases {
		if strings.Contains(lower, p) {
			return true
		}
	}
	return false
}

// Start listens for incoming WebSocket messages and processes them until context is cancelled.
func (g *Gateway) Start(ctx context.Context) error {
	g.mu.Lock()
	g.running = true
	g.mu.Unlock()

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
	if g.conn != nil {
		if err := g.conn.Close(); err != nil {
			slog.Warn("error closing websocket connection on gateway stop", "error", err)
		}
		g.conn = nil
	}
	g.mu.Unlock()

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

	if botID != "" && msg.UserID == botID {
		return nil // Bot is prohibited from approving or manipulating sandboxes
	}

	if sm == nil {
		return g.SendMessage(msg.ChatID, "Sandbox execution is disabled on this server.")
	}

	parts := strings.Fields(strings.TrimSpace(text))
	subcmd := ""
	if len(parts) > 1 {
		subcmd = strings.ToLower(parts[1])
	}

	switch subcmd {
	case "approve":
		sbx, err := sm.ApproveSandbox(ctx, msg.UserID)
		if err != nil {
			return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Failed to approve sandbox: %v", err))
		}
		ackMsg := fmt.Sprintf("Sandbox created successfully, proceeding with %s...", sbx.Reason)
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

		userContinuation := "Sandbox is approved. Please proceed with: " + sbx.Reason
		g.contextManager.Push(msg.ChatID, chatcontext.Entry{
			Role:       "user",
			SenderID:   msg.UserID,
			SenderName: senderName,
			Content:    userContinuation,
			Timestamp:  time.Now().Unix(),
		})

		return g.generateAndSendAgentReply(ctx, msg, true, senderName, sbx.Reason)

	case "deny":
		err := sm.DenySandbox(msg.UserID)
		if err != nil {
			return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Failed to deny sandbox: %v", err))
		}
		reply := "❌ **Sandbox request denied.**"
		g.contextManager.Push(msg.ChatID, chatcontext.Entry{
			Role:       "assistant",
			SenderID:   botID,
			SenderName: botUser.GetDisplayName(),
			Content:    reply,
			Timestamp:  time.Now().Unix(),
		})
		return g.SendMessage(msg.ChatID, reply)

	case "destroy":
		err := sm.Destroy(ctx, msg.UserID)
		if err != nil {
			return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Failed to destroy sandbox: %v", err))
		}
		reply := "🧹 **Sandbox terminated and resources released.**"
		g.contextManager.Push(msg.ChatID, chatcontext.Entry{
			Role:       "assistant",
			SenderID:   botID,
			SenderName: botUser.GetDisplayName(),
			Content:    reply,
			Timestamp:  time.Now().Unix(),
		})
		return g.SendMessage(msg.ChatID, reply)

	case "status":
		sbx, _ := sm.GetStatus(msg.UserID)
		return g.SendMessage(msg.ChatID, g.formatSandboxStatus(sbx))

	default:
		helpText := "🔒 **Sandbox Commands:**\n" +
			"• `/sandbox approve` — Approve pending sandbox request\n" +
			"• `/sandbox deny` — Deny pending sandbox request\n" +
			"• `/sandbox destroy` — Terminate your active sandbox\n" +
			"• `/sandbox status` — View status of your sandbox"
		return g.SendMessage(msg.ChatID, helpText)
	}
}

func (g *Gateway) formatSandboxStatus(sbx *sandbox.UserSandbox) string {
	if sbx == nil || sbx.Status == sandbox.StatusNone {
		return "ℹ️ You do not have an active or pending sandbox."
	}

	switch sbx.Status {
	case sandbox.StatusPendingApproval:
		var b strings.Builder
		b.WriteString("⏳ **Sandbox Request Awaiting Your Approval**\n")
		g.appendSandboxDetails(&b, sbx)
		b.WriteString("\nReply `/sandbox approve` to approve or `/sandbox deny` to reject.")
		return b.String()

	case sandbox.StatusRunning:
		var b strings.Builder
		b.WriteString("🟢 **Active Sandbox Status**\n")
		g.appendSandboxDetails(&b, sbx)
		return b.String()

	case sandbox.StatusExpired:
		return "⌛ **Your previous sandbox has expired.** The agent can request a new sandbox when needed."

	default:
		return fmt.Sprintf("ℹ️ Sandbox status: %s", sbx.Status)
	}
}

func (g *Gateway) appendSandboxDetails(b *strings.Builder, sbx *sandbox.UserSandbox) {
	b.WriteString(fmt.Sprintf("- **Driver:** %s\n", sbx.Driver))
	if sbx.Driver == sandbox.DriverDocker && sbx.DockerImage != "" {
		b.WriteString(fmt.Sprintf("- **Docker Image:** %s\n", sbx.DockerImage))
	}
	b.WriteString(fmt.Sprintf("- **Network:** %s\n", sbx.Network.Mode))
	if len(sbx.Network.AllowedHosts) > 0 {
		b.WriteString(fmt.Sprintf("- **Allowed Domains:** %s\n", strings.Join(sbx.Network.AllowedHosts, ", ")))
	}
	if len(sbx.Mounts) > 0 {
		b.WriteString("- **Mounts (in workspace):**\n")
		for _, m := range sbx.Mounts {
			ro := "read-write"
			if m.ReadOnly {
				ro = "read-only"
			}
			pathStr := m.RelativePath
			if pathStr == "." || pathStr == "" {
				pathStr = "(whole workspace)"
			}
			b.WriteString(fmt.Sprintf("  • %s (%s)\n", pathStr, ro))
		}
	}
	remaining := time.Until(sbx.ExpiresAt).Round(time.Minute)
	if remaining < 0 {
		remaining = 0
	}
	b.WriteString(fmt.Sprintf("- **Time Remaining:** %s\n", remaining))
}
