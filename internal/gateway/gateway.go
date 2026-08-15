package gateway

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"sync"
	"time"

	"bob/internal/config"
	"bob/internal/llm"
	"bob/internal/models"

	"github.com/fasthttp/websocket"
)

const (
	SystemPromptTownhall = "You are a concise, professional AI assistant for the Besedka chat application. Answer directly and accurately. Keep your answer brief (maximum 2 paragraphs) without unnecessary conversational filler."
	SystemPromptDM       = "You are a helpful AI assistant for the Besedka chat application. Answer clearly, accurately, and professionally using markdown formatting."
)

// Gateway handles Besedka ingress (listening for mentions/messages) and egress (posting AI responses).
type Gateway struct {
	cfg                  *config.Config
	llmClient            *llm.Client
	conn                 *websocket.Conn
	mu                   sync.Mutex
	running              bool
	systemPromptTownhall string
	systemPromptDM       string
}

// NewGateway creates a new Besedka Gateway instance.
func NewGateway(cfg *config.Config, llmClient *llm.Client) *Gateway {
	return &Gateway{
		cfg:                  cfg,
		llmClient:            llmClient,
		systemPromptTownhall: SystemPromptTownhall,
		systemPromptDM:       SystemPromptDM,
	}
}

var htmlTagRe = regexp.MustCompile(`<[^>]*>`)

// StripHTML removes HTML tags from a string.
func StripHTML(input string) string {
	cleaned := htmlTagRe.ReplaceAllString(input, "")
	return strings.TrimSpace(cleaned)
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
		prompt := re.ReplaceAllString(plainText, "")
		return true, strings.TrimSpace(prompt)
	}

	if isDM {
		return true, strings.TrimSpace(plainText)
	}

	return false, ""
}

// FormatResponse applies paragraph limits according to channel type.
func FormatResponse(content string, isDM bool, maxTownhallParas, maxDMParas int) string {
	maxParas := maxTownhallParas
	if isDM {
		maxParas = maxDMParas
	}
	if maxParas <= 0 {
		return content
	}

	// Normalise CRLF
	normalized := strings.ReplaceAll(content, "\r\n", "\n")
	paras := strings.Split(normalized, "\n\n")

	nonEmptyParas := make([]string, 0, len(paras))
	for _, p := range paras {
		if strings.TrimSpace(p) != "" {
			nonEmptyParas = append(nonEmptyParas, p)
		}
	}

	if len(nonEmptyParas) <= maxParas {
		return strings.Join(nonEmptyParas, "\n\n")
	}

	return strings.Join(nonEmptyParas[:maxParas], "\n\n")
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

// ProcessMessage handles a single incoming message from Besedka.
func (g *Gateway) ProcessMessage(ctx context.Context, msg models.Message) error {
	// Skip messages with empty content
	if strings.TrimSpace(msg.Content) == "" {
		return nil
	}

	shouldProcess, prompt := IsMentionedOrDM(g.cfg.BotHandle, msg.ChatID, msg.Content)
	if !shouldProcess || prompt == "" {
		return nil
	}

	isDM := msg.ChatID != "townhall"
	systemPrompt := g.systemPromptTownhall
	if isDM {
		systemPrompt = g.systemPromptDM
	}

	slog.Info("processing bot message request", "chatID", msg.ChatID, "prompt", prompt)

	reply, err := g.llmClient.GenerateResponse(ctx, systemPrompt, prompt)
	if err != nil {
		slog.Error("failed to generate LLM response", "error", err)
		reply = "Sorry, I encountered an issue processing your request. Please try again later."
	}

	formattedReply := FormatResponse(reply, isDM, g.cfg.TownhallMaxParagraphs, g.cfg.DMMaxParagraphs)

	if err := g.SendMessage(msg.ChatID, formattedReply); err != nil {
		return fmt.Errorf("failed to send reply to chat %s: %w", msg.ChatID, err)
	}

	return nil
}

// Start listens for incoming WebSocket messages and processes them until context is cancelled.
func (g *Gateway) Start(ctx context.Context) error {
	g.mu.Lock()
	g.running = true
	g.mu.Unlock()

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
	defer g.mu.Unlock()
	g.running = false
	if g.conn != nil {
		_ = g.conn.Close()
		g.conn = nil
	}
}
