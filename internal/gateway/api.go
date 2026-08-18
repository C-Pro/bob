package gateway

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"strings"

	"bob/internal/models"
)

// FetchBotUser fetches current bot user metadata from Besedka /api/me.
func (g *Gateway) FetchBotUser(ctx context.Context) (*models.User, error) {
	reqURL := fmt.Sprintf("%s/api/me", strings.TrimSuffix(g.cfg.BesedkaURL, "/"))
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, reqURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create /api/me request: %w", err)
	}

	if g.cfg.BesedkaAPIKey != "" {
		req.Header.Set("Authorization", "Bearer "+g.cfg.BesedkaAPIKey)
	}

	resp, err := g.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to call /api/me: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("/api/me returned status %d", resp.StatusCode)
	}

	var user models.User
	if err := json.NewDecoder(resp.Body).Decode(&user); err != nil {
		return nil, fmt.Errorf("failed to decode /api/me response: %w", err)
	}

	g.mu.Lock()
	g.botUser = user
	g.botUserID = user.ID
	g.mu.Unlock()

	g.userCache.Set(user)
	slog.Info("fetched bot user metadata", "botUserID", user.ID, "userName", user.GetUserName(), "displayName", user.GetDisplayName())
	return &user, nil
}

// FetchUsers fetches all users from Besedka /api/users and caches them.
func (g *Gateway) FetchUsers(ctx context.Context) ([]models.User, error) {
	reqURL := fmt.Sprintf("%s/api/users", strings.TrimSuffix(g.cfg.BesedkaURL, "/"))
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, reqURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create /api/users request: %w", err)
	}

	if g.cfg.BesedkaAPIKey != "" {
		req.Header.Set("Authorization", "Bearer "+g.cfg.BesedkaAPIKey)
	}

	resp, err := g.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to call /api/users: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("/api/users returned status %d", resp.StatusCode)
	}

	var users []models.User
	var raw json.RawMessage
	if err := json.NewDecoder(resp.Body).Decode(&raw); err != nil {
		return nil, fmt.Errorf("failed to decode /api/users response: %w", err)
	}

	// Support both []User and {"users": []User}
	if err := json.Unmarshal(raw, &users); err != nil {
		var wrapper struct {
			Users []models.User `json:"users"`
		}
		if err2 := json.Unmarshal(raw, &wrapper); err2 == nil {
			users = wrapper.Users
		} else {
			return nil, fmt.Errorf("failed to parse users JSON: %w", err)
		}
	}

	g.userCache.SetAll(users)
	slog.Info("fetched and cached users", "count", len(users))
	return users, nil
}

// FetchChats fetches active chats from Besedka /api/chats.
func (g *Gateway) FetchChats(ctx context.Context) ([]models.Chat, error) {
	reqURL := fmt.Sprintf("%s/api/chats", strings.TrimSuffix(g.cfg.BesedkaURL, "/"))
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, reqURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create /api/chats request: %w", err)
	}

	if g.cfg.BesedkaAPIKey != "" {
		req.Header.Set("Authorization", "Bearer "+g.cfg.BesedkaAPIKey)
	}

	resp, err := g.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to call /api/chats: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("/api/chats returned status %d", resp.StatusCode)
	}

	var chats []models.Chat
	var raw json.RawMessage
	if err := json.NewDecoder(resp.Body).Decode(&raw); err != nil {
		return nil, fmt.Errorf("failed to decode /api/chats response: %w", err)
	}

	// Support both []Chat and {"chats": []Chat}
	if err := json.Unmarshal(raw, &chats); err != nil {
		var wrapper struct {
			Chats []models.Chat `json:"chats"`
		}
		if err2 := json.Unmarshal(raw, &wrapper); err2 == nil {
			chats = wrapper.Chats
		} else {
			return nil, fmt.Errorf("failed to parse chats JSON: %w", err)
		}
	}

	slog.Info("fetched chats", "count", len(chats))
	return chats, nil
}

// FetchChatMessages retrieves historical messages for a chat from Besedka /api/chats/{chat_id}/messages.
func (g *Gateway) FetchChatMessages(ctx context.Context, chatID string, limit int) ([]models.Message, error) {
	reqURL := fmt.Sprintf("%s/api/chats/%s/messages", strings.TrimSuffix(g.cfg.BesedkaURL, "/"), chatID)
	if limit > 0 {
		reqURL = fmt.Sprintf("%s?limit=%d", reqURL, limit)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, reqURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create /api/chats/%s/messages request: %w", chatID, err)
	}

	if g.cfg.BesedkaAPIKey != "" {
		req.Header.Set("Authorization", "Bearer "+g.cfg.BesedkaAPIKey)
	}

	resp, err := g.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to call /api/chats/%s/messages: %w", chatID, err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("/api/chats/%s/messages returned status %d", chatID, resp.StatusCode)
	}

	var messages []models.Message
	var raw json.RawMessage
	if err := json.NewDecoder(resp.Body).Decode(&raw); err != nil {
		return nil, fmt.Errorf("failed to decode messages response: %w", err)
	}

	// Support both []Message and {"messages": []Message}
	if err := json.Unmarshal(raw, &messages); err != nil {
		var wrapper struct {
			Messages []models.Message `json:"messages"`
		}
		if err2 := json.Unmarshal(raw, &wrapper); err2 == nil {
			messages = wrapper.Messages
		} else {
			return nil, fmt.Errorf("failed to parse messages JSON: %w", err)
		}
	}

	return messages, nil
}
