package gateway

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
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
func (g *Gateway) FetchChatMessages(ctx context.Context, chatID string, fromSeq, toSeq int64) ([]models.Message, error) {
	if fromSeq <= 0 {
		fromSeq = 1
	}
	if toSeq <= 0 {
		toSeq = 1000000
	}
	if fromSeq > toSeq {
		fromSeq = toSeq
	}

	reqURL := fmt.Sprintf("%s/api/chats/%s/messages?fromSeq=%d&toSeq=%d", strings.TrimSuffix(g.cfg.BesedkaURL, "/"), chatID, fromSeq, toSeq)

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

// FetchImageThumbnail downloads an image thumbnail from Besedka /api/images/{file_id}?thumb=1.
func (g *Gateway) FetchImageThumbnail(ctx context.Context, fileID string) ([]byte, string, error) {
	if strings.TrimSpace(fileID) == "" {
		return nil, "", errors.New("empty fileID")
	}

	reqURL := fmt.Sprintf("%s/api/images/%s?thumb=1", strings.TrimSuffix(g.cfg.BesedkaURL, "/"), fileID)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, reqURL, nil)
	if err != nil {
		return nil, "", fmt.Errorf("failed to create image request: %w", err)
	}

	if g.cfg.BesedkaAPIKey != "" {
		req.Header.Set("Authorization", "Bearer "+g.cfg.BesedkaAPIKey)
	}

	resp, err := g.httpClient.Do(req)
	if err != nil {
		return nil, "", fmt.Errorf("failed to fetch image %s: %w", fileID, err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return nil, "", fmt.Errorf("/api/images/%s returned status %d", fileID, resp.StatusCode)
	}

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, "", fmt.Errorf("failed to read image body: %w", err)
	}

	contentType := resp.Header.Get("Content-Type")
	if contentType == "" {
		contentType = "image/jpeg"
	}
	if idx := strings.Index(contentType, ";"); idx != -1 {
		contentType = strings.TrimSpace(contentType[:idx])
	}

	return data, contentType, nil
}

// FetchFileContent downloads text file content from Besedka /api/files/{file_id} up to maxBytes.
func (g *Gateway) FetchFileContent(ctx context.Context, fileID string, maxBytes int64) ([]byte, string, error) {
	if strings.TrimSpace(fileID) == "" {
		return nil, "", errors.New("empty fileID")
	}
	if maxBytes <= 0 {
		maxBytes = 16384 // 16KB default
	}

	reqURL := fmt.Sprintf("%s/api/files/%s", strings.TrimSuffix(g.cfg.BesedkaURL, "/"), fileID)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, reqURL, nil)
	if err != nil {
		return nil, "", fmt.Errorf("failed to create file request: %w", err)
	}

	if g.cfg.BesedkaAPIKey != "" {
		req.Header.Set("Authorization", "Bearer "+g.cfg.BesedkaAPIKey)
	}

	resp, err := g.httpClient.Do(req)
	if err != nil {
		return nil, "", fmt.Errorf("failed to fetch file %s: %w", fileID, err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return nil, "", fmt.Errorf("/api/files/%s returned status %d", fileID, resp.StatusCode)
	}

	limitReader := io.LimitReader(resp.Body, maxBytes+1)
	data, err := io.ReadAll(limitReader)
	if err != nil {
		return nil, "", fmt.Errorf("failed to read file body: %w", err)
	}

	if int64(len(data)) > maxBytes {
		data = data[:maxBytes]
	}

	contentType := resp.Header.Get("Content-Type")
	if contentType == "" {
		contentType = "text/plain"
	}
	if idx := strings.Index(contentType, ";"); idx != -1 {
		contentType = strings.TrimSpace(contentType[:idx])
	}

	return data, contentType, nil
}

