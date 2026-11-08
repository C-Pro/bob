package gateway

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"bob/internal/agentapi"
	"bob/internal/chatcontext"
	"bob/internal/config"
	"bob/internal/llm"
	"bob/internal/models"

	"github.com/fasthttp/websocket"
	openai "github.com/sashabaranov/go-openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestExtractAndStripImageDescriptions(t *testing.T) {
	payload := `{"descriptions":[{"filename":"Photo.PNG","file_id":"IMG-123","description":"A cat with {curly} markings."}]}`
	for _, tc := range []struct {
		name, text, clean string
		want              map[string]string
	}{
		{name: "fenced", text: "A cat.\n\n```json\n" + payload + "\n```", clean: "A cat.", want: map[string]string{"photo.png": "A cat with {curly} markings.", "img-123": "A cat with {curly} markings."}},
		{name: "unfenced", text: "A cat.\n" + payload, clean: "A cat.", want: map[string]string{"photo.png": "A cat with {curly} markings.", "img-123": "A cat with {curly} markings."}},
		{name: "prefix", text: "A cat.\njson " + payload, clean: "A cat.", want: map[string]string{"photo.png": "A cat with {curly} markings.", "img-123": "A cat with {curly} markings."}},
		{name: "only metadata", text: "json " + payload, want: map[string]string{"photo.png": "A cat with {curly} markings.", "img-123": "A cat with {curly} markings."}},
		{name: "no json", text: " Just a cat. ", clean: "Just a cat.", want: map[string]string{}},
		{name: "unrelated json", text: "```json\n{\"answer\":42}\n```", clean: "```json\n{\"answer\":42}\n```", want: map[string]string{}},
		{name: "malformed fenced metadata", text: "A cat.\n```json\n{\"descriptions\":[oops]}\n```", clean: "A cat.", want: map[string]string{}},
		{name: "malformed raw metadata", text: "A cat.\njson {\"descriptions\":[oops]}", clean: "A cat.", want: map[string]string{}},
		{name: "incomplete raw metadata", text: "A cat.\n{\"descriptions\":[", clean: "A cat.", want: map[string]string{}},
		{name: "numeric ID", text: `{"descriptions":[{"fileID":123,"description":" Cat "}]}`, want: map[string]string{"123": "Cat"}},
		{name: "empty descriptions", text: `{"descriptions":[{"filename":"photo.png","description":" "}]}`, want: map[string]string{}},
		{name: "multiple blocks", text: payload + "\nAnswer\n" + payload, clean: "Answer", want: map[string]string{"photo.png": "A cat with {curly} markings.", "img-123": "A cat with {curly} markings."}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			clean, descriptions := extractAndStripImageDescriptions(tc.text)
			assert.Equal(t, tc.clean, clean)
			assert.Equal(t, tc.want, descriptions)
		})
	}
}

func TestEnrichAttachmentText(t *testing.T) {
	content := "Look:\n[Attachment: Photo.PNG (id: IMG-123, type: image/png)]\n[Attachment: other.jpg (id: 456, type: image/jpeg)]\n[Attachment: notes.txt (id: 789, type: text/plain)]"
	descriptions := map[string]string{"img-123": "A \"cat\"\non a mat", "photo.png": "Wrong description", "other.jpg": "A dog"}
	got := enrichAttachmentText(content, descriptions)
	assert.Equal(t, "Look:\n[Attachment: Photo.PNG (id: IMG-123, type: image/png, description: \"A \\\"cat\\\"\\non a mat\")]\n[Attachment: other.jpg (id: 456, type: image/jpeg, description: \"A dog\")]\n[Attachment: notes.txt (id: 789, type: text/plain)]", got)
	assert.Equal(t, content, enrichAttachmentText(content, nil))
	assert.Equal(t, got, enrichAttachmentText(got, descriptions))
	marker := `[Attachment: photo.png (id: 123, type: image/png)]`
	for _, description := range []string{"A cat (sitting) on a mat", "A quote: \"hello\"\nwith a new line", `Literal \\ and )] punctuation`, "Actual\nnewline"} {
		enriched := enrichAttachmentText(marker, map[string]string{"123": description})
		assert.Equal(t, enriched, enrichAttachmentText(enriched, map[string]string{"123": "Replacement"}))
		require.Len(t, attachmentMarkerPattern.FindStringSubmatch(enriched), 5)
	}
}

type imageMetadataTransport func(*http.Request) (*http.Response, error)

func (fn imageMetadataTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	return fn(req)
}

func imageTestRun() agentapi.RunDescriptor {
	return agentapi.RunDescriptor{
		Session: agentapi.SessionRef{SessionID: "townhall"}, Actor: agentapi.Actor{ID: "alice"},
		Metadata: map[string]string{"source_message_seq": "1"},
	}
}

func pushImageTestEntry(gw *Gateway) {
	gw.contextManager.Push("townhall", chatcontext.Entry{
		Role: "user", SenderID: "alice", Seq: 1,
		Content: "Look!\n[Attachment: photo.png (id: img-123, type: image/png)]",
		Images:  []chatcontext.ImageAttachment{{URL: "data:image/png;base64,Y2F0"}},
	})
}

type imageTestResponseWriter struct {
	*httptest.ResponseRecorder
	conn   net.Conn
	reader *bufio.Reader
}

func (w imageTestResponseWriter) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	return w.conn, bufio.NewReadWriter(w.reader, bufio.NewWriter(w.conn)), nil
}

func connectImageTestWebSocket(t *testing.T, gw *Gateway) <-chan models.ClientMessage {
	t.Helper()
	client, server := net.Pipe()
	require.NoError(t, client.SetDeadline(time.Now().Add(5*time.Second)))
	require.NoError(t, server.SetDeadline(time.Now().Add(5*time.Second)))
	messages := make(chan models.ClientMessage, 1)
	done := make(chan struct{})
	t.Cleanup(func() {
		assert.NoError(t, client.Close())
		assert.NoError(t, server.Close())
		<-done
	})
	go func() {
		defer close(done)
		reader := bufio.NewReader(server)
		request, err := http.ReadRequest(reader)
		if !assert.NoError(t, err) {
			return
		}
		upgrader := websocket.Upgrader{}
		conn, err := upgrader.Upgrade(imageTestResponseWriter{httptest.NewRecorder(), server, reader}, request, nil)
		if !assert.NoError(t, err) {
			return
		}
		defer func() { assert.NoError(t, conn.Close()) }()
		var message models.ClientMessage
		if assert.NoError(t, conn.ReadJSON(&message)) {
			messages <- message
		}
	}()
	dialer := websocket.Dialer{
		NetDial: func(network, addr string) (net.Conn, error) {
			return client, nil
		},
	}
	conn, _, err := dialer.Dial("ws://besedka.test/api/chat", nil)
	require.NoError(t, err)
	gw.conn = conn
	return messages
}

func TestGateway_DeliverImageDescriptions(t *testing.T) {
	for _, tc := range []struct {
		name, reply string
		fallback    bool
	}{
		{name: "in turn", reply: "A cat.\n```json\n{\"descriptions\":[{\"filename\":\"photo.png\",\"description\":\"A cat on a mat\"}]}\n```"},
		{name: "fallback", reply: "A cat.", fallback: true},
		{name: "malformed", reply: "A cat.\n```json\n{\"descriptions\":[oops]}\n```", fallback: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			var fallbackCalls, thumbnailCalls atomic.Int32
			httpClient := &http.Client{Transport: imageMetadataTransport(func(req *http.Request) (*http.Response, error) {
				if strings.HasPrefix(req.URL.Path, "/api/images/") {
					thumbnailCalls.Add(1)
					assert.Equal(t, "/api/images/img-123", req.URL.Path)
					assert.Equal(t, "1", req.URL.Query().Get("thumb"))
					return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": {"image/png"}}, Body: io.NopCloser(strings.NewReader("cat"))}, nil
				}
				fallbackCalls.Add(1)
				var request openai.ChatCompletionRequest
				if err := json.NewDecoder(req.Body).Decode(&request); err != nil {
					return nil, err
				}
				assert.Equal(t, &openai.ChatCompletionResponseFormat{Type: openai.ChatCompletionResponseFormatTypeJSONObject}, request.ResponseFormat)
				assert.Equal(t, "vision-test", request.Model)
				if assert.Len(t, request.Messages, 2) && assert.Len(t, request.Messages[1].MultiContent, 2) {
					assert.Equal(t, "data:image/png;base64,Y2F0", request.Messages[1].MultiContent[1].ImageURL.URL)
				}
				return &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader(`{"choices":[{"message":{"role":"assistant","content":"{\"descriptions\":[{\"file_id\":\"img-123\",\"description\":\"A cat on a mat\"}]}"}}]}`))}, nil
			})}
			cfg := &config.Config{BesedkaURL: "https://besedka.test", OpenAIBaseURL: "https://llm.test", OpenAIAPIKey: "test", OpenAIModel: "vision-test", TownhallMaxParagraphs: 10, DataDir: t.TempDir()}
			gw := NewGateway(cfg, llm.NewClient(cfg, httpClient))
			defer gw.Stop()
			gw.httpClient = httpClient
			gw.chatCache.Set(models.Chat{ID: "townhall"})
			messages := connectImageTestWebSocket(t, gw)
			pushImageTestEntry(gw)
			require.NoError(t, gw.Deliver(ctx, agentapi.Completion{Run: imageTestRun(), Status: agentapi.RunCompleted, Result: agentapi.Result{Content: tc.reply}}))
			select {
			case message := <-messages:
				assert.Equal(t, "A cat.", message.Content)
			case <-ctx.Done():
				t.Fatal("delivery timed out")
			}
			entries := gw.contextManager.GetOrCreate("townhall").Entries()
			require.Len(t, entries, 2)
			assert.Equal(t, "Look!\n[Attachment: photo.png (id: img-123, type: image/png, description: \"A cat on a mat\")]", entries[0].Content)
			assert.Nil(t, entries[0].Images)
			assert.Equal(t, "A cat.", entries[1].Content)
			assert.Nil(t, entries[1].Images)
			if tc.fallback {
				assert.Equal(t, int32(1), fallbackCalls.Load())
				assert.Equal(t, int32(1), thumbnailCalls.Load())
			} else {
				assert.Zero(t, fallbackCalls.Load())
				assert.Zero(t, thumbnailCalls.Load())
			}
		})
	}
}

func TestGateway_ClearSourceImagesOnRunnerError(t *testing.T) {
	gw := NewGateway(&config.Config{DataDir: t.TempDir()}, nil)
	defer gw.Stop()
	pushImageTestEntry(gw)
	gw.contextManager.Push("townhall", chatcontext.Entry{Role: "user", SenderID: "bob", Seq: 2, Images: []chatcontext.ImageAttachment{{URL: "newer image"}}})
	gw.SetRunner(&mockTurnRunner{err: errors.New("runner failed")})
	err := gw.generateAndSendAgentReply(context.Background(), models.Message{ChatID: "townhall", UserID: "alice", Seq: 1}, false, "Alice", "", 0)
	require.ErrorContains(t, err, "runner failed")
	entries := gw.contextManager.GetOrCreate("townhall").Entries()
	assert.Nil(t, entries[0].Images)
	assert.Len(t, entries[1].Images, 1)
}

func TestGateway_ClearSourceImagesOnDeliveryError(t *testing.T) {
	gw := NewGateway(&config.Config{DataDir: t.TempDir()}, nil)
	defer gw.Stop()
	pushImageTestEntry(gw)
	gw.chatCache.Set(models.Chat{ID: "townhall"})
	err := gw.Deliver(context.Background(), agentapi.Completion{Run: imageTestRun(), Status: agentapi.RunCompleted, Result: agentapi.Result{Content: "A cat."}})
	require.ErrorContains(t, err, "websocket connection is not established")
	assert.Nil(t, gw.contextManager.GetOrCreate("townhall").Entries()[0].Images)
}

func TestGateway_FallbackUnavailable(t *testing.T) {
	gw := NewGateway(&config.Config{DataDir: t.TempDir()}, nil)
	defer gw.Stop()
	assert.Empty(t, gw.fetchImageDescriptionsFallback(context.Background(), []models.Attachment{{Type: models.AttachmentTypeImage, FileID: "img-123"}}))
}

func TestGateway_FinalizeImageReplyPartialAndFailedFallback(t *testing.T) {
	for _, tc := range []struct {
		name, fallback string
		status         int
		thumbnailFails bool
		wantFallback   bool
	}{
		{name: "partial descriptions", status: 200, fallback: `{"descriptions":[{"file_id":"img-456","description":"A dog"}]}`, wantFallback: true},
		{name: "malformed fallback", status: 200, fallback: "not JSON"},
		{name: "request failure", status: 400},
		{name: "thumbnail failure", thumbnailFails: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var thumbnails, requests atomic.Int32
			httpClient := &http.Client{Transport: imageMetadataTransport(func(req *http.Request) (*http.Response, error) {
				if strings.HasPrefix(req.URL.Path, "/api/images/") {
					thumbnails.Add(1)
					assert.Equal(t, "/api/images/img-456", req.URL.Path)
					if tc.thumbnailFails {
						return &http.Response{StatusCode: 404, Body: io.NopCloser(strings.NewReader("missing"))}, nil
					}
					return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": {"image/png"}}, Body: io.NopCloser(strings.NewReader("dog"))}, nil
				}
				requests.Add(1)
				body, err := json.Marshal(openai.ChatCompletionResponse{Choices: []openai.ChatCompletionChoice{{Message: openai.ChatCompletionMessage{Role: "assistant", Content: tc.fallback}}}})
				if err != nil {
					return nil, err
				}
				return &http.Response{StatusCode: tc.status, Body: io.NopCloser(strings.NewReader(string(body)))}, nil
			})}
			cfg := &config.Config{BesedkaURL: "https://besedka.test", OpenAIBaseURL: "https://llm.test", OpenAIAPIKey: "test", DataDir: t.TempDir()}
			gw := NewGateway(cfg, llm.NewClient(cfg, httpClient))
			defer gw.Stop()
			gw.httpClient = httpClient
			pushImageTestEntry(gw)
			gw.contextManager.GetOrCreate("townhall").UpdateLast(func(e *chatcontext.Entry) bool {
				e.Content += "\n[Attachment: dog.png (id: img-456, type: image/png)]"
				return true
			})
			clean := gw.finalizeImageReply(context.Background(), imageTestRun(), "Pets.\n"+`{"descriptions":[{"filename":"photo.png","description":"A cat"}]}`)
			assert.Equal(t, "Pets.", clean)
			entry := gw.contextManager.GetOrCreate("townhall").Entries()[0]
			assert.Nil(t, entry.Images)
			assert.Contains(t, entry.Content, `type: image/png, description: "A cat"`)
			assert.Equal(t, tc.wantFallback, strings.Contains(entry.Content, `description: "A dog"`))
			assert.Equal(t, int32(1), thumbnails.Load())
			if tc.thumbnailFails {
				assert.Zero(t, requests.Load())
			} else {
				assert.Equal(t, int32(1), requests.Load())
			}
		})
	}
}
