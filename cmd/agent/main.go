package main

import (
	"context"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"bob/internal/config"
	"bob/internal/gateway"
	"bob/internal/geoip"
	"bob/internal/llm"
	"bob/internal/store"
)

func main() {
	slog.Info("starting Besedka AI Agent service")

	cfg, err := config.LoadFromEnv()
	if err != nil {
		slog.Error("failed to load configuration", "error", err)
		os.Exit(1)
	}

	if err := cfg.Validate(true); err != nil {
		slog.Error("configuration validation failed", "error", err)
		os.Exit(1)
	}

	slog.Info("configuration loaded",
		"botHandle", cfg.BotHandle,
		"besedkaURL", cfg.BesedkaURL,
		"openAIModel", cfg.OpenAIModel,
		"openAIBaseURL", cfg.OpenAIBaseURL,
		"webSearchEnabled", cfg.TavilyAPIKey != "",
		"townhallMaxParagraphs", cfg.TownhallMaxParagraphs,
		"dmMaxParagraphs", cfg.DMMaxParagraphs,
		"msgRingBufferSize", cfg.MsgRingBufferSize,
		"dataDir", cfg.DataDir,
		"dbPath", cfg.DBPath("bob.db"),
	)

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	// Initialize SQLite database storage
	st, err := store.OpenOrCreate(cfg.DBPath("bob.db"))
	if err != nil {
		slog.Error("failed to initialize SQLite database", "error", err)
		os.Exit(1)
	}
	defer func() {
		if err := st.Close(); err != nil {
			slog.Error("error closing SQLite database", "error", err)
		}
	}()

	schemaVer, err := st.GetSchemaVersion(ctx)
	if err != nil {
		slog.Error("failed to query database schema version", "error", err)
		os.Exit(1)
	}
	slog.Info("database storage initialized", "path", cfg.DBPath("bob.db"), "schemaVersion", schemaVer)

	httpClient := &http.Client{Timeout: 15 * time.Second}

	// Query server GEOIP location on startup (random round-robin across providers, up to 3 attempts)
	loc, err := geoip.FetchLocation(ctx, httpClient)
	if err != nil {
		slog.Warn("could not determine server location via GEOIP", "error", err)
	} else if loc != nil {
		slog.Info("server location determined via GEOIP", "lat", loc.Lat, "lng", loc.Lng)
	}

	llmClient := llm.NewClient(cfg, httpClient)
	gw := gateway.NewGateway(cfg, llmClient)
	if loc != nil {
		gw.SetLocation(loc)
	}

	go func() {
		<-ctx.Done()
		slog.Info("shutdown signal received, closing agent gateway...")
		gw.Stop()
	}()

	slog.Info("starting agent gateway event loop")
	if err := gw.Start(ctx); err != nil && err != context.Canceled {
		slog.Error("gateway exited with error", "error", err)
		os.Exit(1)
	}

	slog.Info("agent service stopped gracefully")
}
