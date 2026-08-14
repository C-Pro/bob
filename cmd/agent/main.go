package main

import (
	"context"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"

	"bob/internal/config"
	"bob/internal/gateway"
	"bob/internal/llm"
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
		"geminiModel", cfg.GeminiModel,
		"geminiBaseURL", cfg.GeminiBaseURL,
		"townhallMaxParagraphs", cfg.TownhallMaxParagraphs,
		"dmMaxParagraphs", cfg.DMMaxParagraphs,
	)

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	httpClient := &http.Client{}
	llmClient := llm.NewClient(cfg, httpClient)
	gw := gateway.NewGateway(cfg, llmClient)

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
