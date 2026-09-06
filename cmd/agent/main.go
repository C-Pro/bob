package main

import (
	"context"
	"database/sql"
	"flag"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"bob/internal/backup"
	"bob/internal/config"
	"bob/internal/gateway"
	"bob/internal/geoip"
	"bob/internal/llm"
	"bob/internal/memory"
	"bob/internal/objectstore"
	"bob/internal/store"

	"github.com/liliang-cn/cortexdb/v2/pkg/cortexdb"
)

const mainDBFname = "bob.db"

func main() {
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	regenerateFlag := flag.Bool("regenerate-vectors", false, "Regenerate vector embeddings for all memory chunks across all chat databases in DATA_DIR and exit")
	reembedFlag := flag.Bool("reembed", false, "Alias for -regenerate-vectors")
	initDBFlag := flag.Bool("init-db", false, "Skip S3 database recovery on startup and initialize fresh empty databases")
	backupFlag := flag.Bool("backup", false, "Trigger an immediate full backup of all databases to S3 and exit")
	dataDirFlag := flag.String("data-dir", "", "Override DATA_DIR directory path")
	flag.Parse()

	cfg, err := config.LoadFromEnv()
	if err != nil {
		slog.Error("failed to load configuration", "error", err)
		os.Exit(1)
	}
	if *dataDirFlag != "" {
		cfg.DataDir = *dataDirFlag
	}

	if *regenerateFlag || *reembedFlag {
		slog.Info("running vector regeneration tool")
		var embedder cortexdb.Embedder
		if cfg.EmbeddingModel != "" {
			llmClient := llm.NewClient(cfg, nil)
			embedder = llm.NewEmbedder(llmClient, cfg.EmbeddingModel)
		} else {
			localEmb, err := llm.NewLocalEmbedder(cfg)
			if err != nil {
				slog.Error("failed to initialize local embedder for vector regeneration", "error", err)
				os.Exit(1)
			}
			defer func() { _ = localEmb.Close() }()
			embedder = localEmb
		}

		modelName := cfg.EmbeddingModel
		if modelName == "" {
			modelName = llm.DefaultLocalModel
		}
		slog.Info("starting vector regeneration", "dataDir", cfg.DataDir, "model", modelName, "dim", embedder.Dim())
		report, err := memory.RegenerateAllVectors(ctx, cfg, embedder, 16)
		if err != nil {
			slog.Error("vector regeneration failed", "error", err)
			os.Exit(1)
		}
		slog.Info("vector regeneration completed",
			"databases", report.TotalDatabases,
			"chunks", report.TotalChunks,
			"reembedded", report.Reembedded,
			"failed", report.Failed,
		)
		if report.Failed > 0 {
			slog.Warn("some memory chunks failed to re-embed", "errors", report.Errors)
			os.Exit(1)
		}
		return
	}

	// Initialize S3 object storage client if configured
	var objClient *objectstore.Client
	if cfg.S3Enabled() {
		objClient, err = objectstore.New(objectstore.Config{
			Endpoint:  cfg.S3Endpoint,
			Region:    cfg.S3Region,
			Bucket:    cfg.S3Bucket,
			AccessKey: cfg.S3AccessKey,
			SecretKey: cfg.S3SecretKey,
			PathStyle: cfg.S3PathStyle,
		})
		if err != nil {
			slog.Error("failed to initialize object storage client", "error", err)
			os.Exit(1)
		}
	}

	// On-demand backup execution via CLI flag
	if *backupFlag {
		if objClient == nil {
			slog.Error("S3 object storage is not configured; cannot take database backup")
			os.Exit(1)
		}
		if err := cfg.Validate(false); err != nil {
			slog.Error("configuration validation failed for backup", "error", err)
			os.Exit(1)
		}

		slog.Info("executing on-demand database backup to object storage", "bucket", cfg.S3Bucket, "prefix", cfg.S3BackupPrefix)
		scheduler := backup.NewScheduler(cfg, objClient, nil)
		if err := scheduler.DoBackup(ctx); err != nil {
			slog.Error("on-demand backup failed", "error", err)
			os.Exit(1)
		}
		slog.Info("on-demand database backup completed successfully")
		return
	}

	slog.Info("starting Besedka AI Agent service")

	if err := cfg.Validate(true); err != nil {
		slog.Error("configuration validation failed", "error", err)
		os.Exit(1)
	}

	slog.Info("configuration loaded",
		"botHandle", cfg.BotHandle,
		"besedkaURL", cfg.BesedkaURL,
		"openAIModel", cfg.OpenAIModel,
		"openAIBaseURL", cfg.OpenAIBaseURL,
		"embeddingModel", cfg.EmbeddingModel,
		"webSearchEnabled", cfg.TavilyAPIKey != "",
		"townhallMaxParagraphs", cfg.TownhallMaxParagraphs,
		"dmMaxParagraphs", cfg.DMMaxParagraphs,
		"msgRingBufferSize", cfg.MsgRingBufferSize,
		"dataDir", cfg.DataDir,
		"dbPath", cfg.DBPath(mainDBFname),
		"s3Enabled", cfg.S3Enabled(),
	)

	// Recover databases from object storage if missing locally
	if objClient != nil {
		recovered, err := backup.RecoverDBsIfMissing(ctx, cfg, objClient, *initDBFlag)
		if err != nil {
			slog.Error("database recovery from object storage failed", "error", err)
			os.Exit(1)
		} else if recovered {
			slog.Info("database recovery from object storage completed", "dataDir", cfg.DataDir)
		}
	}

	// Initialize SQLite database storage
	st, err := store.OpenOrCreate(cfg.DBPath(mainDBFname))
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
	slog.Info("database storage initialized", "path", cfg.DBPath(mainDBFname), "schemaVersion", schemaVer)

	httpClient := &http.Client{Timeout: 15 * time.Second}

	// Query server GEOIP location on startup (random round-robin across providers, up to 3 attempts)
	loc, err := geoip.FetchLocation(ctx, httpClient)
	if err != nil {
		slog.Warn("could not determine server location via GEOIP", "error", err)
	} else if loc != nil {
		slog.Info("server location determined via GEOIP", "lat", loc.Lat, "lng", loc.Lng)
	}

	llmClient := llm.NewClient(cfg, nil)
	gw := gateway.NewGateway(cfg, llmClient)
	if loc != nil {
		gw.SetLocation(loc)
	}

	// Start periodic backup scheduler if S3 is enabled
	var scheduler *backup.Scheduler
	if objClient != nil {
		activeDBsProvider := func() map[string]*sql.DB {
			active := map[string]*sql.DB{
				mainDBFname: st.DB(),
			}
			if gw.MemoryManager() != nil {
				for name, db := range gw.MemoryManager().ActiveDBs() {
					active[name] = db
				}
			}
			return active
		}
		scheduler = backup.NewScheduler(cfg, objClient, activeDBsProvider)
		go func() {
			if err := scheduler.Run(ctx); err != nil && err != context.Canceled {
				slog.Error("backup scheduler stopped with error", "error", err)
			}
		}()
	}

	go func() {
		<-ctx.Done()
		slog.Info("shutdown signal received, closing agent gateway...")
		gw.Stop()

		// Perform final database backup on shutdown
		if scheduler != nil {
			slog.Info("performing final database backup on shutdown...")
			bCtx, bCancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer bCancel()
			if err := scheduler.DoBackup(bCtx); err != nil {
				slog.Error("final shutdown database backup failed", "error", err)
			} else {
				slog.Info("final shutdown database backup completed")
			}
		}
	}()

	slog.Info("starting agent gateway event loop")
	if err := gw.Start(ctx); err != nil && err != context.Canceled {
		slog.Error("gateway exited with error", "error", err)
		os.Exit(1)
	}

	slog.Info("agent service stopped gracefully")
}
