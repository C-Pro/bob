//go:build integration

package backup

import (
	"context"
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"bob/internal/config"
	"bob/internal/objectstore"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	_ "modernc.org/sqlite"
)

func newIntegrationClient(t *testing.T) (*objectstore.Client, *config.Config) {
	t.Helper()
	endpoint := os.Getenv("S3_ENDPOINT")
	bucket := os.Getenv("S3_BUCKET")
	if endpoint == "" || bucket == "" {
		t.Skip("set S3_ENDPOINT and S3_BUCKET to run integration tests")
	}

	accessKey := os.Getenv("S3_ACCESS_KEY")
	secretKey := os.Getenv("S3_SECRET_KEY")
	region := os.Getenv("S3_REGION")
	if region == "" {
		region = "us-east-1"
	}

	c, err := objectstore.New(objectstore.Config{
		Endpoint:  endpoint,
		Region:    region,
		Bucket:    bucket,
		AccessKey: accessKey,
		SecretKey: secretKey,
		PathStyle: true,
	})
	require.NoError(t, err)

	cfg := &config.Config{
		S3Endpoint:       endpoint,
		S3Region:         region,
		S3Bucket:         bucket,
		S3AccessKey:      accessKey,
		S3SecretKey:      secretKey,
		S3PathStyle:      true,
		S3BackupPrefix:   fmt.Sprintf("it-bob-%d/", time.Now().UnixNano()),
		S3BackupKeep:     3,
		S3BackupInterval: time.Hour,
		Secret:           "integration-secret-test",
	}

	return c, cfg
}

func cleanupPrefix(t *testing.T, client *objectstore.Client, prefix string) {
	t.Helper()
	ctx := context.Background()
	objs, err := client.List(ctx, prefix)
	if err == nil {
		for _, o := range objs {
			_ = client.Delete(ctx, o.Key)
		}
	}
}

func TestIntegrationBackupAndRecovery(t *testing.T) {
	client, cfg := newIntegrationClient(t)
	defer cleanupPrefix(t, client, cfg.S3BackupPrefix)

	tempDir := t.TempDir()
	dataDir := filepath.Join(tempDir, "data")
	require.NoError(t, os.MkdirAll(dataDir, 0755))
	cfg.DataDir = dataDir

	// Create test databases
	bobDBPath := filepath.Join(dataDir, "bob.db")
	db1, err := sql.Open("sqlite", bobDBPath)
	require.NoError(t, err)
	_, err = db1.Exec("CREATE TABLE records (id INTEGER PRIMARY KEY, title TEXT); INSERT INTO records (title) VALUES ('bob live data');")
	require.NoError(t, err)

	thDBPath := filepath.Join(dataDir, "townhall.db")
	db2, err := sql.Open("sqlite", thDBPath)
	require.NoError(t, err)
	_, err = db2.Exec("CREATE TABLE memory (id INTEGER PRIMARY KEY, content TEXT); INSERT INTO memory (content) VALUES ('townhall history');")
	require.NoError(t, err)

	activeDBs := func() map[string]*sql.DB {
		return map[string]*sql.DB{
			"bob.db":      db1,
			"townhall.db": db2,
		}
	}

	scheduler := NewScheduler(cfg, client, activeDBs)
	ctx := context.Background()

	// 1. Run backup
	err = scheduler.DoBackup(ctx)
	require.NoError(t, err)

	// Close active DBs
	require.NoError(t, db1.Close())
	require.NoError(t, db2.Close())

	// 2. Verify objects exist in bucket
	objs, err := client.List(ctx, cfg.S3BackupPrefix)
	require.NoError(t, err)
	assert.GreaterOrEqual(t, len(objs), 3) // manifest + bob.db + townhall.db

	manifest, err := FetchManifest(ctx, client, cfg.S3BackupPrefix)
	require.NoError(t, err)
	assert.Contains(t, manifest.Databases, "bob.db")
	assert.Contains(t, manifest.Databases, "townhall.db")

	// 3. Simulate total local disk loss
	require.NoError(t, os.RemoveAll(dataDir))
	assert.NoFileExists(t, bobDBPath)
	assert.NoFileExists(t, thDBPath)

	// 4. Recover from object storage
	recovered, err := RecoverDBsIfMissing(ctx, cfg, client, false)
	require.NoError(t, err)
	assert.True(t, recovered)

	// 5. Verify restored data integrity
	assert.FileExists(t, bobDBPath)
	assert.FileExists(t, thDBPath)

	rdb1, err := sql.Open("sqlite", bobDBPath)
	require.NoError(t, err)
	defer func() { _ = rdb1.Close() }()

	var title string
	err = rdb1.QueryRow("SELECT title FROM records WHERE id = 1;").Scan(&title)
	require.NoError(t, err)
	assert.Equal(t, "bob live data", title)

	rdb2, err := sql.Open("sqlite", thDBPath)
	require.NoError(t, err)
	defer func() { _ = rdb2.Close() }()

	var content string
	err = rdb2.QueryRow("SELECT content FROM memory WHERE id = 1;").Scan(&content)
	require.NoError(t, err)
	assert.Equal(t, "townhall history", content)
}

func TestIntegrationWrongSecretFailsRecovery(t *testing.T) {
	client, cfg := newIntegrationClient(t)
	defer cleanupPrefix(t, client, cfg.S3BackupPrefix)

	dataDir := t.TempDir()
	cfg.DataDir = dataDir

	dbPath := filepath.Join(dataDir, "bob.db")
	db, err := sql.Open("sqlite", dbPath)
	require.NoError(t, err)
	_, err = db.Exec("CREATE TABLE records (id INTEGER PRIMARY KEY, val TEXT); INSERT INTO records (val) VALUES ('test');")
	require.NoError(t, err)
	_ = db.Close()

	scheduler := NewScheduler(cfg, client, nil)
	ctx := context.Background()
	require.NoError(t, scheduler.DoBackup(ctx))

	// Simulate disk loss
	require.NoError(t, os.Remove(dbPath))

	// Try recovering with wrong secret
	cfgWrong := *cfg
	cfgWrong.Secret = "completely-wrong-secret"

	_, err = RecoverDBsIfMissing(ctx, &cfgWrong, client, false)
	assert.Error(t, err)
	assert.NoFileExists(t, dbPath)
}

func TestIntegrationRetentionPruning(t *testing.T) {
	client, cfg := newIntegrationClient(t)
	defer cleanupPrefix(t, client, cfg.S3BackupPrefix)

	dataDir := t.TempDir()
	cfg.DataDir = dataDir
	cfg.S3BackupKeep = 2

	dbPath := filepath.Join(dataDir, "bob.db")
	db, err := sql.Open("sqlite", dbPath)
	require.NoError(t, err)
	_, err = db.Exec("CREATE TABLE records (id INTEGER PRIMARY KEY);")
	require.NoError(t, err)
	_ = db.Close()

	scheduler := NewScheduler(cfg, client, nil)
	ctx := context.Background()

	baseTime := time.Date(2026, 8, 31, 10, 0, 0, 0, time.UTC)
	for i := 0; i < 4; i++ {
		ts := baseTime.Add(time.Duration(i) * time.Hour)
		scheduler.now = func() time.Time { return ts }
		require.NoError(t, scheduler.DoBackup(ctx))
	}

	objs, err := client.List(ctx, cfg.S3BackupPrefix+"bob.db/")
	require.NoError(t, err)
	assert.Len(t, objs, 2, "expected exactly 2 backups retained after pruning")
}
