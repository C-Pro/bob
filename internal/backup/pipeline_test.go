package backup

import (
	"bytes"
	"database/sql"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	_ "modernc.org/sqlite"
)

func TestEncodeAndDecodeSnapshotRoundtrip(t *testing.T) {
	tempDir := t.TempDir()
	dbPath := filepath.Join(tempDir, "source.db")

	db, err := sql.Open("sqlite", dbPath)
	require.NoError(t, err)

	_, err = db.Exec("CREATE TABLE messages (id INTEGER PRIMARY KEY, msg TEXT);")
	require.NoError(t, err)
	_, err = db.Exec("INSERT INTO messages (msg) VALUES ('hello from bob'), ('backup test');")
	require.NoError(t, err)
	_ = db.Close()

	secret := "test-secret-123"
	var buf bytes.Buffer
	err = EncodeSnapshot(dbPath, secret, &buf)
	require.NoError(t, err)
	artifact := buf.Bytes()
	assert.NotEmpty(t, artifact)
	assert.True(t, bytes.HasPrefix(artifact, []byte("BOBB\x01\x10")))

	// Restore to a new location
	restoredPath := filepath.Join(tempDir, "restored.db")
	err = DecodeSnapshot(bytes.NewReader(artifact), secret, restoredPath)
	require.NoError(t, err)

	restoredDB, err := sql.Open("sqlite", restoredPath)
	require.NoError(t, err)
	defer func() { _ = restoredDB.Close() }()

	var count int
	err = restoredDB.QueryRow("SELECT COUNT(*) FROM messages;").Scan(&count)
	require.NoError(t, err)
	assert.Equal(t, 2, count)

	var msg string
	err = restoredDB.QueryRow("SELECT msg FROM messages WHERE id = 1;").Scan(&msg)
	require.NoError(t, err)
	assert.Equal(t, "hello from bob", msg)
}

func TestDecodeSnapshotWrongSecret(t *testing.T) {
	tempDir := t.TempDir()
	dbPath := filepath.Join(tempDir, "source.db")

	err := os.WriteFile(dbPath, []byte("dummy sqlite database content"), 0600)
	require.NoError(t, err)

	secret := "correct-secret"
	var buf bytes.Buffer
	err = EncodeSnapshot(dbPath, secret, &buf)
	require.NoError(t, err)

	restoredPath := filepath.Join(tempDir, "restored.db")
	err = DecodeSnapshot(bytes.NewReader(buf.Bytes()), "wrong-secret", restoredPath)
	assert.ErrorContains(t, err, "failed to decrypt artifact")
	assert.NoFileExists(t, restoredPath)
}
