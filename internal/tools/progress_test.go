package tools

import (
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFormatProgressMessage(t *testing.T) {
	tests := []struct {
		name     string
		task     string
		current  string
		expected string
	}{
		{
			name:     "standard task and command",
			task:     "GetUserCall sites",
			current:  "Grep",
			expected: "Looking for GetUserCall sites. Grep is running.",
		},
		{
			name:     "task with existing Looking for prefix",
			task:     "looking for GetUserCall sites",
			current:  "Grep",
			expected: "Looking for GetUserCall sites. Grep is running.",
		},
		{
			name:     "task with running prefix",
			task:     "running unit tests",
			current:  "Go",
			expected: "Running unit tests. Go is running.",
		},
		{
			name:     "empty task",
			task:     "",
			current:  "Find",
			expected: "Find is running.",
		},
		{
			name:     "empty current",
			task:     "database records",
			current:  "",
			expected: "Looking for database records. Task is running.",
		},
		{
			name:     "model generated description for sandbox_exec",
			task:     "Please calculate factorial of 6",
			current:  "Calculating factorial of 6",
			expected: "Calculating factorial of 6.",
		},
		{
			name:     "model generated description running tests",
			task:     "test the database",
			current:  "Running unit tests suite",
			expected: "Running unit tests suite.",
		},
		{
			name:     "model generated description with trailing period",
			task:     "check disk space",
			current:  "Checking available disk space.",
			expected: "Checking available disk space.",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			out := FormatProgressMessage(tc.task, tc.current)
			assert.Equal(t, tc.expected, out)
		})
	}
}

func TestProgressReporter_TickerExecution(t *testing.T) {
	var mu sync.Mutex
	var sentMessages []string

	sendFunc := func(chatID, text string) error {
		mu.Lock()
		defer mu.Unlock()
		sentMessages = append(sentMessages, text)
		return nil
	}

	reporter := NewProgressReporter("chat_123", "GetUserCall sites", sendFunc, 20*time.Millisecond, false)
	reporter.SetCommand("grep -rn GetUserCall .")
	reporter.Start()

	// Wait for at least 2 ticks
	time.Sleep(55 * time.Millisecond)
	reporter.Stop()

	mu.Lock()
	count := len(sentMessages)
	msgs := append([]string(nil), sentMessages...)
	mu.Unlock()

	require.GreaterOrEqual(t, count, 2)
	assert.Equal(t, ProgressPrefix+"Looking for GetUserCall sites. Grep is running.", msgs[0])
	assert.True(t, IsProgressMessage(msgs[0]))

	// Ensure no more messages sent after stop
	time.Sleep(30 * time.Millisecond)
	mu.Lock()
	afterCount := len(sentMessages)
	mu.Unlock()
	assert.Equal(t, count, afterCount)
}

func TestIsProgressMessage(t *testing.T) {
	assert.True(t, IsProgressMessage("⏳ Looking for files. Grep is running."))
	assert.True(t, IsProgressMessage("  ⏳ Still working on task...  "))
	assert.False(t, IsProgressMessage("Looking for files. Grep is running."))
	assert.False(t, IsProgressMessage("Hello world!"))
	assert.False(t, IsProgressMessage(""))
}

func TestProgressReporter_Disabled(t *testing.T) {
	var sentMessages []string
	sendFunc := func(chatID, text string) error {
		sentMessages = append(sentMessages, text)
		return nil
	}

	reporter := NewProgressReporter("chat_123", "task", sendFunc, 10*time.Millisecond, true)
	reporter.Start()
	time.Sleep(25 * time.Millisecond)
	reporter.Stop()

	assert.Empty(t, sentMessages)
}
