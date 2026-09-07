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
			name:     "imperative verb please",
			task:     "Please check the server logs",
			current:  "Grep",
			expected: "Please check the server logs. Grep is running.",
		},
		{
			name:     "imperative verb calculate",
			task:     "calculate total memory consumption",
			current:  "Bash",
			expected: "Calculate total memory consumption. Bash is running.",
		},
		{
			name:     "imperative verb find",
			task:     "find missing configs",
			current:  "Find",
			expected: "Find missing configs. Find is running.",
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
	assert.Equal(t, "Looking for GetUserCall sites. Grep is running.", msgs[0])

	// Ensure no more messages sent after stop
	time.Sleep(30 * time.Millisecond)
	mu.Lock()
	afterCount := len(sentMessages)
	mu.Unlock()
	assert.Equal(t, count, afterCount)
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
