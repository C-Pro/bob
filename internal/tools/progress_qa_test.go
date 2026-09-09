package tools

import (
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestProgressReporter_RapidStartStop verifies that rapid lifecycle transitions,
// repeated stops, out-of-order starts, and nil pointer receivers do not panic or deadlock.
func TestProgressReporter_RapidStartStop(t *testing.T) {
	sendFunc := func(chatID, text string) error {
		return nil
	}

	t.Run("rapid start and stop in tight loop", func(t *testing.T) {
		for i := 0; i < 100; i++ {
			reporter := NewProgressReporter("chat_1", "task", sendFunc, 5*time.Millisecond, false)
			reporter.Start()
			reporter.Stop()
		}
	})

	t.Run("multiple calls to stop are idempotent", func(t *testing.T) {
		reporter := NewProgressReporter("chat_1", "task", sendFunc, 10*time.Millisecond, false)
		reporter.Start()
		reporter.Stop()
		assert.NotPanics(t, func() {
			reporter.Stop()
			reporter.Stop()
			reporter.Stop()
		})
	})

	t.Run("stop before start is safe", func(t *testing.T) {
		reporter := NewProgressReporter("chat_1", "task", sendFunc, 10*time.Millisecond, false)
		assert.NotPanics(t, func() {
			reporter.Stop()
		})
	})

	t.Run("nil reporter methods are safe no-ops", func(t *testing.T) {
		var nilReporter *ProgressReporter
		assert.NotPanics(t, func() {
			nilReporter.Start()
			nilReporter.Stop()
			nilReporter.SetTask("task")
			nilReporter.SetCurrent("step")
			nilReporter.SetCommand("grep")
		})
	})

	t.Run("disabled reporter start and stop", func(t *testing.T) {
		reporter := NewProgressReporter("chat_1", "task", sendFunc, 10*time.Millisecond, true)
		assert.NotPanics(t, func() {
			reporter.Start()
			reporter.SetTask("new task")
			reporter.SetCurrent("new step")
			reporter.Stop()
		})
	})

	t.Run("non-positive interval defaults to 30s", func(t *testing.T) {
		rZero := NewProgressReporter("chat_1", "task", sendFunc, 0, false)
		assert.Equal(t, 30*time.Second, rZero.interval)

		rNeg := NewProgressReporter("chat_1", "task", sendFunc, -10*time.Second, false)
		assert.Equal(t, 30*time.Second, rNeg.interval)
	})
}

// TestProgressReporter_ConcurrentUpdates verifies thread safety under high contention
// with multiple concurrent writers updating task, step, and command while ticking.
func TestProgressReporter_ConcurrentUpdates(t *testing.T) {
	var sendCount int64
	sendFunc := func(chatID, text string) error {
		atomic.AddInt64(&sendCount, 1)
		return nil
	}

	reporter := NewProgressReporter("chat_concurrent", "Initial Task", sendFunc, 5*time.Millisecond, false)
	reporter.Start()
	defer reporter.Stop()

	const numGoroutines = 15
	const iterationsPerGoroutine = 50

	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		go func(workerID int) {
			defer wg.Done()
			for j := 0; j < iterationsPerGoroutine; j++ {
				reporter.SetTask(fmt.Sprintf("Worker %d task %d", workerID, j))
				reporter.SetCurrent(fmt.Sprintf("Worker %d step %d", workerID, j))
				reporter.SetCommand(fmt.Sprintf("cmd_%d_%d arg1 arg2", workerID, j))
			}
		}(i)
	}

	wg.Wait()
	time.Sleep(20 * time.Millisecond)

	reporter.Stop()
	assert.GreaterOrEqual(t, atomic.LoadInt64(&sendCount), int64(1))
}

// TestProgressReporter_SendFuncFailureHandling verifies that sendFunc delivery errors
// or nil sendFunc do not crash the ticker goroutine or cause deadlocks.
func TestProgressReporter_SendFuncFailureHandling(t *testing.T) {
	t.Run("sendFunc error does not terminate or panic ticker", func(t *testing.T) {
		var callCount int64
		errSendFunc := func(chatID, text string) error {
			atomic.AddInt64(&callCount, 1)
			return errors.New("simulated network connection reset")
		}

		reporter := NewProgressReporter("chat_err", "Error task", errSendFunc, 10*time.Millisecond, false)
		reporter.Start()

		time.Sleep(45 * time.Millisecond)
		reporter.Stop()

		require.GreaterOrEqual(t, atomic.LoadInt64(&callCount), int64(2))
	})

	t.Run("nil sendFunc handled gracefully without panic", func(t *testing.T) {
		reporter := NewProgressReporter("chat_nil", "Nil task", nil, 10*time.Millisecond, false)
		assert.NotPanics(t, func() {
			reporter.Start()
			reporter.SetTask("another task")
			reporter.Stop()
		})
	})
}

// TestProgressReporter_SetCommand_EdgeCases verifies edge case inputs for SetCommand.
func TestProgressReporter_SetCommand_EdgeCases(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"", "Command"},
		{"   ", "Command"},
		{"///", "Command"},
		{"././", "Command"},
		{"/bin/bash", "Bash"},
		{"/usr/local/bin/python3.11 -m unittest", "Python3.11"},
		{"grep -rn pattern .", "Grep"},
		{"./run_tests.sh --flag", "Run_tests.sh"},
		{"12345", "12345"},
		{"🚀launch", "🚀launch"},
		{"curl -fsSL https://example.com", "Curl"},
	}

	for _, tc := range tests {
		t.Run(tc.input, func(t *testing.T) {
			r := NewProgressReporter("c", "t", func(c, text string) error { return nil }, time.Minute, false)
			r.SetCommand(tc.input)
			assert.Equal(t, tc.expected, r.current)
		})
	}
}

// TestProgressReporter_FormatProgressMessage_EdgeCases verifies edge cases in progress formatting.
func TestProgressReporter_FormatProgressMessage_EdgeCases(t *testing.T) {
	tests := []struct {
		name     string
		task     string
		current  string
		expected string
	}{
		{
			name:     "empty task and empty current",
			task:     "",
			current:  "",
			expected: "Task is running.",
		},
		{
			name:     "uppercase LOOKING FOR task preserves task casing after capitalization",
			task:     "LOOKING FOR bug in code",
			current:  "Grep",
			expected: "LOOKING FOR bug in code. Grep is running.",
		},
		{
			name:     "uppercase RUNNING task preserves task casing after capitalization",
			task:     "RUNNING compilation",
			current:  "Go",
			expected: "RUNNING compilation. Go is running.",
		},
		{
			name:     "task with leading and trailing newlines/tabs",
			task:     "\n\t  important task  \n",
			current:  "\n  CustomStep  \t",
			expected: "Looking for important task. CustomStep is running.",
		},
		{
			name:     "task without prefix",
			task:     "database indexing",
			current:  "Sqlite",
			expected: "Looking for database indexing. Sqlite is running.",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			out := FormatProgressMessage(tc.task, tc.current)
			assert.Equal(t, tc.expected, out)
		})
	}
}
