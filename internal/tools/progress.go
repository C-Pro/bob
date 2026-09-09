package tools

import (
	"fmt"
	"log/slog"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// ProgressReporter periodically reports execution progress to the user during long tasks.
type ProgressReporter struct {
	mu       sync.Mutex
	wg       sync.WaitGroup
	chatID   string
	task     string
	current  string
	sendFunc func(chatID, text string) error
	interval time.Duration
	stopCh   chan struct{}
	stopped  bool
	disabled bool
}

// NewProgressReporter creates a new progress reporter.
func NewProgressReporter(chatID, task string, sendFunc func(chatID, text string) error, interval time.Duration, disabled bool) *ProgressReporter {
	if interval <= 0 {
		interval = 30 * time.Second
	}
	return &ProgressReporter{
		chatID:   chatID,
		task:     strings.TrimSpace(task),
		current:  "Process",
		sendFunc: sendFunc,
		interval: interval,
		stopCh:   make(chan struct{}),
		disabled: disabled,
	}
}

// ProgressPrefix is prepended to periodic progress notifications to distinguish
// them from conversational model turns in chat UIs and message processors.
const ProgressPrefix = "⏳ "

// IsProgressMessage reports whether the message content matches a progress notification format.
func IsProgressMessage(content string) bool {
	clean := strings.TrimSpace(content)
	return strings.HasPrefix(clean, ProgressPrefix)
}

// Start initiates the periodic progress reporting background goroutine.
func (p *ProgressReporter) Start() {
	if p == nil || p.disabled || p.sendFunc == nil {
		return
	}

	p.mu.Lock()
	if p.stopped {
		p.mu.Unlock()
		return
	}
	p.wg.Add(1)
	p.mu.Unlock()

	go func() {
		defer p.wg.Done()
		ticker := time.NewTicker(p.interval)
		defer ticker.Stop()

		for {
			select {
			case <-p.stopCh:
				return
			case <-ticker.C:
				p.mu.Lock()
				if p.stopped {
					p.mu.Unlock()
					return
				}
				task := p.task
				current := p.current
				p.mu.Unlock()

				msg := ProgressPrefix + FormatProgressMessage(task, current)
				if err := p.sendFunc(p.chatID, msg); err != nil {
					slog.Warn("failed to send progress notification", "chatID", p.chatID, "error", err)
				}
			}
		}
	}()
}

// SetTask updates the high-level task description being worked on.
func (p *ProgressReporter) SetTask(task string) {
	if p == nil {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	clean := strings.TrimSpace(task)
	if clean != "" {
		p.task = clean
	}
}

// SetCurrent updates the active command or step currently in progress.
func (p *ProgressReporter) SetCurrent(current string) {
	if p == nil {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	clean := strings.TrimSpace(current)
	if clean != "" {
		p.current = clean
	}
}

// SetCommand extracts a friendly summary from a shell command string and updates current.
func (p *ProgressReporter) SetCommand(cmdStr string) {
	if p == nil {
		return
	}
	fields := strings.Fields(strings.TrimSpace(cmdStr))
	if len(fields) == 0 {
		p.SetCurrent("Command")
		return
	}
	base := filepath.Base(fields[0])
	base = strings.TrimLeft(base, "./")
	if base == "" {
		base = "Command"
	}
	// Capitalize first character
	r := []rune(base)
	r[0] = []rune(strings.ToUpper(string(r[0])))[0]
	p.SetCurrent(string(r))
}

// Stop terminates the reporting ticker and waits for any active tick to complete.
func (p *ProgressReporter) Stop() {
	if p == nil || p.disabled {
		return
	}
	p.mu.Lock()
	if p.stopped {
		p.mu.Unlock()
		return
	}
	p.stopped = true
	close(p.stopCh)
	p.mu.Unlock()

	p.wg.Wait()
}

// FormatProgressMessage constructs a human-readable progress report.
// If a multi-word description is provided (e.g. by the model via sandbox_exec),
// it is reported directly. Otherwise, it falls back to reporting task and command status.
func FormatProgressMessage(task, current string) string {
	cleanCurrent := strings.TrimSpace(current)
	cleanTask := strings.TrimSpace(task)

	// If current is a model-generated description (multi-word phrase), report it directly
	if cleanCurrent != "" && cleanCurrent != "Process" && cleanCurrent != "Task" && strings.Contains(cleanCurrent, " ") {
		r := []rune(cleanCurrent)
		r[0] = []rune(strings.ToUpper(string(r[0])))[0]
		res := string(r)
		if !strings.HasSuffix(res, ".") && !strings.HasSuffix(res, "!") && !strings.HasSuffix(res, "?") {
			res += "."
		}
		return res
	}

	if cleanCurrent == "" {
		cleanCurrent = "Task"
	}

	if cleanTask == "" {
		return fmt.Sprintf("%s is running.", cleanCurrent)
	}

	lower := strings.ToLower(cleanTask)
	if strings.HasPrefix(lower, "looking for ") ||
		strings.HasPrefix(lower, "searching for ") ||
		strings.HasPrefix(lower, "running ") ||
		strings.HasPrefix(lower, "executing ") {
		r := []rune(cleanTask)
		r[0] = []rune(strings.ToUpper(string(r[0])))[0]
		return fmt.Sprintf("%s. %s is running.", string(r), cleanCurrent)
	}

	return fmt.Sprintf("Looking for %s. %s is running.", cleanTask, cleanCurrent)
}
