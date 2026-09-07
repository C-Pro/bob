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

				msg := FormatProgressMessage(task, current)
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
// E.g.: "Looking for GetUserCall sites. Grep is running."
func FormatProgressMessage(task, current string) string {
	cleanCurrent := strings.TrimSpace(current)
	if cleanCurrent == "" {
		cleanCurrent = "Task"
	}

	cleanTask := strings.TrimSpace(task)
	if cleanTask == "" {
		return fmt.Sprintf("%s is running.", cleanCurrent)
	}

	lower := strings.ToLower(cleanTask)
	prefix := "Looking for "
	if strings.HasPrefix(lower, "looking for ") ||
		strings.HasPrefix(lower, "searching for ") ||
		strings.HasPrefix(lower, "running ") ||
		strings.HasPrefix(lower, "executing ") ||
		strings.HasPrefix(lower, "finding ") {
		// Capitalize first letter of task
		r := []rune(cleanTask)
		r[0] = []rune(strings.ToUpper(string(r[0])))[0]
		return fmt.Sprintf("%s. %s is running.", string(r), cleanCurrent)
	}

	return fmt.Sprintf("%s%s. %s is running.", prefix, cleanTask, cleanCurrent)
}
