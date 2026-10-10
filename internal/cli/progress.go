package cli

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"regexp"
	"strings"
	"sync"
	"time"

	"bob/internal/agentapi"
	"bob/internal/fsm"

	"github.com/mattn/go-isatty"
)

var exitCodeRegex = regexp.MustCompile(`Exit Code: (-?\d+)`)

// OutputWriter coordinates synchronized writing to stdout for progress events and final answer.
type OutputWriter struct {
	mu          sync.Mutex
	w           io.Writer
	isTTY       bool
	hasProgress bool
	firstErr    error
}

// NewOutputWriter creates an OutputWriter checking if w is a character device (TTY).
func NewOutputWriter(w io.Writer) *OutputWriter {
	tty := false
	if f, ok := w.(*os.File); ok {
		tty = isatty.IsTerminal(f.Fd()) || isatty.IsCygwinTerminal(f.Fd())
	}
	return &OutputWriter{
		w:     w,
		isTTY: tty,
	}
}

// WriteStepLine writes a single progress line under mutex lock.
func (ow *OutputWriter) WriteStepLine(line string) {
	ow.mu.Lock()
	defer ow.mu.Unlock()
	ow.hasProgress = true
	_, err := fmt.Fprintln(ow.w, line)
	if err != nil && ow.firstErr == nil {
		ow.firstErr = err
	}
}

// WriteFinalAnswer writes the final completion answer, preceded by a blank separator line
// if any progress step lines were printed.
func (ow *OutputWriter) WriteFinalAnswer(answer string) error {
	ow.mu.Lock()
	defer ow.mu.Unlock()

	if ow.hasProgress {
		if _, err := fmt.Fprintln(ow.w); err != nil && ow.firstErr == nil {
			ow.firstErr = err
		}
	}

	trimmed := strings.TrimRight(answer, "\r\n")
	if trimmed != "" {
		if _, err := fmt.Fprintln(ow.w, trimmed); err != nil && ow.firstErr == nil {
			ow.firstErr = err
		}
	}
	return ow.firstErr
}

// Err returns the first write error encountered, if any.
func (ow *OutputWriter) Err() error {
	ow.mu.Lock()
	defer ow.mu.Unlock()
	return ow.firstErr
}

// ProgressRenderer implements both agentapi.ProgressObserver and fsm.ProgressObserver.
type ProgressRenderer struct {
	writer *OutputWriter
	quiet  bool
	seen   map[string]struct{}
	mu     sync.Mutex
}

// NewProgressRenderer creates a new progress renderer.
func NewProgressRenderer(writer *OutputWriter, quiet bool) *ProgressRenderer {
	return &ProgressRenderer{
		writer: writer,
		quiet:  quiet,
		seen:   make(map[string]struct{}),
	}
}

// Observe implements agentapi.ProgressObserver.
func (r *ProgressRenderer) Observe(ctx context.Context, ev agentapi.ProgressEvent) {
	if r.quiet || r.writer == nil {
		return
	}
	if ev.Type == agentapi.ProgressStepUpdated && ev.Step != nil {
		step := ev.Step
		if step.Status == agentapi.StepCompleted || step.Status == agentapi.StepFailed || step.Status == agentapi.StepTimedOut {
			r.mu.Lock()
			if _, exists := r.seen[step.ID]; exists {
				r.mu.Unlock()
				return
			}
			r.seen[step.ID] = struct{}{}
			r.mu.Unlock()

			r.renderStepLine(step.Call.Name, step.Call.Arguments, string(step.Status), step.Error, "", 0)
		}
	}
}

// OnStepsPrepared implements fsm.ProgressObserver.
func (r *ProgressRenderer) OnStepsPrepared(ctx context.Context, steps []fsm.FSMStep) {}

// OnStepUpdate implements fsm.ProgressObserver.
func (r *ProgressRenderer) OnStepUpdate(ctx context.Context, step *fsm.FSMStep) {
	if r.quiet || r.writer == nil || step == nil {
		return
	}
	if step.Status == fsm.StepStatusCompleted || step.Status == fsm.StepStatusFailed || step.Status == fsm.StepStatusTimedOut {
		r.mu.Lock()
		if _, exists := r.seen[step.ID]; exists {
			r.mu.Unlock()
			return
		}
		r.seen[step.ID] = struct{}{}
		r.mu.Unlock()

		var duration time.Duration
		if step.StartedAt != nil && step.CompletedAt != nil && *step.CompletedAt >= *step.StartedAt {
			duration = time.Duration(*step.CompletedAt-*step.StartedAt) * time.Millisecond
		}

		r.renderStepLine(step.ToolName, step.ArgsJSON, string(step.Status), step.ErrorText, step.ResultJSON, duration)
	}
}

// OnRunFinished implements fsm.ProgressObserver.
func (r *ProgressRenderer) OnRunFinished(ctx context.Context, status fsm.RunStatus, runErr error) {}

func (r *ProgressRenderer) renderStepLine(toolName, argsJSON, status, errorText, resultJSON string, dur time.Duration) {
	summary := summarizeArgs(toolName, argsJSON)

	durStr := ""
	if dur > 0 {
		durStr = dur.Round(time.Millisecond).String()
	}

	exitCodeStr := ""
	if resultJSON != "" && toolName == "execute" {
		if matches := exitCodeRegex.FindStringSubmatch(resultJSON); len(matches) > 1 {
			if matches[1] != "0" {
				exitCodeStr = fmt.Sprintf("exit %s", matches[1])
			}
		}
	}

	suffixParts := make([]string, 0, 2)
	if exitCodeStr != "" {
		suffixParts = append(suffixParts, exitCodeStr)
	}
	if durStr != "" {
		suffixParts = append(suffixParts, durStr)
	}
	suffix := ""
	if len(suffixParts) > 0 {
		suffix = fmt.Sprintf("(%s)", strings.Join(suffixParts, ", "))
	}

	isSuccess := status == string(fsm.StepStatusCompleted) || status == string(agentapi.StepCompleted)
	if exitCodeStr != "" {
		isSuccess = false
	}

	var line string
	if r.writer.isTTY {
		if isSuccess {
			line = fmt.Sprintf("\x1b[36m•\x1b[0m \x1b[1m%s\x1b[0m", toolName)
			if summary != "" {
				line += " " + summary
			}
			if suffix != "" {
				line += " \x1b[2m" + suffix + "\x1b[0m"
			}
		} else {
			line = fmt.Sprintf("\x1b[31m✗\x1b[0m \x1b[1m%s\x1b[0m", toolName)
			if summary != "" {
				line += " " + summary
			}
			if errorText != "" {
				line += fmt.Sprintf(" \x1b[31m(%s)\x1b[0m", sanitizeString(errorText, 60))
			} else if suffix != "" {
				line += " \x1b[31m" + suffix + "\x1b[0m"
			}
		}
	} else {
		if isSuccess {
			line = "• " + toolName
			if summary != "" {
				line += " " + summary
			}
			if suffix != "" {
				line += " " + suffix
			}
		} else {
			line = "✗ " + toolName
			if summary != "" {
				line += " " + summary
			}
			if errorText != "" {
				line += fmt.Sprintf(" (%s)", sanitizeString(errorText, 60))
			} else if suffix != "" {
				line += " " + suffix
			}
		}
	}

	r.writer.WriteStepLine(line)
}

func summarizeArgs(toolName, argsJSON string) string {
	clean := strings.TrimSpace(argsJSON)
	if clean == "" || clean == "{}" {
		return ""
	}
	switch toolName {
	case "execute":
		var m struct {
			Command string `json:"command"`
		}
		if err := json.Unmarshal([]byte(clean), &m); err == nil && m.Command != "" {
			return sanitizeString(m.Command, 60)
		}
	case "web_search":
		var m struct {
			Query string `json:"query"`
		}
		if err := json.Unmarshal([]byte(clean), &m); err == nil && m.Query != "" {
			return fmt.Sprintf("%q", sanitizeString(m.Query, 60))
		}
	case "web_fetch":
		var m struct {
			URL string `json:"url"`
		}
		if err := json.Unmarshal([]byte(clean), &m); err == nil && m.URL != "" {
			return sanitizeString(m.URL, 60)
		}
	case "discover_memories", "discover_skills":
		var m struct {
			Query string `json:"query"`
		}
		if err := json.Unmarshal([]byte(clean), &m); err == nil && m.Query != "" {
			return fmt.Sprintf("%q", sanitizeString(m.Query, 60))
		}
	case "load_memory":
		var m struct {
			Key string `json:"key"`
		}
		if err := json.Unmarshal([]byte(clean), &m); err == nil && m.Key != "" {
			return sanitizeString(m.Key, 60)
		}
	case "load_skill":
		var m struct {
			Name string `json:"name"`
		}
		if err := json.Unmarshal([]byte(clean), &m); err == nil && m.Name != "" {
			return sanitizeString(m.Name, 60)
		}
	}
	return sanitizeString(clean, 60)
}

func sanitizeString(s string, maxLen int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	s = strings.ReplaceAll(s, "\r", " ")
	s = strings.ReplaceAll(s, "\t", " ")
	s = strings.TrimSpace(s)
	runes := []rune(s)
	if len(runes) > maxLen {
		return string(runes[:maxLen-3]) + "..."
	}
	return s
}
