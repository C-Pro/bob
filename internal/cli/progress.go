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
	mu           sync.Mutex
	w            io.Writer
	isTTY        bool
	hasProgress  bool
	firstErr     error
	activeStepID string
	lastLiveLen  int
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

// SetTTY manually overrides the TTY flag (mainly for testing).
func (ow *OutputWriter) SetTTY(isTTY bool) {
	ow.mu.Lock()
	defer ow.mu.Unlock()
	ow.isTTY = isTTY
}

// WriteLiveStep renders a live progress line for activeStepID on the same terminal line.
func (ow *OutputWriter) WriteLiveStep(stepID, line string) {
	ow.mu.Lock()
	defer ow.mu.Unlock()

	if ow.activeStepID != "" && ow.activeStepID != stepID {
		if _, err := fmt.Fprintln(ow.w); err != nil && ow.firstErr == nil {
			ow.firstErr = err
		}
		ow.lastLiveLen = 0
	}
	ow.activeStepID = stepID
	ow.hasProgress = true

	var err error
	if ow.isTTY {
		_, err = fmt.Fprintf(ow.w, "\r\x1b[K%s", line)
	} else {
		pad := ""
		lineLen := len([]rune(line))
		if lineLen < ow.lastLiveLen {
			pad = strings.Repeat(" ", ow.lastLiveLen-lineLen)
		} else {
			ow.lastLiveLen = lineLen
		}
		_, err = fmt.Fprintf(ow.w, "\r%s%s", line, pad)
	}
	if err != nil && ow.firstErr == nil {
		ow.firstErr = err
	}
}

// FinishStepLine finalizes a step line with a trailing newline, overwriting any live line in-place.
func (ow *OutputWriter) FinishStepLine(stepID, line string) {
	ow.mu.Lock()
	defer ow.mu.Unlock()

	ow.hasProgress = true
	var err error

	if ow.activeStepID != "" && ow.activeStepID == stepID {
		if ow.isTTY {
			_, err = fmt.Fprintf(ow.w, "\r\x1b[K%s\n", line)
		} else {
			pad := ""
			lineLen := len([]rune(line))
			if lineLen < ow.lastLiveLen {
				pad = strings.Repeat(" ", ow.lastLiveLen-lineLen)
			}
			_, err = fmt.Fprintf(ow.w, "\r%s%s\n", line, pad)
		}
		ow.activeStepID = ""
		ow.lastLiveLen = 0
	} else {
		if ow.activeStepID != "" {
			if _, e := fmt.Fprintln(ow.w); e != nil && ow.firstErr == nil {
				ow.firstErr = e
			}
			ow.activeStepID = ""
			ow.lastLiveLen = 0
		}
		_, err = fmt.Fprintln(ow.w, line)
	}

	if err != nil && ow.firstErr == nil {
		ow.firstErr = err
	}
}

// WriteStepLine writes a single progress line under mutex lock.
func (ow *OutputWriter) WriteStepLine(line string) {
	ow.FinishStepLine("", line)
}

// WriteFinalAnswer writes the final completion answer, preceded by a blank separator line
// if any progress step lines were printed.
func (ow *OutputWriter) WriteFinalAnswer(answer string) error {
	ow.mu.Lock()
	defer ow.mu.Unlock()

	if ow.activeStepID != "" {
		if _, err := fmt.Fprintln(ow.w); err != nil && ow.firstErr == nil {
			ow.firstErr = err
		}
		ow.activeStepID = ""
		ow.lastLiveLen = 0
	}

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

type liveStep struct {
	id        string
	toolName  string
	argsJSON  string
	startTime time.Time
	stopCh    chan struct{}
	doneCh    chan struct{}
}

// ProgressRenderer implements both agentapi.ProgressObserver and fsm.ProgressObserver.
type ProgressRenderer struct {
	writer         *OutputWriter
	quiet          bool
	seen           map[string]struct{}
	mu             sync.Mutex

	activeMu       sync.Mutex
	active         *liveStep
	tickerInterval time.Duration
	timeNow        func() time.Time
}

// NewProgressRenderer creates a new progress renderer.
func NewProgressRenderer(writer *OutputWriter, quiet bool) *ProgressRenderer {
	return &ProgressRenderer{
		writer: writer,
		quiet:  quiet,
		seen:   make(map[string]struct{}),
	}
}

// SetTickerInterval configures the tick interval for live duration updates (defaults to 1s).
func (r *ProgressRenderer) SetTickerInterval(d time.Duration) {
	r.activeMu.Lock()
	defer r.activeMu.Unlock()
	r.tickerInterval = d
}

// Close stops any active live progress ticker and cleans up resources.
func (r *ProgressRenderer) Close() {
	r.stopActiveStep()
}

func (r *ProgressRenderer) now() time.Time {
	if r.timeNow != nil {
		return r.timeNow()
	}
	return time.Now()
}

func (r *ProgressRenderer) stopActiveStep() {
	r.activeMu.Lock()
	live := r.active
	r.active = nil
	r.activeMu.Unlock()

	if live != nil {
		close(live.stopCh)
		<-live.doneCh
	}
}

func (r *ProgressRenderer) startLiveStep(id, toolName, argsJSON string, startTime time.Time) {
	if r.quiet || r.writer == nil {
		return
	}

	r.mu.Lock()
	if _, exists := r.seen[id]; exists {
		r.mu.Unlock()
		return
	}
	r.mu.Unlock()

	r.stopActiveStep()

	if startTime.IsZero() {
		startTime = r.now()
	}

	live := &liveStep{
		id:        id,
		toolName:  toolName,
		argsJSON:  argsJSON,
		startTime: startTime,
		stopCh:    make(chan struct{}),
		doneCh:    make(chan struct{}),
	}

	r.activeMu.Lock()
	r.active = live
	interval := r.tickerInterval
	r.activeMu.Unlock()

	if interval <= 0 {
		interval = time.Second
	}

	r.renderLiveLine(live, 0)
	go r.runLiveTicker(live, interval)
}

func (r *ProgressRenderer) runLiveTicker(live *liveStep, interval time.Duration) {
	defer close(live.doneCh)
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-live.stopCh:
			return
		case <-ticker.C:
			now := r.now()
			elapsed := int64(now.Sub(live.startTime) / time.Second)
			if elapsed < 0 {
				elapsed = 0
			}
			r.renderLiveLine(live, elapsed)
		}
	}
}

func (r *ProgressRenderer) renderLiveLine(live *liveStep, elapsedSeconds int64) {
	summary := summarizeArgs(live.toolName, live.argsJSON)
	suffix := fmt.Sprintf("(%ds)", elapsedSeconds)
	line := r.formatStepLine(live.toolName, summary, true, "", suffix)
	r.writer.WriteLiveStep(live.id, line)
}

func (r *ProgressRenderer) finishStep(id, toolName, argsJSON, status, errorText, resultJSON, durStr string) {
	r.mu.Lock()
	if _, exists := r.seen[id]; exists {
		r.mu.Unlock()
		return
	}
	r.seen[id] = struct{}{}
	r.mu.Unlock()

	r.activeMu.Lock()
	var live *liveStep
	if r.active != nil && r.active.id == id {
		live = r.active
		r.active = nil
	}
	r.activeMu.Unlock()

	if live != nil {
		close(live.stopCh)
		<-live.doneCh
		if durStr == "" {
			diff := int64(r.now().Sub(live.startTime) / time.Second)
			if diff == 0 {
				durStr = "<1s"
			} else {
				durStr = fmt.Sprintf("%ds", diff)
			}
		}
	}

	line := r.formatCompletedStepLine(toolName, argsJSON, status, errorText, resultJSON, durStr)
	r.writer.FinishStepLine(id, line)
}

// Observe implements agentapi.ProgressObserver.
func (r *ProgressRenderer) Observe(ctx context.Context, ev agentapi.ProgressEvent) {
	if r.quiet || r.writer == nil {
		return
	}
	if ev.Type == agentapi.ProgressRunFinished {
		r.stopActiveStep()
		return
	}
	if ev.Type == agentapi.ProgressStepUpdated && ev.Step != nil {
		step := ev.Step
		if step.Status == agentapi.StepRunning {
			r.startLiveStep(step.ID, step.Call.Name, step.Call.Arguments, r.now())
			return
		}
		if step.Status == agentapi.StepCompleted || step.Status == agentapi.StepFailed || step.Status == agentapi.StepTimedOut {
			r.finishStep(step.ID, step.Call.Name, step.Call.Arguments, string(step.Status), step.Error, "", "")
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
	if step.Status == fsm.StepStatusRunning {
		startTime := r.now()
		if step.StartedAt != nil && *step.StartedAt > 0 {
			startTime = time.Unix(*step.StartedAt, 0)
		}
		r.startLiveStep(step.ID, step.ToolName, step.ArgsJSON, startTime)
		return
	}
	if step.Status == fsm.StepStatusCompleted || step.Status == fsm.StepStatusFailed || step.Status == fsm.StepStatusTimedOut {
		var durStr string
		if step.StartedAt != nil && step.CompletedAt != nil && *step.CompletedAt >= *step.StartedAt {
			diff := *step.CompletedAt - *step.StartedAt
			if diff == 0 {
				durStr = "<1s"
			} else {
				durStr = fmt.Sprintf("%ds", diff)
			}
		}

		r.finishStep(step.ID, step.ToolName, step.ArgsJSON, string(step.Status), step.ErrorText, step.ResultJSON, durStr)
	}
}

// OnRunFinished implements fsm.ProgressObserver.
func (r *ProgressRenderer) OnRunFinished(ctx context.Context, status fsm.RunStatus, runErr error) {
	r.stopActiveStep()
}

func (r *ProgressRenderer) formatCompletedStepLine(toolName, argsJSON, status, errorText, resultJSON, durStr string) string {
	summary := summarizeArgs(toolName, argsJSON)

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

	return r.formatStepLine(toolName, summary, isSuccess, errorText, suffix)
}

func (r *ProgressRenderer) formatStepLine(toolName, summary string, isSuccess bool, errorText, suffix string) string {
	if r.writer.isTTY {
		if isSuccess {
			line := fmt.Sprintf("\x1b[36m•\x1b[0m \x1b[1m%s\x1b[0m", toolName)
			if summary != "" {
				line += " " + summary
			}
			if suffix != "" {
				line += " \x1b[2m" + suffix + "\x1b[0m"
			}
			return line
		}
		line := fmt.Sprintf("\x1b[31m✗\x1b[0m \x1b[1m%s\x1b[0m", toolName)
		if summary != "" {
			line += " " + summary
		}
		if errorText != "" {
			line += fmt.Sprintf(" \x1b[31m(%s)\x1b[0m", sanitizeString(errorText, 60))
		} else if suffix != "" {
			line += " \x1b[31m" + suffix + "\x1b[0m"
		}
		return line
	}

	if isSuccess {
		line := "• " + toolName
		if summary != "" {
			line += " " + summary
		}
		if suffix != "" {
			line += " " + suffix
		}
		return line
	}
	line := "✗ " + toolName
	if summary != "" {
		line += " " + summary
	}
	if errorText != "" {
		line += fmt.Sprintf(" (%s)", sanitizeString(errorText, 60))
	} else if suffix != "" {
		line += " " + suffix
	}
	return line
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
			MemoryID string `json:"memory_id"`
			Key      string `json:"key"`
			ItemID   string `json:"item_id"`
		}
		if err := json.Unmarshal([]byte(clean), &m); err == nil {
			id := m.MemoryID
			if id == "" {
				id = m.Key
			}
			if id == "" {
				id = m.ItemID
			}
			if id != "" {
				return sanitizeString(id, 60)
			}
		}
	case "load_skill":
		var m struct {
			SkillID string `json:"skill_id"`
			Name    string `json:"name"`
			ItemID  string `json:"item_id"`
		}
		if err := json.Unmarshal([]byte(clean), &m); err == nil {
			id := m.SkillID
			if id == "" {
				id = m.Name
			}
			if id == "" {
				id = m.ItemID
			}
			if id != "" {
				return sanitizeString(id, 60)
			}
		}
	case "propose_memory":
		var m struct {
			Content string `json:"content"`
		}
		if err := json.Unmarshal([]byte(clean), &m); err == nil && m.Content != "" {
			return sanitizeString(m.Content, 60)
		}
	case "propose_skill":
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
