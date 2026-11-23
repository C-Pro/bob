package tools

import (
	"sync"

	"bob/internal/agentapi"
)

// ActionCollector accumulates agent actions produced during tool execution in a thread-safe manner.
type ActionCollector struct {
	mu      sync.Mutex
	actions []agentapi.Action
}

// NewActionCollector creates an empty ActionCollector.
func NewActionCollector() *ActionCollector {
	return &ActionCollector{}
}

// Add appends an action to the collector.
func (c *ActionCollector) Add(act agentapi.Action) {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.actions = append(c.actions, act)
}

// All returns a snapshot copy of all collected actions.
func (c *ActionCollector) All() []agentapi.Action {
	if c == nil {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if len(c.actions) == 0 {
		return nil
	}
	out := make([]agentapi.Action, len(c.actions))
	copy(out, c.actions)
	return out
}
