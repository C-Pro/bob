package models

import (
	"strings"
	"unicode"
)

// UIActionKind identifies a reserved Go-handled interactive UI button action.
type UIActionKind string

const (
	UIActionSandboxApprove  UIActionKind = "sandbox_approve"
	UIActionSandboxDeny     UIActionKind = "sandbox_deny"
	UIActionSandboxDestroy  UIActionKind = "sandbox_destroy"
	UIActionSandboxKeep     UIActionKind = "sandbox_keep"
	UIActionMemoryApprove   UIActionKind = "mem_approve"
	UIActionMemoryReject    UIActionKind = "mem_reject"
	UIActionSkillApprove    UIActionKind = "skill_approve"
	UIActionSkillReject     UIActionKind = "skill_reject"
	UIActionScheduleApprove UIActionKind = "sched_approve"
	UIActionScheduleDeny    UIActionKind = "sched_deny"
)

const (
	uiActionSandboxApprovePrefix = "bob:sandbox:approve:"
	uiActionSandboxDenyPrefix    = "bob:sandbox:deny:"
	uiActionSandboxDestroyPrefix = "bob:sandbox:destroy:"
	uiActionSandboxKeepPrefix    = "bob:sandbox:keep:"

	uiActionMemoryApprovePrefix   = "bob:mem:approve:"
	uiActionMemoryRejectPrefix    = "bob:mem:reject:"
	uiActionSkillApprovePrefix    = "bob:skill:approve:"
	uiActionSkillRejectPrefix     = "bob:skill:reject:"
	uiActionScheduleApprovePrefix = "bob:sched:approve:"
	uiActionScheduleDenyPrefix    = "bob:sched:deny:"
)

// ParsedUIAction represents a parsed reserved "bob:" button action ID and its target identifier.
type ParsedUIAction struct {
	Kind     UIActionKind
	TargetID string
}

func hasWhitespace(s string) bool {
	return strings.IndexFunc(s, unicode.IsSpace) >= 0
}

// ParseReservedUIAction parses a reserved "bob:" UI element ID into a strongly-typed action.
func ParseReservedUIAction(id string) (ParsedUIAction, bool) {
	if id == "" || hasWhitespace(id) {
		return ParsedUIAction{}, false
	}

	prefixes := []struct {
		prefix string
		kind   UIActionKind
	}{
		{prefix: uiActionSandboxApprovePrefix, kind: UIActionSandboxApprove},
		{prefix: uiActionSandboxDenyPrefix, kind: UIActionSandboxDeny},
		{prefix: uiActionSandboxDestroyPrefix, kind: UIActionSandboxDestroy},
		{prefix: uiActionSandboxKeepPrefix, kind: UIActionSandboxKeep},
		{prefix: uiActionMemoryApprovePrefix, kind: UIActionMemoryApprove},
		{prefix: uiActionMemoryRejectPrefix, kind: UIActionMemoryReject},
		{prefix: uiActionSkillApprovePrefix, kind: UIActionSkillApprove},
		{prefix: uiActionSkillRejectPrefix, kind: UIActionSkillReject},
		{prefix: uiActionScheduleApprovePrefix, kind: UIActionScheduleApprove},
		{prefix: uiActionScheduleDenyPrefix, kind: UIActionScheduleDeny},
	}

	for _, p := range prefixes {
		if strings.HasPrefix(id, p.prefix) {
			target := strings.TrimPrefix(id, p.prefix)
			if target == "" {
				return ParsedUIAction{}, false
			}
			return ParsedUIAction{Kind: p.kind, TargetID: target}, true
		}
	}

	return ParsedUIAction{}, false
}

func sanitizeActionTarget(targetID string) (string, bool) {
	if targetID == "" || hasWhitespace(targetID) {
		return "", false
	}
	return targetID, true
}

func buildTargetActionPair(approveID, approveLabel, rejectID, rejectLabel string) *UIData {
	if len(approveID) > MaxUIElementIDBytes || len(rejectID) > MaxUIElementIDBytes {
		return nil
	}
	return &UIData{
		Elements: []UIElement{
			{
				ID:    approveID,
				Type:  UIElementTypeButton,
				Label: approveLabel,
				Style: UIElementStylePrimary,
			},
			{
				ID:    rejectID,
				Type:  UIElementTypeButton,
				Label: rejectLabel,
				Style: UIElementStyleDanger,
			},
		},
	}
}

// NewSandboxApprovalUI builds the Approve/Deny button card bound to a pending sandbox ID.
func NewSandboxApprovalUI(sandboxID string) *UIData {
	target, ok := sanitizeActionTarget(sandboxID)
	if !ok {
		return nil
	}
	return buildTargetActionPair(
		uiActionSandboxApprovePrefix+target, "Approve",
		uiActionSandboxDenyPrefix+target, "Deny",
	)
}

// NewSandboxRunningUI builds Destroy/Keep buttons bound to an active running sandbox ID.
func NewSandboxRunningUI(sandboxID string) *UIData {
	target, ok := sanitizeActionTarget(sandboxID)
	if !ok {
		return nil
	}
	destroyID := uiActionSandboxDestroyPrefix + target
	keepID := uiActionSandboxKeepPrefix + target
	if len(destroyID) > MaxUIElementIDBytes || len(keepID) > MaxUIElementIDBytes {
		return nil
	}
	return &UIData{
		Elements: []UIElement{
			{
				ID:    destroyID,
				Type:  UIElementTypeButton,
				Label: "Destroy Sandbox",
				Style: UIElementStyleDanger,
			},
			{
				ID:    keepID,
				Type:  UIElementTypeButton,
				Label: "Keep Running",
				Style: UIElementStyleDefault,
			},
		},
	}
}

// NewSandboxDestroyOnlyUI builds a single Destroy Sandbox action button bound to a sandbox ID for status output.
func NewSandboxDestroyOnlyUI(sandboxID string) *UIData {
	target, ok := sanitizeActionTarget(sandboxID)
	if !ok {
		return nil
	}
	destroyID := uiActionSandboxDestroyPrefix + target
	if len(destroyID) > MaxUIElementIDBytes {
		return nil
	}
	return &UIData{
		Elements: []UIElement{
			{
				ID:    destroyID,
				Type:  UIElementTypeButton,
				Label: "Destroy Sandbox",
				Style: UIElementStyleDanger,
			},
		},
	}
}

// NewMemoryApprovalUI builds Approve/Reject buttons for a proposed structured memory version.
func NewMemoryApprovalUI(versionID string) *UIData {
	target, ok := sanitizeActionTarget(versionID)
	if !ok {
		return nil
	}
	return buildTargetActionPair(
		uiActionMemoryApprovePrefix+target, "Approve Memory",
		uiActionMemoryRejectPrefix+target, "Reject",
	)
}

// NewSkillApprovalUI builds Approve/Reject buttons for a proposed procedural skill version.
func NewSkillApprovalUI(versionID string) *UIData {
	target, ok := sanitizeActionTarget(versionID)
	if !ok {
		return nil
	}
	return buildTargetActionPair(
		uiActionSkillApprovePrefix+target, "Approve Skill",
		uiActionSkillRejectPrefix+target, "Reject",
	)
}

// NewScheduleApprovalUI builds Approve/Deny buttons for a pending scheduled task.
func NewScheduleApprovalUI(scheduleID string) *UIData {
	target, ok := sanitizeActionTarget(scheduleID)
	if !ok {
		return nil
	}
	return buildTargetActionPair(
		uiActionScheduleApprovePrefix+target, "Approve Schedule",
		uiActionScheduleDenyPrefix+target, "Deny",
	)
}
