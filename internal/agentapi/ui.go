package agentapi

// UIElementType classifies an interactive UI element rendered by a supporting frontend.
type UIElementType string

const (
	UIElementButton   UIElementType = "button"
	UIElementCheckbox UIElementType = "checkbox"
	UIElementRadio    UIElementType = "radio"
)

// UIElementStyle classifies the visual style of a UI element.
type UIElementStyle string

const (
	UIElementStyleDefault UIElementStyle = "default"
	UIElementStylePrimary UIElementStyle = "primary"
	UIElementStyleDanger  UIElementStyle = "danger"
)

// UIElement describes a frontend-agnostic interactive UI option or action button.
type UIElement struct {
	ID          string         `json:"id"`
	Type        UIElementType  `json:"type"`
	Label       string         `json:"label"`
	Description string         `json:"description,omitempty"`
	Style       UIElementStyle `json:"style,omitempty"`
	Default     bool           `json:"default,omitempty"`
}

// InteractivePromptPayload is carried in ActionInteractivePrompt actions to render prompt text with UI controls.
type InteractivePromptPayload struct {
	Message  string      `json:"message"`
	Elements []UIElement `json:"elements"`
}

// ApprovalKind identifies the domain of a Go-controlled approval request.
type ApprovalKind string

const (
	ApprovalKindSandbox  ApprovalKind = "sandbox"
	ApprovalKindMemory   ApprovalKind = "memory"
	ApprovalKindSkill    ApprovalKind = "skill"
	ApprovalKindSchedule ApprovalKind = "schedule"
)

// ApprovalRequestPayload describes a structured approval request for frontend rendering.
type ApprovalRequestPayload struct {
	Kind     ApprovalKind `json:"kind"`
	TargetID string       `json:"target_id,omitempty"`
	Label    string       `json:"label,omitempty"`
	Summary  string       `json:"summary,omitempty"`
}
