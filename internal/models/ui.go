package models

import (
	"errors"
	"fmt"
	"strings"
)

const (
	// MaxUIElements is the maximum number of UI controls Besedka allows on a single message.
	MaxUIElements = 50
	// MaxUIElementIDBytes is the maximum byte length for a UI element identifier in Besedka.
	MaxUIElementIDBytes = 64
	// MaxUIElementLabelBytes is the maximum byte length for a UI element label in Besedka.
	MaxUIElementLabelBytes = 255
	// MaxUIElementDescBytes is the maximum byte length for a UI element description in Besedka.
	MaxUIElementDescBytes = 1024
	// ReservedUIActionPrefix marks element IDs reserved for hardcoded Go approval and command actions.
	ReservedUIActionPrefix = "bob:"
)

// HasReservedUIActionPrefix reports whether id uses the reserved "bob:" action namespace.
func HasReservedUIActionPrefix(id string) bool {
	return strings.HasPrefix(strings.ToLower(strings.TrimSpace(id)), ReservedUIActionPrefix)
}

// ValidateUIData validates an interactive UI payload against Besedka server rules.
func ValidateUIData(ui *UIData) error {
	if ui == nil {
		return errors.New("ui data cannot be nil")
	}
	if len(ui.Elements) == 0 || len(ui.Elements) > MaxUIElements {
		return fmt.Errorf("invalid ui elements count: must be between 1 and %d", MaxUIElements)
	}

	seenIDs := make(map[string]struct{}, len(ui.Elements))
	defaultRadios := 0

	for _, el := range ui.Elements {
		id := strings.TrimSpace(el.ID)
		if id == "" {
			return errors.New("ui element ID cannot be empty")
		}
		if len(id) > MaxUIElementIDBytes {
			return fmt.Errorf("ui element ID %q exceeds %d bytes", id, MaxUIElementIDBytes)
		}
		if _, exists := seenIDs[id]; exists {
			return fmt.Errorf("duplicate ui element ID %q", id)
		}
		seenIDs[id] = struct{}{}

		label := strings.TrimSpace(el.Label)
		if label == "" {
			return fmt.Errorf("ui element %q label cannot be empty", id)
		}
		if len(label) > MaxUIElementLabelBytes {
			return fmt.Errorf("ui element %q label exceeds %d bytes", id, MaxUIElementLabelBytes)
		}

		desc := strings.TrimSpace(el.Description)
		if len(desc) > MaxUIElementDescBytes {
			return fmt.Errorf("ui element %q description exceeds %d bytes", id, MaxUIElementDescBytes)
		}

		switch el.Type {
		case UIElementTypeButton, UIElementTypeCheckbox, UIElementTypeRadio:
		default:
			return fmt.Errorf("invalid ui element type %q", el.Type)
		}

		switch el.Style {
		case "", UIElementStyleDefault, UIElementStylePrimary, UIElementStyleDanger:
		default:
			return fmt.Errorf("invalid ui element style %q", el.Style)
		}

		if el.Type == UIElementTypeButton && el.Default {
			return errors.New("button element cannot have default state")
		}
		if el.Type == UIElementTypeRadio && el.Default {
			defaultRadios++
		}
	}

	if defaultRadios > 1 {
		return errors.New("at most one radio element can have default state")
	}

	return nil
}

// SanitizeUIDataForSend validates ui and returns a clean copy safe for outgoing bot messages,
// without mutating the caller's slice.
func SanitizeUIDataForSend(ui *UIData) (*UIData, error) {
	if err := ValidateUIData(ui); err != nil {
		return nil, err
	}

	elements := make([]UIElement, len(ui.Elements))
	for i, el := range ui.Elements {
		elements[i] = UIElement{
			ID:          strings.TrimSpace(el.ID),
			Type:        el.Type,
			Label:       strings.TrimSpace(el.Label),
			Description: strings.TrimSpace(el.Description),
			Style:       el.Style,
			Default:     el.Default,
		}
	}

	return &UIData{
		Elements: elements,
	}, nil
}

// FormatUIForContext renders interactive UI elements deterministically for chat context and deduplication.
// Mutable answer metadata (Answered, SelectedIDs, ResponderID) is intentionally omitted so outgoing pushes,
// WebSocket self-echoes, and post-answer warmup snapshots stay identical.
func FormatUIForContext(ui *UIData) string {
	if ui == nil || len(ui.Elements) == 0 {
		return ""
	}

	parts := make([]string, 0, len(ui.Elements))
	for _, el := range ui.Elements {
		id := strings.TrimSpace(el.ID)
		label := strings.TrimSpace(el.Label)
		desc := strings.TrimSpace(el.Description)

		defaultSuffix := ""
		if el.Default {
			defaultSuffix = ", default"
		}

		part := fmt.Sprintf("%s %q (id=%s%s)", el.Type, label, id, defaultSuffix)
		if desc != "" {
			part += ": " + desc
		}
		parts = append(parts, part)
	}

	return "[Interactive UI options: " + strings.Join(parts, "; ") + "]"
}

// FormatUIResponseForContext enriches a user's ui_response message with parent sequence and selected IDs.
func FormatUIResponseForContext(text string, resp *UIResponseData) string {
	trimmed := strings.TrimSpace(text)
	if resp == nil {
		return trimmed
	}

	selected := "none"
	if len(resp.SelectedIDs) > 0 {
		selected = strings.Join(resp.SelectedIDs, ", ")
	}
	meta := fmt.Sprintf("[UI response to seq %d | selected_ids: %s]", resp.ParentSeq, selected)
	if trimmed == "" {
		return meta
	}
	return trimmed + "\n" + meta
}
