package tools

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"bob/internal/agentapi"
	"bob/internal/models"

	openai "github.com/sashabaranov/go-openai"
)

// AskUserToolName is the registered tool name for presenting free-form interactive UI questions.
const AskUserToolName = "ask_user"

// AskUserElementArg describes a single clickable option or button passed to ask_user.
type AskUserElementArg struct {
	ID          string `json:"id"`
	Type        string `json:"type"`
	Label       string `json:"label"`
	Description string `json:"description,omitempty"`
	Style       string `json:"style,omitempty"`
	Default     bool   `json:"default,omitempty"`
}

// AskUserArgs defines arguments for the ask_user tool.
type AskUserArgs struct {
	Message  string              `json:"message"`
	Elements []AskUserElementArg `json:"elements"`
}

// AskUserToolDefinition returns the OpenAI tool schema for ask_user.
func AskUserToolDefinition() openai.Tool {
	return openai.Tool{
		Type: openai.ToolTypeFunction,
		Function: &openai.FunctionDefinition{
			Name:        AskUserToolName,
			Description: "Ask the user a question accompanied by interactive Besedka UI controls (clickable buttons, checkboxes, or radio options) so the user can click instead of typing while still remaining able to reply with free text. Explicit 'button' elements submit immediately when clicked. When presenting only 'checkbox' and/or 'radio' elements without buttons, Besedka renders an automatic Submit button. Concluding a turn with ask_user sends this message and UI directly to the user; do NOT call ask_user for system approvals handled automatically by sandbox_request, propose_memory, propose_skill, or schedule_task.",
			Parameters: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"message": map[string]interface{}{
						"type":        "string",
						"description": "The prompt or question text to display above the interactive UI controls.",
					},
					"elements": map[string]interface{}{
						"type":        "array",
						"description": "Ordered list of 1 to 50 interactive UI elements.",
						"minItems":    1,
						"maxItems":    models.MaxUIElements,
						"items": map[string]interface{}{
							"type": "object",
							"properties": map[string]interface{}{
								"id": map[string]interface{}{
									"type":        "string",
									"description": "Unique option identifier (1-64 bytes). Must not start with 'bob:'.",
								},
								"type": map[string]interface{}{
									"type":        "string",
									"enum":        []string{"button", "checkbox", "radio"},
									"description": "Control type: button, checkbox, or radio.",
								},
								"label": map[string]interface{}{
									"type":        "string",
									"description": "Human-readable label shown on the control (1-255 bytes).",
								},
								"description": map[string]interface{}{
									"type":        "string",
									"description": "Optional helper text shown under checkbox or radio options (max 1024 bytes).",
								},
								"style": map[string]interface{}{
									"type":        "string",
									"enum":        []string{"default", "primary", "danger"},
									"description": "Optional visual style, typically used on button elements.",
								},
								"default": map[string]interface{}{
									"type":        "boolean",
									"description": "Initial checked state for checkbox or radio elements (not allowed on buttons; at most one radio may be default).",
								},
							},
							"required": []string{"id", "type", "label"},
						},
					},
				},
				"required": []string{"message", "elements"},
			},
		},
	}
}

func (r *Registry) executeAskUser(ctx context.Context, argsJSON string) (string, error) {
	session, ok := ChatSessionFromContext(ctx)
	if !ok || !session.SupportsInteractiveUI || session.IsScheduled {
		return "", errors.New("ask_user is only available in interactive UI sessions")
	}
	if session.Actions == nil {
		return "", errors.New("interactive prompt delivery is not configured for this session")
	}

	var args AskUserArgs
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		return "", fmt.Errorf("failed to parse ask_user arguments: %w", err)
	}

	msg := strings.TrimSpace(args.Message)
	if msg == "" {
		return "", errors.New("message cannot be empty")
	}
	if len(args.Elements) == 0 {
		return "", errors.New("elements cannot be empty")
	}

	rawElems := make([]models.UIElement, len(args.Elements))
	for i, el := range args.Elements {
		rawElems[i] = models.UIElement{
			ID:          el.ID,
			Type:        models.UIElementType(el.Type),
			Label:       el.Label,
			Description: el.Description,
			Style:       models.UIElementStyle(el.Style),
			Default:     el.Default,
		}
	}

	sanitized, err := models.SanitizeUIDataForSend(&models.UIData{Elements: rawElems})
	if err != nil {
		return "", err
	}

	apiElements := make([]agentapi.UIElement, len(sanitized.Elements))
	for i, el := range sanitized.Elements {
		if models.HasReservedUIActionPrefix(el.ID) {
			return "", fmt.Errorf("element id %q uses reserved prefix %q", el.ID, models.ReservedUIActionPrefix)
		}
		apiElements[i] = agentapi.UIElement{
			ID:          el.ID,
			Type:        agentapi.UIElementType(el.Type),
			Label:       el.Label,
			Description: el.Description,
			Style:       agentapi.UIElementStyle(el.Style),
			Default:     el.Default,
		}
	}

	payloadBytes, err := json.Marshal(agentapi.InteractivePromptPayload{
		Message:  msg,
		Elements: apiElements,
	})
	if err != nil {
		return "", fmt.Errorf("failed to marshal interactive prompt payload: %w", err)
	}

	session.Actions.Add(agentapi.Action{
		Type:    agentapi.ActionInteractivePrompt,
		Content: msg,
		Payload: payloadBytes,
	})

	return fmt.Sprintf(`{"status":"prompt_queued","elements":%d}`, len(apiElements)), nil
}
