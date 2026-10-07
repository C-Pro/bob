package commands

import (
	"context"
	"fmt"
	"strings"
	"time"

	"bob/internal/sandbox"
)

// SandboxHandler handles /sandbox slash commands.
type SandboxHandler struct {
	Manager *sandbox.Manager
}

// NewSandboxHandler creates a new SandboxHandler.
func NewSandboxHandler(manager *sandbox.Manager) *SandboxHandler {
	return &SandboxHandler{
		Manager: manager,
	}
}

// Handle executes the /sandbox command specified by req.
func (h *SandboxHandler) Handle(ctx context.Context, req Request) (Result, error) {
	if h.Manager == nil {
		return Result{Reply: "Sandbox execution is disabled on this server."}, nil
	}

	subcmd := strings.ToLower(req.Subcommand)
	switch subcmd {
	case "approve":
		return h.handleApprove(ctx, req)
	case "deny":
		return h.handleDeny(ctx, req)
	case "destroy":
		return h.handleDestroy(ctx, req)
	case "status":
		return h.handleStatus(ctx, req)
	default:
		helpText := "🔒 **Sandbox Commands:**\n" +
			"• `/sandbox approve` — Approve pending sandbox request\n" +
			"• `/sandbox deny` — Deny pending sandbox request\n" +
			"• `/sandbox destroy` — Terminate your active sandbox\n" +
			"• `/sandbox status` — View status of your sandbox"
		return Result{Reply: helpText}, nil
	}
}

func (h *SandboxHandler) handleApprove(ctx context.Context, req Request) (Result, error) {
	sbx, err := h.Manager.ApproveSandbox(ctx, req.Actor.ID)
	if err != nil {
		return Result{Reply: fmt.Sprintf("⚠️ Failed to approve sandbox: %v", err)}, nil
	}
	if !sbx.ClaimContinuation() {
		return Result{Reply: "Sandbox is already approved and running."}, nil
	}

	return Result{
		Continuation: &SandboxContinuation{
			ActorID: req.Actor.ID,
			Reason:  sbx.Reason,
		},
	}, nil
}

func (h *SandboxHandler) handleDeny(ctx context.Context, req Request) (Result, error) {
	err := h.Manager.DenySandbox(req.Actor.ID)
	if err != nil {
		return Result{Reply: fmt.Sprintf("⚠️ Failed to deny sandbox: %v", err)}, nil
	}
	return Result{Reply: "❌ **Sandbox request denied.**", RecordAssistantEntry: true}, nil
}

func (h *SandboxHandler) handleDestroy(ctx context.Context, req Request) (Result, error) {
	err := h.Manager.Destroy(ctx, req.Actor.ID)
	if err != nil {
		return Result{Reply: fmt.Sprintf("⚠️ Failed to destroy sandbox: %v", err)}, nil
	}
	return Result{Reply: "🧹 **Sandbox terminated and resources released.**", RecordAssistantEntry: true}, nil
}

func (h *SandboxHandler) handleStatus(_ context.Context, req Request) (Result, error) {
	sbx, _ := h.Manager.GetStatus(req.Actor.ID)
	return Result{Reply: FormatSandboxStatus(sbx)}, nil
}

// FormatSandboxStatus formats user sandbox details into a status markdown string.
func FormatSandboxStatus(sbx *sandbox.UserSandbox) string {
	if sbx == nil || sbx.Status == sandbox.StatusNone {
		return "ℹ️ You do not have an active or pending sandbox."
	}

	switch sbx.Status {
	case sandbox.StatusPendingApproval:
		var b strings.Builder
		b.WriteString("⏳ **Sandbox Request Awaiting Your Approval**\n")
		AppendSandboxDetails(&b, sbx)
		b.WriteString("\nReply `/sandbox approve` to approve or `/sandbox deny` to reject.")
		return b.String()

	case sandbox.StatusRunning:
		var b strings.Builder
		b.WriteString("🟢 **Active Sandbox Status**\n")
		AppendSandboxDetails(&b, sbx)
		return b.String()

	case sandbox.StatusExpired:
		return "⌛ **Your previous sandbox has expired.** The agent can request a new sandbox when needed."

	default:
		return fmt.Sprintf("ℹ️ Sandbox status: %s", sbx.Status)
	}
}

// AppendSandboxDetails appends sandbox parameters to a string builder.
func AppendSandboxDetails(b *strings.Builder, sbx *sandbox.UserSandbox) {
	fmt.Fprintf(b, "- **Driver:** %s\n", sbx.Driver)
	if sbx.Driver == sandbox.DriverDocker && sbx.DockerImage != "" {
		fmt.Fprintf(b, "- **Docker Image:** %s\n", sbx.DockerImage)
	}
	fmt.Fprintf(b, "- **Network:** %s\n", sbx.Network.Mode)
	if len(sbx.Network.AllowedHosts) > 0 {
		fmt.Fprintf(b, "- **Allowed Domains:** %s\n", strings.Join(sbx.Network.AllowedHosts, ", "))
	}
	if len(sbx.Mounts) > 0 {
		b.WriteString("- **Mounts (in workspace):**\n")
		for _, m := range sbx.Mounts {
			ro := "read-write"
			if m.ReadOnly {
				ro = "read-only"
			}
			pathStr := m.RelativePath
			if pathStr == "." || pathStr == "" {
				pathStr = "(whole workspace)"
			}
			fmt.Fprintf(b, "  • %s (%s)\n", pathStr, ro)
		}
	}
	remaining := time.Until(sbx.ExpiresAt).Round(time.Minute)
	if remaining < 0 {
		remaining = 0
	}
	fmt.Fprintf(b, "- **Time Remaining:** %s\n", remaining)
}
