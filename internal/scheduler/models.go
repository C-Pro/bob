package scheduler

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"regexp"
	"strings"
	"time"

	"github.com/robfig/cron/v3"
)

// ScheduleType identifies the trigger mechanism.
type ScheduleType string

const (
	ScheduleTypeOnce     ScheduleType = "ONCE"
	ScheduleTypeCron     ScheduleType = "CRON"
	ScheduleTypeInterval ScheduleType = "INTERVAL"
)

// ScheduleStatus represents the lifecycle state of a schedule.
type ScheduleStatus string

const (
	ScheduleStatusPendingApproval ScheduleStatus = "PENDING_APPROVAL"
	ScheduleStatusActive          ScheduleStatus = "ACTIVE"
	ScheduleStatusPaused          ScheduleStatus = "PAUSED"
	ScheduleStatusCompleted       ScheduleStatus = "COMPLETED"
	ScheduleStatusCancelled       ScheduleStatus = "CANCELLED"
	ScheduleStatusExpired         ScheduleStatus = "EXPIRED"
	ScheduleStatusDenied          ScheduleStatus = "DENIED"
)

// IsTerminal reports whether the status is a final state that will not trigger again.
func (s ScheduleStatus) IsTerminal() bool {
	return s == ScheduleStatusCompleted || s == ScheduleStatusCancelled || s == ScheduleStatusExpired || s == ScheduleStatusDenied
}

// Schedule defines a persistent recurring or delayed task execution.
type Schedule struct {
	ID                string         `json:"id"`
	Name              string         `json:"name"`
	ChatID            string         `json:"chat_id"`
	UserID            string         `json:"user_id"`
	ScheduleType      ScheduleType   `json:"schedule_type"`
	CronExpr          string         `json:"cron_expr,omitempty"`
	IntervalSeconds   int            `json:"interval_seconds,omitempty"`
	Instruction       string         `json:"instruction"`
	Status            ScheduleStatus `json:"status"`
	NextRunAt         int64          `json:"next_run_at"`
	LastRunAt         *int64         `json:"last_run_at,omitempty"`
	ExpiresAt         *int64         `json:"expires_at,omitempty"`
	MaxRuns           int            `json:"max_runs"`
	RunCount          int            `json:"run_count"`
	MissedCount       int            `json:"missed_count"`
	RunTimeoutSeconds int            `json:"run_timeout_seconds"`
	MaxTurns          int            `json:"max_turns"`
	LastStatus        string         `json:"last_status,omitempty"`
	CreatedAt         int64          `json:"created_at"`
	UpdatedAt         int64          `json:"updated_at"`
}

// ScheduleGrant represents the explicit authorization granted by a human user for a schedule.
type ScheduleGrant struct {
	ID                    string `json:"id"`
	ScheduleID            string `json:"schedule_id"`
	PermissionRequestJSON string `json:"permission_request_json"`
	ParamsHash            string `json:"params_hash"`
	GrantedBy             string `json:"granted_by"`
	GrantedAt             int64  `json:"granted_at"`
	ValidUntil            int64  `json:"valid_until"`
}

// Verify checks that the grant has valid contents and that its stored params_hash
// strictly matches the SHA-256 digest of its verbatim raw permission request JSON.
func (g *ScheduleGrant) Verify() bool {
	if g == nil || g.PermissionRequestJSON == "" || g.ParamsHash == "" {
		return false
	}
	return ComputeRawJSONHash(g.PermissionRequestJSON) == g.ParamsHash
}

// PermissionEnvelope is an extensible container for capability grants requested by a schedule.
type PermissionEnvelope struct {
	Sandbox *SandboxPermissions `json:"sandbox,omitempty"`
}

// SandboxPermissions defines sandbox environment options requested for scheduled runs.
type SandboxPermissions struct {
	Driver         string             `json:"driver,omitempty"`
	Image          string             `json:"image,omitempty"`
	Network        string             `json:"network,omitempty"`
	Domains        []string           `json:"domains,omitempty"`
	Mounts         []SandboxUserMount `json:"mounts,omitempty"`
	TimeoutSeconds int                `json:"timeout_seconds,omitempty"`
}

// SandboxUserMount defines a user workspace directory mount inside the sandbox.
type SandboxUserMount struct {
	Path     string `json:"path"`
	ReadOnly bool   `json:"read_only,omitempty"`
}

// ScheduleNameRegex enforces lowercase alphanumeric identifiers with max 4 underscore-separated words.
var ScheduleNameRegex = regexp.MustCompile(`^[a-z0-9]+(_[a-z0-9]+){0,3}$`)

// ValidateScheduleName checks that a schedule name complies with naming policy.
func ValidateScheduleName(name string) error {
	name = strings.TrimSpace(name)
	if name == "" {
		return errors.New("schedule_name cannot be empty")
	}
	if len(name) > 64 {
		return fmt.Errorf("schedule_name %q is too long (maximum 64 characters)", name)
	}
	if !ScheduleNameRegex.MatchString(name) {
		return fmt.Errorf("invalid schedule_name %q: must be lowercase alphanumeric words separated by underscores (max 4 words, e.g. 'daily_brief')", name)
	}
	return nil
}

// ComputeRawJSONHash computes the SHA-256 digest of the exact original raw permission request JSON string.
func ComputeRawJSONHash(rawJSON string) string {
	h := sha256.Sum256([]byte(rawJSON))
	return hex.EncodeToString(h[:])
}

// Standard 5-field cron parser supporting descriptors (@daily, @hourly, etc.).
var standardCronParser = cron.NewParser(cron.Minute | cron.Hour | cron.Dom | cron.Month | cron.Dow | cron.Descriptor)

// ParseCronSchedule parses and validates a standard 5-field cron expression.
func ParseCronSchedule(expr string) (cron.Schedule, error) {
	expr = strings.TrimSpace(expr)
	if expr == "" {
		return nil, errors.New("cron expression cannot be empty")
	}
	return standardCronParser.Parse(expr)
}

// CalculateNextRun computes the next execution time for a schedule relative to fromTime.
// If the schedule reaches its expiration or max run limit, returns time.Time{} with shouldEnd=true.
func CalculateNextRun(s *Schedule, fromTime time.Time) (next time.Time, shouldEnd bool, err error) {
	if s == nil {
		return time.Time{}, true, errors.New("schedule cannot be nil")
	}
	if s.Status.IsTerminal() {
		return time.Time{}, true, nil
	}
	if s.ExpiresAt != nil && fromTime.Unix() > *s.ExpiresAt {
		return time.Time{}, true, nil
	}
	if s.MaxRuns > 0 && s.RunCount >= s.MaxRuns {
		return time.Time{}, true, nil
	}

	switch s.ScheduleType {
	case ScheduleTypeOnce:
		if s.RunCount > 0 || s.LastRunAt != nil {
			return time.Time{}, true, nil
		}
		t := time.Unix(s.NextRunAt, 0)
		if s.ExpiresAt != nil && t.Unix() > *s.ExpiresAt {
			return time.Time{}, true, nil
		}
		return t, false, nil

	case ScheduleTypeInterval:
		if s.IntervalSeconds <= 0 {
			return time.Time{}, false, errors.New("interval_seconds must be greater than 0")
		}
		next = fromTime.Add(time.Duration(s.IntervalSeconds) * time.Second)

	case ScheduleTypeCron:
		sched, err := ParseCronSchedule(s.CronExpr)
		if err != nil {
			return time.Time{}, false, fmt.Errorf("invalid cron expression %q: %w", s.CronExpr, err)
		}
		next = sched.Next(fromTime)
		if next.IsZero() {
			return time.Time{}, true, nil
		}

	default:
		return time.Time{}, false, fmt.Errorf("unknown schedule type %q", s.ScheduleType)
	}

	if s.ExpiresAt != nil && next.Unix() > *s.ExpiresAt {
		return time.Time{}, true, nil
	}

	return next, false, nil
}
