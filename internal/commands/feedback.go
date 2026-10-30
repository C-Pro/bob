package commands

import (
	"context"
	"errors"
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"bob/internal/trajectory"

	"github.com/google/uuid"
)

var uuidRegex = regexp.MustCompile(`^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$`)

func isTrajectoryID(s string) bool {
	if uuidRegex.MatchString(s) {
		return true
	}
	if strings.HasPrefix(s, "traj-") && !strings.Contains(s, " ") {
		return true
	}
	return false
}

// TrajectoryStoreResolver resolves trajectory stores per chat context.
type TrajectoryStoreResolver interface {
	GetTrajectoryStore(ctx context.Context, chatID string, isDM bool) (*trajectory.SQLiteStore, error)
}

// FeedbackHandler processes user feedback slash commands (/good, /bad, /rate).
type FeedbackHandler struct {
	stores TrajectoryStoreResolver
}

// NewFeedbackHandler creates an initialized FeedbackHandler.
func NewFeedbackHandler(stores TrajectoryStoreResolver) *FeedbackHandler {
	return &FeedbackHandler{
		stores: stores,
	}
}

// Handle routes and processes a feedback command.
func (h *FeedbackHandler) Handle(ctx context.Context, req Request) (Result, error) {
	if h.stores == nil {
		return Result{Reply: "⚠️ Feedback storage is not configured."}, nil
	}

	cmd := strings.ToLower(strings.TrimPrefix(req.Command, "/"))
	switch cmd {
	case "good":
		return h.handleRating(ctx, req, 1.0, req.Args)
	case "bad":
		return h.handleRating(ctx, req, -1.0, req.Args)
	case "rate":
		return h.handleRate(ctx, req)
	default:
		return Result{Reply: "⚠️ Supported feedback commands: `/good [note]`, `/bad [note]`, `/rate <+1|-1|0> [note]`"}, nil
	}
}

func (h *FeedbackHandler) handleRate(ctx context.Context, req Request) (Result, error) {
	if len(req.Args) == 0 {
		return Result{Reply: "⚠️ Usage: `/rate <+1|-1|0> [trajectory_id] [note]`"}, nil
	}

	rawRating := strings.ToLower(req.Args[0])
	var rating float64

	switch rawRating {
	case "+1", "1", "+", "good", "positive":
		rating = 1.0
	case "-1", "-", "bad", "negative":
		rating = -1.0
	case "0", "neutral":
		rating = 0.0
	default:
		parsed, err := strconv.ParseFloat(rawRating, 64)
		if err != nil || parsed < -1.0 || parsed > 1.0 {
			return Result{Reply: "⚠️ Invalid rating value. Please specify `+1`, `-1`, or `0` (or a float within [-1.0, 1.0])."}, nil
		}
		rating = parsed
	}

	return h.handleRating(ctx, req, rating, req.Args[1:])
}

func (h *FeedbackHandler) handleRating(ctx context.Context, req Request, rating float64, remainingArgs []string) (Result, error) {
	var targetID string
	var noteParts []string

	if len(remainingArgs) > 0 && isTrajectoryID(remainingArgs[0]) {
		targetID = remainingArgs[0]
		noteParts = remainingArgs[1:]
	} else {
		noteParts = remainingArgs
	}

	note := strings.TrimSpace(strings.Join(noteParts, " "))

	msgID := strings.TrimSpace(req.MessageID)
	if msgID == "" {
		msgID = uuid.NewString()
	}

	chatID := req.Session.SessionID
	if chatID == "" {
		chatID = req.Session.ScopeID
	}

	store, err := h.stores.GetTrajectoryStore(ctx, chatID, req.Session.IsDM)
	if err != nil {
		return Result{Reply: fmt.Sprintf("⚠️ Unable to access feedback storage for this chat: %v", err)}, nil
	}

	fbReq := trajectory.FeedbackRequest{
		MessageID:    msgID,
		FrontendID:   req.Session.FrontendID,
		SessionID:    req.Session.SessionID,
		ScopeID:      req.Session.ScopeID,
		ActorID:      req.Actor.ID,
		TrajectoryID: targetID,
		Rating:       rating,
		Note:         note,
	}

	if fbReq.FrontendID == "" {
		fbReq.FrontendID = "besedka"
	}

	traj, err := store.UpdateFeedback(ctx, fbReq)
	if err != nil {
		if errors.Is(err, trajectory.ErrTrajectoryNotFound) {
			return Result{Reply: "⚠️ No recent assistant turn found to rate in this chat."}, nil
		}
		if errors.Is(err, trajectory.ErrTrajectoryExpired) {
			return Result{Reply: "⚠️ That turn has expired from local history and can no longer be rated."}, nil
		}
		if errors.Is(err, trajectory.ErrUnauthorized) {
			return Result{Reply: "⚠️ You are not authorized to rate that turn."}, nil
		}
		return Result{Reply: fmt.Sprintf("⚠️ Failed to record feedback: %v", err)}, nil
	}

	var reply string
	if rating > 0 {
		reply = "👍 Feedback recorded. Thank you!"
	} else if rating < 0 {
		reply = "👎 Feedback recorded. We'll work on improving!"
	} else {
		reply = "👌 Neutral feedback recorded."
	}

	if note != "" {
		reply += fmt.Sprintf(" Note: %q", note)
	}
	if traj != nil && targetID != "" {
		reply += fmt.Sprintf(" (turn: `%s`)", traj.ID)
	}

	return Result{
		Reply: reply,
	}, nil
}
