package trajectory

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestCalculateReward(t *testing.T) {
	tests := []struct {
		name     string
		input    EvalBreakdown
		expected float64
	}{
		{
			name: "Normal conversational turn (0 tools)",
			input: EvalBreakdown{
				TotalToolCalls:    0,
				TerminationReason: "completed",
			},
			expected: 0.20,
		},
		{
			name: "Conversational turn + positive feedback",
			input: EvalBreakdown{
				TotalToolCalls:    0,
				TerminationReason: "completed",
				HasUserFeedback:   true,
				UserFeedback:      1.0,
			},
			expected: 0.80,
		},
		{
			name: "Conversational turn + negative feedback",
			input: EvalBreakdown{
				TotalToolCalls:    0,
				TerminationReason: "completed",
				HasUserFeedback:   true,
				UserFeedback:      -1.0,
			},
			expected: -0.40,
		},
		{
			name: "All tools succeed (1.0), normal",
			input: EvalBreakdown{
				TotalToolCalls:      2,
				SuccessfulToolCalls: 2,
				TerminationReason:   "completed",
			},
			expected: 0.40,
		},
		{
			name: "All tools succeed (1.0), positive feedback",
			input: EvalBreakdown{
				TotalToolCalls:      2,
				SuccessfulToolCalls: 2,
				TerminationReason:   "completed",
				HasUserFeedback:     true,
				UserFeedback:        1.0,
			},
			expected: 1.00,
		},
		{
			name: "All tools succeed (1.0), negative feedback",
			input: EvalBreakdown{
				TotalToolCalls:      2,
				SuccessfulToolCalls: 2,
				TerminationReason:   "completed",
				HasUserFeedback:     true,
				UserFeedback:        -1.0,
			},
			expected: -0.20,
		},
		{
			name: "Half tools fail (0.5), normal",
			input: EvalBreakdown{
				TotalToolCalls:      2,
				SuccessfulToolCalls: 1,
				FailedToolCalls:     1,
				TerminationReason:   "completed",
			},
			expected: 0.00,
		},
		{
			name: "All tools fail (0.0), normal completion",
			input: EvalBreakdown{
				TotalToolCalls:      2,
				SuccessfulToolCalls: 0,
				FailedToolCalls:     2,
				TerminationReason:   "completed",
			},
			expected: -0.40,
		},
		{
			name: "All tools succeed, budget exhausted",
			input: EvalBreakdown{
				TotalToolCalls:      5,
				SuccessfulToolCalls: 5,
				BudgetExhausted:     true,
				TerminationReason:   "budget_exhausted",
			},
			expected: 0.10,
		},
		{
			name: "Zero tools, FSM hard failure",
			input: EvalBreakdown{
				TotalToolCalls:    0,
				FSMFailed:         true,
				TerminationReason: "fsm_failed",
			},
			expected: -0.50,
		},
		{
			name: "FSM crash / hard failure with tool failure",
			input: EvalBreakdown{
				TotalToolCalls:      1,
				SuccessfulToolCalls: 0,
				FailedToolCalls:     1,
				FSMFailed:           true,
				TerminationReason:   "fsm_failed",
			},
			expected: -0.90,
		},
		{
			name: "Hard failure + negative feedback (clamp floor)",
			input: EvalBreakdown{
				TotalToolCalls:      1,
				SuccessfulToolCalls: 0,
				FailedToolCalls:     1,
				FSMFailed:           true,
				TerminationReason:   "fsm_failed",
				HasUserFeedback:     true,
				UserFeedback:        -1.0,
			},
			expected: -1.00,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			actual := CalculateReward(tc.input)
			assert.InDelta(t, tc.expected, actual, 0.0001)
		})
	}
}
