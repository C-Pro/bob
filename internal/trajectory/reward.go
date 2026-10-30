package trajectory

import "math"

const (
	// RewardVersionV1 is the canonical version identifier for v1 reward formulation.
	RewardVersionV1 = "v1"
)

// CalculateReward computes the composite execution heuristic score R_eval in [-1.0, 1.0].
//
// Formulation (v1):
// R_eval = clamp(S_base - 0.5*I_fail - 0.3*I_budget + 0.6*F_user, -1.0, 1.0)
func CalculateReward(b EvalBreakdown) float64 {
	var sBase float64
	if b.TotalToolCalls <= 0 {
		if !b.FSMFailed && (b.TerminationReason == "" || b.TerminationReason == "completed") {
			sBase = 0.20
		} else {
			sBase = 0.00
		}
	} else {
		successRate := float64(b.SuccessfulToolCalls) / float64(b.TotalToolCalls)
		if successRate < 0.0 {
			successRate = 0.0
		} else if successRate > 1.0 {
			successRate = 1.0
		}
		sBase = 0.40 * (2.0*successRate - 1.0)
	}

	var iFail float64
	if b.FSMFailed || b.TerminationReason == "fsm_failed" || b.TerminationReason == "cancelled" {
		iFail = 1.0
	}

	var iBudget float64
	if b.BudgetExhausted || b.TerminationReason == "budget_exhausted" {
		iBudget = 1.0
	}

	var fUser float64
	if b.HasUserFeedback {
		fUser = clamp(b.UserFeedback, -1.0, 1.0)
	}

	score := sBase - 0.50*iFail - 0.30*iBudget + 0.60*fUser
	// Round to 4 decimal places to prevent floating-point noise
	return math.Round(clamp(score, -1.0, 1.0)*10000) / 10000
}

func clamp(val, min, max float64) float64 {
	if val < min {
		return min
	}
	if val > max {
		return max
	}
	return val
}
