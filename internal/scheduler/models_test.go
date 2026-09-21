package scheduler

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestValidateScheduleName(t *testing.T) {
	validNames := []string{
		"daily_brief",
		"stock_watcher",
		"run_hourly_sync_test",
		"backup",
		"job1_step2",
	}
	for _, name := range validNames {
		assert.NoError(t, ValidateScheduleName(name), "expected valid name: %s", name)
	}

	invalidNames := []string{
		"",
		"   ",
		"Daily_Brief",           // Uppercase not allowed
		"daily-brief",           // Hyphen not allowed
		"daily__brief",          // Consecutive underscores not allowed
		"_daily_brief",          // Leading underscore not allowed
		"daily_brief_",          // Trailing underscore not allowed
		"one_two_three_four_five", // More than 4 words
		"daily brief",           // Spaces not allowed
		"daily@brief",           // Special characters
	}
	for _, name := range invalidNames {
		assert.Error(t, ValidateScheduleName(name), "expected invalid name: %s", name)
	}
}

func TestComputeRawJSONHash(t *testing.T) {
	raw1 := `{"sandbox":{"driver":"docker","network":"restricted"}}`
	raw2 := `{"sandbox":{"driver":"docker","network":"restricted"}}`
	rawDiff := `{"sandbox":{"driver":"docker","network":"none"}}`

	h1 := ComputeRawJSONHash(raw1)
	h2 := ComputeRawJSONHash(raw2)
	hDiff := ComputeRawJSONHash(rawDiff)

	assert.NotEmpty(t, h1)
	assert.Equal(t, h1, h2, "identical raw strings must produce identical hashes")
	assert.NotEqual(t, h1, hDiff, "different raw strings must produce different hashes")
}

func TestScheduleGrant_Verify(t *testing.T) {
	raw := `{"sandbox":{"driver":"docker"}}`
	validHash := ComputeRawJSONHash(raw)

	grant := &ScheduleGrant{
		ID:                    "grant_1",
		ScheduleID:            "sched_1",
		PermissionRequestJSON: raw,
		ParamsHash:            validHash,
	}
	assert.True(t, grant.Verify())

	// Tampered raw JSON
	tampered := &ScheduleGrant{
		ID:                    "grant_1",
		ScheduleID:            "sched_1",
		PermissionRequestJSON: `{"sandbox":{"driver":"docker","network":"host"}}`,
		ParamsHash:            validHash,
	}
	assert.False(t, tampered.Verify())

	// Empty fields
	assert.False(t, (&ScheduleGrant{PermissionRequestJSON: raw, ParamsHash: ""}).Verify())
	assert.False(t, (&ScheduleGrant{PermissionRequestJSON: "", ParamsHash: validHash}).Verify())

	// Nil receiver
	var nilGrant *ScheduleGrant
	assert.False(t, nilGrant.Verify())
}

func TestParseCronSchedule(t *testing.T) {
	valid := []string{
		"0 9 * * *",      // Every day at 9am
		"*/15 * * * *",   // Every 15 minutes
		"0 0 1 * *",      // 1st of every month
		"@daily",         // Descriptor
		"@hourly",        // Descriptor
	}
	for _, expr := range valid {
		sched, err := ParseCronSchedule(expr)
		assert.NoError(t, err, "cron expr should be valid: %s", expr)
		assert.NotNil(t, sched)
	}

	invalid := []string{
		"",
		"invalid",
		"* * * *",        // Only 4 fields
		"* * * * * *",    // 6 fields (non-standard for standard 5-field cron)
		"65 * * * *",     // Minute out of range
	}
	for _, expr := range invalid {
		_, err := ParseCronSchedule(expr)
		assert.Error(t, err, "cron expr should be invalid: %s", expr)
	}
}

func TestCalculateNextRun(t *testing.T) {
	now := time.Date(2026, 9, 19, 10, 0, 0, 0, time.UTC)

	t.Run("ONCE schedule", func(t *testing.T) {
		future := now.Add(2 * time.Hour).Unix()
		s := &Schedule{
			ScheduleType: ScheduleTypeOnce,
			Status:       ScheduleStatusActive,
			NextRunAt:    future,
			RunCount:     0,
		}
		next, shouldEnd, err := CalculateNextRun(s, now)
		require.NoError(t, err)
		assert.False(t, shouldEnd)
		assert.Equal(t, future, next.Unix())

		// Already executed ONCE ends
		s.RunCount = 1
		_, shouldEnd, err = CalculateNextRun(s, now)
		require.NoError(t, err)
		assert.True(t, shouldEnd)
	})

	t.Run("INTERVAL schedule", func(t *testing.T) {
		s := &Schedule{
			ScheduleType:    ScheduleTypeInterval,
			Status:          ScheduleStatusActive,
			IntervalSeconds: 600, // 10 minutes
		}
		next, shouldEnd, err := CalculateNextRun(s, now)
		require.NoError(t, err)
		assert.False(t, shouldEnd)
		assert.Equal(t, now.Add(10*time.Minute), next)

		// With max runs reached
		s.MaxRuns = 5
		s.RunCount = 5
		_, shouldEnd, err = CalculateNextRun(s, now)
		require.NoError(t, err)
		assert.True(t, shouldEnd)
	})

	t.Run("CRON schedule", func(t *testing.T) {
		s := &Schedule{
			ScheduleType: ScheduleTypeCron,
			Status:       ScheduleStatusActive,
			CronExpr:     "0 11 * * *", // next 11:00 AM
		}
		next, shouldEnd, err := CalculateNextRun(s, now)
		require.NoError(t, err)
		assert.False(t, shouldEnd)
		expected := time.Date(2026, 9, 19, 11, 0, 0, 0, time.UTC)
		assert.Equal(t, expected, next)
	})

	t.Run("Schedule expiration cutoff", func(t *testing.T) {
		exp := now.Add(5 * time.Minute).Unix()
		s := &Schedule{
			ScheduleType:    ScheduleTypeInterval,
			Status:          ScheduleStatusActive,
			IntervalSeconds: 600, // 10m > 5m expiration
			ExpiresAt:       &exp,
		}
		_, shouldEnd, err := CalculateNextRun(s, now)
		require.NoError(t, err)
		assert.True(t, shouldEnd, "should end when next occurrence exceeds expires_at")
	})

	t.Run("Terminal status ends immediately", func(t *testing.T) {
		for _, st := range []ScheduleStatus{ScheduleStatusCompleted, ScheduleStatusCancelled, ScheduleStatusExpired, ScheduleStatusDenied} {
			s := &Schedule{
				Status: st,
			}
			_, shouldEnd, err := CalculateNextRun(s, now)
			require.NoError(t, err)
			assert.True(t, shouldEnd)
		}
	})

	t.Run("Nil schedule returns error", func(t *testing.T) {
		_, shouldEnd, err := CalculateNextRun(nil, now)
		assert.Error(t, err)
		assert.True(t, shouldEnd)
	})

	t.Run("Already expired fromTime ends immediately", func(t *testing.T) {
		past := now.Add(-10 * time.Minute).Unix()
		s := &Schedule{
			ScheduleType:    ScheduleTypeInterval,
			Status:          ScheduleStatusActive,
			IntervalSeconds: 60,
			ExpiresAt:       &past,
		}
		_, shouldEnd, err := CalculateNextRun(s, now)
		require.NoError(t, err)
		assert.True(t, shouldEnd)
	})
}
