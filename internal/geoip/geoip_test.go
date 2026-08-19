package geoip

import (
	"context"
	"encoding/json"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"bob/internal/models"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestParseIPAPI(t *testing.T) {
	tests := []struct {
		name        string
		body        string
		wantLoc     *models.Location
		expectError bool
	}{
		{
			name: "valid ip-api success response",
			body: `{
				"status": "success",
				"country": "United States",
				"lat": 37.7749,
				"lon": -122.4194,
				"query": "1.2.3.4"
			}`,
			wantLoc: &models.Location{
				Lat: 37.7749,
				Lng: -122.4194,
			},
			expectError: false,
		},
		{
			name: "ip-api fail status",
			body: `{
				"status": "fail",
				"message": "invalid query",
				"query": "256.256.256.256"
			}`,
			wantLoc:     nil,
			expectError: true,
		},
		{
			name:        "ip-api empty/zero coordinates",
			body:        `{"status":"success","lat":0,"lon":0}`,
			wantLoc:     nil,
			expectError: true,
		},
		{
			name:        "ip-api invalid json",
			body:        `{invalid json`,
			wantLoc:     nil,
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			loc, err := ParseIPAPI([]byte(tt.body))
			if tt.expectError {
				assert.Error(t, err)
				assert.Nil(t, loc)
			} else {
				require.NoError(t, err)
				assert.Equal(t, tt.wantLoc.Lat, loc.Lat)
				assert.Equal(t, tt.wantLoc.Lng, loc.Lng)
			}
		})
	}
}

func TestParseIPAPICo(t *testing.T) {
	tests := []struct {
		name        string
		body        string
		wantLoc     *models.Location
		expectError bool
	}{
		{
			name: "valid ipapi.co success response",
			body: `{
				"ip": "8.8.8.8",
				"latitude": 37.386,
				"longitude": -122.0838,
				"city": "Mountain View"
			}`,
			wantLoc: &models.Location{
				Lat: 37.386,
				Lng: -122.0838,
			},
			expectError: false,
		},
		{
			name: "ipapi.co error response",
			body: `{
				"error": true,
				"reason": "Rate limited"
			}`,
			wantLoc:     nil,
			expectError: true,
		},
		{
			name:        "ipapi.co empty/zero coordinates",
			body:        `{"latitude":0,"longitude":0}`,
			wantLoc:     nil,
			expectError: true,
		},
		{
			name:        "ipapi.co invalid json",
			body:        `not json`,
			wantLoc:     nil,
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			loc, err := ParseIPAPICo([]byte(tt.body))
			if tt.expectError {
				assert.Error(t, err)
				assert.Nil(t, loc)
			} else {
				require.NoError(t, err)
				assert.Equal(t, tt.wantLoc.Lat, loc.Lat)
				assert.Equal(t, tt.wantLoc.Lng, loc.Lng)
			}
		})
	}
}

func TestParseIPInfo(t *testing.T) {
	tests := []struct {
		name        string
		body        string
		wantLoc     *models.Location
		expectError bool
	}{
		{
			name: "valid ipinfo.io success response",
			body: `{
				"ip": "8.8.8.8",
				"city": "Mountain View",
				"region": "California",
				"country": "US",
				"loc": "37.3860,-122.0838"
			}`,
			wantLoc: &models.Location{
				Lat: 37.3860,
				Lng: -122.0838,
			},
			expectError: false,
		},
		{
			name: "ipinfo.io error response",
			body: `{
				"status": 429,
				"error": {
					"title": "Rate limit exceeded",
					"message": "You have exceeded the request quota"
				}
			}`,
			wantLoc:     nil,
			expectError: true,
		},
		{
			name:        "ipinfo.io empty loc string",
			body:        `{"ip":"8.8.8.8","loc":""}`,
			wantLoc:     nil,
			expectError: true,
		},
		{
			name:        "ipinfo.io invalid loc format",
			body:        `{"ip":"8.8.8.8","loc":"invalid-coord"}`,
			wantLoc:     nil,
			expectError: true,
		},
		{
			name:        "ipinfo.io non-float coordinates",
			body:        `{"ip":"8.8.8.8","loc":"abc,xyz"}`,
			wantLoc:     nil,
			expectError: true,
		},
		{
			name:        "ipinfo.io zero coordinates",
			body:        `{"ip":"8.8.8.8","loc":"0,0"}`,
			wantLoc:     nil,
			expectError: true,
		},
		{
			name:        "ipinfo.io invalid json",
			body:        `corrupted`,
			wantLoc:     nil,
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			loc, err := ParseIPInfo([]byte(tt.body))
			if tt.expectError {
				assert.Error(t, err)
				assert.Nil(t, loc)
			} else {
				require.NoError(t, err)
				assert.Equal(t, tt.wantLoc.Lat, loc.Lat)
				assert.Equal(t, tt.wantLoc.Lng, loc.Lng)
			}
		})
	}
}

func TestClient_FetchLocation_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "Besedka-Bot/1.0 (https://besedka.ai)", r.Header.Get("User-Agent"))
		assert.Equal(t, "application/json", r.Header.Get("Accept"))

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"status": "success",
			"lat":    51.5074,
			"lon":    -0.1278,
		})
	}))
	defer server.Close()

	providers := []Provider{
		{
			Name:   "mock-ip-api",
			URL:    server.URL,
			Parser: ParseIPAPI,
		},
	}

	client := NewClient(server.Client(), WithProviders(providers), WithMaxAttempts(3))
	loc, err := client.FetchLocation(context.Background())

	require.NoError(t, err)
	require.NotNil(t, loc)
	assert.Equal(t, 51.5074, loc.Lat)
	assert.Equal(t, -0.1278, loc.Lng)
}

func TestClient_FetchLocation_FailoverAndRoundRobin(t *testing.T) {
	var s1Calls, s2Calls, s3Calls int32

	// Server 1 always fails with 500
	s1 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&s1Calls, 1)
		http.Error(w, "internal server error", http.StatusInternalServerError)
	}))
	defer s1.Close()

	// Server 2 fails with error JSON
	s2 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&s2Calls, 1)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"error":  true,
			"reason": "rate limit",
		})
	}))
	defer s2.Close()

	// Server 3 succeeds
	s3 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&s3Calls, 1)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"loc": "40.7128,-74.0060",
		})
	}))
	defer s3.Close()

	providers := []Provider{
		{Name: "p1", URL: s1.URL, Parser: ParseIPAPI},
		{Name: "p2", URL: s2.URL, Parser: ParseIPAPICo},
		{Name: "p3", URL: s3.URL, Parser: ParseIPInfo},
	}

	// Use deterministic rand seed where p1, p2, p3 order is tried
	r := rand.New(rand.NewSource(42))
	client := NewClient(
		&http.Client{Timeout: 2 * time.Second},
		WithProviders(providers),
		WithMaxAttempts(3),
		WithRand(r),
	)

	loc, err := client.FetchLocation(context.Background())
	require.NoError(t, err)
	require.NotNil(t, loc)
	assert.Equal(t, 40.7128, loc.Lat)
	assert.Equal(t, -74.0060, loc.Lng)

	// Verify that at least one failed server was called before falling back to s3
	assert.GreaterOrEqual(t, atomic.LoadInt32(&s3Calls), int32(1))
}

func TestClient_FetchLocation_AllFail(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "down", http.StatusServiceUnavailable)
	}))
	defer s.Close()

	providers := []Provider{
		{Name: "p1", URL: s.URL, Parser: ParseIPAPI},
		{Name: "p2", URL: s.URL, Parser: ParseIPAPICo},
		{Name: "p3", URL: s.URL, Parser: ParseIPInfo},
	}

	client := NewClient(s.Client(), WithProviders(providers), WithMaxAttempts(3))
	loc, err := client.FetchLocation(context.Background())

	require.Error(t, err)
	assert.Nil(t, loc)
	assert.Contains(t, err.Error(), "failed to determine GEOIP location after 3 attempts")
}

func TestClient_FetchLocation_NoProviders(t *testing.T) {
	client := NewClient(&http.Client{}, WithProviders([]Provider{}))
	loc, err := client.FetchLocation(context.Background())
	require.Error(t, err)
	assert.Nil(t, loc)
	assert.Contains(t, err.Error(), "no GEOIP providers configured")
}

func TestClient_FetchLocation_ContextCanceled(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(100 * time.Millisecond)
		w.WriteHeader(http.StatusOK)
	}))
	defer s.Close()

	providers := []Provider{
		{Name: "p1", URL: s.URL, Parser: ParseIPAPI},
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	client := NewClient(s.Client(), WithProviders(providers))
	loc, err := client.FetchLocation(ctx)
	require.Error(t, err)
	assert.Nil(t, loc)
}

func TestDefaultProviders(t *testing.T) {
	dp := DefaultProviders()
	require.Len(t, dp, 3)

	names := []string{dp[0].Name, dp[1].Name, dp[2].Name}
	assert.Contains(t, names, "ip-api.com")
	assert.Contains(t, names, "ipapi.co")
	assert.Contains(t, names, "ipinfo.io")
}
