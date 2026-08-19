package geoip

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math/rand" //nolint:gosec // nosemgrep: go.lang.security.audit.crypto.math_random.math-random-used
	"net/http"
	"strconv"
	"strings"
	"time"

	"bob/internal/models"
)

// Provider represents a GEOIP service provider and parser.
type Provider struct {
	Name   string
	URL    string
	Parser func([]byte) (*models.Location, error)
}

// DefaultProviders returns the default list of supported GEOIP providers.
func DefaultProviders() []Provider {
	return []Provider{
		{
			Name:   "ip-api.com",
			URL:    "http://ip-api.com/json/",
			Parser: ParseIPAPI,
		},
		{
			Name:   "ipapi.co",
			URL:    "https://ipapi.co/json/",
			Parser: ParseIPAPICo,
		},
		{
			Name:   "ipinfo.io",
			URL:    "https://ipinfo.io/json",
			Parser: ParseIPInfo,
		},
	}
}

// ParseIPAPI parses a response from http://ip-api.com/json/.
func ParseIPAPI(body []byte) (*models.Location, error) {
	var resp struct {
		Status  string  `json:"status"`
		Message string  `json:"message"`
		Lat     float64 `json:"lat"`
		Lon     float64 `json:"lon"`
	}
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("ip-api unmarshal error: %w", err)
	}
	if resp.Status != "" && strings.ToLower(resp.Status) != "success" {
		return nil, fmt.Errorf("ip-api failed status: %s", resp.Message)
	}
	if resp.Lat == 0 && resp.Lon == 0 {
		return nil, errors.New("ip-api returned empty coordinates (0, 0)")
	}
	return &models.Location{
		Lat: resp.Lat,
		Lng: resp.Lon,
	}, nil
}

// ParseIPAPICo parses a response from https://ipapi.co/json/.
func ParseIPAPICo(body []byte) (*models.Location, error) {
	var resp struct {
		Latitude  float64 `json:"latitude"`
		Longitude float64 `json:"longitude"`
		Error     bool    `json:"error"`
		Reason    string  `json:"reason"`
	}
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("ipapi.co unmarshal error: %w", err)
	}
	if resp.Error {
		return nil, fmt.Errorf("ipapi.co error: %s", resp.Reason)
	}
	if resp.Latitude == 0 && resp.Longitude == 0 {
		return nil, errors.New("ipapi.co returned empty coordinates (0, 0)")
	}
	return &models.Location{
		Lat: resp.Latitude,
		Lng: resp.Longitude,
	}, nil
}

// ParseIPInfo parses a response from https://ipinfo.io/json.
func ParseIPInfo(body []byte) (*models.Location, error) {
	var resp struct {
		Loc   string `json:"loc"`
		Error *struct {
			Title   string `json:"title"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("ipinfo.io unmarshal error: %w", err)
	}
	if resp.Error != nil && (resp.Error.Title != "" || resp.Error.Message != "") {
		return nil, fmt.Errorf("ipinfo.io error: %s - %s", resp.Error.Title, resp.Error.Message)
	}
	if strings.TrimSpace(resp.Loc) == "" {
		return nil, errors.New("ipinfo.io loc coordinate field is empty")
	}

	parts := strings.Split(resp.Loc, ",")
	if len(parts) != 2 {
		return nil, fmt.Errorf("ipinfo.io invalid loc format: %s", resp.Loc)
	}

	lat, err := strconv.ParseFloat(strings.TrimSpace(parts[0]), 64)
	if err != nil {
		return nil, fmt.Errorf("ipinfo.io invalid latitude %q: %w", parts[0], err)
	}

	lng, err := strconv.ParseFloat(strings.TrimSpace(parts[1]), 64)
	if err != nil {
		return nil, fmt.Errorf("ipinfo.io invalid longitude %q: %w", parts[1], err)
	}

	if lat == 0 && lng == 0 {
		return nil, errors.New("ipinfo.io returned empty coordinates (0, 0)")
	}

	return &models.Location{
		Lat: lat,
		Lng: lng,
	}, nil
}

// Client handles GEOIP lookups using random round-robin across configured providers.
type Client struct {
	httpClient  *http.Client
	providers   []Provider
	maxAttempts int
	randSource  *rand.Rand
}

// Option configures a GEOIP Client.
type Option func(*Client)

// WithProviders sets custom providers for GEOIP queries.
func WithProviders(providers []Provider) Option {
	return func(c *Client) {
		c.providers = providers
	}
}

// WithMaxAttempts sets the maximum number of query attempts across providers.
func WithMaxAttempts(attempts int) Option {
	return func(c *Client) {
		if attempts > 0 {
			c.maxAttempts = attempts
		}
	}
}

// WithRand sets a custom random generator (useful for deterministic tests).
func WithRand(r *rand.Rand) Option {
	return func(c *Client) {
		c.randSource = r
	}
}

// NewClient creates a new GEOIP Client.
func NewClient(httpClient *http.Client, opts ...Option) *Client {
	if httpClient == nil {
		httpClient = &http.Client{Timeout: 5 * time.Second}
	}
	c := &Client{
		httpClient:  httpClient,
		providers:   DefaultProviders(),
		maxAttempts: 3,
		randSource:  rand.New(rand.NewSource(time.Now().UnixNano())), //nolint:gosec // nosemgrep: go.lang.security.audit.crypto.math_random.math-random-used
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// FetchLocation queries one of the GEOIP providers using random round-robin (up to maxAttempts).
func (c *Client) FetchLocation(ctx context.Context) (*models.Location, error) {
	if len(c.providers) == 0 {
		return nil, errors.New("no GEOIP providers configured")
	}

	// Create a randomized order of providers
	shuffled := make([]Provider, len(c.providers))
	copy(shuffled, c.providers)
	c.randSource.Shuffle(len(shuffled), func(i, j int) {
		shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
	})

	attempts := c.maxAttempts
	if attempts > len(shuffled) {
		attempts = len(shuffled)
	}

	var errs []error
	for i := 0; i < attempts; i++ {
		provider := shuffled[i]
		slog.Debug("attempting GEOIP query", "provider", provider.Name, "attempt", i+1, "maxAttempts", attempts)

		loc, err := c.queryProvider(ctx, provider)
		if err == nil && loc != nil {
			slog.Info("successfully determined server location via GEOIP",
				"provider", provider.Name,
				"lat", loc.Lat,
				"lng", loc.Lng,
				"attempt", i+1,
			)
			return loc, nil
		}

		slog.Warn("GEOIP query attempt failed", "provider", provider.Name, "attempt", i+1, "error", err)
		errs = append(errs, fmt.Errorf("%s: %w", provider.Name, err))
	}

	return nil, fmt.Errorf("failed to determine GEOIP location after %d attempts: %w", attempts, errors.Join(errs...))
}

func (c *Client) queryProvider(ctx context.Context, p Provider) (*models.Location, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, p.URL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Standard User-Agent header (prevents 403 Forbidden on providers like ipapi.co)
	req.Header.Set("User-Agent", "Besedka-Bot/1.0 (https://besedka.ai)")
	req.Header.Set("Accept", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("http request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("received non-200 status code: %d", resp.StatusCode)
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, 64*1024))
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	return p.Parser(body)
}

// FetchLocation queries GEOIP using the default client and random round-robin.
func FetchLocation(ctx context.Context, httpClient *http.Client) (*models.Location, error) {
	return NewClient(httpClient).FetchLocation(ctx)
}
