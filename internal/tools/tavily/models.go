package tavily

// SearchRequest defines parameters sent to the Tavily Search API (/search).
type SearchRequest struct {
	APIKey        string `json:"api_key,omitempty"`
	Query         string `json:"query"`
	SearchDepth   string `json:"search_depth,omitempty"`   // "basic" (default) or "advanced"
	Topic         string `json:"topic,omitempty"`          // "general" (default) or "news"
	MaxResults    int    `json:"max_results,omitempty"`    // 1 to 20 (default 5)
	IncludeAnswer bool   `json:"include_answer,omitempty"` // true by default
}

// SearchResult represents an individual search result item from Tavily.
type SearchResult struct {
	Title   string  `json:"title"`
	URL     string  `json:"url"`
	Content string  `json:"content"`
	Score   float64 `json:"score,omitempty"`
}

// SearchResponse represents the response payload from the Tavily Search API.
type SearchResponse struct {
	Query        string         `json:"query"`
	Answer       string         `json:"answer,omitempty"`
	Results      []SearchResult `json:"results"`
	ResponseTime float64        `json:"response_time,omitempty"`
}
