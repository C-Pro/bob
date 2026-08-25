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

// ExtractRequest defines parameters sent to the Tavily Extract API (/extract).
type ExtractRequest struct {
	APIKey string   `json:"api_key,omitempty"`
	URLs   []string `json:"urls"`
}

// ExtractResult represents an individual extraction item from Tavily Extract.
type ExtractResult struct {
	URL        string   `json:"url"`
	RawContent string   `json:"raw_content"`
	Images     []string `json:"images,omitempty"`
}

// ExtractFailedResult represents a failed URL in Tavily Extract.
type ExtractFailedResult struct {
	URL   string `json:"url"`
	Error string `json:"error"`
}

// ExtractResponse represents the response payload from the Tavily Extract API.
type ExtractResponse struct {
	Results       []ExtractResult       `json:"results"`
	FailedResults []ExtractFailedResult `json:"failed_results,omitempty"`
	ResponseTime  float64               `json:"response_time,omitempty"`
}

