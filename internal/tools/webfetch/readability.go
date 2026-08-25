package webfetch

import (
	"bytes"
	"fmt"
	"net/url"
	"strings"

	"github.com/go-shiori/go-readability"
)

// ReadabilityResult holds the parsed readability output.
type ReadabilityResult struct {
	Title       string
	TextContent string
	Excerpt     string
	Byline      string
	Length      int
}

// ParseReadability parses HTML content using go-readability.
func ParseReadability(htmlContent []byte, pageURL string) (*ReadabilityResult, error) {
	if len(bytes.TrimSpace(htmlContent)) == 0 {
		return nil, fmt.Errorf("empty html content")
	}

	parsedURL, err := url.Parse(pageURL)
	if err != nil {
		parsedURL = &url.URL{}
	}

	article, err := readability.FromReader(bytes.NewReader(htmlContent), parsedURL)
	if err != nil {
		return nil, fmt.Errorf("readability extraction failed: %w", err)
	}

	return &ReadabilityResult{
		Title:       strings.TrimSpace(article.Title),
		TextContent: strings.TrimSpace(article.TextContent),
		Excerpt:     strings.TrimSpace(article.Excerpt),
		Byline:      strings.TrimSpace(article.Byline),
		Length:      article.Length,
	}, nil
}
