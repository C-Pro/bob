package webfetch

import (
	"fmt"
	"strings"
)

// MinTextLengthThreshold is the minimum character length required for direct readability text.
const MinTextLengthThreshold = 200

// ChallengeKeywords contains common lowercase substrings indicating bot challenges, captchas, or blocking.
var ChallengeKeywords = []string{
	"javascript is required",
	"please enable javascript",
	"you need to enable javascript to run this app",
	"enable javascript and cookies",
	"javascript must be enabled",
	"just a moment...",
	"attention required! | cloudflare",
	"checking your browser before accessing",
	"ddos protection by cloudflare",
	"cf-browser-verification",
	"cf-im-under-attack",
	"verify you are human",
	"press & hold to confirm you are human",
	"datadome",
	"px-captcha",
	"g-recaptcha",
	"blocked by incapsula",
	"imperva incapsula",
	"access denied",
	"403 forbidden",
	"security verification",
	"security challenge",
	"aws waf",
}

// NeedsDynamicFallback evaluates whether readability output is insufficient or blocked, requiring dynamic extraction fallback.
func NeedsDynamicFallback(result *ReadabilityResult, parseErr error, rawHTML string) (bool, string) {
	if parseErr != nil {
		return true, fmt.Sprintf("HTML parsing failed: %v", parseErr)
	}

	if result == nil {
		return true, "no readability content extracted"
	}

	trimmedText := strings.TrimSpace(result.TextContent)
	if len(trimmedText) < MinTextLengthThreshold {
		return true, fmt.Sprintf("extracted text too short (%d chars < %d threshold)", len(trimmedText), MinTextLengthThreshold)
	}

	lowerText := strings.ToLower(result.TextContent)
	lowerTitle := strings.ToLower(result.Title)
	lowerRaw := strings.ToLower(rawHTML)

	for _, kw := range ChallengeKeywords {
		if strings.Contains(lowerTitle, kw) || strings.Contains(lowerText, kw) || (len(trimmedText) < 500 && strings.Contains(lowerRaw, kw)) {
			return true, fmt.Sprintf("detected challenge/interstitial keyword: %q", kw)
		}
	}

	return false, ""
}
