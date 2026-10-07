package commands

import (
	"fmt"
	"strings"
)

// FormatAdaptiveFence wraps content in markdown code fences whose length exceeds
// any backtick run contained inside content.
func FormatAdaptiveFence(content, lang string) string {
	maxRun := 2
	cur := 0
	for _, r := range content {
		if r == '`' {
			cur++
			if cur > maxRun {
				maxRun = cur
			}
		} else {
			cur = 0
		}
	}
	fence := strings.Repeat("`", maxRun+1)
	return fmt.Sprintf("%s%s\n%s\n%s", fence, lang, content, fence)
}
