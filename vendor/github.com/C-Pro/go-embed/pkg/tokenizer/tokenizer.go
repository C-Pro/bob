package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strings"
)

const (
	BOS_ID  = 0      // <s>
	PAD_ID  = 1      // <pad>
	EOS_ID  = 2      // </s>
	UNK_ID  = 3      // <unk>
	MASK_ID = 250001 // <mask>

	Metaspace = '\u2581'
	MaxSeqLen = 512
)

// TrieNode represents a node in the prefix search trie for unigram tokenization.
type TrieNode struct {
	Children map[rune]*TrieNode
	TokenID  int
	Score    float32
	IsEnd    bool
}

// Tokenizer implements a pure-Go SentencePiece / Unigram tokenizer for XLM-RoBERTa.
type Tokenizer struct {
	root         *TrieNode
	vocab        []string
	scores       []float32
	pieceToID    map[string]int
	multiSpaceRe *regexp.Regexp
}

type tokenizerJSON struct {
	Model struct {
		Type  string          `json:"type"`
		Vocab [][]interface{} `json:"vocab"`
	} `json:"model"`
	AddedTokens []struct {
		ID      int    `json:"id"`
		Content string `json:"content"`
	} `json:"added_tokens"`
}

// LoadFromBytes parses a tokenizer from tokenizer.json raw byte payload.
func LoadFromBytes(data []byte) (*Tokenizer, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("tokenizer data is empty")
	}

	var tj tokenizerJSON
	if err := json.Unmarshal(data, &tj); err != nil {
		return nil, fmt.Errorf("failed to unmarshal tokenizer json: %w", err)
	}

	numVocab := len(tj.Model.Vocab)
	tok := &Tokenizer{
		root: &TrieNode{
			Children: make(map[rune]*TrieNode),
			TokenID:  -1,
			Score:    -1e9,
			IsEnd:    false,
		},
		vocab:        make([]string, numVocab),
		scores:       make([]float32, numVocab),
		pieceToID:    make(map[string]int, numVocab+len(tj.AddedTokens)),
		multiSpaceRe: regexp.MustCompile(` {2,}`),
	}

	for i, item := range tj.Model.Vocab {
		if len(item) < 2 {
			continue
		}
		piece, ok1 := item[0].(string)
		scoreFloat, ok2 := item[1].(float64)
		if !ok1 || !ok2 {
			continue
		}
		if piece == "" {
			continue
		}

		score := float32(scoreFloat)
		tok.vocab[i] = piece
		tok.scores[i] = score
		tok.pieceToID[piece] = i

		// Insert into Trie
		curr := tok.root
		for _, r := range piece {
			child, exists := curr.Children[r]
			if !exists {
				child = &TrieNode{
					Children: make(map[rune]*TrieNode),
					TokenID:  -1,
					Score:    -1e9,
					IsEnd:    false,
				}
				curr.Children[r] = child
			}
			curr = child
		}
		curr.TokenID = i
		curr.Score = score
		curr.IsEnd = true
	}

	for _, at := range tj.AddedTokens {
		tok.pieceToID[at.Content] = at.ID
	}

	return tok, nil
}

// LoadFromFile loads the tokenizer from tokenizer.json.
func LoadFromFile(path string) (*Tokenizer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read tokenizer file %s: %w", path, err)
	}
	return LoadFromBytes(data)
}

// VocabSize returns the total vocabulary size.
func (t *Tokenizer) VocabSize() int {
	if t == nil {
		return 0
	}
	return len(t.vocab)
}

// Normalize applies standard SentencePiece Metaspace normalization.
func (t *Tokenizer) Normalize(text string) string {
	// 1. Map common full-width punctuation to standard ASCII (SentencePiece NFKC normalizer behavior)
	var sb strings.Builder
	for _, r := range text {
		switch r {
		case '？':
			sb.WriteRune('?')
		case '！':
			sb.WriteRune('!')
		case '：':
			sb.WriteRune(':')
		case '；':
			sb.WriteRune(';')
		case '，':
			sb.WriteRune(',')
		case '（':
			sb.WriteRune('(')
		case '）':
			sb.WriteRune(')')
		case '【', '〔', '［':
			sb.WriteRune('[')
		case '】', '〕', '］':
			sb.WriteRune(']')
		case '“', '”':
			sb.WriteRune('"')
		case '‘', '’':
			sb.WriteRune('\'')
		default:
			sb.WriteRune(r)
		}
	}
	s := sb.String()

	// 2. Collapse multi-spaces
	if t != nil && t.multiSpaceRe != nil {
		s = t.multiSpaceRe.ReplaceAllString(s, " ")
	}

	// 3. Trim right space if any
	s = strings.TrimRight(s, " ")

	return s
}

type DPState struct {
	Score   float32
	PrevIdx int
	TokenID int
}

// EncodeRawIntoZeroAlloc tokenizes text into raw content tokens (without BOS/EOS) using provided runeBuf and dpBuf without heap allocations.
func (t *Tokenizer) EncodeRawIntoZeroAlloc(text string, runeBuf []rune, tokens []int, dpBuf []DPState) ([]rune, []int) {
	if t == nil || t.root == nil {
		return runeBuf[:0], tokens[:0]
	}
	runeBuf = runeBuf[:0]

	// 1. In-place rune decoding and normalization into runeBuf
	hasNonSpace := false
	for _, r := range text {
		if r != ' ' && r != '\t' && r != '\n' && r != '\r' {
			hasNonSpace = true
			break
		}
	}

	if !hasNonSpace {
		return runeBuf, tokens[:0]
	}

	runeBuf = append(runeBuf, Metaspace)
	lastWasSpace := false

	for _, r := range text {
		// Map full-width punctuation
		switch r {
		case '？':
			r = '?'
		case '！':
			r = '!'
		case '：':
			r = ':'
		case '；':
			r = ';'
		case '，':
			r = ','
		case '（':
			r = '('
		case '）':
			r = ')'
		case '【', '〔', '［':
			r = '['
		case '】', '〕', '］':
			r = ']'
		case '“', '”':
			r = '"'
		case '‘', '’':
			r = '\''
		}

		if r == ' ' || r == '\t' || r == '\n' || r == '\r' {
			if !lastWasSpace {
				runeBuf = append(runeBuf, Metaspace)
				lastWasSpace = true
			}
		} else {
			runeBuf = append(runeBuf, r)
			lastWasSpace = false
		}
	}

	// Trim trailing space if any
	for len(runeBuf) > 1 && runeBuf[len(runeBuf)-1] == Metaspace {
		runeBuf = runeBuf[:len(runeBuf)-1]
	}

	n := len(runeBuf)
	if cap(dpBuf) < n+1 {
		dpBuf = make([]DPState, n+1)
	} else {
		dpBuf = dpBuf[:n+1]
	}

	for i := 0; i <= n; i++ {
		dpBuf[i] = DPState{
			Score:   -1e9,
			PrevIdx: -1,
			TokenID: -1,
		}
	}
	dpBuf[0].Score = 0.0

	// Viterbi forward search
	for i := 0; i < n; i++ {
		if dpBuf[i].Score < -1e8 {
			continue
		}

		curr := t.root
		matched := false

		for j := i; j < n; j++ {
			r := runeBuf[j]
			child, ok := curr.Children[r]
			if !ok {
				break
			}
			curr = child
			if curr.IsEnd {
				score := dpBuf[i].Score + curr.Score
				if score > dpBuf[j+1].Score {
					dpBuf[j+1].Score = score
					dpBuf[j+1].PrevIdx = i
					dpBuf[j+1].TokenID = curr.TokenID
				}
				matched = true
			}
		}

		if !matched && dpBuf[i+1].Score < -1e8 {
			dpBuf[i+1].Score = dpBuf[i].Score - 100.0
			dpBuf[i+1].PrevIdx = i
			dpBuf[i+1].TokenID = UNK_ID
		}
	}

	// Backtrack into tokens
	tokens = tokens[:0]
	currIdx := n
	for currIdx > 0 {
		prev := dpBuf[currIdx]
		if prev.PrevIdx < 0 || prev.PrevIdx >= currIdx {
			break
		}
		tokens = append(tokens, prev.TokenID)
		currIdx = prev.PrevIdx
	}

	// Reverse tokens
	for i, j := 0, len(tokens)-1; i < j; i, j = i+1, j-1 {
		tokens[i], tokens[j] = tokens[j], tokens[i]
	}

	return runeBuf, tokens
}

// EncodeRaw tokenizes text into raw content tokens (without BOS/EOS).
func (t *Tokenizer) EncodeRaw(text string) []int {
	if t == nil {
		return []int{}
	}
	runeBuf := make([]rune, 0, len(text)+2)
	tokens := make([]int, 0, 1024)
	dpBuf := make([]DPState, 0, 1024)
	_, tokens = t.EncodeRawIntoZeroAlloc(text, runeBuf, tokens, dpBuf)
	return tokens
}

// EncodeIntoZeroAlloc encodes text into inputIDs and attnMask with BOS/EOS wrapping and optional truncation.
func (t *Tokenizer) EncodeIntoZeroAlloc(text string, runeBuf []rune, inputIDs []int, attnMask []int8, dpBuf []DPState, maxLen int) ([]rune, []int, []int8) {
	if maxLen <= 0 {
		maxLen = MaxSeqLen
	}

	var rawTokens []int
	runeBuf, rawTokens = t.EncodeRawIntoZeroAlloc(text, runeBuf, inputIDs, dpBuf)

	numTokens := len(rawTokens)
	finalLen := numTokens + 2
	if finalLen > maxLen {
		finalLen = maxLen
	}
	if finalLen < 2 {
		finalLen = 2
	}

	maxContent := finalLen - 2
	if maxContent > numTokens {
		maxContent = numTokens
	}

	if cap(inputIDs) < finalLen {
		newIDs := make([]int, finalLen)
		newIDs[0] = BOS_ID
		if maxContent > 0 {
			copy(newIDs[1:1+maxContent], rawTokens[:maxContent])
		}
		newIDs[finalLen-1] = EOS_ID
		inputIDs = newIDs
	} else {
		inputIDs = inputIDs[:finalLen]
		if maxContent > 0 {
			copy(inputIDs[1:1+maxContent], rawTokens[:maxContent])
		}
		inputIDs[0] = BOS_ID
		inputIDs[finalLen-1] = EOS_ID
	}

	attnMask = attnMask[:0]
	for i := 0; i < finalLen; i++ {
		attnMask = append(attnMask, 1)
	}

	return runeBuf, inputIDs, attnMask
}

// EncodeInto encodes text into the provided inputIDs and attnMask slices.
func (t *Tokenizer) EncodeInto(text string, inputIDs []int, attnMask []int8, dpBuf []DPState, maxLen int) ([]int, []int8) {
	runeBuf := make([]rune, 0, len(text)+2)
	_, inputIDs, attnMask = t.EncodeIntoZeroAlloc(text, runeBuf, inputIDs, attnMask, dpBuf, maxLen)
	return inputIDs, attnMask
}

// Encode converts input text into token IDs, with optional max sequence length truncation.
// Includes <s> at index 0 and </s> at the end.
func (t *Tokenizer) Encode(text string, maxLen int) ([]int, []int8) {
	dpBuf := make([]DPState, 0, 1024)
	inputIDs := make([]int, 0, MaxSeqLen)
	attnMask := make([]int8, 0, MaxSeqLen)
	return t.EncodeInto(text, inputIDs, attnMask, dpBuf, maxLen)
}

// EncodeQuery tokenizes with standard 'query: ' prefix.
func (t *Tokenizer) EncodeQuery(text string, maxLen int) ([]int, []int8) {
	if !strings.HasPrefix(text, "query: ") {
		text = "query: " + text
	}
	return t.Encode(text, maxLen)
}

// EncodePassage tokenizes with standard 'passage: ' prefix.
func (t *Tokenizer) EncodePassage(text string, maxLen int) ([]int, []int8) {
	if !strings.HasPrefix(text, "passage: ") {
		text = "passage: " + text
	}
	return t.Encode(text, maxLen)
}

// Decode converts token IDs back into string.
func (t *Tokenizer) Decode(tokens []int) string {
	if t == nil || len(tokens) == 0 {
		return ""
	}
	var sb strings.Builder
	for _, id := range tokens {
		if id == BOS_ID || id == EOS_ID || id == PAD_ID {
			continue
		}
		if id >= 0 && id < len(t.vocab) {
			sb.WriteString(t.vocab[id])
		}
	}
	res := sb.String()
	res = strings.ReplaceAll(res, string(Metaspace), " ")
	return strings.TrimSpace(res)
}
