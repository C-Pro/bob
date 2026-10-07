package scheduled

// MessageSenderFunc is an adapter for MessageSender.
type MessageSenderFunc func(chatID, content string) error

// SendMessage calls f(chatID, content).
func (f MessageSenderFunc) SendMessage(chatID, content string) error {
	return f(chatID, content)
}
