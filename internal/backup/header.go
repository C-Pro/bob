package backup

import (
	"bytes"
	"fmt"
	"io"
)

// Magic is the 4-byte identifier for Bob backup artifacts.
var Magic = [4]byte{'B', 'O', 'B', 'B'}

const (
	HeaderVersion1 byte = 1
)

// Header describes the metadata header of an encrypted backup artifact.
type Header struct {
	Version byte
	Salt    []byte
}

// WriteHeader writes the backup artifact header followed by payload to w.
func WriteHeader(w io.Writer, h Header, payload []byte) error {
	if len(h.Salt) > 255 {
		return fmt.Errorf("backup: salt too long: %d", len(h.Salt))
	}
	version := h.Version
	if version == 0 {
		version = HeaderVersion1
	}
	if version != HeaderVersion1 {
		return fmt.Errorf("backup: unsupported header version %d", version)
	}

	var buf bytes.Buffer
	buf.Write(Magic[:])
	buf.WriteByte(version)
	buf.WriteByte(byte(len(h.Salt)))
	buf.Write(h.Salt)

	if _, err := w.Write(buf.Bytes()); err != nil {
		return err
	}
	_, err := w.Write(payload)
	return err
}

// ReadHeader parses the header from data and returns the parsed Header and remaining payload bytes.
func ReadHeader(data []byte) (Header, []byte, error) {
	const fixedLen = 4 + 1 + 1 // magic (4) + version (1) + saltLen (1)
	if len(data) < fixedLen {
		return Header{}, nil, fmt.Errorf("backup: artifact too short")
	}
	if !bytes.Equal(data[:4], Magic[:]) {
		return Header{}, nil, fmt.Errorf("backup: bad magic")
	}
	version := data[4]
	if version != HeaderVersion1 {
		return Header{}, nil, fmt.Errorf("backup: unsupported header version %d", version)
	}
	saltLen := int(data[5])
	if len(data) < fixedLen+saltLen {
		return Header{}, nil, fmt.Errorf("backup: truncated salt in header")
	}

	salt := make([]byte, saltLen)
	copy(salt, data[fixedLen:fixedLen+saltLen])

	h := Header{
		Version: version,
		Salt:    salt,
	}
	return h, data[fixedLen+saltLen:], nil
}
