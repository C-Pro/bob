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

// WriteHeader writes the backup artifact header to w. If payload is provided, it is appended to w.
func WriteHeader(w io.Writer, h Header, payload ...[]byte) error {
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
	for _, p := range payload {
		if _, err := w.Write(p); err != nil {
			return err
		}
	}
	return nil
}

// ReadHeaderFrom reads and parses the Header from r.
func ReadHeaderFrom(r io.Reader) (Header, error) {
	const fixedLen = 4 + 1 + 1 // magic (4) + version (1) + saltLen (1)
	var prefix [fixedLen]byte
	if _, err := io.ReadFull(r, prefix[:]); err != nil {
		if err == io.EOF || err == io.ErrUnexpectedEOF {
			return Header{}, fmt.Errorf("backup: artifact too short")
		}
		return Header{}, fmt.Errorf("backup: failed to read header: %w", err)
	}
	if !bytes.Equal(prefix[:4], Magic[:]) {
		return Header{}, fmt.Errorf("backup: bad magic")
	}
	version := prefix[4]
	if version != HeaderVersion1 {
		return Header{}, fmt.Errorf("backup: unsupported header version %d", version)
	}
	saltLen := int(prefix[5])
	salt := make([]byte, saltLen)
	if _, err := io.ReadFull(r, salt); err != nil {
		return Header{}, fmt.Errorf("backup: truncated salt in header")
	}

	return Header{
		Version: version,
		Salt:    salt,
	}, nil
}

// ReadHeader parses the header from data and returns the parsed Header and remaining payload bytes.
func ReadHeader(data []byte) (Header, []byte, error) {
	r := bytes.NewReader(data)
	h, err := ReadHeaderFrom(r)
	if err != nil {
		return Header{}, nil, err
	}
	remaining, _ := io.ReadAll(r)
	return h, remaining, nil
}
