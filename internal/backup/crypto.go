package backup

import (
	"bytes"
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"encoding/binary"
	"fmt"
	"io"
	"os"

	"golang.org/x/crypto/argon2"
)

const (
	// SaltLen is the default byte length of the Argon2 salt.
	SaltLen = 16
	keyLen  = 32

	chunkSize              = 64 * 1024 // 64 KiB chunk size for streaming AEAD
	maxChunkCiphertextSize = chunkSize + 1024
	nonceLen               = 12
)

var (
	argonTime uint32 = 15
	argonMem  uint32 = 64 * 1024
	argonThr  uint8  = 4
)

func init() {
	// In CI or testing environments, reduce the number of rounds to speed up execution.
	if os.Getenv("CI") != "" {
		argonTime = 1
		argonMem = 16 * 1024
		argonThr = 1
	}
}

// Crypter handles AES-256-GCM streaming chunked encryption and decryption with derived Argon2id keys.
type Crypter struct {
	salt []byte
	aead cipher.AEAD
}

// NewCrypter creates an instance of Crypter.
// If salt is empty, a cryptographically random salt is generated.
func NewCrypter(secret []byte, salt []byte) (*Crypter, error) {
	var err error
	if len(salt) == 0 {
		if salt, err = genSalt(); err != nil {
			return nil, err
		}
	} else if len(salt) != SaltLen {
		return nil, fmt.Errorf("invalid salt length: expected %d, got %d", SaltLen, len(salt))
	}

	key := deriveKey(secret, salt)

	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, err
	}

	aead, err := cipher.NewGCM(block)
	if err != nil {
		return nil, err
	}

	return &Crypter{
		salt: salt,
		aead: aead,
	}, nil
}

// Salt returns a copy of the salt used for key derivation.
func (c *Crypter) Salt() []byte {
	saltCopy := make([]byte, len(c.salt))
	copy(saltCopy, c.salt)
	return saltCopy
}

// EncryptWriter returns an io.WriteCloser that encrypts written data in 64 KiB AES-256-GCM chunks to w.
func (c *Crypter) EncryptWriter(w io.Writer) (io.WriteCloser, error) {
	var baseNonce [nonceLen]byte
	if _, err := rand.Read(baseNonce[:]); err != nil {
		return nil, fmt.Errorf("failed to generate nonce: %w", err)
	}
	if _, err := w.Write(baseNonce[:]); err != nil {
		return nil, fmt.Errorf("failed to write nonce: %w", err)
	}

	return &encryptWriter{
		w:         w,
		aead:      c.aead,
		baseNonce: baseNonce,
		buf:       make([]byte, 0, chunkSize),
	}, nil
}

// DecryptReader returns an io.Reader that streams and decrypts 64 KiB AES-256-GCM chunks from r.
func (c *Crypter) DecryptReader(r io.Reader) (io.Reader, error) {
	var baseNonce [nonceLen]byte
	if _, err := io.ReadFull(r, baseNonce[:]); err != nil {
		if err == io.EOF || err == io.ErrUnexpectedEOF {
			return nil, fmt.Errorf("backup: truncated encrypted stream: missing nonce")
		}
		return nil, fmt.Errorf("failed to read nonce: %w", err)
	}

	return &decryptReader{
		r:         r,
		aead:      c.aead,
		baseNonce: baseNonce,
	}, nil
}

type encryptWriter struct {
	w          io.Writer
	aead       cipher.AEAD
	baseNonce  [nonceLen]byte
	chunkIndex uint64
	buf        []byte
	closed     bool
}

func (ew *encryptWriter) Write(p []byte) (int, error) {
	if ew.closed {
		return 0, fmt.Errorf("write on closed encryptWriter")
	}
	total := len(p)
	for len(p) > 0 {
		avail := chunkSize - len(ew.buf)
		if avail > len(p) {
			avail = len(p)
		}
		ew.buf = append(ew.buf, p[:avail]...)
		p = p[avail:]
		if len(ew.buf) == chunkSize {
			if err := ew.flushChunk(false); err != nil {
				return 0, err
			}
		}
	}
	return total, nil
}

func (ew *encryptWriter) Close() error {
	if ew.closed {
		return nil
	}
	ew.closed = true
	return ew.flushChunk(true)
}

func (ew *encryptWriter) flushChunk(isLast bool) error {
	nonce := deriveChunkNonce(ew.baseNonce, ew.chunkIndex)
	ew.chunkIndex++

	ciphertext := ew.aead.Seal(nil, nonce[:], ew.buf, nil)
	ew.buf = ew.buf[:0]

	var hdr [5]byte
	binary.BigEndian.PutUint32(hdr[:4], uint32(len(ciphertext)))
	if isLast {
		hdr[4] = 1
	} else {
		hdr[4] = 0
	}

	if _, err := ew.w.Write(hdr[:]); err != nil {
		return fmt.Errorf("failed to write chunk header: %w", err)
	}
	if _, err := ew.w.Write(ciphertext); err != nil {
		return fmt.Errorf("failed to write chunk ciphertext: %w", err)
	}
	return nil
}

type decryptReader struct {
	r          io.Reader
	aead       cipher.AEAD
	baseNonce  [nonceLen]byte
	chunkIndex uint64
	buf        []byte
	lastSeen   bool
	err        error
}

func (dr *decryptReader) Read(p []byte) (int, error) {
	for len(dr.buf) == 0 {
		if dr.err != nil {
			return 0, dr.err
		}
		if dr.lastSeen {
			dr.err = io.EOF
			return 0, io.EOF
		}
		if err := dr.readNextChunk(); err != nil {
			dr.err = err
			return 0, err
		}
	}
	n := copy(p, dr.buf)
	dr.buf = dr.buf[n:]
	return n, nil
}

func (dr *decryptReader) readNextChunk() error {
	var hdr [5]byte
	if _, err := io.ReadFull(dr.r, hdr[:]); err != nil {
		if err == io.EOF || err == io.ErrUnexpectedEOF {
			return fmt.Errorf("backup: truncated encrypted stream")
		}
		return fmt.Errorf("failed to read chunk header: %w", err)
	}

	chunkLen := binary.BigEndian.Uint32(hdr[:4])
	if chunkLen > maxChunkCiphertextSize {
		return fmt.Errorf("backup: chunk size %d exceeds maximum limit", chunkLen)
	}
	isLast := hdr[4] == 1

	ciphertext := make([]byte, chunkLen)
	if _, err := io.ReadFull(dr.r, ciphertext); err != nil {
		return fmt.Errorf("backup: truncated chunk ciphertext: %w", err)
	}

	nonce := deriveChunkNonce(dr.baseNonce, dr.chunkIndex)
	dr.chunkIndex++

	plaintext, err := dr.aead.Open(nil, nonce[:], ciphertext, nil)
	if err != nil {
		return fmt.Errorf("failed to decrypt artifact (wrong SECRET?): %w", err)
	}

	dr.buf = plaintext
	dr.lastSeen = isLast
	return nil
}

func deriveChunkNonce(base [nonceLen]byte, index uint64) [nonceLen]byte {
	out := base
	idx := binary.BigEndian.Uint64(base[4:]) ^ index
	binary.BigEndian.PutUint64(out[4:], idx)
	return out
}

func genSalt() ([]byte, error) {
	salt := make([]byte, SaltLen)
	if _, err := rand.Read(salt); err != nil {
		return nil, err
	}
	return salt, nil
}

func deriveKey(secret, salt []byte) []byte {
	return argon2.IDKey(secret, salt, argonTime, argonMem, argonThr, keyLen)
}

// Encrypt encrypts data into a byte slice using the streaming chunked AEAD format.
func (c *Crypter) Encrypt(data []byte) ([]byte, error) {
	var buf bytes.Buffer
	ew, err := c.EncryptWriter(&buf)
	if err != nil {
		return nil, err
	}
	if _, err := ew.Write(data); err != nil {
		_ = ew.Close()
		return nil, err
	}
	if err := ew.Close(); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// Decrypt decrypts ciphertext produced by Encrypt.
func (c *Crypter) Decrypt(data []byte) ([]byte, error) {
	dr, err := c.DecryptReader(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	return io.ReadAll(dr)
}
