package backup

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"fmt"
	"os"

	"golang.org/x/crypto/argon2"
)

const (
	// SaltLen is the default byte length of the Argon2 salt.
	SaltLen = 16
	keyLen  = 32
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

// Crypter handles AES-256-GCM encryption and decryption with derived Argon2id keys.
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

	aead, err := cipher.NewGCMWithRandomNonce(block)
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

// Encrypt encrypts data using AES-256-GCM with a random nonce prepended to the ciphertext.
func (c *Crypter) Encrypt(data []byte) ([]byte, error) {
	return c.aead.Seal(nil, nil, data, nil), nil
}

// Decrypt decrypts ciphertext produced by Encrypt.
func (c *Crypter) Decrypt(data []byte) ([]byte, error) {
	return c.aead.Open(nil, nil, data, nil)
}
