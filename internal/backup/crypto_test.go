package backup

import (
	"bytes"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCryptoEncryptDecrypt(t *testing.T) {
	secret := []byte("test-super-secret-passphrase")
	crypter, err := NewCrypter(secret, nil)
	require.NoError(t, err)
	assert.Len(t, crypter.Salt(), SaltLen)

	plaintext := []byte("SQLite format 3\x00this is a secret database content with sensitive memory vectors")
	ciphertext, err := crypter.Encrypt(plaintext)
	require.NoError(t, err)
	assert.NotEqual(t, plaintext, ciphertext)

	// Recreate crypter with same salt
	decryptCrypter, err := NewCrypter(secret, crypter.Salt())
	require.NoError(t, err)

	decrypted, err := decryptCrypter.Decrypt(ciphertext)
	require.NoError(t, err)
	assert.Equal(t, plaintext, decrypted)
}

func TestCryptoWrongSecret(t *testing.T) {
	secret := []byte("correct-secret")
	wrongSecret := []byte("wrong-secret")

	crypter, err := NewCrypter(secret, nil)
	require.NoError(t, err)

	plaintext := []byte("confidential backup data")
	ciphertext, err := crypter.Encrypt(plaintext)
	require.NoError(t, err)

	wrongCrypter, err := NewCrypter(wrongSecret, crypter.Salt())
	require.NoError(t, err)

	_, err = wrongCrypter.Decrypt(ciphertext)
	assert.Error(t, err)
}

func TestCryptoInvalidSaltLength(t *testing.T) {
	secret := []byte("secret")
	shortSalt := []byte("too-short")

	_, err := NewCrypter(secret, shortSalt)
	assert.ErrorContains(t, err, "invalid salt length")
}

func TestCryptoSaltCopy(t *testing.T) {
	secret := []byte("secret")
	crypter, err := NewCrypter(secret, nil)
	require.NoError(t, err)

	s1 := crypter.Salt()
	s2 := crypter.Salt()
	assert.True(t, bytes.Equal(s1, s2))

	// Modifying s1 should not mutate crypter internal salt
	s1[0] ^= 0xFF
	s3 := crypter.Salt()
	assert.False(t, bytes.Equal(s1, s3))
}
