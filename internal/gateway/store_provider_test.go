package gateway

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestGateway_StoreProviderAlias(t *testing.T) {
	provider := NewMemoryStoreProvider(nil, "")
	assert.NotNil(t, provider)
}
