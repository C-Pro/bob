package sandbox

import (
	"fmt"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestBoundedBuffer_ExactLimit(t *testing.T) {
	buf := NewBoundedBuffer(10)
	n, err := buf.Write([]byte("0123456789"))
	assert.NoError(t, err)
	assert.Equal(t, 10, n)
	assert.False(t, buf.Truncated())
	assert.Equal(t, "0123456789", buf.String())
	assert.Equal(t, int64(10), buf.TotalBytesWritten())
}

func TestBoundedBuffer_OverflowSingleWrite(t *testing.T) {
	buf := NewBoundedBuffer(10)
	n, err := buf.Write([]byte("0123456789extra"))
	assert.NoError(t, err)
	assert.Equal(t, 15, n)
	assert.True(t, buf.Truncated())
	assert.Contains(t, buf.String(), "0123456789")
	assert.Contains(t, buf.String(), "[output truncated: exceeded 10 bytes limit]")
	assert.Equal(t, int64(15), buf.TotalBytesWritten())
}

func TestBoundedBuffer_OverflowMultipleWrites(t *testing.T) {
	buf := NewBoundedBuffer(10)
	_, _ = buf.Write([]byte("01234"))
	assert.False(t, buf.Truncated())
	_, _ = buf.Write([]byte("56789"))
	assert.False(t, buf.Truncated())
	_, _ = buf.Write([]byte("overflow"))
	assert.True(t, buf.Truncated())
	assert.Contains(t, buf.String(), "0123456789")
	assert.Contains(t, buf.String(), "[output truncated: exceeded 10 bytes limit]")
	assert.Equal(t, int64(18), buf.TotalBytesWritten())
}

func TestBoundedBuffer_ConcurrentWrites(t *testing.T) {
	buf := NewBoundedBuffer(1000)
	var wg sync.WaitGroup
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for j := 0; j < 10; j++ {
				_, _ = fmt.Fprintf(buf, "worker-%d-msg-%d\n", id, j)
			}
		}(i)
	}
	wg.Wait()
	assert.True(t, buf.Truncated())
	assert.True(t, strings.HasSuffix(buf.String(), "[output truncated: exceeded 1000 bytes limit] ..."))
}
