//go:build linux

package docker

import (
	"fmt"

	"golang.org/x/sys/unix"
)

func checkNoExec(path string) error {
	var stat unix.Statfs_t
	if err := unix.Statfs(path, &stat); err != nil {
		return nil
	}
	if uint64(stat.Flags)&unix.MS_NOEXEC != 0 {
		return fmt.Errorf("filesystem at %q is mounted with noexec; cannot execute proxy forwarder in sandbox container (ensure data directory is mounted with exec permissions)", path)
	}
	return nil
}
