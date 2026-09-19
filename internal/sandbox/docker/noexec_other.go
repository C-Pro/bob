//go:build !linux

package docker

func checkNoExec(path string) error {
	return nil
}
