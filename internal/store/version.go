package store

var version = struct {
	Version     int
	Description string
}{1, "initial schema"}

// CurrentVersion returns the active schema version.
func CurrentVersion() int {
	return version.Version
}
