package store

var version = struct {
	Version     int
	Description string
}{2, "add trajectory tables"}

// CurrentVersion returns the active schema version.
func CurrentVersion() int {
	return version.Version
}
