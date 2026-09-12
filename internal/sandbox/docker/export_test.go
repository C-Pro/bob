package docker

// SetCustomBlockedCIDRs overrides default mandatory blocked CIDRs for testing.
func (d *Driver) SetCustomBlockedCIDRs(cidrs []string) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.customBlockedCIDRs = cidrs
}
