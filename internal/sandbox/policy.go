package sandbox

import (
	"errors"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"strings"
)

var (
	// MandatoryBlockedCIDRs protects cloud metadata, loopback, private RFC 1918, and Docker networks.
	MandatoryBlockedCIDRs = []string{
		"127.0.0.0/8",    // IPv4 loopback
		"10.0.0.0/8",     // RFC 1918 Class A
		"172.16.0.0/12",  // RFC 1918 Class B (including Docker bridge 172.17.0.0/16)
		"192.168.0.0/16", // RFC 1918 Class C
		"169.254.0.0/16", // Link-local & cloud metadata (169.254.169.254/32)
		"::1/128",        // IPv6 loopback
		"fc00::/7",       // IPv6 ULA
		"fe80::/10",      // IPv6 Link-local
	}

	// MandatoryBlockedHosts protects cloud metadata and local hostnames.
	MandatoryBlockedHosts = []string{
		"metadata.google.internal",
		"metadata",
		"169.254.169.254",
		"localhost",
	}
)

// ValidateMountPath verifies that relativePath resides strictly inside userWorkspaceDir.
// If relativePath is ".", "", or "/", it mounts the whole workspace (isWholeWorkspace = true).
// Otherwise, it verifies the path using os.OpenRoot to ensure no traversal or symlink escapes.
func ValidateMountPath(userWorkspaceDir, relativePath string) (string, bool, error) {
	if strings.TrimSpace(userWorkspaceDir) == "" {
		return "", false, errors.New("user workspace directory cannot be empty")
	}
	absWorkspace, err := filepath.Abs(userWorkspaceDir)
	if err != nil {
		return "", false, fmt.Errorf("failed to resolve user workspace path: %w", err)
	}

	trimmedRel := strings.TrimSpace(relativePath)
	cleanRel := filepath.Clean(trimmedRel)

	if cleanRel == "." || cleanRel == "" || trimmedRel == "/" {
		return absWorkspace, true, nil
	}

	if filepath.IsAbs(cleanRel) || strings.HasPrefix(cleanRel, "/") || strings.HasPrefix(cleanRel, "\\") {
		return "", false, fmt.Errorf("mount path must be relative to user workspace, got absolute: %s", relativePath)
	}
	if cleanRel == ".." || strings.HasPrefix(cleanRel, ".."+string(filepath.Separator)) {
		return "", false, fmt.Errorf("mount path cannot traverse outside user workspace: %s", relativePath)
	}

	if err := os.MkdirAll(absWorkspace, 0o755); err != nil {
		return "", false, fmt.Errorf("failed to prepare user workspace directory: %w", err)
	}

	root, err := os.OpenRoot(absWorkspace)
	if err != nil {
		return "", false, fmt.Errorf("failed to open user workspace root: %w", err)
	}
	defer func() { _ = root.Close() }()

	targetPath := filepath.Join(absWorkspace, cleanRel)
	if fi, err := root.Stat(cleanRel); err == nil {
		if !fi.IsDir() {
			return "", false, fmt.Errorf("mount path must be a directory: %s", relativePath)
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return "", false, fmt.Errorf("invalid mount path %q: %w", relativePath, err)
	}

	return targetPath, false, nil
}

// ValidateNetworkPolicy validates and sanitizes a requested network policy,
// unconditionally injecting mandatory cloud metadata blocks.
func ValidateNetworkPolicy(policy NetworkPolicy) (NetworkPolicy, error) {
	sanitized := NetworkPolicy{
		Mode: policy.Mode,
	}

	switch policy.Mode {
	case NetworkNone, "":
		sanitized.Mode = NetworkNone
		return sanitized, nil

	case NetworkRestricted, NetworkFull:
		// Collect and deduplicate allowed hosts
		hostSet := make(map[string]struct{})
		for _, h := range policy.AllowedHosts {
			clean := strings.ToLower(strings.TrimSpace(h))
			if clean != "" {
				hostSet[clean] = struct{}{}
			}
		}
		for h := range hostSet {
			sanitized.AllowedHosts = append(sanitized.AllowedHosts, h)
		}

		// Collect and deduplicate blocked hosts, ensuring mandatory blocks are always included
		blockedSet := make(map[string]struct{})
		for _, h := range policy.BlockedHosts {
			clean := strings.ToLower(strings.TrimSpace(h))
			if clean != "" {
				blockedSet[clean] = struct{}{}
			}
		}
		for _, h := range MandatoryBlockedHosts {
			blockedSet[strings.ToLower(h)] = struct{}{}
		}
		for h := range blockedSet {
			sanitized.BlockedHosts = append(sanitized.BlockedHosts, h)
		}

		// Collect and validate CIDRs
		cidrSet := make(map[string]struct{})
		for _, c := range policy.BlockedCIDRs {
			clean := strings.TrimSpace(c)
			if clean != "" {
				if _, _, err := net.ParseCIDR(clean); err != nil {
					return sanitized, fmt.Errorf("invalid blocked CIDR: %s (%w)", clean, err)
				}
				cidrSet[clean] = struct{}{}
			}
		}
		for _, c := range MandatoryBlockedCIDRs {
			cidrSet[c] = struct{}{}
		}
		for c := range cidrSet {
			sanitized.BlockedCIDRs = append(sanitized.BlockedCIDRs, c)
		}

		return sanitized, nil

	default:
		return sanitized, fmt.Errorf("invalid network mode: %s", policy.Mode)
	}
}

// ValidateImage checks if the requested Docker image is allowed by configuration.
func ValidateImage(image string, allowedImages []string) error {
	trimmed := strings.TrimSpace(image)
	if trimmed == "" {
		return errors.New("docker image cannot be empty")
	}
	for _, allowed := range allowedImages {
		if trimmed == strings.TrimSpace(allowed) {
			return nil
		}
	}
	return fmt.Errorf("docker image %q is not in allowed images whitelist: %v", trimmed, allowedImages)
}
