# Implementation Plan: Isolated Sandbox Execution Subsystem (`sandbox-execution`)

## Phase 1: Core Architecture, Types & Security Policy
- [x] Task: Establish core sandbox interfaces, data models, and mount path validation in `internal/sandbox`
    - [x] Define `DriverType`, `NetworkMode`, `UserMount`, `UserSandbox`, `ExecResult`, and `Driver` interface in `internal/sandbox/types.go`
    - [x] Implement mount path validation (`ValidateMountPath`) enforcing boundary checks and preventing path traversal (`../`) in `internal/sandbox/policy.go`
    - [x] Implement image whitelist validation (`ValidateImage`) and network policy sanitization (`ValidateNetworkPolicy`) in `internal/sandbox/policy.go`
    - [x] Write unit tests for mount paths, image validation, and network policy rules in `internal/sandbox/policy_test.go`
    - [x] Implement `sandbox.Manager` with thread-safe lifecycle tracking, dynamic workspace resolution from `DataDir`, pending request queue, and background expiration cleanup in `internal/sandbox/manager.go`
    - [x] Write unit tests covering `Manager` creation, approval, execution dispatch, expiration, and destruction in `internal/sandbox/manager_test.go`
    - [x] Run `go test -race ./internal/sandbox/...` and verify tests pass

## Phase 2: Bubblewrap (`bwrap`) Unprivileged Sandbox Driver
- [x] Task: Implement unprivileged Linux user namespace execution using `bwrap` in `internal/sandbox/bwrap`
    - [x] Check host binary availability via `exec.LookPath("bwrap")` in `driver.go`
    - [x] Implement workspace preparation, read-only system binds (`/usr`, `/lib`, `/lib64`, `/bin`), and pseudo filesystem mounts (`/proc`, `/dev`, `/tmp`)
    - [x] Implement `Exec` using `exec.CommandContext` with `--die-with-parent` and user workspace mounts (`/workspace`)
    - [x] Implement `Destroy` cleaning up allocated user resources and terminating child processes
    - [x] Write unit tests in `internal/sandbox/bwrap/driver_test.go` testing basic command execution, filesystem isolation, and workspace write persistence
    - [x] Run `go test -race ./internal/sandbox/bwrap/...` and verify tests pass

## Phase 3: Pure-Go Docker Engine Sibling Driver
- [x] Task: Implement CGO-free Docker Engine API driver over `/var/run/docker.sock` in `internal/sandbox/docker`
    - [x] Configure standard library `http.Client` with custom `DialContext` connecting to Docker Unix domain socket
    - [x] Implement `Available` checking daemon responsiveness via `GET /_ping`
    - [x] Implement container creation (`POST /containers/create`) with image validation, workspace binds, memory limit, and CPU quota (`NanoCPUs`)
    - [x] Implement container startup (`POST /containers/{id}/start`) and container removal (`DELETE /containers/{id}?v=1&force=true`)
    - [x] Implement `Exec` (`POST /containers/{id}/exec`, `POST /exec/{id}/start`) and 8-byte framed stdout/stderr stream demultiplexing (`demuxDockerStream`)
    - [x] Write unit tests with a mock HTTP server in `internal/sandbox/docker/driver_test.go` verifying container creation payload, stream demuxing, and exec inspect exit codes
    - [x] Run `go test -race ./internal/sandbox/docker/...` and verify tests pass

## Phase 4: Network Isolation & In-Process Filtering Proxy
- [x] Task: Implement HTTP/CONNECT filtering proxy enforcing domain whitelists and CIDR blocks in `internal/sandbox/proxy.go`
    - [x] Support HTTP `GET`/`POST` forwarding and HTTPS `CONNECT` bidirectional tunneling with `http.Hijacker`
    - [x] Enforce mandatory blocked CIDRs (loopback `127.0.0.0/8`, metadata `169.254.0.0/16`, RFC1918 private subnets)
    - [x] Enforce domain whitelist filtering for `NetworkRestricted` mode and blacklist filtering for `BlockedHosts`
    - [x] Write unit tests in `internal/sandbox/proxy_test.go` testing HTTP proxying, CONNECT tunneling, allowed domains, and blocked private subnets
    - [x] Run `go test -race ./internal/sandbox/...` and verify tests pass

## Phase 5: Direct Message Tool Definitions & User Approval Protocol
- [x] Task: Expose sandbox management tools exclusively in 1-on-1 Direct Messages in `internal/tools`
    - [x] Add session-aware tool scoping: `ToolDefinitionsForSession(session ChatSessionContext)` returning sandbox tools only when `session.IsDM == true`
    - [x] Implement `sandbox_request` tool in `internal/tools/registry.go`: validate DM context, queue pending request in `sandbox.Manager`, and post permission request card to user chat via `session.Notifier`
    - [x] Implement `sandbox_exec` tool in `internal/tools/registry.go`: verify active approved sandbox, execute command with timeout, and format exit code, duration, stdout, and stderr
    - [x] Implement `sandbox_destroy` tool in `internal/tools/registry.go`: destroy sandbox and release resources
    - [x] Write unit tests in `internal/tools/tools_test.go` verifying Townhall rejection, DM execution, approval gating, and lifecycle dispatch
    - [x] Run `go test -race ./internal/tools/...` and verify tests pass

## Phase 6: Gateway Command Interception & Rich-Text Normalization
- [x] Task: Intercept `/sandbox approve` and `/sandbox deny` commands in `internal/gateway`
    - [x] Hook into gateway event loop before prompt rendering and message context storage
    - [x] Sanitize incoming text by stripping HTML tags (`<p>`, `<code>`) and backticks produced by Besedka rich-text formatting
    - [x] Implement `/sandbox approve` handling: call `manager.ApproveSandbox(ctx, userID)` and reply with status confirmation
    - [x] Implement `/sandbox deny` handling: call `manager.DenySandbox(ctx, userID)` and notify user
    - [x] Write unit tests in `internal/gateway/gateway_test.go` verifying command parsing, HTML tag stripping, approval, and denial flows
    - [x] Run `go test -race ./internal/gateway/...` and verify tests pass

## Phase 7: Security Hardening & Review Remediation
- [x] Task: Remediate 5 critical review findings (High & Medium severity)
    - [x] **DNS Rebinding & TOCTOU Prevention (Finding 1.2 - High):** Refactor `FilteringProxy` to resolve hostnames once via `net.DefaultResolver.LookupIP`, validate all resolved IPs against CIDR blocks, pin the verified IP, and dial the target IP directly in `handleConnect` and `handleHTTP` (`DialContext`) while retaining virtual `Host` header
    - [x] **Hop-by-Hop Header Sanitization (Finding 2.2 - Medium):** Implement RFC 7230 / RFC 9110 hop-by-hop header removal (`removeHopByHopHeaders`) for `Connection`, `Keep-Alive`, `Proxy-Authenticate`, `Upgrade`, etc., applied to outbound client requests and inbound upstream responses
    - [x] **Dynamic Absolute Workspace Path Resolution (Finding 2.3 - Medium):** Add `WorkspaceDir` to `UserSandbox`, dynamically compute workspace paths via `filepath.Join(DataDir, "sandboxes", userID)`, and replace hardcoded `./data/sandboxes` in `bwrap` and `docker` drivers
    - [x] **Kernel-Enforced Network Namespace Airgap (Finding 1.1 - High):** Enforce `--unshare-all` unconditionally in `bwrap` for both `NetworkNone` and `NetworkRestricted` modes; establish Unix domain socket listener (`/run/proxy.sock`) and in-container loopback forwarder (`127.0.0.1:18080`) so raw TCP/UDP egress is blocked at the kernel level with `ENETUNREACH`
    - [x] **Docker Container Proxy Reachability & Client Access Control (Finding 2.1 - Medium):** Bind `FilteringProxy` to `0.0.0.0:0` in Docker restricted mode, inject `"ExtraHosts": ["host.docker.internal:host-gateway"]`, set `HTTP_PROXY=http://host.docker.internal:<port>`, and enforce client IP authorization in `handleRequest`
    - [x] Write dedicated tests: `TestFilteringProxyDNSRebindingPrevention`, `TestFilteringProxyHopByHopHeaders`, `TestFilteringProxyUnixSocket`, `TestBwrap_NetworkRestrictedAirgap`, and `TestDockerDriver_NetworkRestrictedProxyReachability`
    - [x] Run `go test -race ./internal/sandbox/...` and verify all tests pass cleanly

## Phase 8: Comprehensive Verification, Auditing, & Quality Checks
- [x] Task: Run complete repository verification and quality checks
    - [x] Run `go test -race ./...` across all packages (PASS, 0 race conditions)
    - [x] Run `go vet ./...` (PASS, 0 errors)
    - [x] Run `osv-scanner scan -r .` (PASS, 0 vulnerabilities)
    - [x] Verify local live agent execution and Besedka gateway connection
