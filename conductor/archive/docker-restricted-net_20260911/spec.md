# Specification: Restricted Docker Network Isolation & Injected Proxy Forwarder

## 1. Overview
The current implementation of Docker restricted network mode (`sandbox.NetworkRestricted`) relies on Docker's default bridge network and advisory environment variables (`http_proxy`, `https_proxy`, `all_proxy`). This leaves a significant security gap: processes within the sandbox can ignore proxy variables to achieve direct WAN egress, perform DNS exfiltration, and reach lateral containers or host services.

This track establishes true kernel-level network airgapping for Docker restricted mode (`NetworkMode: "none"`) with an isolated Unix domain socket filtering proxy, matching Bubblewrap's security posture. Network communication is enabled via an injected, statically linked `bob-proxy-fwd` helper binary that bridges loopback TCP (`127.0.0.1:18080`) to the host Unix socket (`/run/proxy/proxy.sock`). Furthermore, both Docker and Bubblewrap drivers are unified to use this standard forwarder binary.

## 2. Functional Requirements

### 2.1 Forwarder Binary (`bob-proxy-fwd`)
- **Location & Technology:** Pure Go implementation in `cmd/bob-proxy-fwd/main.go`, compiled with `CGO_ENABLED=0` for static execution on any Linux rootfs (Alpine/musl, Debian/Ubuntu, or distroless).
- **Architecture:** Matches the architecture of the Bob binary (`linux/amd64` or `linux/arm64`).
- **Operating Modes:**
  1. **Daemon Mode (`-daemon`):** Listens continuously on loopback TCP (default `127.0.0.1:18080`) and forwards full-duplex traffic to a Unix domain socket (default `/run/proxy/proxy.sock`). Handles `SIGTERM`/`SIGINT` for clean termination.
  2. **Wrapper Mode (`-- cmd args...`):** Starts TCP listener, spawns the child process with arguments, forwards connections, and exits with the child's exit code upon completion.

### 2.2 Docker Driver Isolation (`internal/sandbox/docker`)
- **Kernel-Enforced Airgap:** In `sandbox.NetworkRestricted`, set `HostConfig["NetworkMode"] = "none"`. No external `eth0` network interface is provisioned.
- **Proxy Socket Placement & Mount:** Create Unix domain socket under `<DataDir>/sandboxes/<user>/.proxy/proxy.sock` (or `<DataDir>/proxies/<user>/proxy.sock`). Bind-mount the proxy directory or socket into the container at `/run/proxy` (translated via `toHostPath` for sibling Docker setups).
- **Binary Injection & Execution:** Bind-mount `bob-proxy-fwd` into the container as `/run/proxy/fwd:ro`. Configure container `Cmd`/`Entrypoint` to run `/run/proxy/fwd -daemon -tcp 127.0.0.1:18080 -sock /run/proxy/proxy.sock` as PID 1, replacing `sleep infinity`.
- **Exec Environment:** In `driver.Exec`, inject proxy environment variables pointing to `http://127.0.0.1:18080`. Commands run directly without shell wrappers.
- **Cleanup:** Remove `ExtraHosts` and external bridge IP discovery logic. Terminating the container cleanly stops the forwarder daemon.

### 2.3 Bubblewrap Driver Unification (`internal/sandbox/bwrap`)
- Update `internal/sandbox/bwrap/driver.go` to bind-mount the same `bob-proxy-fwd` binary.
- Use `bob-proxy-fwd -- <cmd...>` to eliminate brittle shell scripts and external dependencies on `socat` or `python3`.

### 2.4 Build and Distribution Lifecycle
- **No Binaries in Git:** Git repository tracks zero compiled binaries.
- **Build Automation (`Makefile`):** Target `build-fwd` cross-compiles `cmd/bob-proxy-fwd` with `CGO_ENABLED=0 GOOS=linux GOARCH=$(go env GOARCH)`. Integrated into `make build` and `make check`.
- **Embedding / Self-Extraction:** Bob embeds the forwarder binary during build, or extracts/caches it into `<DataDir>/bin/bob-proxy-fwd` with `0755` permissions.
- **Dev/Test On-Demand Fallback:** If `bob-proxy-fwd` is missing during local development or unit testing, Bob automatically compiles it using the host `go` toolchain.
- **Custom Path Override:** Support `SANDBOX_PROXY_FWD_PATH` environment variable for custom forwarder binary locations.

## 3. Non-Functional Requirements
- **Security:** Strict airgap. Any raw TCP/UDP or DNS traffic to non-loopback destinations fails with `ENETUNREACH`. No listening ports opened on host network interfaces (`0.0.0.0` or `docker0`).
- **Performance:** Instant command exec in Docker because the daemon is already running as PID 1 (no per-exec startup latency).
- **Concurrency:** Supports concurrent execs inside the same sandbox container without TCP port conflicts.

## 4. Acceptance Criteria
1. `docker.Driver` running in `NetworkRestricted` mode creates containers with `NetworkMode: "none"`.
2. Direct connection attempts (e.g. `nc -w 1 1.1.1.1 80`, DNS lookups) within the container fail with `ENETUNREACH`.
3. Allowed HTTP/HTTPS requests (via `http_proxy=http://127.0.0.1:18080`) succeed and are routed through `FilteringProxy`.
4. Blocked domains or IP addresses are denied according to sandbox policy.
5. Bubblewrap driver functions properly using `bob-proxy-fwd` without requiring `socat` or `python3`.
6. Sibling Docker setups (`SANDBOX_HOST_DATA_DIR`) map socket and forwarder binary paths correctly.
7. `make check` passes cleanly (linting, tests with `-race`, semgrep, osv-scanner).

## 5. Out of Scope
- Multi-architecture cross-compilation within a single Bob binary (forwarder strictly follows host/Bob binary architecture).
- Transparent iptables redirection inside containers.
