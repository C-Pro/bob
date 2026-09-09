# Specification: Isolated Sandbox Execution Subsystem (`sandbox-execution`)

## 1. Overview
The Isolated Sandbox Execution Subsystem allows the Bob AI Agent to execute shell commands, scripts, code snippets, and lightweight tools inside secure, disposable, isolated environments on behalf of authenticated Besedka users.

To safeguard the host system and the Besedka chat platform from untrusted code execution, arbitrary network access, and data exfiltration, the subsystem enforces:
1. **Multi-Engine Driver Abstraction:** Seamless execution via unprivileged Linux user namespaces (`bubblewrap` / `bwrap`) or containerized daemon instances (`docker` sibling API) without CGO or third-party SDK dependencies.
2. **Strict Direct Message Exclusivity:** Sandbox management tools are exposed exclusively in 1-on-1 Direct Messages (DMs). Execution requests in public Townhall chats are categorically prohibited.
3. **Interactive User Approval Protocol:** Tool execution requires explicit, out-of-band user authorization. Sandboxes start in a pending state, Bob delivers an informative permission request card to the user's chat, and execution remains blocked until the user types `/sandbox approve`.
4. **Kernel-Enforced Network Airgap & Domain-Filtering Proxy:** In `restricted` and `none` network modes, containers have their network namespace completely unshared (`--unshare-all`). All outbound network traffic is blocked at the kernel level (`ENETUNREACH: Network is unreachable`). In `restricted` mode, HTTP/HTTPS traffic is tunneled through a verified in-process filtering proxy over a local Unix domain socket, enforcing domain whitelists, single-resolution DNS rebinding / TOCTOU protection, RFC 7230/9110 hop-by-hop header sanitization, and CIDR blocklists.
5. **Dynamic Workspace Isolation:** Sandboxes operate on dedicated per-user workspace directories dynamically resolved from `DATA_DIR` with strict path traversal validation.

---

## 2. Architecture & Components

```
                      +-----------------------------+
                      |       Besedka User (DM)     |
                      +--------------+--------------+
                                     |
                          [Chat Event / Mentions]
                                     v
                      +-----------------------------+
                      |   Gateway Command Filter    |
                      |  (/sandbox approve | deny)  |
                      +--------------+--------------+
                                     |
                       [Normal Tool / Prompt Flow]
                                     v
                      +-----------------------------+
                      |   LLM & Tool Registry       |
                      |  (DM-scoped sandbox tools)  |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |       Sandbox Manager       |
                      |  (Lifecycle, Expiration,    |
                      |   Workspace Resolution)     |
                      +-------+-------------+-------+
                              |             |
                 +------------+             +------------+
                 |                                       |
                 v                                       v
    +-------------------------+             +-------------------------+
    |      bwrap.Driver       |             |      docker.Driver      |
    | (Unprivileged User NS,  |             | (Pure-Go Engine API,    |
    |  Read-only Host Mounts) |             |  Resource Constraints)  |
    +------------+------------+             +------------+------------+
                 |                                       |
                 +-------------------+-------------------+
                                     |
                                     v
                      +-----------------------------+
                      |       FilteringProxy        |
                      |  - Domain Whitelisting      |
                      |  - DNS Rebinding / TOCTOU   |
                      |  - Hop-by-Hop Sanitization  |
                      |  - Unix Socket / TCP        |
                      |  - Mandatory Blocked CIDRs  |
                      +-----------------------------+
```

### 2.1 Driver Abstraction (`internal/sandbox`)
The core interface decouples lifecycle and execution logic from the underlying sandboxing implementation:
```go
type Driver interface {
    Type() DriverType
    Available(ctx context.Context) bool
    Create(ctx context.Context, sbx *UserSandbox, userWorkspaceDir string) error
    Exec(ctx context.Context, sbx *UserSandbox, cmd []string, timeout time.Duration) (*ExecResult, error)
    Destroy(ctx context.Context, sbx *UserSandbox) error
}
```

### 2.2 Bubblewrap Driver (`internal/sandbox/bwrap`)
- Primary unprivileged sandbox driver for Linux environments.
- Leverages `/usr/bin/bwrap` with `--die-with-parent` to ensure zero orphaned processes.
- Read-only mounts host `/usr`, `/lib`, `/lib64`, `/bin`, and pseudo filesystems (`/proc`, `/dev`, `/tmp`).
- Enforces `--unshare-all` (user, IPC, PID, UTS, and network namespaces) for both `none` and `restricted` network modes.
- In `restricted` mode, passes proxy communication through an isolated Unix domain socket (`/run/proxy.sock`) with an in-container loopback forwarder (`127.0.0.1:18080`).

### 2.3 Docker Sibling Driver (`internal/sandbox/docker`)
- Secondary sandbox driver interacting directly with the Docker Engine daemon via HTTP over Unix socket (`/var/run/docker.sock`).
- Completely CGO-free, requiring no 3rd-party Docker Go SDK.
- Controls resource limits via `HostConfig` (`Memory` in bytes, `NanoCPUs` for CPU quota).
- Demultiplexes standard 8-byte framed Docker stdout/stderr streams.
- Connects containers to bridge networks with `ExtraHosts: ["host.docker.internal:host-gateway"]`, dynamically pointing container `HTTP_PROXY` to the host filtering proxy.

---

## 3. Security Model & Isolation Boundaries

### 3.1 DM-Only Tool Scoping & Public Chat Defense
- The tools `sandbox_request`, `sandbox_exec`, and `sandbox_destroy` are registered exclusively for direct message contexts via `ToolDefinitionsForSession(session ChatSessionContext)`.
- If invoked from a Townhall public chat, the registry rejects execution immediately with an explicit error: `"sandbox tools are strictly available in 1-on-1 direct messages"`.

### 3.2 Interactive Approval Protocol
- Creation requests initiated by `sandbox_request` register a `pending` sandbox in `sandbox.Manager`.
- Bob posts a structured permission request card into the user's DM detailing:
  - Driver type (e.g. `bwrap`, `docker`)
  - Network policy (mode, allowed domains)
  - Requested mounts and permissions
  - Sandbox duration / expiry time
  - Reason / intended goal
- The user must explicitly approve by replying `/sandbox approve` or reject by replying `/sandbox deny`.
- The Gateway intercepts approval commands before message ingestion into LLM context, strips any rich-text formatting (e.g. `<p>/sandbox approve</p>`), activates the sandbox, and posts a confirmation status card.

### 3.3 Kernel-Enforced Network Airgap
- In `NetworkNone` and `NetworkRestricted` modes:
  - The network namespace is completely unshared (`--unshare-all`).
  - No physical or virtual network interfaces exist in the container except loopback (`lo`).
  - Raw TCP/UDP socket attempts to external internet or local network IPs fail instantly at the kernel level with `ENETUNREACH: Network is unreachable`.
  - Scripts cannot bypass network restrictions by unsetting `HTTP_PROXY` or creating custom network sockets.

### 3.4 In-Process Filtering Proxy (`FilteringProxy`)
- **DNS Rebinding & TOCTOU Prevention:**
  - Performs a single DNS resolution (`LookupIP`) during validation.
  - Verifies all resolved IP addresses against blacklisted CIDRs (`127.0.0.0/8`, `169.254.0.0/16`, `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`, `fe80::/10`).
  - Pins the verified IP and directly connects using `net.JoinHostPort(targetIP.String(), port)`.
  - Plaintext HTTP requests dial the pinned IP via custom `http.Transport.DialContext` while maintaining the original `Host` header for HTTP virtual host routing.
- **Hop-by-Hop Header Sanitization:**
  - Strips RFC 7230 / RFC 9110 hop-by-hop headers (`Connection`, `Keep-Alive`, `Proxy-Authenticate`, `Proxy-Authorization`, `Proxy-Connection`, `TE`, `Trailer`, `Trailers`, `Transfer-Encoding`, `Upgrade`) from both outbound client requests and inbound server responses to prevent HTTP smuggling.
- **Client IP Authorization:**
  - Verifies that incoming connections originate strictly from loopback or authorized private Docker bridge subnets, returning `403 Forbidden` to external client IPs.

### 3.5 Dynamic Workspace Path Resolution
- User workspaces are dynamically located at `<DATA_DIR>/sandboxes/<userID>`.
- Path traversal verification (`ValidateMountPath`) ensures that all mount points resolve strictly within the user's workspace boundaries, preventing directory traversal vulnerabilities (`../`).

---

## 4. Configuration Specification (`internal/config`)

| Environment Variable | Type | Default | Description |
|---|---|---|---|
| `SANDBOX_ENABLED` | bool | `true` | Globally enables or disables sandbox execution tools |
| `SANDBOX_DRIVERS` | []string | `bwrap,docker` | Comma-separated list of prioritized drivers |
| `SANDBOX_MAX_LIFETIME` | duration | `30m` | Maximum permitted sandbox lifespan |
| `SANDBOX_DEFAULT_EXEC_TIMEOUT` | duration | `1m` | Default per-command execution timeout |
| `SANDBOX_MAX_EXEC_TIMEOUT` | duration | `10m` | Hard upper bound on command execution duration |
| `SANDBOX_DOCKER_SOCKET` | string | `/var/run/docker.sock` | Path to host Docker daemon socket |
| `SANDBOX_DOCKER_ALLOWED_IMAGES`| []string | `alpine:latest` | Whitelist of permitted Docker images |
| `SANDBOX_DOCKER_CPU_LIMIT` | float64 | `1.0` | Maximum CPU cores allocated to Docker sandboxes |
| `SANDBOX_DOCKER_MEMORY_LIMIT_MB`| int | `512` | Memory limit in MB allocated to Docker sandboxes |

---

## 5. Non-Functional Requirements
- **Zero CGO:** Built with 100% pure Go standard library and existing dependencies.
- **Concurrency Safety:** All state operations protected with read/write and mutual exclusion locks; background cleaner routine evicts expired sandboxes safely.
- **Resource Discipline:** Clean process teardown using context timeouts, `--die-with-parent` in bubblewrap, and container force-delete on destroy.
- **Audit Logging:** Structured logging (`slog`) of every sandbox lifecycle event, proxy access attempt, approval, denial, and denied network access.

---

## 6. Acceptance Criteria
- Unit tests for `Manager` lifecycle, timeouts, expiration cleanup, and mount validation.
- Unit tests for `FilteringProxy` covering HTTP proxying, CONNECT tunneling, domain whitelisting, CIDR blacklisting, hop-by-hop header removal, and DNS rebinding prevention.
- Unit tests for `bwrap.Driver` verifying basic command execution, filesystem isolation, workspace persistence, and kernel-level network airgap rejection.
- Mock server unit tests for `docker.Driver` verifying container creation payload, ExtraHosts injection, stream demultiplexing, and container destruction.
- Unit tests for gateway command sanitization (`/sandbox approve`, `/sandbox deny` with HTML/backticks).
- Full repository test pass with `go test -race ./...`, `go vet ./...`, and `osv-scanner scan -r .` with 0 errors.
