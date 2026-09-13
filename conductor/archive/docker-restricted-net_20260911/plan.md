# Implementation Plan: Restricted Docker Network Isolation & Injected Proxy Forwarder

## Phase 1: Forwarder Binary (`bob-proxy-fwd`) & Build Pipeline
- [x] Task: Implement `bob-proxy-fwd` in `cmd/bob-proxy-fwd/main.go`
    - [x] Implement bidirectional TCP-to-Unix socket forwarding with error handling and buffer pooling
    - [x] Implement `-daemon` mode with graceful `SIGTERM`/`SIGINT` handling
    - [x] Implement wrapper mode (`-- <cmd...>`) with process execution and exit code forwarding
    - [x] Write unit tests in `cmd/bob-proxy-fwd/main_test.go` covering daemon and wrapper forwarding modes
- [x] Task: Build Pipeline & Makefile Automation
    - [x] Add `build-fwd` target in `Makefile` compiling `cmd/bob-proxy-fwd` with `CGO_ENABLED=0 GOOS=linux GOARCH=$(go env GOARCH)`
    - [x] Integrate forwarder compilation into `make check`, `make test`, and `Dockerfile`
    - [x] Update `.gitignore` to ignore the compiled forwarder binary

## Phase 2: Forwarder Manager & Extraction Logic
- [x] Task: Implement Forwarder Manager in `internal/sandbox/forwarder.go`
    - [x] Implement `EnsureForwarderBinary(dataDir string)` resolving binary location:
        1. Check `SANDBOX_PROXY_FWD_PATH` environment variable override
        2. Check target path `<dataDir>/bin/bob-proxy-fwd`
        3. If missing, compile on demand using host `go build` toolchain in dev/test environment
    - [x] Set `0755` permissions on the binary
    - [x] Write unit tests in `internal/sandbox/forwarder_test.go`

## Phase 3: Docker Driver Network Isolation
- [x] Task: Refactor Docker Restricted Network Mode in `internal/sandbox/docker/driver.go`
    - [x] Update proxy creation to use Unix domain socket under `<dataDir>/sandboxes/<user>/.proxy/proxy.sock` (no host TCP bind)
    - [x] Configure `HostConfig["NetworkMode"] = "none"` for `sandbox.NetworkRestricted`
    - [x] Mount forwarder binary into container at `/run/proxy/fwd:ro` (translated via `toHostPath`)
    - [x] Mount proxy socket into container at `/run/proxy:rw` (translated via `toHostPath`)
    - [x] Set container `Cmd`/`Entrypoint` to `/run/proxy/fwd -daemon -tcp 127.0.0.1:18080 -sock /run/proxy/proxy.sock`
    - [x] Remove `ExtraHosts` (`host.docker.internal`) and bridge IP lookup logic
    - [x] In `driver.Exec`, inject `http_proxy=http://127.0.0.1:18080`, `https_proxy=...`, `all_proxy=...`
- [x] Task: Unit & Mock Integration Tests for Docker Driver
    - [x] Update `internal/sandbox/docker/driver_test.go`
    - [x] Verify `NetworkMode: "none"` in container create payload
    - [x] Verify bind mounts for `/run/proxy/fwd` and `/run/proxy`
    - [x] Verify container start command and exec environment

## Phase 4: Bubblewrap Driver Unification
- [x] Task: Update Bubblewrap Driver to Use `bob-proxy-fwd`
    - [x] Update `internal/sandbox/bwrap/driver.go` to mount `bob-proxy-fwd` at `/run/proxy/fwd:ro`
    - [x] Replace `socat`/`python3` shell forwarder script with `/run/proxy/fwd -tcp 127.0.0.1:18080 -sock /run/proxy.sock -- <cmd...>`
    - [x] Update `internal/sandbox/bwrap/driver_test.go` to test execution with the unified forwarder

## Phase 5: Verification & Quality Assurance
- [x] Task: End-to-End Airgap and Proxy Verification Tests
    - [x] Add test verifying that in `NetworkRestricted` mode, direct WAN/LAN connections fail immediately with `ENETUNREACH`
    - [x] Add test verifying HTTP/HTTPS proxying via loopback forwarder succeeds for allowed domains and fails for blocked domains
- [x] Task: Run Quality and Security Checks
    - [x] Run `make check` (`lint-go`, `test-go -race`, `semgrep`, `osv-scanner`) and ensure 0 errors
