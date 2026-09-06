.PHONY: all check test test-go lint-go semgrep osv-scanner docker-build

all: check

check: lint-go test-go semgrep osv-scanner

lint-go:
	docker run --rm \
		-v $(PWD):/app \
		-v $(shell go env GOCACHE):/root/.cache/go-build \
		-v $(shell go env GOMODCACHE):/go/pkg/mod \
		-v $(HOME)/.cache/golangci-lint:/root/.cache/golangci-lint \
		-w /app \
		-e GOFLAGS="-mod=vendor" \
		-e GOEXPERIMENT="simd" \
		golangci/golangci-lint:latest \
		golangci-lint run

test: test-go

test-go:
	GOEXPERIMENT=simd go test -v -covermode=atomic -coverprofile=coverage.out -race ./...

semgrep:
	docker run --rm -v $(PWD):/src returntocorp/semgrep:1.106.0 semgrep scan --config=p/default

osv-scanner:
	docker run --rm -e GOTOOLCHAIN=auto -v $(PWD):/src -w /src ghcr.io/google/osv-scanner:latest scan source --no-call-analysis=go -r .

test-backup-integration:
	go test -v -tags integration ./internal/backup/...

docker-build:
	docker build -t bob:latest .
