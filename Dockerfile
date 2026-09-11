# Build stage
FROM golang:1.27-alpine AS builder

WORKDIR /app

# Create a non-root user and data directory
RUN adduser -D -u 10001 -g '' appuser && \
    mkdir -p /data && \
    chown -R appuser:appuser /data

# Copy source code (includes vendor directory)
COPY . .

# Build the agent application with SIMD enabled
ARG TARGETOS
ARG TARGETARCH
ENV GOEXPERIMENT=simd
RUN CGO_ENABLED=0 GOOS=${TARGETOS} GOARCH=${TARGETARCH} go build -mod=vendor -ldflags="-w -s" -o agent ./cmd/agent
RUN mkdir -p /data/bin && CGO_ENABLED=0 GOOS=linux GOARCH=${TARGETARCH} go build -mod=vendor -ldflags="-w -s" -o /data/bin/bob-proxy-fwd ./cmd/bob-proxy-fwd && chmod 0755 /data/bin/bob-proxy-fwd && chown -R appuser:appuser /data

# Final stage
FROM scratch

# Copy SSL certificates
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/

# Copy the user from the builder stage
COPY --from=builder /etc/passwd /etc/passwd

# Copy data directory owned by appuser
COPY --from=builder --chown=appuser:appuser /data /data

# Copy the binary
COPY --from=builder /app/agent /agent

# Use the non-root user
USER appuser

# Run the binary
ENTRYPOINT ["/agent"]
