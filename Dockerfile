# Build stage
FROM golang:1.24-alpine AS builder

WORKDIR /app

# Create a non-root user
RUN adduser -D -g '' appuser

# Copy source code (includes vendor directory)
COPY . .

# Build the agent application
ARG TARGETOS
ARG TARGETARCH
RUN CGO_ENABLED=0 GOOS=${TARGETOS} GOARCH=${TARGETARCH} go build -mod=vendor -ldflags="-w -s" -o agent ./cmd/agent

# Final stage
FROM scratch

# Copy SSL certificates
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/

# Copy the user from the builder stage
COPY --from=builder /etc/passwd /etc/passwd

# Copy the binary
COPY --from=builder /app/agent /agent

# Use the non-root user
USER appuser

# Run the binary
ENTRYPOINT ["/agent"]
