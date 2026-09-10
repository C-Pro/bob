# Bob — Besedka AI Agent

Bob is an AI agent for the [Besedka](https://github.com/c-pro/besedka) self-hosted chat platform. It connects over WebSocket, listens for mentions (`@bot`) in the public Townhall chat and private DMs, maintains multi-turn conversation context, and generates responses using any OpenAI-compatible LLM API (defaults to Google Gemini).

## Features

- **WebSocket ingress** with automatic reconnect and ping keepalive
- **Multi-turn context & chunked eviction** — per-chat in-memory ring buffers with historical backfill and stepped batch eviction (pruning 1/3 of the buffer on overflow) to keep the prompt prefix static and maximize LLM prefix caching
- **Long-term isolated memory & RAG (`recall_memory`)** — persistent SQLite vector and FTS5 memory per chat with watermark tracking, background indexing upon ring buffer eviction, startup sequence catch-up, and strict privacy isolation (Townhall searches townhall memory; DMs search own DM + townhall memory)
- **Multimodal image & file attachments** — automatic ingestion and base64 encoding of image thumbnails via OpenAI `image_url` payloads, plus inlined markdown code blocks for text-based code/config attachments
- **Isolated sandbox execution (`bash_exec`, `file_manage`)** — secure tool-assisted command execution and per-user workspace file management with multi-driver support (Bubblewrap `bwrap` or Docker `docker`), fine-grained submounts, resource limits (CPU, memory, timeouts), and restricted network modes with an egress filtering proxy
- **Automated S3 / object storage backups** — periodic snapshots of SQLite databases uploaded to S3-compatible storage (AWS S3, MinIO, Cloudflare R2, GCS) with retention management and optional database encryption
- **Live web search** via [Tavily](https://tavily.com/) (when `TAVILY_API_KEY` is set)
- **Web page fetch & extraction** with readability parsing and dynamic rendering fallback
- **SQLite storage & schema migrations** — pure Go embedded SQLite database with automatic single-step version migrations
- **GEOIP location reporting** — round-robin across public providers
- **OpenAI-compatible LLM client** with exponential retry backoff and tool/function calling

## Quick Start

### Prerequisites

- Go 1.27+
- A running [Besedka](https://github.com/c-pro/besedka) server
- An API key for an OpenAI-compatible LLM provider
- A writable directory on the host/container filesystem for SQLite database files (`./data` by default)
- *(Optional for Sandboxing)* `bwrap` (Bubblewrap) on Linux hosts, or access to the Docker daemon (`/var/run/docker.sock`)

### Configuration

Create a `.env` file (gitignored) or export environment variables:

```sh
BESEDKA_URL=http://127.0.0.1:8080
BESEDKA_API_KEY=<your-besedka-api-key>
OPENAI_API_KEY=<your-llm-api-key>
OPENAI_MODEL=<preferred model name>
OPENAI_BASE_URL=<openai-compatible endpoint url>
BOT_HANDLE=@botname
```

#### Core & Context Options

| Variable | Default | Description |
| --- | --- | --- |
| `BESEDKA_URL` / `BASE_URL` | `http://127.0.0.1:8080` | URL of the Besedka chat server |
| `BESEDKA_API_KEY` | *(required)* | Bot account API key or session token |
| `BOT_HANDLE` | `@bot` | Bot username mention handle |
| `OPENAI_API_KEY` / `GEMINI_API_KEY` | *(required)* | LLM provider API key |
| `OPENAI_MODEL` / `GEMINI_MODEL` | `gemini-3.7-flash` | Model identifier |
| `OPENAI_BASE_URL` / `GEMINI_BASE_URL` | Google Gemini OpenAI endpoint | Base URL for OpenAI-compatible API |
| `MSG_RING_BUFFER_SIZE` | `100` | In-memory message ring buffer capacity per chat |
| `TOWNHALL_MAX_PARAGRAPHS` | `2` | Maximum paragraph length for Townhall responses |
| `DM_MAX_PARAGRAPHS` | `10` | Maximum paragraph length for Direct Message responses |

#### Storage & Long-Term Memory

| Variable | Default | Description |
| --- | --- | --- |
| `DATA_DIR` | `./data` | Directory for SQLite databases, user workspaces, and models |
| `EMBEDDING_MODEL` | `""` *(local MiniLM-L12-v2)* | HuggingFace embedding model ID or empty for local embedding |
| `EMBEDDING_PRECISION` | `bf16` | Precision mode for local embeddings (`bf16`, `fp32`, `int8`) |
| `SECRET` / `AUTH_SECRET` | `""` | Secret key used for database backup encryption |

#### Web Tools

| Variable | Default | Description |
| --- | --- | --- |
| `TAVILY_API_KEY` | `""` | Tavily API key; enables the `tavily_search` web search tool |
| `TAVILY_BASE_URL` | `https://api.tavily.com` | Base URL for Tavily search API requests |

#### S3-Compatible Automated Backups

Bob can automatically create and upload snapshots of chat SQLite databases to S3-compatible object storage:

| Variable | Default | Description |
| --- | --- | --- |
| `S3_ENDPOINT` | `""` | Object storage endpoint URL (e.g. AWS S3, MinIO, or Cloudflare R2) |
| `S3_REGION` | `us-east-1` | S3 region |
| `S3_BUCKET` | `""` | Target S3 bucket name (backups disabled if empty) |
| `S3_ACCESS_KEY` | `""` | Object storage access key ID |
| `S3_SECRET_KEY` | `""` | Object storage secret access key |
| `S3_PATH_STYLE` | `true` | Use path-style S3 addressing (`true` or `false`) |
| `S3_BACKUP_INTERVAL` | `1h` | Duration between automatic backup snapshots (e.g. `30m`, `1h`, `24h`) |
| `S3_BACKUP_KEEP` | `7` | Number of most recent backups to retain in storage |
| `S3_BACKUP_PREFIX` | `bob_agent/` | S3 key prefix for uploaded backup archives |

#### Sandbox Execution (`bash_exec`, `file_manage`)

Bob supports running bash commands and editing files in isolated sandboxes:

| Variable | Default | Description |
| --- | --- | --- |
| `SANDBOX_ENABLED` | `true` | Enable or disable sandbox execution tools |
| `SANDBOX_DRIVERS` | `bwrap,docker` | Comma-separated list of drivers to try in order (`bwrap`, `docker`) |
| `SANDBOX_DOCKER_SOCKET` | `/var/run/docker.sock` | Path to Docker daemon Unix socket (for `docker` driver) |
| `SANDBOX_HOST_DATA_DIR` | `""` | Host path corresponding to `DATA_DIR` when running Bob inside Docker |
| `SANDBOX_ALLOWED_IMAGES` | `alpine:latest,golang:alpine,python:3.11-slim,node:20-slim` | Comma-separated list of container images allowed for execution |
| `SANDBOX_ALLOWED_NETWORK_MODES` | `none,restricted` | Allowed network modes (`none`, `restricted`, `full`) |
| `SANDBOX_MAX_LIFETIME` | `30m` | Inactivity lifetime before an idle sandbox is destroyed |
| `SANDBOX_DEFAULT_EXEC_TIMEOUT` | `1m` | Default timeout for command execution |
| `SANDBOX_MAX_EXEC_TIMEOUT` | `10m` | Maximum allowed timeout for command execution |
| `SANDBOX_CPU_LIMIT` | `1.0` | Maximum CPU cores allocated to a sandbox |
| `SANDBOX_MEMORY_LIMIT_MB` | `512` | Memory limit per sandbox in megabytes |

### Run

```sh
# Run agent service
go run ./cmd/agent

# Regenerate all vector embeddings across all chat databases (e.g. after changing embedding models)
go run ./cmd/agent -regenerate-vectors
```

### Docker

```sh
docker build -t bob:latest .
docker run --env-file .env -v $(pwd)/data:/data bob:latest
```

#### Running Docker Sandbox from inside Docker

When running Bob inside Docker and using the `docker` sandbox driver (sibling container pattern):

1. Mount the host Docker socket: `-v /var/run/docker.sock:/var/run/docker.sock`
2. Ensure the container process has permission to access the socket (e.g. `--group-add $(stat -c '%g' /var/run/docker.sock)` or matching the host docker group GID)
3. Set `SANDBOX_HOST_DATA_DIR` to the path of your host data directory (e.g. `/path/to/host/data`) so user workspace volumes mounted into sandbox sibling containers resolve correctly on the host.

> **Note on Storage Permissions:** Bob requires a writable data directory (configured via `DATA_DIR`, defaulting to `./data`, or `/data` inside the container). When mounting a volume in containerized environments, ensure the mount point is writable by `appuser` (UID `10001`).

## Development

```sh
# Run tests
make test

# Or full check (lint + test + semgrep + osv-scanner)
make check
```

## Project Structure

```
cmd/agent/          — entrypoint
internal/
  backup/           — automated S3/object storage backups
  chatcontext/      — per-chat ring buffer context management
  config/           — environment-based configuration
  gateway/          — WebSocket/REST gateway to Besedka
  geoip/            — server location lookup
  llm/              — OpenAI-compatible LLM & embeddings client
  memory/           — isolated SQLite vector + FTS5 long-term memory store
  models/           — shared data types
  objectstore/      — S3-compatible client wrapper
  prompt/           — system prompt rendering
  sandbox/          — secure code execution isolation (bwrap, docker, network proxy)
  store/            — SQLite database storage & single-step migrations
  tools/            — tool registry, memory recall, Tavily search, web fetch
```

## License

[MIT](LICENSE)
