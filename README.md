# Bob — Besedka AI Agent

Bob is an AI agent for the [Besedka](https://github.com/c-pro/besedka) self-hosted chat platform. It connects over WebSocket, listens for mentions (`@bot`) in the public Townhall chat and private DMs, maintains multi-turn conversation context, and generates responses using any OpenAI-compatible LLM API (defaults to Google Gemini).

## Features

- **WebSocket ingress** with automatic reconnect and ping keepalive
- **Multi-turn context & chunked eviction** — per-chat in-memory ring buffers with historical backfill and stepped batch eviction (pruning 1/3 of the buffer on overflow) to keep the prompt prefix static and maximize LLM prefix caching
- **Long-term isolated memory & RAG (`recall_memory`)** — persistent SQLite vector and FTS5 memory per chat with watermark tracking, background indexing upon ring buffer eviction, startup sequence catch-up, and strict privacy isolation (Townhall searches townhall memory; DMs search own DM + townhall memory)
- **Multimodal image & file attachments** — automatic ingestion and base64 encoding of image thumbnails via OpenAI `image_url` payloads, plus inlined markdown code blocks for text-based code/config attachments
- **Live web search** via [Tavily](https://tavily.com/) (when `TAVILY_API_KEY` is set)
- **Web page fetch & extraction** with readability parsing and dynamic rendering fallback
- **SQLite storage & schema migrations** — pure Go embedded SQLite database with automatic single-step version migrations
- **GEOIP location reporting** — round-robin across public providers
- **OpenAI-compatible LLM client** with exponential retry backoff and tool/function calling

## Quick Start

### Prerequisites

- Go 1.24+
- A running [Besedka](https://github.com/c-pro/besedka) server
- An API key for an OpenAI-compatible LLM provider
- A writable directory on the host/container filesystem for SQLite database files (`./data` by default)

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

Optional:

```sh
DATA_DIR=./data                           # directory for SQLite databases (must be writable)
EMBEDDING_MODEL=text-embedding-004        # embedding model for vector semantic search (defaults to FTS5 lexical search if unset)
TAVILY_API_KEY=<your-tavily-key>          # enables web search tool
MSG_RING_BUFFER_SIZE=100                  # context window size per chat
TOWNHALL_MAX_PARAGRAPHS=2                 # response length limit in Townhall
DM_MAX_PARAGRAPHS=10                      # response length limit in DMs
```

### Run

```sh
go run ./cmd/agent
```

### Docker

```sh
docker build -t bob:latest .
docker run --env-file .env -v $(pwd)/data:/data bob:latest
```

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
  chatcontext/      — per-chat ring buffer context management
  config/           — environment-based configuration
  gateway/          — WebSocket/REST gateway to Besedka
  geoip/            — server location lookup
  llm/              — OpenAI-compatible LLM & embeddings client
  memory/           — isolated SQLite vector + FTS5 long-term memory store
  models/           — shared data types
  prompt/           — system prompt rendering
  store/            — SQLite database storage & single-step migrations
  tools/            — tool registry, memory recall, Tavily search, web fetch
```

## License

[MIT](LICENSE)
