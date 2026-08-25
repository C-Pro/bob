# Bob — Besedka AI Agent

Bob is an AI agent for the [Besedka](https://github.com/c-pro/besedka) self-hosted chat platform. It connects over WebSocket, listens for mentions (`@bot`) in the public Townhall chat and private DMs, maintains multi-turn conversation context, and generates responses using any OpenAI-compatible LLM API (defaults to Google Gemini).

## Features

- **WebSocket ingress** with automatic reconnect and ping keepalive
- **Multi-turn context & chunked eviction** — per-chat in-memory ring buffers with historical backfill and stepped batch eviction (pruning 1/3 of the buffer on overflow) to keep the prompt prefix static and maximize LLM prefix caching
- **Multimodal image & file attachments** — automatic ingestion and base64 encoding of image thumbnails via OpenAI `image_url` payloads, plus inlined markdown code blocks for text-based code/config attachments
- **Live web search** via [Tavily](https://tavily.com/) (when `TAVILY_API_KEY` is set)
- **Web page fetch & extraction** with readability parsing and dynamic rendering fallback
- **GEOIP location reporting** — round-robin across public providers
- **OpenAI-compatible LLM client** with exponential retry backoff and tool/function calling

## Quick Start

### Prerequisites

- Go 1.24+
- A running [Besedka](https://github.com/c-pro/besedka) server
- An API key for an OpenAI-compatible LLM provider

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
docker run --env-file .env bob:latest
```

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
  llm/              — OpenAI-compatible LLM client
  models/           — shared data types
  prompt/           — system prompt rendering
  tools/            — tool registry, Tavily search, web fetch
```

## License

[MIT](LICENSE)
