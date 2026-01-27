# jb-embed

Local embedding service powered by [jumpboot](https://github.com/richinsley/jumpboot) and [sentence-transformers](https://www.sbert.net/).

Single Go binary that bootstraps its own Python environment and serves embeddings via CLI or HTTP.

## Installation

```bash
go build -o jb-embed .
cp jb-embed ~/bin/  # or wherever you like
```

First run will create a Python environment at `~/.jb-embed/envs/` and install sentence-transformers + torch.

## Usage

### CLI Mode

```bash
# Embed text, output JSON
jb-embed "Hello world"

# Batch mode (stdin → stdout)
echo -e "line one\nline two" | jb-embed batch
```

### Server Mode

```bash
# Start HTTP server (default port 8420)
jb-embed serve

# Custom port
jb-embed serve --port 9000
```

### HTTP API

```bash
# Single text
curl -X POST localhost:8420/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "hello world"}'

# Batch
curl -X POST localhost:8420/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["hello", "world", "foo"]}'

# Health check
curl localhost:8420/health

# Switch model
curl -X POST localhost:8420/model \
  -H "Content-Type: application/json" \
  -d '{"model": "all-mpnet-base-v2"}'
```

## Options

| Flag | Description | Default |
|------|-------------|---------|
| `--model` | Embedding model | `all-MiniLM-L6-v2` |
| `--port` | Server port | `8420` |
| `--env` | Environment path | `~/.jb-embed/envs` |
| `--python` | Python version | `3.11` |

## Models

Default model (`all-MiniLM-L6-v2`) is fast and produces 384-dim vectors. For higher quality:

- `all-mpnet-base-v2` — 768 dims, better quality
- `multi-qa-MiniLM-L6-cos-v1` — optimized for semantic search
- Any model from [SBERT models](https://www.sbert.net/docs/pretrained_models.html)

## Network Service Design

Built with LAN deployment in mind:
- `/health` endpoint for load balancers
- Request counting and uptime metrics
- Model hot-swap without restart
- Stateless — scale horizontally

Future: auth tokens, TLS, Prometheus metrics.
