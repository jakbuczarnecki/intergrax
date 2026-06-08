# Intergrax Infrastructure Layer

Local Docker services for development, integration tests, and CI. **Decoupled from** `intergrax/` runtime code.

**Port matrix:** [PORTS.md](PORTS.md)

---

## Layout

```text
infra/
  PORTS.md                    # Host ports + conflict resolutions
  README.md
  docker/                     # Per-service compose (manage.ps1 / manage.sh)
    redis/, postgresql/, kafka/, …
    docling/                  # Custom image (Dockerfile)
  integration/                # Unified profile-based stack
    docker-compose.yml
    manage.sh / manage.ps1
    initdb/                   # Extra Postgres DBs (langfuse, temporal)
    prometheus/prometheus.yml
    .env.example
```

---

## Quick start (recommended — unified stack)

```bash
cd infra/integration

# Default: core + queue + rag + data + secrets
./manage.sh start

# Full platform (all profiles)
./manage.sh start all

# RAG only
./manage.sh start rag

# Build Docling image first (rag profile)
./manage.sh build rag
./manage.sh start rag
```

Windows:

```powershell
cd infra\integration
.\manage.ps1 start
.\manage.ps1 start all
```

---

## Compose profiles

| Profile | Services |
|---------|----------|
| `core` | redis, postgresql |
| `queue` | kafka, rabbitmq, nats |
| `rag` | qdrant, chroma, weaviate, neo4j, milvus, ollama, docling |
| `data` | mongodb, mysql, cassandra, minio, memcached |
| `secrets` | vault |
| `observability` | elasticsearch, prometheus, clickhouse, langfuse, phoenix, mailpit |
| `cloud` | localstack, azurite, pubsub-emulator |
| `heavy` | temporal, vespa, selenium |
| `all` | Enables every profile (alias) |

**Default profile set:** `core` + `queue` + `rag` + `data` + `secrets`.

---

## Per-tool management

Start a **single** service (same images/ports as unified stack):

```bash
cd infra/docker
./manage.sh redis start
./manage.sh neo4j start
./manage.ps1 weaviate start
```

Docling (custom build):

```bash
./manage.sh docling build
./manage.sh docling start
```

Health: Docling `http://localhost:8081/health`

---

## Environment defaults

Copy `infra/integration/.env.example` to `infra/integration/.env` to override credentials.

| Service | Default credentials |
|---------|---------------------|
| PostgreSQL | `intergrax` / `intergrax`, DB `intergrax` |
| RabbitMQ | `intergrax` / `intergrax` |
| MinIO | `intergrax` / `intergrax` |
| Neo4j | `neo4j` / `intergrax` |
| Vault dev | token `intergrax-dev-token` |

---

## Integration test mapping

| Test area | Start profile |
|-----------|---------------|
| `tests/integration/distributed/*` (Redis) | `core` |
| `tests/integration/queueing/*` (Kafka, RabbitMQ) | `queue` |
| `tests/integration/rag/vectorstore/*` (Qdrant, Chroma) | `rag` |
| `tests/integration/rag/embedding/test_ollama*` | `rag` (ollama) |
| GraphRAG Neo4j | `rag` |
| Harness conformance (Mongo, Cassandra, MinIO, Vault) | `data`, `secrets` |
| Tools observability (ES, Prometheus) | `observability` |
| AWS/Azure/GCP emulators | `cloud` |

---

## CI usage

```bash
./infra/integration/manage.sh start default
# run pytest tests/integration/ ...
./infra/integration/manage.sh stop default
```

---

## Design principles

- **Profile-based** — avoid running 30+ containers when only Redis is needed.
- **Single port matrix** — documented in [PORTS.md](PORTS.md); conflicts resolved (ClickHouse native → host `9002`, Vespa → `8089`).
- **Tool isolation** — each service under `infra/docker/<name>/`.
- **Optional build** — only Docling uses `Dockerfile`; `manage build` targets Docling in unified stack.

---

## Related documentation

- [docs/architecture/INTEGRATIONS.md](../docs/architecture/INTEGRATIONS.md) — provider slugs and `INTERGRAX_*` env vars
- [docs/intergrax_runtime_architecture.md](../docs/intergrax_runtime_architecture.md) §7.1.2, §33 — RAG/LLM metrics and observability backends (Prometheus/Langfuse/Phoenix)
- [intergrax/integrations/providers/*/USAGE.md](../intergrax/integrations/providers/) — per-provider connection examples
