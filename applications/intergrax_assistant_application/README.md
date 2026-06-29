# Intergrax Assistant Application (IAA)

Harness-native conversational lab — hub agent, swappable LLM (local Ollama default), optional platform specialist delegation.

**Architecture:** [`ARCHITECTURE.md`](docs/ARCHITECTURE.md) · **ADR:** [`adr/ADR-INTERGRAX_ASSISTANT-001.md`](docs/adr/ADR-INTERGRAX_ASSISTANT-001.md)

## Quick start

```bash
uv run pytest applications/intergrax_assistant_application/tests -q
cp applications/intergrax_assistant_application/.env.example applications/intergrax_assistant_application/.env
uv run uvicorn intergrax_assistant_application.host.main:app --host 127.0.0.1 --port 8096
```

## Smoke

```bash
curl -s http://127.0.0.1:8096/v1/intergrax_assistant/agents
curl -s -X POST http://127.0.0.1:8096/v1/intergrax_assistant/run \
  -H "Content-Type: application/json" \
  -d '{"message":"hello","capability":"platform.assist"}'
```

## LLM swap

```bash
# Local (default)
INTERGRAX_LLM_PROVIDER=ollama
INTERGRAX_LLM_MODEL=llama3.1:latest

# Cloud
INTERGRAX_LLM_PROVIDER=openai
INTERGRAX_LLM_MODEL=gpt-4o-mini
```

See [`docs/architecture/LLM_ADAPTERS.md`](../../docs/architecture/LLM_ADAPTERS.md).

## Optional specialists

```bash
INTERGRAX_ASSISTANT_INCLUDE_LEGAL=true
INTERGRAX_ASSISTANT_INCLUDE_RESEARCH=true
```

Nexus delegates to mounted agents — see [`ARCHITECTURE.md` §3](docs/ARCHITECTURE.md).
