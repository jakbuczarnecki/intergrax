# Intergrax LLM Adapters

**Last updated:** 2026-05-30

The **LLM adapter layer** (`intergrax/llm_adapters/`) is Intergrax’s Tier-0 module for calling large language model providers through one runtime contract. Agents, `ChatAgent`, `ToolsAgent`, and Nexus runtime components depend on `LLMAdapter` — not vendor SDKs directly.

**Related docs:**

| Document | Purpose |
|----------|---------|
| [intergrax_runtime_architecture.md](intergrax_runtime_architecture.md) §5.2.2 | Canon — LLM adapters are **outside** the Integration Library |
| [TOOLS.md](TOOLS.md) | Agent tools; native tool-calling uses `generate_with_tools` |
| [AGENT_CREATION_GUIDE.md](AGENT_CREATION_GUIDE.md) | Wire `LLMAdapterRegistry.create()` in Tier-3 factories |
| [INTEGRATIONS.md](INTEGRATIONS.md) | External backends (DB, queues, …) — **not** LLM providers |
| [INTERGRAX_IMPLEMENTATION_PLAN.md](INTERGRAX_IMPLEMENTATION_PLAN.md) | Phase **M-LLM** status |

---

## Design principles

| Principle | What it means |
|-----------|---------------|
| **Single contract** | All providers implement `LLMAdapter`: chat, streaming, optional tools, structured output. |
| **Registry, not integrations** | Use `LLMAdapterRegistry` — not `IntegrationRegistry`. |
| **Lazy loading** | Provider modules load on first `create(provider)`. |
| **Shared mapping** | `_shared/` normalizes `ChatMessage` → provider payloads (including tool rounds). |
| **Retries** | `LLMCallConfig` + `_execute()` for transient API errors (all shipped adapters). |
| **Usage tracking** | Per-`run_id` token and latency stats for `LLMUsageTracker`. |
| **CI gate** | `tests/unit/llm_adapters/` included in GitHub Actions regression gate. |

---

## Supported providers (9)

| Provider key | Class | SDK / API | Native tools | Streaming | Structured |
|--------------|-------|-----------|--------------|-----------|------------|
| `openai` | `OpenAIChatResponsesAdapter` | OpenAI Responses | ✅ | ✅ text + tools stream | ✅ JSON schema |
| `claude` | `ClaudeChatAdapter` | Anthropic Messages | ✅ | ✅ | ✅ prompt + schema |
| `azure_openai` | `AzureOpenAIChatAdapter` | Azure Chat Completions | ✅ | ✅ tools stream | ✅ json_schema |
| `gemini` | `GeminiChatAdapter` | google-genai | ✅ | ✅ tools stream | ✅ prompt + schema |
| `mistral` | `MistralChatAdapter` | mistralai | ✅ | ✅ tools stream | ✅ prompt + schema |
| `aws_bedrock` | `BedrockChatAdapter` | boto3 Invoke + **Converse** | ✅ Converse / Anthropic | ✅ | — |
| `ollama` | `LangChainOllamaAdapter` | langchain-ollama | ❌ JSON planner | ✅ + fallback | ✅ prompt + schema |
| `groq` | `GroqChatAdapter` | OpenAI-compatible (Groq) | ✅ | ✅ tools stream | ✅ json_schema |
| `vllm` | `VllmChatAdapter` | OpenAI-compatible (vLLM server) | ✅ | ✅ tools stream | ✅ json_schema |

**Ollama** uses `supports_tools() == False`; `ToolsAgent` uses the JSON planner branch.

**Groq** — fast inference (Llama, Mixtral, Gemma, …) via [Groq OpenAI-compatible API](https://console.groq.com/docs/openai).

**vLLM** — self-hosted OpenAI-compatible server (on-prem, Kubernetes, GPU clusters).

---

## Architecture

```text
Tier-2/3  ChatAgent / ToolsAgent / RuntimeConfig
        │
        ▼
LLMAdapterRegistry.create("groq", model="llama-3.3-70b-versatile")
        │
        ▼
LLMAdapter (ABC)  —  call_config, _execute(retry), usage log
        │
        ├── _shared/  messages, tool_schema, conformance, bedrock_converse
        │
        ▼
Provider SDK or HTTP (openai, anthropic, google.genai, boto3, …)
```

---

## Quick start

```python
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry
from intergrax.llm.messages import ChatMessage

llm = LLMAdapterRegistry.create(
    LLMProvider.GROQ,
    model="llama-3.3-70b-versatile",
    max_retries=2,
)

answer = llm.generate_messages(
    [ChatMessage(role="user", content="Hello")],
    run_id="demo-run",
)

for chunk in llm.stream_messages(
    [ChatMessage(role="user", content="Stream this")],
    run_id="demo-run",
):
    print(chunk, end="")
```

---

## Environment variables

| Variable | Provider | Purpose |
|----------|----------|---------|
| `OPENAI_API_KEY` | OpenAI | Required |
| `INTERGRAX_DEFAULT_OPENAI_MODEL` | OpenAI | Default model |
| `ANTHROPIC_API_KEY` | Claude | Required |
| `INTERGRAX_DEFAULT_CLAUDE_MODEL` | Claude | Default model |
| `GOOGLE_API_KEY` | Gemini | Required |
| `INTERGRAX_DEFAULT_GEMINI_MODEL` | Gemini | Default model |
| `MISTRAL_API_KEY` | Mistral | Required |
| `INTERGRAX_DEFAULT_MISTRAL_MODEL` | Mistral | Default model |
| `INTERGRAX_DEFAULT_AZURE_OPENAI_*` | Azure | Endpoint, API version, deployment |
| `INTERGRAX_DEFAULT_AWS_REGION` | Bedrock | Required |
| `INTERGRAX_DEFAULT_BEDROCK_MODEL_ID` | Bedrock | Required |
| `INTERGRAX_BEDROCK_USE_CONVERSE` | Bedrock | `true` → Converse API (+ tools) |
| `INTERGRAX_DEFAULT_OLLAMA_MODEL` | Ollama | Model tag |
| `GROQ_API_KEY` | Groq | Required |
| `INTERGRAX_DEFAULT_GROQ_MODEL` | Groq | Default model |
| `INTERGRAX_DEFAULT_GROQ_BASE_URL` | Groq | Default `https://api.groq.com/openai/v1` |
| `INTERGRAX_DEFAULT_VLLM_BASE_URL` | vLLM | Default `http://127.0.0.1:8000/v1` |
| `INTERGRAX_DEFAULT_VLLM_MODEL` | vLLM | Model id on server |
| `VLLM_API_KEY` | vLLM | Often `EMPTY` for local servers |

---

## Capabilities API

```python
llm.supports_streaming()
llm.supports_tools()
llm.supports_structured_output()
llm.context_window_tokens
llm.usage.get_run_stats(run_id)
```

---

## Retries and call policy

```python
llm = LLMAdapterRegistry.create(
    "openai",
    model="gpt-4o-mini",
    max_retries=2,
    retry_backoff_sec=0.5,
)
```

All shipped adapters call provider SDKs through `LLMAdapter._execute()` when `max_retries > 0`.

---

## Bedrock

- **InvokeModel** — per-family codecs (`anthropic`, `meta`, `mistral`, `amazon`).
- **Converse API** — set `INTERGRAX_BEDROCK_USE_CONVERSE=true` for unified messages and **native tools** via `toolConfig`.
- Unknown model family → error at adapter construction.

---

## Testing and CI

```bash
uv run pytest tests/unit/llm_adapters/ -q
```

Included in **regression gate** (`.github/workflows/unit-tests.yml`):

```bash
uv run pytest tests/unit/llm_adapters/ ... -m gate -q
```

Conformance helpers: `intergrax/llm_adapters/_shared/conformance.py`.

Real API smoke tests: mark with `@pytest.mark.network` (not in PR CI).

---

## Optional install extras

```bash
uv sync --extra llm-groq
uv sync --extra llm-vllm
uv sync --extra llm-all
```

---

## Roadmap — additional providers

| Provider | Rationale | Suggested approach |
|----------|-----------|-------------------|
| **vLLM** | Self-hosted GPU inference | **Done** — `VllmChatAdapter` |
| **Groq** | Low-latency hosted Llama/Gemma | **Done** — `GroqChatAdapter` |
| **Together AI** | OpenAI-compatible hosting | Extend `OpenAIChatCompletionsAdapter` |
| **Fireworks AI** | Fast open models | OpenAI-compatible base URL |
| **Cohere** | Command R+, embed + chat | `cohere` SDK or OpenAI-compatible layer |
| **Vertex AI (Gemini)** | GCP IAM, no API key | `google-genai` with ADC / service account |
| **OpenRouter** | Multi-model gateway | Single OpenAI-compatible adapter + model string |
| **DeepSeek** | Cost-effective reasoning | OpenAI-compatible endpoint |
| **xAI (Grok)** | Grok models | OpenAI-compatible when available |
| **Azure AI Inference** | Unified Azure model catalog | Extend Azure adapter or separate slug |
| **Local llama.cpp** | Edge / CPU | HTTP OpenAI-compatible server (like vLLM) |

Implementation pattern for OpenAI-compatible hosts:

1. Subclass or configure `OpenAIChatCompletionsAdapter`.
2. Register slug in `LLMProvider` + `_BUILTIN_ADAPTERS`.
3. Add env vars and a row in this doc.
4. Add mocked conformance tests.

---

## Adding a new provider

1. Implement under `providers/<name>_adapter.py`.
2. Register in `llm_provider_registry._BUILTIN_ADAPTERS`.
3. Extend `LLMProvider` enum.
4. Add `_shared` mappers if needed.
5. Tests + `LLM_ADAPTERS.md` env table.
6. Optional `pyproject.toml` extra `llm-<name>`.

---

## Out of scope

- LLM adapters are **not** Integration Library slugs.
- Cloud facades must **not** wrap LLM adapters.
- Billing-grade token counts — prefer provider `usage` fields; else tiktoken estimates.
