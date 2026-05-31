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

---

## Design principles

| Principle | What it means |
|-----------|---------------|
| **Single contract** | All providers implement `LLMAdapter`: chat, optional streaming, tools, structured output. |
| **Registry, not integrations** | Use `LLMAdapterRegistry` — not `IntegrationRegistry`. |
| **Lazy loading** | Provider SDKs load on first `create(provider)` — faster cold start. |
| **Shared mapping** | `_shared/` normalizes `ChatMessage` → provider payloads (including tool rounds). |
| **Usage tracking** | Each adapter aggregates tokens/latency per `run_id` for `LLMUsageTracker`. |
| **No secrets in code** | API keys and endpoints come from environment variables. |

---

## Supported providers (7)

| Provider key | Class | SDK | Native tools | Streaming | Structured output |
|--------------|-------|-----|--------------|-----------|-----------------|
| `openai` | `OpenAIChatResponsesAdapter` | OpenAI Responses API | ✅ | ✅ text + tools partial | ✅ JSON schema |
| `claude` | `ClaudeChatAdapter` | Anthropic Messages | ✅ | ✅ text + tools partial | ✅ prompt + schema |
| `azure_openai` | `AzureOpenAIChatAdapter` | Azure OpenAI Chat Completions | ✅ | ✅ | ✅ json_schema |
| `gemini` | `GeminiChatAdapter` | google-genai | ✅ | ✅ | ✅ prompt + schema |
| `mistral` | `MistralChatAdapter` | mistralai | ✅ | ✅ | ✅ prompt + schema |
| `aws_bedrock` | `BedrockChatAdapter` | boto3 bedrock-runtime | ✅ Anthropic models | ✅ InvokeModel stream | — |
| `ollama` | `LangChainOllamaAdapter` | langchain-ollama | ❌ (JSON planner) | ✅ + invoke fallback | ✅ prompt + schema |

**Ollama** intentionally uses `supports_tools() == False`; `ToolsAgent` falls back to JSON planner mode.

---

## Architecture

```text
Tier-2/3  ChatAgent / ToolsAgent / RuntimeConfig
        │
        ▼
LLMAdapterRegistry.create("openai", model="gpt-4o-mini")
        │
        ▼
LLMAdapter (ABC)
  ├── generate_messages / stream_messages
  ├── generate_with_tools / stream_with_tools  (optional)
  ├── generate_structured                    (optional)
  └── usage: LLMAdapterUsageLog
        │
        ▼
Provider SDK (openai, anthropic, google.genai, …)
```

**Shared helpers** (`intergrax/llm_adapters/_shared/`):

| Module | Role |
|--------|------|
| `messages.py` | Chat Completions message mapping with tool history |
| `responses_input.py` | OpenAI Responses API input items |
| `anthropic_messages.py` | Anthropic `tool_use` / `tool_result` blocks |
| `tool_schema.py` | OpenAI tool schema → Anthropic / Gemini |
| `tool_results.py` | Standard `{content, tool_calls, finish_reason}` |
| `call_config.py` | `LLMCallConfig` (timeout, retries) |
| `retry.py` | Transient error retries |
| `bedrock_converse.py` | Bedrock Converse API (optional path) |
| `conformance.py` | Contract assertions for tests |

---

## Quick start

```python
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry
from intergrax.llm.messages import ChatMessage

# Requires OPENAI_API_KEY (and optional INTERGRAX_DEFAULT_OPENAI_MODEL)
llm = LLMAdapterRegistry.create(LLMProvider.OPENAI, model="gpt-4o-mini")

answer = llm.generate_messages(
    [ChatMessage(role="user", content="Hello")],
    temperature=0.2,
    max_tokens=256,
    run_id="demo-run",
)

for chunk in llm.stream_messages(
    [ChatMessage(role="user", content="Stream this")],
    run_id="demo-run",
):
    print(chunk, end="")
```

**Custom factory (override built-in):**

```python
LLMAdapterRegistry.register("openai", my_factory, override=True)
```

---

## Environment variables

| Variable | Provider | Purpose |
|----------|----------|---------|
| `OPENAI_API_KEY` | OpenAI | API key (required) |
| `INTERGRAX_DEFAULT_OPENAI_MODEL` | OpenAI | Default model (fallback: `gpt-5-mini`) |
| `ANTHROPIC_API_KEY` | Claude | API key (required) |
| `INTERGRAX_DEFAULT_CLAUDE_MODEL` | Claude | Default model |
| `GOOGLE_API_KEY` | Gemini | API key (required) |
| `INTERGRAX_DEFAULT_GEMINI_MODEL` | Gemini | Default model |
| `MISTRAL_API_KEY` | Mistral | API key (required) |
| `INTERGRAX_DEFAULT_MISTRAL_MODEL` | Mistral | Default model |
| `INTERGRAX_DEFAULT_AZURE_OPENAI_ENDPOINT` | Azure | Endpoint URL (required) |
| `INTERGRAX_DEFAULT_AZURE_OPENAI_API_VERSION` | Azure | API version (required) |
| `INTERGRAX_DEFAULT_AZURE_OPENAI_DEPLOYMENT` | Azure | Deployment name (required) |
| `AZURE_OPENAI_API_KEY` | Azure | Key (SDK default) |
| `INTERGRAX_DEFAULT_AWS_REGION` | Bedrock | AWS region (required) |
| `INTERGRAX_DEFAULT_BEDROCK_MODEL_ID` | Bedrock | Model ID (required) |
| `INTERGRAX_BEDROCK_USE_CONVERSE` | Bedrock | `true` → use Converse API when available |
| `INTERGRAX_DEFAULT_OLLAMA_MODEL` | Ollama | Model tag (fallback: `llama3.1:latest`) |

---

## Capabilities API

```python
llm.supports_streaming()          # default True for shipped adapters
llm.supports_tools()              # True for native tool providers
llm.supports_structured_output()  # True when generate_structured is wired
llm.context_window_tokens         # cached estimate for budgeting
llm.usage.get_run_stats(run_id)   # aggregated LLMRunStats
```

**ToolsAgent** checks `supports_tools()` and calls `generate_with_tools` / `stream_with_tools` with OpenAI-format tool schemas from `intergrax.tools.exporters.openai`.

**ChatAgent** uses `supports_structured_output()` before `generate_structured`.

---

## Call policy (retries)

Pass retry settings via adapter constructor kwargs (stored in `LLMCallConfig`):

```python
llm = LLMAdapterRegistry.create(
    "openai",
    model="gpt-4o-mini",
    max_retries=2,
    retry_backoff_sec=0.5,
)
```

`LLMAdapter._execute()` wraps SDK calls when `max_retries > 0` (OpenAI and Bedrock invoke/converse paths use this today; other adapters can adopt the same pattern).

---

## Bedrock notes

- **InvokeModel codecs** — `anthropic`, `meta`, `mistral`, `amazon` families via native JSON bodies.
- **Unknown model family** — fails at adapter construction (not at first request).
- **Converse API** — set `INTERGRAX_BEDROCK_USE_CONVERSE=true` for unified `converse()` when the boto3 client supports it; otherwise InvokeModel codecs are used.
- **Tools** — native tools only for `BedrockModelFamily.ANTHROPIC` (Anthropic message format on Bedrock).

---

## Optional install extras

Smaller deployments can depend on subsets (see `pyproject.toml`):

```bash
uv sync --extra llm-openai
uv sync --extra llm-anthropic
uv sync --extra llm-all
```

The default `dependencies` list still includes all LLM SDKs for the full monorepo gate.

---

## Testing

| Path | Purpose |
|------|---------|
| `tests/unit/llm_adapters/` | Contract, registry, mapping, conformance mocks |
| `intergrax/llm_adapters/_shared/conformance.py` | Reusable assertions for new adapters |

Run:

```bash
uv run pytest tests/unit/llm_adapters/ -q
```

Network tests against real APIs should use `@pytest.mark.network` — not part of the PR gate.

---

## Adding a new provider

1. Implement `LLMAdapter` under `intergrax/llm_adapters/providers/<name>_adapter.py`.
2. Register in `llm_provider_registry.py` → `_BUILTIN_ADAPTERS`.
3. Add `LLMProvider` enum value if needed.
4. Extend `_shared/` mappers if the API differs from OpenAI Chat Completions.
5. Add conformance tests with mocked SDK.
6. Document env vars in this file.

---

## Out of scope

- LLM adapters are **not** registered in the Integration Library (§5.2.2).
- Cloud platform facades (`aws`, `azure`, `gcp`) must **not** re-export LLM adapters.
- Token billing accuracy for all models — use provider usage fields when available; otherwise tiktoken estimates.
