# Intergrax LLM Adapters

**Last updated:** 2026-05-30

The **LLM adapter layer** (`intergrax/llm_adapters/`) is Intergrax’s Tier-0 module for calling large language model providers through one runtime contract. Agents, `ChatAgent`, `ToolsAgent`, and Nexus runtime components depend on `LLMAdapter` — not vendor SDKs directly.

**Related docs:**

| Document | Purpose |
|----------|---------|
| [intergrax_runtime_architecture.md](intergrax_runtime_architecture.md) §5.2.2 | Canon — LLM adapters are **outside** the Integration Library |
| [TOOLS.md](TOOLS.md) | Agent tools; native tool-calling uses `generate_with_tools` |
| [AGENT_CREATION_GUIDE.md](AGENT_CREATION_GUIDE.md) | Wire `LLMAdapterRegistry.create()` or `LLMProfile` in Tier-3 |
| [INTEGRATIONS.md](INTEGRATIONS.md) | External backends (DB, queues, …) — **not** LLM providers |
| [INTERGRAX_IMPLEMENTATION_PLAN.md](INTERGRAX_IMPLEMENTATION_PLAN.md) | Phase **M-LLM** status |

---

## Design principles

| Principle | What it means |
|-----------|---------------|
| **Single contract** | All providers implement `LLMAdapter`: chat, streaming, optional tools, structured output. |
| **Registry, not integrations** | Use `LLMAdapterRegistry` — not `IntegrationRegistry`. |
| **Lazy loading** | Provider modules load on first `create(provider)`. |
| **OpenAI-compatible factory** | Groq, vLLM, Together, Fireworks, OpenRouter, DeepSeek, xAI, llama.cpp, Cohere share `openai_compat_factory.py`. |
| **Tier-3 profiles** | `LLMProfile` + `llm_profile_from_env()` mirror `IntegrationProfile`. |
| **Shared mapping** | `_shared/` normalizes `ChatMessage` → provider payloads (including tool rounds). |
| **Retries** | `LLMCallConfig` + `_execute()` for transient API errors (all shipped adapters). |
| **Usage tracking** | Per-`run_id` token and latency stats for `LLMUsageTracker`. |
| **CI gate** | `tests/unit/llm_adapters/` (except `@pytest.mark.network`) in GitHub Actions regression gate. |

---

## Supported providers (17)

| Provider key | Class | SDK / API | Native tools | Streaming | Structured |
|--------------|-------|-----------|--------------|-----------|------------|
| `openai` | `OpenAIChatResponsesAdapter` | OpenAI Responses | ✅ | ✅ text + tools | ✅ JSON schema |
| `claude` | `ClaudeChatAdapter` | Anthropic Messages | ✅ | ✅ | ✅ |
| `azure_openai` | `AzureOpenAIChatAdapter` | Azure Chat Completions | ✅ | ✅ tools | ✅ json_schema |
| `gemini` | `GeminiChatAdapter` | google-genai (API key) | ✅ | ✅ tools | ✅ |
| `vertex_gemini` | `VertexGeminiChatAdapter` | google-genai (Vertex ADC) | ✅ | ✅ tools | ✅ |
| `mistral` | `MistralChatAdapter` | mistralai | ✅ | ✅ tools | ✅ |
| `aws_bedrock` | `BedrockChatAdapter` | InvokeModel + **Converse** | ✅ Converse / Anthropic | ✅ Converse stream | — |
| `ollama` | `LangChainOllamaAdapter` | langchain-ollama | ❌ JSON planner | ✅ | ✅ |
| `groq` | `GroqChatAdapter` | OpenAI-compatible | ✅ | ✅ tools | ✅ |
| `vllm` | `VllmChatAdapter` | OpenAI-compatible (self-hosted) | ✅ | ✅ tools | ✅ |
| `together` | `TogetherChatAdapter` | OpenAI-compatible | ✅ | ✅ tools | ✅ |
| `fireworks` | `FireworksChatAdapter` | OpenAI-compatible | ✅ | ✅ tools | ✅ |
| `openrouter` | `OpenRouterChatAdapter` | OpenAI-compatible gateway | ✅ | ✅ tools | ✅ |
| `deepseek` | `DeepSeekChatAdapter` | OpenAI-compatible | ✅ | ✅ tools | ✅ |
| `xai` | `XaiChatAdapter` | OpenAI-compatible (Grok) | ✅ | ✅ tools | ✅ |
| `llama_cpp` | `LlamaCppChatAdapter` | OpenAI-compatible (llama.cpp server) | ✅ | ✅ tools | ✅ |
| `cohere` | `CohereChatAdapter` | Cohere OpenAI-compat layer | ✅ | ✅ tools | ✅ |

**Ollama** — `supports_tools() == False`; `ToolsAgent` uses the JSON planner branch.

**Bedrock** — set `INTERGRAX_BEDROCK_USE_CONVERSE=true` for unified messages, native tools, and **Converse streaming** (`converse_stream`).

**Vertex Gemini** — requires `INTERGRAX_VERTEX_PROJECT` and GCP Application Default Credentials (no `GOOGLE_API_KEY`).

---

## Architecture

```text
Tier-3  LLMProfile.from_env() / ApplicationBuildContext
        │
        ▼
LLMAdapterRegistry.create("together", model="...")
        │
        ▼
LLMAdapter (ABC)  —  call_config, _execute(retry), usage log
        │
        ├── _shared/  messages, tool_schema, bedrock_converse, conformance
        ├── registry/profile.py  (Tier-3 selection)
        ├── openai_compat_factory.py  (hosted OpenAI-compatible APIs)
        │
        ▼
Provider SDK or HTTP
```

---

## Quick start

### Registry

```python
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry
from intergrax.llm.messages import ChatMessage

llm = LLMAdapterRegistry.create(
    LLMProvider.OPENROUTER,
    model="anthropic/claude-3.5-sonnet",
    max_retries=2,
)

answer = llm.generate_messages(
    [ChatMessage(role="user", content="Hello")],
    run_id="demo-run",
)
```

### Tier-3 profile (env-driven)

```python
from intergrax.llm_adapters.registry import LLMProfile, llm_profile_from_env

# INTERGRAX_LLM_PROVIDER=groq  INTERGRAX_LLM_MODEL=llama-3.3-70b-versatile
llm = llm_profile_from_env().create_adapter()
```

---

## Environment variables

### Core providers

| Variable | Provider | Purpose |
|----------|----------|---------|
| `OPENAI_API_KEY` | OpenAI | Required |
| `INTERGRAX_DEFAULT_OPENAI_MODEL` | OpenAI | Default model |
| `ANTHROPIC_API_KEY` | Claude | Required |
| `INTERGRAX_DEFAULT_CLAUDE_MODEL` | Claude | Default model |
| `GOOGLE_API_KEY` | Gemini | Required |
| `INTERGRAX_DEFAULT_GEMINI_MODEL` | Gemini | Default model |
| `INTERGRAX_VERTEX_PROJECT` | Vertex Gemini | GCP project id |
| `INTERGRAX_DEFAULT_VERTEX_LOCATION` | Vertex Gemini | e.g. `us-central1` |
| `INTERGRAX_DEFAULT_VERTEX_GEMINI_MODEL` | Vertex Gemini | Default model |
| `MISTRAL_API_KEY` | Mistral | Required |
| `INTERGRAX_DEFAULT_AZURE_OPENAI_*` | Azure | Endpoint, API version, deployment |
| `INTERGRAX_DEFAULT_AWS_REGION` | Bedrock | Required |
| `INTERGRAX_DEFAULT_BEDROCK_MODEL_ID` | Bedrock | Required |
| `INTERGRAX_BEDROCK_USE_CONVERSE` | Bedrock | `true` → Converse + stream + tools |
| `INTERGRAX_DEFAULT_OLLAMA_MODEL` | Ollama | Model tag |

### OpenAI-compatible (factory)

| Variable | Provider |
|----------|----------|
| `GROQ_API_KEY`, `INTERGRAX_DEFAULT_GROQ_*` | Groq |
| `VLLM_API_KEY`, `INTERGRAX_DEFAULT_VLLM_*` | vLLM |
| `TOGETHER_API_KEY`, `INTERGRAX_DEFAULT_TOGETHER_*` | Together |
| `FIREWORKS_API_KEY`, `INTERGRAX_DEFAULT_FIREWORKS_*` | Fireworks |
| `OPENROUTER_API_KEY`, `INTERGRAX_DEFAULT_OPENROUTER_*` | OpenRouter |
| `DEEPSEEK_API_KEY`, `INTERGRAX_DEFAULT_DEEPSEEK_*` | DeepSeek |
| `XAI_API_KEY`, `INTERGRAX_DEFAULT_XAI_*` | xAI |
| `LLAMA_CPP_API_KEY`, `INTERGRAX_DEFAULT_LLAMA_CPP_*` | llama.cpp server |
| `COHERE_API_KEY`, `INTERGRAX_DEFAULT_COHERE_*` | Cohere |

### Tier-3 profile env

| Variable | Purpose |
|----------|---------|
| `INTERGRAX_LLM_PROVIDER` | Slug for `llm_profile_from_env()` |
| `INTERGRAX_LLM_MODEL` | Optional model override |

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

## Bedrock

- **InvokeModel** — per-family codecs (`anthropic`, `meta`, `mistral`, `amazon`).
- **Converse API** — `INTERGRAX_BEDROCK_USE_CONVERSE=true`: `converse`, `converse_stream`, `toolConfig`.
- Unknown model family → error at adapter construction.

---

## Testing and CI

```bash
uv run pytest tests/unit/llm_adapters/ -q
uv run pytest tests/unit/llm_adapters/ -m gate -q
```

**Network smoke** (optional, not in PR gate):

```bash
GROQ_API_KEY=... uv run pytest tests/unit/llm_adapters/test_network_smoke.py -m network -q
```

Or GitHub Actions: workflow **LLM network smoke** (`workflow_dispatch`, secrets `GROQ_API_KEY` / `OPENAI_API_KEY`).

---

## Optional install extras

```bash
uv sync --extra llm-groq
uv sync --extra llm-compat    # all OpenAI-compatible hosted adapters (openai SDK)
uv sync --extra llm-vertex    # Vertex Gemini (google-genai)
uv sync --extra llm-all
```

---

## Roadmap

| Item | Status |
|------|--------|
| OpenAI-compatible factory (8 slugs) | **Done** |
| `LLMProfile` / env factory | **Done** |
| Vertex Gemini (ADC) | **Done** |
| Bedrock Converse streaming | **Done** |
| Azure thin wrapper on Chat Completions | **Done** |
| Per-provider observability dashboards | Backlog |
| Conformance gate on every adapter PR | Backlog |
| Azure AI Inference unified catalog | Backlog |
| Dedicated Cohere native SDK adapter | Backlog (compat layer shipped) |

---

## Adding a new provider

1. Add `LLMProvider` enum value.
2. Implement adapter under `providers/` (or `OpenAICompatProviderConfig` + thin class).
3. Register in `llm_provider_registry._BUILTIN_ADAPTERS`.
4. Add `_shared` mappers if non-standard message/tool shape.
5. Mocked tests in `tests/unit/llm_adapters/`.
6. Document env vars here; optional `pyproject.toml` extra.

---

## Out of scope

- LLM adapters are **not** Integration Library slugs.
- Cloud facades must **not** wrap LLM adapters.
- Billing-grade token counts — prefer provider `usage` fields; else tiktoken estimates.
