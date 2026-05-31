# Intergrax LLM Adapters

**Last updated:** 2026-05-30

The **LLM adapter layer** (`intergrax/llm_adapters/`) is Intergrax’s Tier-0 module for calling large language model providers through one runtime contract.

**Related:** [Architecture §5.2.2](intergrax_runtime_architecture.md) · [Tier-3 deployment](applications/USAGE.md) · Phase **M-LLM** in [INTERGRAX_IMPLEMENTATION_PLAN.md](INTERGRAX_IMPLEMENTATION_PLAN.md)

---

## Supported providers (19)

| Key | Class | Notes |
|-----|-------|-------|
| `openai` | `OpenAIChatResponsesAdapter` | Responses API, tools + stream |
| `claude` | `ClaudeChatAdapter` | Native tools stream |
| `azure_openai` | `AzureOpenAIChatAdapter` | Chat Completions on deployment |
| `azure_ai_inference` | `AzureAiInferenceChatAdapter` | Azure AI Inference OpenAI-compatible endpoint |
| `gemini` | `GeminiChatAdapter` | API key |
| `vertex_gemini` | `VertexGeminiChatAdapter` | GCP ADC |
| `mistral` | `MistralChatAdapter` | |
| `aws_bedrock` | `BedrockChatAdapter` | Converse + **converse_stream** (tools) |
| `ollama` | `LangChainOllamaAdapter` | JSON planner (no native tools) |
| `groq` / `vllm` / `together` / `fireworks` / `openrouter` / `deepseek` / `xai` / `llama_cpp` | `*ChatAdapter` in `openai_compat_providers.py` | OpenAI-compatible factory |
| `cohere` | `CohereChatAdapter` | OpenAI-compat layer |
| `cohere_native` | `CohereNativeChatAdapter` | Cohere SDK v2 |

---

## Tier-3 wiring

```python
from intergrax.llm_adapters.registry import LLMProfile, llm_profile_from_env
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider

# Legal application pattern (see legal_application/host/wiring.py)
profile = LLMProfile(provider=LLMProvider.GROQ, model="llama-3.3-70b-versatile", options={"max_retries": 2})
llm = profile.create_adapter()

# Env: INTERGRAX_LLM_PROVIDER / INTERGRAX_LLM_MODEL
llm = llm_profile_from_env().create_adapter()
```

**Legal host env:** `LEGAL_LLM_PROVIDER`, optional `LEGAL_LLM_MODEL`.

---

## Observability (metrics)

Per-call metrics when `INTERGRAX_LLM_METRICS_ENABLED=true`:

- Module: `intergrax/llm_adapters/tracking/metrics.py`
- Snapshot: `get_llm_metrics_collector().snapshot()`
- Prometheus text: `get_llm_metrics_collector().prometheus_lines()`
- Metrics: `intergrax_llm_calls_total`, `intergrax_llm_input_tokens_total`, …

Calls record provider/model from `usage.begin_call(..., adapter=self)`.

---

## Conformance (CI gate)

Every builtin provider is covered by parametrized mocked tests:

```bash
uv run pytest tests/unit/llm_adapters/test_builtin_conformance.py -q
```

Helper: `run_adapter_conformance(adapter)` in `_shared/conformance.py`.

---

## Bedrock

`INTERGRAX_BEDROCK_USE_CONVERSE=true` enables `converse`, `converse_stream`, and **native `stream_with_tools`**.

---

## Environment variables (selected)

| Variable | Provider |
|----------|----------|
| `INTERGRAX_LLM_METRICS_ENABLED` | All — Prometheus-style counters |
| `AZURE_AI_INFERENCE_API_KEY` | `azure_ai_inference` |
| `INTERGRAX_DEFAULT_AZURE_AI_INFERENCE_BASE_URL` | Azure AI Inference base URL |
| `INTERGRAX_DEFAULT_COHERE_NATIVE_MODEL` | `cohere_native` |
| `INTERGRAX_VERTEX_PROJECT` | `vertex_gemini` |
| See previous tables for Groq, OpenRouter, Bedrock, … | |

---

## Testing

```bash
uv run pytest tests/unit/llm_adapters/ -m gate -q
```

**Network smoke** (weekly CI + manual): `.github/workflows/llm-network-smoke.yml`

```bash
GROQ_API_KEY=... uv run pytest tests/unit/llm_adapters/test_network_smoke.py -m network -q
```

---

## Extras (`pyproject.toml`)

`llm-openai`, `llm-compat`, `llm-vertex`, `llm-cohere` (main deps include `cohere`), `llm-all`

---

## Production roadmap (next)

| Item | Priority |
|------|----------|
| OTLP export from `LLMMetricsCollector` | P1 |
| Provider billing dashboards per tenant | P1 |
| Full Bedrock tool-stream on InvokeModel families | P2 |
| Rate-limit / circuit-breaker per provider | P2 |
| Secret rotation via Integration-style profile | P2 |
