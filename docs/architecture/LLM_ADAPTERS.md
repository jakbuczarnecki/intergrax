# Intergrax LLM Adapters

**Status:** Canonical architecture (domain pair 1:1)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Plan (1:1):** [`plan/LLM_ADAPTERS.md`](../plan/LLM_ADAPTERS.md)  
**Target:** [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md)  
**Audit layers:** 6  
**Audit instruction:** [`guides/audit/LLM_ADAPTERS.md`](../guides/audit/LLM_ADAPTERS.md)  
---

## Response envelope (M-LLM-R)

All adapter completion methods return typed envelopes — **not** bare `str` or untyped dicts.

| Method | Return type |
|--------|-------------|
| `generate_messages` | `LLMAdapterResponse` |
| `generate_with_tools` | `LLMAdapterResponse` |
| `stream_messages` / `stream_with_tools` | `Iterable[LLMStreamEvent]` |
| `generate_structured` | `LLMStructuredResult[T]` |

### `LLMAdapterResponse`

Primary fields:

| Field | Type | Notes |
|-------|------|-------|
| `content` | `str` | Assistant text (alias: `.text`) |
| `finish_reason` | `LLMFinishReason` | Normalized stop reason |
| `usage` | `LLMTokenUsage` | Per-call token accounting |
| `model` / `provider` | `str` | Identity metadata |
| `response_id` | `str \| None` | Provider correlation id |
| `refusal` | `str \| None` | Provider-native safety/refusal signal when present |
| *(post-adapter)* | `GuardrailScanResult` | Optional Tier-3 `llm_guardrail` scan via middleware (`AFTER_LLM_OUTPUT`) — complements `refusal`; see [`INTEGRATIONS.md`](INTEGRATIONS.md) §47 |
| `tool_calls` | `tuple[LLMToolCall, ...]` | Native tool calls |
| `provider_extensions` | `LLMProviderExtensions` | Optional provider-specific slices |

Example:

```python
from intergrax.llm_adapters import LLMAdapter, LLMAdapterResponse

completion: LLMAdapterResponse = adapter.generate_messages(messages, run_id=run_id)
answer = completion.content
if completion.usage:
    print(completion.usage.total_tokens)
for tc in completion.tool_calls:
    plan_args = tc.arguments_json
```

Build helpers (adapter internals): `build_adapter_response`, `partial_stream_event`, `final_stream_event` in `intergrax/llm_adapters/_shared/adapter_response_builders.py`.

Call lifecycle helper (M-LLM-R.2.6): `LLMCallLifecycle` in `intergrax/llm_adapters/_shared/call_lifecycle.py` — shared `begin_call` / `end_call` + usage sync for provider adapters.

### Trace and replay bridge (M-LLM-R.7.2)

`CoreLLMStep` emits `CoreLLMCallRecordedDiagV1` on each successful LLM call. Persisted Nexus traces map to replay DTOs via:

- `intergrax/runtime/replay/trace_replay_bridge.py` — `serialized_trace_events_to_replay_dtos`
- `intergrax/runtime/replay/persisted_trace_event_store.py` — `PersistedRunTraceEventStore` (`TraceEventStore` over `RunTraceReader`)
- `intergrax/runtime/replay/llm_call_mapper.py` — `llm_call_info_from_adapter_response`

### Adaptive harness hook (M-LLM-R.7.5)

Optional per-call metadata on harness outcome signals: pass `LLMCallSummary` into `SignalAssemblyInput.last_llm_call` (from `intergrax/runtime/adaptive/llm_call_summary.py`). Fields land on `HarnessOutcomeSignal` as `last_llm_*` columns.

### CI guards

| Script | Purpose |
|--------|---------|
| `scripts/check_llm_adapter_typed_returns.py` | ABC public methods must not return bare `str` / dict |
| `scripts/check_agents_llm_adapter_response.py` | Tier-2 agents must not annotate adapter returns as `str` |

---

## Modality plane A — generative multimodal (LLM)

LLM adapters own **Plane A** of the Model & Modality architecture (canon §7.1.9). Dedicated CV (YOLO, ONNX, …), classical ML, and SaaS TTS (ElevenLabs, …) belong to **Plane C** / integrations — not this module.

| Concern | Owner | Notes |
|---------|-------|-------|
| Chat reasoning over text | `llm_adapters/` | Existing |
| Native vendor vision/audio in dialog | `llm_adapters/` | Capability flags + content parts (W-ML.1) |
| Dedicated vision CV / TTS-STT tools | `model_inference/` + `speech_adapters/` | `VisionProfile` / `SpeechProfile` (same pattern as `LLMProfile`) — see [architecture/MODALITY.md](architecture/MODALITY.md) |
| Audio/file → text for RAG | `document_parser` + `rag/` | Plane B — [architecture/MODALITY.md](architecture/MODALITY.md) |
| Object detection / segmentation | `model_inference/` (planned) | Plane C — tools `vision.*` |

### Message attachments

`intergrax/llm/messages.py` defines `AttachmentRef` (`type`: `image`, `audio`, `video`, `pdf`, …; `uri`; metadata). Target behavior:

- Adapters with `supports_vision()` / `supports_audio_input()` map attachments to vendor message parts.
- `ContextBudgetPolicy` and `ModalityProfile.max_media_bytes` cap attachment volume (Phase W-ML).
- Traces record attachment types and sizes — not necessarily raw bytes.

### Capability flags (W-ML.1)

| Method | Meaning |
|--------|---------|
| `supports_vision()` | Image (and optionally video frame) input |
| `supports_audio_input()` | Audio input in chat |
| `supports_audio_output()` | Spoken response generation via vendor API |

Defaults remain **false** until an adapter implements mapping and conformance tests pass.

### When not to use LLM for vision

Use Plane C tools (`vision.detect`, …) when outputs must be **deterministic**, **geometric**, or **auditably reproducible** (safety, manufacturing, compliance). Use Plane A when the product needs **semantic interpretation** in natural language.

---

## Providers (19)

OpenAI-compatible slugs share `openai_compat_factory.py`. Override `supports_streaming()` / `supports_structured_output()` per adapter (ABC defaults: streaming **false**, structured **false**).

| Slug | Adapter module | Primary env | Stream | Structured | Notes |
|------|----------------|-------------|--------|------------|-------|
| `openai` | `openai_responses_adapter` | `OPENAI_API_KEY` | yes | yes | Native Responses API |
| `gemini` | `gemini_adapter` | `GEMINI_API_KEY` | yes | yes | |
| `ollama` | `ollama_adapter` | `OLLAMA_BASE_URL` | yes | partial | Local |
| `mistral` | `mistral_adapter` | `MISTRAL_API_KEY` | yes | yes | |
| `claude` | `claude_adapter` | `ANTHROPIC_API_KEY` | yes | yes | |
| `azure_openai` | `azure_openai_adapter` | `AZURE_OPENAI_*` | yes | yes | |
| `aws_bedrock` | `aws_bedrock_adapter` | `AWS_*` | yes | partial | |
| `groq` | `openai_compat` | `GROQ_API_KEY` | compat | compat | |
| `vllm` | `openai_compat` | `VLLM_BASE_URL` | compat | compat | |
| `together` | `openai_compat` | `TOGETHER_API_KEY` | compat | compat | |
| `fireworks` | `openai_compat` | `FIREWORKS_API_KEY` | compat | compat | |
| `openrouter` | `openai_compat` | `OPENROUTER_API_KEY` | compat | compat | |
| `deepseek` | `openai_compat` | `DEEPSEEK_API_KEY` | compat | compat | |
| `xai` | `openai_compat` | `XAI_API_KEY` | compat | compat | |
| `llama_cpp` | `openai_compat` | `LLAMA_CPP_BASE_URL` | compat | compat | |
| `cohere` | `openai_compat` | `COHERE_API_KEY` | compat | compat | Chat Completions shim |
| `cohere_native` | `cohere_native_adapter` | `COHERE_API_KEY` | yes | partial | |
| `vertex_gemini` | `vertex_gemini_adapter` | `GOOGLE_APPLICATION_CREDENTIALS` | yes | yes | |
| `azure_ai_inference` | `azure_ai_inference_adapter` | `AZURE_AI_*` | yes | partial | |

Central env appendix: `INTERGRAX_LLM_PROVIDER`, `INTERGRAX_LLM_MODEL`, `INTERGRAX_LLM_TENANT_MAX_TOKENS`, `INTERGRAX_LLM_METRICS_ENABLED`, `INTERGRAX_LLM_PROMETHEUS_PUSHGATEWAY_URL`. Per-provider secrets: `llm/<provider>/api_key` via `SecretsStore`.

---

## Tier-3 wiring

```python
from intergrax.llm_adapters.registry import LLMProfile, llm_profile_from_env
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider

profile = LLMProfile(provider=LLMProvider.GROQ, model="llama-3.3-70b-versatile")
llm = profile.create_adapter(secrets={"api_key": key})  # or create_adapter_from_secrets_store(vault)
```

---

## Nexus runtime (automatic)

| Feature | Mechanism |
|---------|-----------|
| Tenant scope | `UnifiedTaskRunner` → `llm_tenant_scope` |
| Task-complete export | `bootstrap_nexus_platform()` → plugin `runtime.llm_metrics_export` |
| Hard quota | `INTERGRAX_LLM_TENANT_MAX_TOKENS` |
| Soft governance warn | `INTERGRAX_LLM_GOVERNANCE_WARN_TOKENS` |
| Pushgateway | `INTERGRAX_LLM_PROMETHEUS_PUSHGATEWAY_URL` |
| Distributed rate limit | `set_llm_distributed_rate_limiter` + `use_distributed_rate_limit` |

---

## Observability (Prometheus & governance)

Tier-0 metrics from `intergrax/llm_adapters/tracking/`.

### Scrape (recommended)

```python
from intergrax.llm_adapters.tracking.exposition import register_llm_metrics_routes

register_llm_metrics_routes(app)  # GET /metrics/llm
```

Configure Prometheus to scrape `http://<host>/metrics/llm`. Lab factory registers routes when `INTERGRAX_LLM_METRICS_ENABLED=true`.

### Pushgateway (optional)

```bash
INTERGRAX_LLM_METRICS_ENABLED=true
INTERGRAX_LLM_PROMETHEUS_PUSHGATEWAY_URL=http://pushgateway:9091
```

Pushes on each `TASK_COMPLETED` via runtime plugin (grouping `tenant/<tenant_id>`).

### Example PromQL (SLO)

```promql
sum by (tenant_id) (rate(intergrax_llm_calls_total[5m]))
sum by (provider) (rate(intergrax_llm_errors_total[5m]))
  / sum by (provider) (rate(intergrax_llm_calls_total[5m]))
sum by (tenant_id, model) (rate(intergrax_llm_output_tokens_total[5m]))
```

Query via Integration `observability_backend` = `prometheus` (`create_prometheus_observability_backend`).

### Governance signals

Correlate logs with Nexus trace using `run_id` / `task_id` in `llm_metrics_export` structured fields.

### Distributed rate limit (multi-replica)

```python
from intergrax.llm_adapters._shared.resilience import set_llm_distributed_rate_limiter
from intergrax.integrations.providers.key_value_cache.redis.bundle import create_redis_rate_limiter

set_llm_distributed_rate_limiter(create_redis_rate_limiter(url="redis://..."))
profile = LLMProfile(provider=..., options={"use_distributed_rate_limit": True, "calls_per_minute": 120})
```

Falls back to in-process limiter when Redis limiter is not set.

### Usage tracking: two layers

| Layer | Type | When |
|-------|------|------|
| **Adapter** | `LLMAdapter.usage` (`LLMAdapterUsageLog`) | Per SDK call inside `generate_messages` / tools |
| **Runtime** | `LLMUsageTracker` on `RuntimeState` | Nexus pipeline steps aggregate into run/trace finalize |

Do not merge counters without explicit bridge code.

### Prometheus: scrape vs integration backend

| Mode | Mechanism | Use when |
|------|-----------|----------|
| **In-process scrape** | `register_llm_metrics_routes(app)` → `GET /metrics/llm` | Single replica, lab, local dev |
| **Pushgateway** | Plugin pushes on `TASK_COMPLETED` | Ephemeral workers without scrape target |
| **PromQL backend** | Integration slug `observability_backend=prometheus` | Central queries against long-lived Prometheus |

---

## Resilience & secrets

- `LLMCallConfig`: retries, in-process rate limit, circuit breaker, optional Redis rate limit.
- `registry/secrets.py`: env + `SecretsStore` paths (`llm/<provider>/api_key`).

## Environment appendix (central)

| Variable | Purpose |
|----------|---------|
| `INTERGRAX_LLM_PROVIDER` | Default provider slug for `LLMProfile.from_env()` |
| `INTERGRAX_LLM_MODEL` | Default model id |
| `INTERGRAX_LLM_METRICS_ENABLED` | Enable metrics plugin + `/metrics/llm` |
| `INTERGRAX_LLM_PROMETHEUS_PUSHGATEWAY_URL` | Optional push on `TASK_COMPLETED` |
| `INTERGRAX_LLM_TENANT_MAX_TOKENS` | Hard per-tenant quota |
| `INTERGRAX_LLM_GOVERNANCE_WARN_TOKENS` | Soft warn via `PolicyEngine` on task complete |
| `INTERGRAX_BEDROCK_USE_CONVERSE` | Bedrock Converse API toggle |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, … | Per-provider secrets (see each `providers/*/USAGE.md`) |

---

## CI (unit gate only — no product E2E)

```bash
uv run pytest tests/unit/llm_adapters/ -m gate -q
```

Workflows: `unit-tests.yml`, `llm-adapters-guard.yml`, optional `llm-network-smoke.yml`.

---

## Harness-aligned next steps

| Item | Canon / goal |
|------|----------------|
| Wire Prometheus scrape in Tier-3 host Helm/K8s | §7.1 observability |
| PolicyEngine rules consuming `llm_cost_evaluation` logs | §governance replay |
| Central LLM gateway service (single egress) | §5.2.4 — needs architecture approval |
| Model routing / fallback chains in `LLMProfile` | Agent harness flexibility |
| Multimodal attachment mapping + capability flags | Phase W-ML.1 · §7.1.9 |
| Cost envelopes and quota policy integration | Phase V `V-COST.*` |
| Evaluation score baselines for model/profile changes | Phase V `V-EVAL.*` |
| Adversarial prompt/tool defense validation on model paths | Phase V `V-SEC.*` |

**Out of scope:** product E2E gates, per-business-agent adapter code in `llm_adapters/`, YOLO/ONNX/CV engines (see [architecture/MODALITY.md](architecture/MODALITY.md) Plane C).
