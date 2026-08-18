# Llm Adapters — Implementation Plan

**Architecture (1:1):** [`architecture/LLM_ADAPTERS.md`](../../architecture/LLM_ADAPTERS.md)
**Hub:** [`intergrax_runtime_architecture.md`](../../architecture/intergrax_runtime_architecture.md)
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../../technical/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> When implementing this layer, read **only** the architecture doc and **this plan hub** (`plan/satellites` satellites on demand).

**Cross-feature — Token Optimization:** feature architecture [`features/architecture/TOKEN_OPTIMIZATION.md`](../../capabilities/architecture/TOKEN_OPTIMIZATION.md) · feature plan [`features/plan/TOKEN_OPTIMIZATION.md`](../../capabilities/plan/TOKEN_OPTIMIZATION.md). LLM_ADAPTERS provides tokenizer-consistent token counting, context window metadata, usage accounting, and model/cost signals consumed by Token Optimization. Do not create a parallel tokenizer or duplicate adapter usage accounting.

---

## LCI-6A — Native Ollama adapter architecture and parity matrix

**Status:** `APPROVED`
**Owner:** LLM_ADAPTERS
**Feature satellite:** [`../../capabilities/architecture/satellites/OLLAMA_NATIVE_ADAPTER_PARITY_MATRIX.md`](../../capabilities/architecture/satellites/OLLAMA_NATIVE_ADAPTER_PARITY_MATRIX.md)

The target `NativeOllamaAdapter` implements the existing `LLMAdapter` ABI.
`LangChainOllamaAdapter` remains the explicit compatibility parity baseline
after LCI-6D. The
satellite freezes message, response, streaming, tool, structured-output,
capability, context-window, usage, error, Token Optimization, LKW, transport,
and LCI-6B harness requirements.

LCI-6A changes no production code, resolver, LKW path, Token Optimization
policy, dependency, or tests. LCI-6B implements the native adapter behind a
non-default path; LCI-6C is the mandatory live proof; LCI-6D owns controlled
cutover; LCI-6E owns compatibility packaging.

## LCI-6B — Native Ollama adapter implementation

**Status:** `APPROVED`
**Owner:** LLM_ADAPTERS
**Implementation:** `intergrax/llm_adapters/providers/native_ollama_adapter.py`

The official `ollama` client now has a direct, injectable native transport
implementation of the existing `LLMAdapter` ABI. It preserves the existing
LangChain adapter as the default baseline and adds deterministic unit and
side-by-side request/response evidence for messages, streaming fallback,
tools, structured output, usage, capabilities, context windows, and blocked
LangChain imports. Live rows remain assigned to LCI-6C; default resolver,
LKW, and Token Optimization behavior are unchanged.

## LCI-6C — Native Ollama live parity proof

**Status:** `APPROVED`
**Owner:** LLM_ADAPTERS
**Evidence:** [`OLLAMA_NATIVE_ADAPTER_LIVE_PARITY_EVIDENCE.md`](../../capabilities/architecture/satellites/OLLAMA_NATIVE_ADAPTER_LIVE_PARITY_EVIDENCE.md)

The mandatory live proof against Ollama `0.32.5` is recorded for the native
adapter's plain, tools, structured-output, streaming, capability, usage, and
error surfaces. Rows `040`–`042` remain `LIVE_NOT_REPRODUCIBLE`, rows `043`–
`044` remain `PROVIDER_PREVENTS_REPRODUCTION`, and the no-tools chat model
remains `BLOCKED_MODEL_AVAILABILITY`; deterministic LCI-6B evidence supports
those non-live rows.

## LCI-6D — LKW and Token Optimization native Ollama cutover

**Status:** `APPROVED`
**Owner:** LLM_ADAPTERS

The canonical `LLMProvider.OLLAMA` registry entry now resolves to
`NativeOllamaAdapter`. LKW model-runtime proof and Token Optimization continue
to consume only the existing `LLMAdapter` ABI, capability signals, structured
output, usage, and context-window contracts. `LangChainOllamaAdapter` remains
explicitly constructible for compatibility and parity tests; it is no longer a
production default. Native Ollama regression gating is complete for LCI-6E.

## LCI-6E — LangChain Ollama compatibility optionalization

**Status:** `APPROVED`
**Owner:** LLM_ADAPTERS

`NativeOllamaAdapter` remains the canonical/default Ollama adapter.
`LangChainOllamaAdapter` is explicitly constructed behind the
`llm-langchain-ollama` extra, with lazy LangChain imports and a stable missing
extra error. Multimedia Ollama vision detection uses provider identity and no
longer imports the compatibility class. LCI-7A–LCI-7C are accepted; LCI-7D
documents the resulting core/default and compatibility boundary.

---

## Cursor read scope (token budget)

**Do not read this entire file in one session** (LLM_ADAPTERS plan).

- **Implement / audit default:** Hub §6 · [`plan/satellites`](plan/satellites) satellites on demand. Phase AUDIT-IDEAL — **Planned** / open rows only. §6.1 maintenance queues — open P0/P1 only
- **Token Optimization:** read feature pair + row `TOKEN-LLM-1`; validate consumption of existing `count_messages_tokens`, `context_window_tokens`, `LLMAdapterResponse.usage`, and ModelCatalog signals only.
- **Use** `Read` with offset/limit — open `### 6.1*` / Phase rows (**P0/P1**, Status ≠ Done) only.
- **Skip** `(closed)`, `(complete)`, `Archived`, **Done** unless re-validating a cited gap.
- **Architecture hub:** [`architecture/LLM_ADAPTERS.md`](../../architecture/LLM_ADAPTERS.md) read-scope block only.
- **Platform audit:** [`docs/audit_results/AUDIT_PROTOCOL.md`](../../audit_results/AUDIT_PROTOCOL.md).
- **Satellites:** at most **one** `plan/satellites` file per session unless RESUME cites more.

---

## Satellite registers (read on demand)

Large historical registers moved out of the hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited gap ID.

| Satellite | Contents |
|-----------|----------|
| [`plan/LLM_ADAPTERS_implementation_history.md`](plan/LLM_ADAPTERS_implementation_history.md) | implementation history |

> **Cursor context budget:** read hub read-scope block + **at most one** satellite per session.

---

## Phase TOKEN-LLM — Token Optimization adapter guardrail (Planned)

**Feature:** [`features/plan/TOKEN_OPTIMIZATION.md`](../../capabilities/plan/TOKEN_OPTIMIZATION.md)
**Architecture:** [`features/architecture/TOKEN_OPTIMIZATION.md`](../../capabilities/architecture/TOKEN_OPTIMIZATION.md)
**Priority:** P1 validation row, not a new adapter feature  
**Delivery rule:** keep Token Optimization dependent on existing adapter token/cost contracts.

| ID | Type | Priority | Status | Deliverable | Acceptance |
|----|------|----------|--------|-------------|------------|
| **TOKEN-LLM-1** | Guardrail | P1 | Planned | Verify Token Optimization integrations consume existing `LLMAdapter.count_messages_tokens`, `context_window_tokens`, `LLMAdapterResponse.usage`, and ModelCatalog context metadata | No parallel tokenizer or per-feature cost tracker; CE/context preflight still uses adapter token path; token savings attribution can consume usage envelope; `uv run python scripts/maintenance/check_context_preflight_uses_adapter_tokens.py`; `uv run python scripts/check_token_optimization_contracts.py` |
| **TOKEN-LLM-2** | Capability | P1 | Implemented / Ready for review | Prompt-cache provider capability and usage contract integration — expose cache mode, cached/uncached input tokens, prefix-cache signals through `LLMAdapterResponse.usage` and provider extensions without a Token Optimization private client | Managed and self-hosted paths documented; Token Optimization reads signals only; no second tokenizer; feature plan §TOKEN-10C |
| **TOKEN-LLM-3** | Proof path | P1 | Implemented / Ready for review | vLLM prefix-cache request/metrics proof path — pin image/version, enable automatic prefix caching, health/readiness, cache metrics, cold/warm/changed-prefix controls for universal proof | Reuse `VllmChatAdapter`, `infra/docker/vllm/docker-compose.yml`; fail closed when metrics unavailable; live proof gated by `INTERGRAX_TOKEN_OPTIMIZATION_VLLM_E2E=1`; feature plan §TOKEN-10C |

---

## Phase AUDIT-IDEAL — Ideal architecture gap register (2026-06-09)

**Source:** Post-L3 audit vs [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../../technical/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §3.5 · baseline **32/32 L3**
**Master register:** [`plan/AUDIT_IDEAL_2026.md`](AUDIT_IDEAL_2026.md) · Band **2ay** · queue **§6.1au**  
**Status:** **Done** (2026-06-18) — all AUDIT-IDEAL §6 rows closed

| ID | AUDIT § | Gap | Priority | Status |
|----|---------|-----|----------|--------|
| AUDIT-IDEAL-6.1 | §6 LLM | Structured output validation on 100% reference + certified agent paths | P1 | **Done** |
| AUDIT-IDEAL-6.2 | §6 LLM | Live cost/latency/quality model routing (AHI prod path) | P2 | **Done** — `check_live_model_routing_wiring.py` |
| AUDIT-IDEAL-6.3 | §6 LLM | Central `ModelCatalog` + unified context window resolution | P0 | **Done** — `CatalogCapabilityAdapter` |
| AUDIT-IDEAL-6.4 | §6 LLM | Tokenizer-consistent context preflight (adapter path) | P0 | **Done** — `count_message_tokens(adapter=)` |
| AUDIT-IDEAL-6.5 | §6 LLM | Profile failover chain on retriable provider errors | P1 | **Done** — LC-3 |
| AUDIT-IDEAL-6.6 | §6 LLM | ACP `StepLLMRouter` backed by `LLMAdapter` (single DX) | P1 | **Done** — M-LLM-X.5.4 |
| AUDIT-IDEAL-6.7 | §6 LLM | Developer `USAGE.md` + startup validation | P2 | **Done** — `check_llm_profile_runtime.py` + doctor |

**Delivery rule:** One **AUDIT-IDEAL-*** ID per PR → update this table + master register → gate green.

---

## Phase M-LLM-X — LLM Developer Excellence (post-audit 2026-06-14)

**Source:** Deep production audit 2026-06-14 — contract L3, model metadata L1–L2, routing L1–L2, DX L2.  
**Canon:** [`architecture/LLM_ADAPTERS.md`](../../architecture/LLM_ADAPTERS.md) §Model catalog · §Routing · §Audit register
**Goal:** Elevate Tier-0 LLM layer from **production L3 foundation** to **best-in-class developer engine** — correct context for any model string, unified token accounting, runtime routing/failover, single agent API.  
**Status:** **LC baseline Done** (2026-06-14) — P0/P1 closed · **M-LLM-X partial waves** = P2+ backlog (not blocking layer maturity)  
**Priority ladder:** Band **2ba** (after M-LLM-R closeout) · queue [§6.1ax](.#61ax-harness-implementation-queue--llm-developer-excellence-m-llm-x)
**Execution order:** [§6.2af](.#62af-phase-m-llm-x-execution-order)
**Target maturity:** Model metadata **L3**, routing **L3**, DX **L3+** (see architecture maturity table).

**Hard rules (non-negotiable):**

- **No** new per-adapter hardcoded context dicts without catalog entry or prefix rule — migrate to `ModelCatalog`.
- **`LLMProfile.options["context_window_tokens"]`** MUST override catalog for **all** providers (not Ollama-only).
- **Preflight / history budget** MUST use `adapter.count_messages_tokens` when adapter is in scope.
- **No** vendor SDK imports in Tier-2 — unchanged tier boundary.
- One **M-LLM-X.*** task group per PR → update master table + architecture audit register → gate green.
- **ADR:** [ADR-LLM-002](../../technical/adr/entries/2026-06-14/ADR-LLM-002.md) **Done** — prerequisite for M-LLM-X.1 code merge.
**ADR:** [ADR-LLM-003](../../technical/adr/entries/2026-06-19/ADR-LLM-003.md) **Accepted** — prerequisite for M-LLM-X.9 code merge.

**Explicitly excluded:** Central LLM gateway microservice (needs separate platform ADR), rewriting all 19 SDK clients, product HTTP DTOs, Phase K agents.

---

## Phase LLM-PROVIDER-PLUGIN — Provider plugin registration layer (Backlog)

**Architecture:** [`architecture/LLM_ADAPTERS.md`](../../architecture/LLM_ADAPTERS.md) §Provider selection · §Provider plugin layer (planned)
**Priority:** P2 (P1 if external provider packages become a near-term requirement) — **not blocking** current layer maturity  
**Status:** Planned / Backlog  
**Goal:** Add a provider metadata and plugin registration layer for LLM providers **without** replacing the existing `LLMAdapter` execution contract.

| ID | Name | Priority | Status | Deliverable |
|----|------|----------|--------|-------------|
| **LLM-PROVIDER-PLUGIN-1** | LLM provider contract and plugin registration layer | P2 | Planned | Thin provider metadata / registration layer above `LLMAdapterRegistry`, mirroring runtime integrations registry v2 strengths |

### Assumptions

- `LLMAdapter` remains the hot-path execution contract (generate, stream, tools, structured output, token counting, quota/resilience).
- `LLMProvider` enum remains valid for stable built-in providers.
- Custom providers MUST NOT require editing `LLMProvider` enum or core `_BUILTIN_ADAPTERS`.
- Provider packages are discoverable / registrable through a deterministic provider registration contract.
- Mirror integrations registry v2: `provider_id`, provider kind, config class, factory, capabilities, health check support, security posture, metadata, default disabled behavior.
- The new layer creates or describes `LLMAdapter` instances — it does **not** replace them.
- Do **not** replace `LLMAdapter` with `PlatformIntegrationContract`.

### Proposed target objects

- `LLMProviderRegistration`
- `LLMProviderContract` or `LLMProviderMetadataContract`
- `LLMProviderConfig` base class (if needed)
- Optional LLM provider package convention / discovery entrypoint
- Unit tests for custom provider registration without enum/core edits

### Acceptance criteria

1. Built-in LLM providers still work exactly as today.
2. `LLMProvider` enum remains available for stable built-ins.
3. A custom provider package can register itself without modifying `LLMProvider` enum.
4. A custom provider package can register itself without modifying `_BUILTIN_ADAPTERS`.
5. Registry lists provider metadata: `provider_id`, `display_name`, supported protocol, config class, factory, capabilities, default model, secret/env requirements.
6. Registry can instantiate an `LLMAdapter` from the provider registration.
7. Provider metadata supports safe public view without exposing secrets.
8. Unit tests prove: built-in compatibility; custom registration; duplicate rejection; invalid contract rejection; factory returns `LLMAdapter`; metadata does not expose secrets.
9. No Tier-2 agent imports vendor SDKs directly.
10. No rewrite of existing provider adapters required unless explicitly scoped later.

### Non-goals

- Do not rewrite all existing adapters.
- Do not remove `LLMProvider` enum.
- Do not replace `LLMAdapter` with `PlatformIntegrationContract`.
- Do not introduce a central HTTP LLM gateway.
- Do not change model routing behavior, `ModelCatalog` behavior, or response envelope contracts.

**Delivery rule:** One **LLM-PROVIDER-PLUGIN-*** ID per PR → update this table + architecture §Provider plugin layer → gate green.

---
