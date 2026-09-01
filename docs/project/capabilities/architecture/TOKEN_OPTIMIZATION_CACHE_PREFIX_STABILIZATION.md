<!--
© Artur Czarnecki. All rights reserved.
Intergrax framework – proprietary and confidential.
Use, modification, or distribution without written permission is prohibited.
-->

# Token Optimization - Cache-prefix Stabilization Architecture and Contract Addendum

**Status:** Helper-level contracts and policy **Done / Closed** (`TOKEN-OPT-5A`–`TOKEN-OPT-5E`); runtime wiring and universal proof planned under **TOKEN-10**

**Parent feature architecture:** [`TOKEN_OPTIMIZATION.md`](TOKEN_OPTIMIZATION.md) (§8.3.1 Cache-prefix stabilization)

**Parent feature plan:** [`../plan/TOKEN_OPTIMIZATION.md`](../plan/TOKEN_OPTIMIZATION.md) (§TOKEN-10)

**Scope:** cache-aware prompt/thread architecture, provider cache attribution, cache-safe optimization sequencing, and TOKEN-10 implementation canon.

---

## 1. Why this addendum exists

Prompt caching is not just a cheaper way to count tokens. For long-running agent workloads it becomes an architectural constraint on prompt assembly, tool exposure, thread history, compaction timing, provider adapters, and observability.

This addendum captures the Token Optimization implications from cache-heavy agent-runtime designs such as Viktor's prompt-caching architecture research:

```text
https://viktor.com/research/how-we-built-viktor-around-prompt-caching
```

Use as **architectural inspiration only** - not an external dependency or copied implementation. Do not call the Intergrax implementation “VIKTOR algorithm/cache/runtime.”

The key architectural conclusion for Intergrax:

```text
Cache-prefix stabilization is a first-class Token Optimization surface.
It must be measured separately from content-reduction strategies.
```

`TOKEN-OPT-5A`–`TOKEN-OPT-5E` closed helper-level architecture/contracts. **TOKEN-10** wires them into runtime assembly, provider integration, orchestration, and universal proof.

---

## 2. Core distinction

Prompt caching is a **cost and latency optimization**. It is not the same thing as source-content reduction.

| Category | Examples |
|----------|----------|
| Content reduction | deduplication, budget-aware packing, extractive filtering, structural compaction, output policy |
| Provider prefix-cache reuse | cached_input_tokens, prefix-cache hits, lower prefill work, latency improvement |
| Managed-provider billing (Claude-style) | explicit cache breakpoints, billing discounts where offered - **not** identical to vLLM self-hosted semantics |

Do not mix these families in receipts, telemetry, benchmark summaries, or public claims. Do not add cached tokens to content-saved tokens.

---

## 3. Stable prefix and dynamic tail

### Stable prefix may contain

```text
system policy
stable agent role
stable runtime conventions
stable safety instructions
stable model-facing tool envelope
stable product rules
intentionally cacheable long-lived thread context
```

### Dynamic tail contains

```text
current user request
current tool results
fresh RAG evidence
new attachments or source payloads
current run and step metadata
timestamps, trace IDs, request IDs
transient diagnostics
dynamic tool availability data
current optimization request data
```

### Hard invariants (runtime - TOKEN-10B)

1. Stable prefix must be byte/token stable for shared-prefix requests.
2. Stable block order must remain deterministic.
3. Stable block content must not be rewritten during the active cache window.
4. Dynamic data must not appear in the stable prefix.
5. New conversation content must be appended after the stable prefix.
6. Existing historical messages must not be silently reordered or rewritten.
7. Tool schemas must remain stable for the same effective tool set.
8. Prefix and tool-envelope fingerprints must be measurable.
9. Prefix invalidation must produce an explicit reason.
10. Cache stability is a runtime property, not only a test helper.

Helper-level contracts are extended by the production assembler at `intergrax/runtime/token_optimization/prompt_assembly.py` (**TOKEN-10B** - implemented / ready for review).

---

## 4. Append-only thread invariant

```text
stable prefix
  + append-only messages / observations / tool results
  + dynamic tail for the current step
```

Cache-safe: append new information; preserve byte-stable prefix; preserve block order; keep dynamic data after prefix.

Cache-hostile: rewriting old messages; reordering history; inserting metadata into prefix; regenerating tool catalog per step; compacting hot prefix while cache is valuable.

---

## 5. Stable tool-envelope architecture

Two independent objectives:

```text
reduce the model-facing tool catalog size
preserve a deterministic cache-stable tool envelope
```

Canonical rules:

- canonical `ToolContract` objects remain immutable;
- model-facing exported schemas are deterministic;
- identical effective tool sets → identical ordered tool envelopes;
- tool order cannot depend on non-deterministic registry iteration;
- descriptions/schemas cannot contain per-request metadata;
- dynamic availability reasons do not belong in the stable prefix;
- effective tool-set fingerprint must be reported;
- changed effective tool set invalidates cache identity;
- smaller unstable catalog may be worse than slightly larger stable catalog.

Native tool calling remains valid when the exposed envelope is deterministic and cache stable.

---

## 6. As-built runtime (TOKEN-10B / TOKEN-10B-R1 / TOKEN-10B-R2)

Production assembler: `intergrax/runtime/token_optimization/prompt_assembly.py`.

Public contracts:

- `PromptAssemblyMessageBlock`, `PromptCacheBlockFingerprint`, `CacheStablePromptState`, `CacheStableToolEnvelope`, `CacheStablePromptAssemblyReport`, `CacheStablePromptAssembly`
- `assemble_cache_stable_prompt`, `build_cache_stable_tool_envelope`, `cache_stable_prompt_assembly_to_safe_dict`
- **TOKEN-10B-R1:** `CacheStablePromptSendPayload`, `CacheStablePromptIntegrityError`, `materialize_cache_stable_send_payload`
- **TOKEN-10B-R2:** exact-send tool-schema hash order sensitivity; materialization and `ToolPlanningService` reject noncanonical prepared order

### TOKEN-10B-R1 send integrity (as-built correction)

- **Defensive model-facing snapshots:** assembly deep-copies only `to_dict()` fields (`role`, `content`, `name`, `tool_call_id`, `tool_calls`); caller-owned messages and nested structures are not retained.
- **Defensive tool-schema snapshots:** `build_cache_stable_tool_envelope` deep-copies, canonicalizes, and fingerprints copied entries only.
- **Full message integrity hash:** `CacheStablePromptAssembly.messages_hash` - SHA-256 over canonical `to_dict()` sequence; separate from `prefix_hash` (cacheable prefix identity only).
- **Send-time validation:** `materialize_cache_stable_send_payload` recomputes and compares `messages_hash`, stable-prefix fingerprints, and tool-envelope hash/ordering before adapter invocation; returns fresh defensive copies; raises `CacheStablePromptIntegrityError` on mismatch (fail closed).
- **Canonical hashing helpers:** `compute_model_facing_messages_hash` (`intergrax/llm/messages.py`), `compute_openai_tools_schema_hash` (`intergrax/tools/exporters/openai.py`) - shared by assembler and `ToolPlanningService` post-pruning validation.
- **Complete tool-envelope transitions:** `None↔hash` and `hash A↔hash B` report `tool_envelope_stable=false` and `TOOL_ENVELOPE_CHANGED` when prefix is otherwise reusable; prompt-safety invalidation retains precedence.

### TOKEN-10B-R2 exact schema sequence integrity (as-built correction)

- **Order-sensitive exact-send schema hash:** `compute_openai_tools_schema_hash` preserves outer tool-list order; dictionary keys are canonicalized via `sort_keys=True` only.
- **Canonical tool order established once:** `build_cache_stable_tool_envelope` sorts by `function.name` before hashing; returned envelope is the exact sequence hashed.
- **Materialization rejects reordered envelope schema:** `materialize_cache_stable_send_payload` compares observed tool IDs and order-sensitive hash against recorded envelope state without re-canonicalizing tampered input.
- **`ToolPlanningService` rejects noncanonical prepared order:** `plan_native_round` validates prepared schema against expected lexicographic tool-ID sequence before adapter invocation.

Router integration (`TokenOptimizationLLMRouter`):

- stable prefix block: `token_optimization.router.system`
- dynamic tail: request facts + untrusted analyzed content
- caller-owned continuity: `previous_prompt_cache_state` on `TokenOptimizationLLMRouterRequest`; `prompt_cache_state` + `prompt_assembly_report` on `TokenOptimizationLLMRouterResult`

Tool envelope:

- deterministic export: `build_tool_planning_schema` in `intergrax/runtime/nexus/tools/tool_planning_service.py`
- exact schema forwarding: `prepared_tools_schema` on `plan_native_round`

Invalidation reasons include `append_only_violation` and `tool_envelope_changed` alongside existing helper-level reasons.

---

## 8. Provider ownership boundary

### `LLM_ADAPTERS` owns

- provider-specific prompt-cache capabilities;
- automatic prefix caching support;
- explicit cache breakpoints where applicable;
- provider cache keys;
- provider retention or TTL data where available;
- session or replica affinity requirements;
- provider request parameters;
- provider cache usage mapping;
- cached-token accounting;
- provider-specific latency and cost interpretation;
- provider health and capability discovery.

### `TOKEN_OPTIMIZATION` owns

- cache-stable prompt strategy;
- stable prefix and dynamic tail contracts;
- append-only policy;
- tool-envelope stability requirements;
- cache-aware execution policy;
- separation of cache reuse and content reduction;
- orchestration of provider signals with the deterministic pipeline;
- proof configuration and proof evaluation;
- receipts and safe reports;
- application-neutral integration contract.

### `OBSERVABILITY` owns

- approved domain-signal/HOS emission;
- cache hit/miss/invalidation metrics;
- content-reduction metrics;
- proof/run attribution;
- no private Token Optimization telemetry bus.

Token Optimization must not create: a second tokenizer; a private vLLM HTTP client outside `LLM_ADAPTERS`; a parallel provider abstraction; hidden provider-cache configuration; a private cache metrics exporter; provider-specific logic inside LKW.

Implementation rows: `TOKEN-LLM-2`, `TOKEN-LLM-3` in [`docs/project/maintainers/plans/LLM_ADAPTERS.md`](../../maintainers/plans/LLM_ADAPTERS.md).

---

## 7. Cache-aware execution gate (TOKEN-10D - accepted / closed)

`TOKEN-OPT-5E` delivered helper-level timing policy. **TOKEN-10D-1** wired the runtime consumer. **TOKEN-10D-3** adds evidence reconciliation before routing. **Cache-aware runtime composition contract:** accepted / closed under TOKEN-10D.

```text
LLM adapter → typed usage
  → PromptCacheUsageSnapshot
  → evidence reconciliation with PromptCacheAttribution
  → cache signal normalizer (TOKEN-10D-2)
  → CacheAwareCompactionTimingInput
TokenOptimizationLLMRouter.route()
  → CacheAwareTokenOptimizationOrchestrator.orchestrate()
  → decide_cache_aware_compaction_timing()   # prompt_cache.py - policy only
  → pipeline execution only on RUN
```

**Source-of-truth rules (TOKEN-10D-2 / TOKEN-10D-3):**

- evidence reconciliation runs before router invocation;
- provider/model mismatch fails closed (`SIGNALS_REJECTED`);
- normalization rejection stops before LLM and pipeline;
- reported zero is not the same as unknown cache state;
- missing provider cache details must not be coerced into cache miss;
- TTL remaining is an explicit runtime signal - never inferred from requested/default/max TTL;
- global provider KV metrics do not prove per-request prefix hotness;
- the normalizer and runtime perform no provider I/O and no provider polling;
- no TTL inference;
- no global-metric-to-request mapping;
- no automatic in-cache mutation.

TOKEN-10E adds in-cache compaction (architecture defined; creates new prefix lineage - does not mutate provider cache).

Do not claim mixed character/token estimates as measured savings.

---

## 8. In-cache compaction (TOKEN-10E - architecture defined / ready for review)

Promoted from future commentary to explicit planned phase. Cross-domain lifecycle architecture frozen under **CTX-UCL-ARCH-1** ([`UNIFIED_CONTEXT_LIFECYCLE.md`](../../architecture/UNIFIED_CONTEXT_LIFECYCLE.md)); **TOKEN-10E-ARCH-1** superseded. Runtime implementation has **not** started and is **blocked** pending UCL foundation.

**Logical meaning (not provider KV-cache mutation):**

```text
existing logical context version
  → build shorter compaction candidate
  → validate candidate
  → accept or reject
  → if accepted, new context version and new cache lineage
```

Accepted compaction creates a **new prefix lineage** - it does not mutate provider-owned cache entries. The system declares that subsequent requests no longer assume compatibility with the previous stable prefix; it does not claim immediate provider cache deletion.

Canonical detail: [TOKEN_OPTIMIZATION.md §8.10](TOKEN_OPTIMIZATION.md#810-policy-governed-in-cache-compaction-token-10e).

Requires (at implementation): explicit policy opt-in; protected-region preservation; receipts with old/new hashes; cache attribution separate from content reduction; operator-visible fallback; no destructive overwrite before validation; rollback metadata; no automatic production enablement by default.

For vLLM timing, use actual cache-state evidence from TOKEN-10D - not invented Claude-style fixed TTL.

**Historical note:** `TOKEN-OPT-5A` excluded in-cache compaction by design. That scope boundary is superseded by **TOKEN-10E** planning.

---

## 9. Measurement separation

**Content reduction:** `baseline_input_tokens`, `optimized_input_tokens`, `content_saved_tokens`, `content_saved_ratio`, `content_saved_chars`, `content_reduction_strategy`, `per_layer_saved_*`.

**Provider prefix-cache reuse:** `prompt_tokens`, `cached_input_tokens`, `uncached_input_tokens`, `prefix_cache_queries`, `prefix_cache_hits`, `cache_hit_ratio`, `prefix_hash`, `prefix_stability_status`, `cache_invalidation_reason`, `prefill_duration`, `time_to_first_token`, `total_duration`.

---

## 10. vLLM universal proof runtime (TOKEN-10C)

First canonical self-hosted prefix-cache proof runtime. Reuse: `LLMProvider.VLLM`, `VllmChatAdapter`, `infra/docker/vllm/docker-compose.yml`, `LLMAdapterResponse.usage`, `LLMTokenUsage.cached_input_tokens`.

Proof must distinguish cold, warm, and changed-prefix negative control. Fail when cache evidence unavailable. Model configurable via TOML - no permanently canonical model.

vLLM proves compute reuse and latency/prefill improvement - not Claude billing discounts or guaranteed retention.

---

## 11. Roadmap

| Phase | Status |
|-------|--------|
| `TOKEN-OPT-5A`–`TOKEN-OPT-5E` | **Done / Closed** (contracts + helpers + synthetic policy tests) |
| `TOKEN-8`, `TOKEN-9` | **Accepted / Closed** (execution engine + router) |
| `TOKEN-10A` | **Accepted / Closed** (docs-only canon) |
| `TOKEN-10B` | **Implemented / Ready for review after R2** (`prompt_assembly.py`, router wiring) |
| `TOKEN-10B-R1` | **Implemented / Ready for review** (send-payload integrity, envelope transitions) |
| `TOKEN-10B-R2` | **Implemented / Ready for review** (exact tool-schema sequence integrity) |
| `TOKEN-10C`–`TOKEN-10H` | **Planned** |

See [`../plan/TOKEN_OPTIMIZATION.md`](../plan/TOKEN_OPTIMIZATION.md) §TOKEN-10 for subtask acceptance criteria.

---

## 12. Public claim guardrails

See [`../TOKEN_OPTIMIZATION_CLAIMS.md`](../TOKEN_OPTIMIZATION_CLAIMS.md). README promotion deferred to **TOKEN-10H**.

---

## 13. Current decision

Next implementation: **TOKEN-10C** (vLLM prefix-cache provider integration). **TOKEN-10B** assembler is implemented at `intergrax/runtime/token_optimization/prompt_assembly.py`. Universal platform proof precedes LKW product proof (**LKW-PF6-A**–**C**).
