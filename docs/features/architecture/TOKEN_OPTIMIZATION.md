<!--
© Artur Czarnecki. All rights reserved.
Intergrax framework – proprietary and confidential.
Use, modification, or distribution without written permission is prohibited.
-->

# Token Optimization — Multi-layer Feature Architecture

**Status:** Planned multi-layer feature architecture  
**Feature plan (1:1):** [`../plan/TOKEN_OPTIMIZATION.md`](../plan/TOKEN_OPTIMIZATION.md)  
**Source audit instruction:** [`../../audit/TOKEN_OPTIMIZATION.md`](../../audit/TOKEN_OPTIMIZATION.md)  
**Primary anchor domain:** `CONTEXT_ENGINEERING`  
**Related domains:** `LLM_ADAPTERS`, `TOOLS`, `MEMORY`, `RAG`, `OBSERVABILITY`, `UNIFIED_EXECUTION_RUNTIME`, `AGENT_CONTRACTS_AND_ASSEMBLY`, `ADAPTIVE_HARNESS_INTELLIGENCE`

---

## Cursor read scope (token budget)

**Do not read this entire file in one session** (TOKEN_OPTIMIZATION feature architecture).

- **Implement / audit default:** §1–§8 engine lifecycle, mechanisms, and extensibility. **On demand (one max):** [`architecture/satellites/TOKEN_OPTIMIZATION_domain_architecture_cross_references.md`](satellites/TOKEN_OPTIMIZATION_domain_architecture_cross_references.md).
- **Plan hub:** [`../plan/TOKEN_OPTIMIZATION.md`](../plan/TOKEN_OPTIMIZATION.md) read-scope block only.
- **Satellites:** at most **one** `architecture/satellites/` file per session unless RESUME cites more.

---

## Architecture satellites (read on demand)

Large cross-domain sync registers moved out of the hub to reduce Cursor context use.
Load **only** the satellite matching your task.

| Satellite | Contents |
|-----------|----------|
| [`satellites/TOKEN_OPTIMIZATION_domain_architecture_cross_references.md`](satellites/TOKEN_OPTIMIZATION_domain_architecture_cross_references.md) | domain architecture cross-reference map and sync checklist |

> **Cursor context budget:** read hub read-scope block + **at most one** satellite per session.

---

## 1. Purpose

Token Optimization is a multi-layer Intergrax platform capability that minimizes unnecessary token usage while preserving correctness, safety, provenance, auditability, and developer control.

It is not a prompt-only brevity mode or a generic prompt-shortening feature. It is a **source-aware**, **measurable**, and **auditable** runtime-controlled token economy capability spanning:

```text
input
  → context assembly
  → memory
  → RAG
  → tool schemas
  → model routing signals
  → output shaping
  → telemetry and receipts
```

Strategic product statement:

```text
Intergrax provides runtime-controlled token economy for governed AI agents.
```

---

## 2. Documentation placement

Token Optimization is documented as a multi-layer feature, not as a new architecture domain by default.

It therefore lives under:

```text
docs/features/architecture/TOKEN_OPTIMIZATION.md
docs/features/architecture/satellites/
docs/features/plan/TOKEN_OPTIMIZATION.md
docs/features/plan/satellites/
```

Feature hubs are Cursor entry points. Satellites hold bulky cross-domain sync registers and are in `.cursorignore` — load with explicit `Read` or `@` only.

The feature coordinates updates across existing domain pairs:

```text
docs/architecture/<DOMAIN>.md
docs/plan/<DOMAIN>.md
```

Do not create `docs/plan/TOKEN_OPTIMIZATION.md` unless Token Optimization is later promoted into a full domain with a matching `docs/architecture/TOKEN_OPTIMIZATION.md`.

---

## 3. Design principles

| Principle | Meaning |
|----------|---------|
| Budget-first | Every LLM call has explicit input/output budget semantics. |
| Quality-preserving | Savings are invalid if answer quality or safety drops below threshold. |
| Policy-governed | Compression can be disabled, downgraded, or scoped by runtime policy. |
| Step-aware | Different execution steps may use different context/output profiles. |
| Source-aware | RAG, memory, tools, history, attachments, and policy text compress differently. |
| Protected regions | Code, paths, URLs, API names, errors, IDs, hashes, dates, and warnings are preserved. |
| Observable | Every optimization is measurable and auditable. |
| Reversible where persistent | Persistent compression uses staging, validation, receipts, and rollback metadata. |
| Fail-safe | Validation failure falls back to original content or lower compression. |
| No silent degradation | Dropped/compressed content is reflected in provenance or receipts. |

---

## 4. Capability boundaries

### 4.1 In scope

- input token budgeting,
- output token budgeting,
- context assembly efficiency,
- structural and semantic context compression,
- memory summary compression,
- tool description/schema presentation optimization,
- output verbosity policy,
- token-aware model routing signals,
- token savings telemetry,
- compression receipts,
- token-vs-quality regression gates,
- policy-governed compression safety.

### 4.2 Out of scope

- direct control of private model reasoning tokens,
- private chain-of-thought compression,
- executable code mutation,
- strict JSON schema semantic rewrite,
- removal of required audit evidence,
- replacement of RAG ranking,
- replacement of memory lifecycle management,
- replacement of LLM adapter token counting,
- replacement of model routing.

---

## 5. Domain ownership matrix

| Domain | Responsibility |
|--------|----------------|
| `CONTEXT_ENGINEERING` | Owns `ContextPackOptimizer`, source-aware context compression, post-compression token recalculation, provenance links to receipts, and fallback behavior. |
| `LLM_ADAPTERS` | Provides tokenizer-consistent counting, context window metadata, output budget metadata, usage accounting, and cost/latency signals. Token Optimization consumes these signals and must not create a parallel tokenizer. |
| `TOOLS` | Owns `ToolSchemaOptimizer` and compact tool catalog presentation. |
| `MEMORY` | Owns `MemorySummaryCompressor`, safe persistent compression, rollback metadata, and memory receipt storage. |
| `RAG` | Allows post-retrieval/post-ranking chunk compression where citations and grounding remain intact. |
| `OBSERVABILITY` | Owns token optimization telemetry, metrics, diagnostic payloads, spans, and savings attribution. |
| `UNIFIED_EXECUTION_RUNTIME` | Resolves runtime token policy, compression level, output profile, and safety bypass rules. |
| `AGENT_CONTRACTS_AND_ASSEMBLY` | Allows agent-level hints for output profile and context compactness without manual prompt assembly. |
| `ADAPTIVE_HARNESS_INTELLIGENCE` | Later learns optimal budgets and compression strategies from telemetry. |

---

## 6. Runtime component placement

Token Optimization uses shared contracts plus domain-owned adapters. The shared package owns policy, receipts, validation, and telemetry contracts; domain packages own the domain-specific integration points.

| Component | Planned module | Owner | Notes |
|-----------|----------------|-------|-------|
| `TokenOptimizationPolicy` | `intergrax/runtime/token_optimization/contracts.py` | `UNIFIED_EXECUTION_RUNTIME` + feature | Top-level policy envelope: input budget, output policy, compression level, protected regions, validation, telemetry. |
| `OutputPolicy` / `OutputPolicyResolver` | `intergrax/runtime/token_optimization/output_policy.py` | `UNIFIED_EXECUTION_RUNTIME` | Runtime-selected output profile; not prompt-only wording. |
| `CompressionReceipt` | `intergrax/runtime/token_optimization/receipts.py` | feature + `OBSERVABILITY` | Shared receipt contract used by context, tools, memory, and telemetry. |
| `ProtectedRegionValidator` | `intergrax/runtime/token_optimization/protected_regions.py` | feature | Exact-preservation validator for code, paths, URLs, API names, enum values, hashes, dates, warnings, and schema-sensitive regions. |
| `TokenOptimizationTelemetryEmitter` | `intergrax/runtime/token_optimization/telemetry.py` | `OBSERVABILITY` | Emits domain signal / diagnostic payload + counters through HOS, not a private telemetry bus. |
| `TokenOptimizer` | `intergrax/runtime/token_optimization/optimizer.py` | feature | Thin policy-driven orchestrator for safe text optimization; domain integrations call it. |
| `ContextPackOptimizer` | `intergrax/runtime/nexus/context/context_pack_optimizer.py` | `CONTEXT_ENGINEERING` | Runs after rank/budget and before format/preflight. |
| `ToolSchemaOptimizer` | `intergrax/runtime/nexus/tools/tool_schema_optimizer.py` | `TOOLS` | Produces compact LLM-facing tool catalog view without mutating schema semantics. |
| `MemorySummaryCompressor` | `intergrax/memory/summary_compressor.py` | `MEMORY` | Persistent summary compression with staging, validation, receipt, and rollback metadata. |
| `TokenRegressionBenchmarkRunner` | `intergrax/runtime/token_optimization/regression.py` | `OBSERVABILITY` + feature | Token-vs-quality benchmark helper; CI scripts consume it. |

No Tier-2 agent may import these internals directly for prompt assembly. Agents may declare hints through contracts; runtime/profile layers resolve effective policy.

---

## 7. Core capability model

```text
TokenOptimizationPolicy
  ├─ InputBudgetPolicy
  ├─ OutputPolicy
  ├─ CompressionPolicy
  ├─ ToolCatalogOptimizationPolicy
  ├─ MemoryCompressionPolicy
  ├─ ProtectedRegionPolicy
  ├─ ValidationPolicy
  └─ TelemetryPolicy
```

### 7.1 Planned shared contracts

```python
@dataclass(frozen=True, slots=True)
class TokenOptimizationRequest:
    run_id: str
    step_id: str | None
    source_type: str
    content: str
    policy: TokenOptimizationPolicy
    model_id: str | None = None
    provider: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class TokenOptimizationResult:
    content: str
    receipt: CompressionReceipt
    validation: TokenOptimizationValidation
    telemetry: TokenOptimizationTelemetry
    fallback_used: bool = False
```

```python
@dataclass(frozen=True, slots=True)
class OutputPolicy:
    profile: OutputProfile
    max_output_tokens: int | None
    require_sections: tuple[str, ...] = ()
    forbid_sections: tuple[str, ...] = ()
    preserve_exact: tuple[str, ...] = ()
```

```python
@dataclass(frozen=True, slots=True)
class CompressionReceipt:
    receipt_id: str
    run_id: str | None
    step_id: str | None
    source_type: str
    strategy: str
    lossy: bool
    original_hash: str
    compressed_hash: str
    original_tokens: int
    compressed_tokens: int
    saved_tokens: int
    saved_ratio: float
    protected_regions_count: int
    validation_status: str
    quality_score: float | None = None
    fallback_used: bool = False
```

The first implementation slice must define these as stable contracts before deeper context/tool/memory integrations depend on them.

---

## 8. Token Optimization Engine lifecycle, mechanisms, and extensibility

The **Token Optimization Engine** is a policy-governed orchestrator, not a single hard-coded compressor. It coordinates classification, policy resolution, mechanism selection, protected-region handling, token measurement, validation, receipts, observability, and safe fallback across domain-owned adapters and optional plugins.

Platform capability model:

```text
Token Optimization Engine
=
source-aware optimization
+ protected-region validation
+ compression receipts
+ fallback
+ telemetry
+ evals/regression gates
+ measurable savings
```

Different source categories require different optimization strategies. The engine classifies content and routes to domain-owned adapters; it does not apply one generic compression algorithm across all sources.

| Source category | Optimization strategy | Safety rule |
|-----------------|----------------------|-------------|
| `tools` | Compact LLM-facing schema/catalog view | No schema semantics mutation; canonical `ToolContract` immutability |
| `context` / `RAG` | Structural/light compression after ranking | Provenance and evidence preservation; citation/grounding checks |
| `memory` | Staged persistent summary compression | Validation and rollback metadata before any live overwrite |
| `output` | Runtime output shaping (verbosity, section structure) | Shapes model completion length — not source compression |
| `policy` / `legal` / `security` text | Exact preservation or explicit opt-in only | No silent lossy compression unless policy explicitly allows |

**Semantic validation / LLM-as-a-Judge (forward-looking, not implemented today):** future regression and evals gates (for example TOKEN-6B) may use semantic validation or LLM-as-a-Judge to assess lossy/semantic compression quality across measured workflows. That layer must **not** replace deterministic protected-region validation or source-specific structural validation in the optimization path. First implementation slices use deterministic validators only; semantic judges belong to the regression/evals layer, not to helper-only compressors.

### 8.1 Engine lifecycle

```text
input/source payload
  → classify content/source type
  → resolve token optimization policy
  → select mechanism and strategy
  → detect protected regions
  → estimate or read baseline token count
  → apply optimization
  → re-count optimized tokens
  → validate protected regions and safety rules
  → build receipt / measurement record
  → emit observability signal through HOS or approved domain-signal path
  → return optimized payload or fallback to original
```

Rules:

- policy resolution happens before any lossy or structural change,
- baseline token counts use the existing LLM adapter token path when available; the engine must not create a second tokenizer,
- validation failure must fallback to the original payload or a safer/lower optimization level — never return silently degraded content,
- every applied optimization produces a receipt or explicit bypass/fallback record when measurement is enabled,
- observability emission uses the Harness Observability Spine or an approved domain-signal path only.

### 8.2 Mechanism catalog

Each mechanism class below is a **policy-selectable optimization surface**. Savings must be measured, when applicable, as:

```text
baseline_tokens
optimized_tokens
saved_tokens
saved_ratio
```

Not every mechanism belongs in the first public proof. The first proof should prioritize mechanisms that are measurable, safe, and easy to validate (see §8.7).

| Mechanism | What it optimizes | Typical source/category | Expected measurable metric | Safety risk | Required validator / guardrail | Likely implementation phase | First public proof candidate |
|-----------|-------------------|-------------------------|----------------------------|-------------|-------------------------------|------------------------------|------------------------------|
| Tool output compaction | Verbose tool results, logs, and command output replayed into context | `tool_result`, terminal/log output | `saved_tokens` on tool-result category | High — may drop errors, paths, IDs | Protected-region validator; exact error/path preservation | TOKEN-4 light + domain hooks | **Yes** — primary proof candidate |
| Terminal/log/test-output filtering | Noisy shell, CI, and test transcripts | `terminal_output`, `test_output` | `saved_tokens` on filtered transcript category | Medium — may hide failure signals | Extractive filter rules; protected-region validator; fallback on ambiguity | TOKEN-4 / application hooks | Later — after tool-output compaction proves receipts |
| Tool catalog/schema compaction | LLM-facing tool descriptions and catalog prose | `tool_catalog` | `saved_tokens` on tool-catalog category | Medium — must not alter schema semantics | Schema-preservation validator; canonical `ToolContract` immutability | TOKEN-3 | **Yes** — primary proof candidate |
| RAG/context-pack light compression | Retrieved evidence fragments after ranking | `context_pack`, `rag_chunk` | `saved_tokens` on RAG/evidence category | High — may break citations/grounding | Citation/grounding checks; protected-region validator; ranking-before-compression rule | TOKEN-4 light | **Yes** — light/structural only |
| Memory/context pruning | Low-value history, duplicate blocks, stale summaries | `memory`, `history` | `saved_tokens` on memory/history category | High — persistent loss risk | Staging + rollback; receipt; protected-region validator; quality gate | TOKEN-5 | No — gated until regression gates exist |
| Cache alignment / stable prompt prefixing | Repeated stable prefixes across steps/runs | `prompt_prefix`, `system_policy` | Cache-hit / prefix-stability signal + `saved_tokens` where measurable | Low when prefix-only | Prefix immutability check; no mutation of dynamic tail | TOKEN-2 / TOKEN-6 | **Yes** — where cache/prefix savings are measurable |
| Output policy / verbosity shaping | Model completion length and section structure | `model_output` | `output_tokens` vs budget; `saved_tokens` on output category | Medium — may reduce audit clarity | OutputPolicyResolver; high-risk bypass; explicit profile comparison | TOKEN-2 | **Yes** — only where baseline/optimized comparison is explicit |
| Structured data compression | JSON/YAML/tabular blobs in context | `structured_data` | `saved_tokens` on structured-data category | Medium — schema-sensitive | Schema-shape preservation; protected keys/enums | TOKEN-4 extension | Later |
| Reversible machine-to-machine representation | Machine-facing payloads that can be re-expanded | `m2m_payload` | `saved_tokens` with reversibility flag | Low–medium — expansion contract must hold | Reversibility validator; round-trip check | TOKEN-4 / plugin | Later — plugin-friendly |
| Retrieval-on-demand instead of full replay | Full document/chunk replay replaced by retrieval handles | `document_replay`, `chunk_replay` | `saved_tokens` on replay category | Medium — grounding risk | Retrieval handle integrity; citation checks | TOKEN-4 / RAG integration | Later |
| Deduplication and repeated-context suppression | Repeated paragraphs, tool prose, and duplicate evidence | `context_pack`, `history`, `tool_catalog` | `saved_tokens`; `input_tokens_after_dedup_total` | Low–medium — ordering/semantics | Dedup provenance; mandatory-fragment preservation | TOKEN-4 light | **Yes** — when provenance is recorded |

### 8.3 Strategy / algorithm taxonomy

**Sequencing roadmap:** [`../plan/TOKEN_OPTIMIZATION.md`](../plan/TOKEN_OPTIMIZATION.md) §TOKEN-OPT-3A defines the stronger-optimizer rollout order, per-algorithm measurement expectations, and savings attribution rules. Do not build one monolithic stronger optimizer; introduce each strategy as a separate measurable platform step.

Strategies are the algorithms a mechanism may apply. A single mechanism may compose multiple strategies under policy control.

| Strategy type | Lossless | Lossy | Reversible | Measurement-only | Policy-only | Requires protected-region validation | Requires quality/regression benchmark | Safe for initial proof | Experimental |
|---------------|----------|-------|------------|------------------|-------------|--------------------------------------|---------------------------------------|--------------------------|--------------|
| Lossless normalization | Yes | No | Yes | No | No | Yes | No | **Yes** | No |
| Lossless structural compression | Yes | No | Yes | No | No | Yes | Light | **Yes** | No |
| Deduplication | Yes | No | Partial | No | No | Yes | Light | **Yes** | No |
| Extractive filtering | Partial | Partial | No | No | No | Yes | Yes | Later | No |
| Schema minimization | Yes | No | Yes | No | No | Yes | Yes | **Yes** | No |
| Ranking/pruning | No | Yes | No | No | No | Yes | Yes | No | No |
| Cache-prefix stabilization | Yes | No | Yes | No | No | Light | No | **Yes** | No |
| Safe lossy summarization | No | Yes | No | No | No | Yes | **Yes** | No | No |
| Semantic compression | No | Yes | No | No | No | **Yes** | **Yes** | No | **Yes** |
| Reversible M2M encoding | Yes | No | **Yes** | No | No | Yes | Light | Later | No |
| Retrieval-on-demand | Partial | Partial | Partial | No | No | Yes | Yes | Later | No |
| Output verbosity shaping | Partial | Partial | No | No | **Yes** | Light | Yes | **Yes** — explicit comparison only | No |

Rules:

- aggressive lossy or semantic compression must **not** be enabled by default before protected-region validation, receipts, telemetry, and regression gates exist,
- `measurement-only` and `policy-only` strategies may run without mutating payload content,
- experimental strategies require an explicit `experimental` profile and operator opt-in.

### 8.3.1 Cache-prefix stabilization

**Status:** architecture / contract closed by `TOKEN-OPT-5A` (**Done / Closed**). Runtime and provider implementation remain deferred to `TOKEN-OPT-5B+`.

**Detailed supporting contract:** [`TOKEN_OPTIMIZATION_CACHE_PREFIX_STABILIZATION.md`](TOKEN_OPTIMIZATION_CACHE_PREFIX_STABILIZATION.md).

Cache-prefix stabilization defines a future provider-cache-aware optimization surface. Prompt caching is a **cost/latency optimization**, not content reduction. It is a **first-class Token Optimization surface** and must be measured separately from content-reduction strategies.

#### Stable prefix and dynamic tail

Prompt assembly must treat **stable prefix** and **dynamic tail** as separate zones.

The stable prefix may contain:

```text
system policy
agent role and stable instructions
stable safety policy
stable product/runtime conventions
stable tool envelope / deterministic compact tool catalog view
long-lived workspace/thread context only when intentionally cacheable
```

The stable prefix must **not** contain volatile values such as:

```text
wall-clock timestamps
run_id
trace_id
request_id
random IDs
provider-generated metadata
per-step diagnostic counters
per-request user text
fresh retrieved evidence
dynamic source payloads
ephemeral tool results
```

```text
Dynamic content belongs in the prompt tail, not in the stable prefix.
```

Where provider cache reuse is intended, the stable prefix must be **byte-stable**.

#### Append-only prompt/thread invariant

```text
When provider cache reuse is active, prompt/thread assembly should preserve already-sent cacheable prefix blocks and append new information after them.
```

**Cache-safe behavior:**

```text
append new information
preserve byte-stable already-sent prefix
preserve ordering of cacheable blocks
keep dynamic per-step data after the cacheable prefix
```

**Cache-hostile behavior:**

```text
rewriting old thread messages
reordering historical blocks
inserting timestamps/run IDs/trace IDs into the prefix
regenerating tool catalog text differently per step
compacting hot cacheable prefix blocks while cache value is still high
```

#### Tool-surface cache-stability rule

Tool optimization has two independent goals:

```text
1. reduce LLM-facing tool catalog size
2. keep the LLM-facing tool envelope cache-stable
```

```text
A shorter tool catalog that changes frequently can be worse than a slightly longer stable tool catalog that achieves high provider-cache reuse.
```

Boundary:

- `ToolSchemaOptimizer` must **not** mutate canonical `ToolContract` definitions.
- Future compact tool views must be **deterministic**, **cache-stable for the same effective tool set**, and **separate from per-request tool metadata**.

#### Provider ownership boundary

| Owner | Responsibility |
|-------|----------------|
| `LLM_ADAPTERS` | Provider-specific prompt cache capabilities; explicit cache breakpoints; automatic/implicit provider caching behavior; cache key / retention / TTL support; cache read/write usage fields; cached-token accounting; provider-specific price/latency interpretation; session affinity requirements where applicable |
| `TOKEN_OPTIMIZATION` | Shared cache-prefix stabilization strategy; shared attribution vocabulary; policy and safety boundaries; cache-vs-content-reduction separation; compaction timing rules |
| `OBSERVABILITY` | Cache hit/miss/invalidation attribution; safe counters/signals through existing approved HOS/domain-signal paths |

Token Optimization must **not** create:

```text
a second tokenizer
a provider-specific prompt cache client
a private provider-cache telemetry bus
hidden prompt-cache configuration
```

Provider-cache attribution must be measured separately from content-reduction savings. Runtime/provider implementation is deferred to later `TOKEN-OPT-5B+` tasks.

#### Cache attribution vocabulary

**Cache-related terms** (contract vocabulary only — not claimed as implemented):

```text
cache_read_tokens
cache_creation_tokens
cached_input_tokens
uncached_input_tokens
cache_hit_ratio
prefix_hash
prefix_stability_status
cache_invalidation_reason
cache_discount_estimate
cache_latency_delta_estimate
```

**Content-reduction terms** (separate attribution family):

```text
content_saved_chars
content_saved_tokens
content_reduction_strategy
```

```text
cache_* metrics describe provider cache behavior.
content_saved_* metrics describe content reduction.
They must remain separable in telemetry, receipts, benchmark summaries, and public proof claims.
```

#### Cache-aware compaction timing

Content compaction is not always beneficial when prompt caching is active.

```text
A compaction step that rewrites a hot cached prefix may destroy more value than it saves.
```

Future compaction policy must compare:

```text
estimated content-reduction benefit
estimated cache invalidation cost
```

**Prefer:**

```text
filtering dynamic tool/log output before it becomes stable prefix
packing/reducing dynamic tail content
preserving hot cacheable prefix while cache value is high
compacting cold or idle thread history
```

**Avoid:**

```text
rewriting stable prefix during active multi-step runs
semantic summarization of hot cacheable history
repacking old thread history on every step
changing stable tool catalog formatting per request
```

##### Cache-aware compaction timing policy

`TOKEN-OPT-5E` adds a provider-neutral helper/policy layer that decides whether compaction should **run**, **defer**, **bypass**, or **require manual review**.

Intergrax now has a provider-neutral cache-aware compaction timing policy. Runtime/provider integration remains deferred. The policy helps avoid rewriting hot cacheable prefixes when estimated cache invalidation cost outweighs estimated content-reduction benefit.

Key rules:

```text
compaction can destroy more cache value than it saves
dynamic tail compaction is preferred over stable prefix rewriting
stable prefix / full-thread compaction is conservative
protected/semantic risk requires manual review
near-expiry or cold-history compaction may be allowed
policy decisions are helper-level / provider-neutral
no runtime provider caching is implemented in TOKEN-OPT-5E
```

Contracts and helper live under `intergrax/runtime/token_optimization/` (`CacheAwareCompaction*` types and `decide_cache_aware_compaction_timing`). Decisions must not include raw prompt/thread content and must not treat mixed-unit estimates as measured token savings.

##### Advisory recommendation layer (`TOKEN-7A`)

`TOKEN-7A` adds a **recommendation-only**, policy-only advisory layer. Intergrax now has a policy-only advisory recommendation layer for Token Optimization posture. Recommendations are redaction-safe and non-auto-apply. Runtime/adaptive integration remains deferred.

Key properties:

```text
recommendation-only first — no autonomous production auto-apply
recommendations use redaction-safe scalar signals only
may suggest conservative/balanced profile, full context, strategy enable/disable,
  dynamic-tail reduction, cache-prefix preservation, or manual review
must not include raw prompt/context/evidence/tool output
must not compute token savings
runtime/adaptive integration remains deferred
```

Contracts and helper live under `intergrax/runtime/token_optimization/` (`TokenOptimizationAdvisory*` types and `recommend_token_optimization_action`). Every recommendation keeps `auto_apply_allowed=False` and `raw_content_included=False`.

##### Advisory evaluation and reporting (`TOKEN-7B`)

`TOKEN-7B` adds a redaction-safe advisory evaluation and reporting layer on top of the policy-only recommender. Advisory reports evaluate recommendation outcomes, not runtime behavior. Reports use safe scalar fields only, must not include raw prompt/context/evidence/tool output, and must prove non-auto-apply status across evaluated cases. Runtime/adaptive integration remains deferred.

Key properties:

```text
evaluation/reporting only — no autonomous production auto-apply
deterministic per-case evaluation against expected action/reason/confidence
aggregate pass/fail/manual-review/insufficient-data/non-auto-apply/raw-content-safe counts
redaction-safe dict and text report formatters
must not include raw signal or recommendation objects in report output
must not include saved_tokens / optimized_tokens / baseline_tokens / compressed_tokens fields
runtime/adaptive integration remains deferred
```

Contracts and helpers live under `intergrax/runtime/token_optimization/` (`TokenOptimizationAdvisoryEvaluation*` types, `evaluate_advisory_recommendation_case`, `evaluate_advisory_recommendation_cases`, `token_optimization_advisory_report_to_dict`, `format_token_optimization_advisory_report`).

##### Advisory policy-gated integration surface (`TOKEN-7C`)

`TOKEN-7C` adds a deterministic policy gate around the advisory recommender. Policy is explicitly passed in the request; no global config resolver, env resolver, or YAML config is added in this task.

Supported modes: `disabled`, `report_only`, `dry_run`, `review_only`, `advisory_allowed`. Recommendations may be blocked, returned as report-only, returned as dry-run, escalated to review, or marked recommendation-ready. Recommendation-ready still does not mean auto-apply; `auto_apply_allowed` remains `False`. Result serialization is redaction-safe.

Key properties:

```text
policy-gated integration only — no autonomous production auto-apply
policy is explicitly passed per request (no global/env/YAML resolver)
deterministic gate over recommend_token_optimization_action(...)
modes: disabled, report_only, dry_run, review_only, advisory_allowed
recommendations may be blocked, report-only, dry-run, review-required, or recommendation-ready
recommendation-ready remains non-auto-apply
redaction-safe dict and text result formatters
must not include raw signal or recommendation objects in serialized output
no runtime prompt assembly changes
no provider calls
no adaptive runtime integration
no observability/HOS emission
```

Contracts and helpers live under `intergrax/runtime/token_optimization/` (`TokenOptimizationAdvisoryIntegration*` types, `evaluate_policy_gated_advisory_request`, `token_optimization_advisory_integration_result_to_dict`, `format_token_optimization_advisory_integration_result`).

#### In-cache compaction boundary (`TOKEN-OPT-5A`)

```text
No in-cache compaction in TOKEN-OPT-5A.
No LLM summarization in TOKEN-OPT-5A.
No semantic compression in TOKEN-OPT-5A.
No adaptive rewriting in TOKEN-OPT-5A.
```

Future in-cache compaction requires:

```text
explicit lossy/semantic opt-in
protected-region validation
quality/regression gates
receipts and rollback metadata where persistent
provider-cache attribution separated from content-reduction attribution
operator-visible fallback behavior
```

### 8.4 Configuration model

External product configuration should expose a simple operator-facing switch:

```yaml
token_optimization:
  enabled: true
```

The platform must support richer policy-controlled configuration underneath, for example:

```yaml
token_optimization:
  enabled: true
  profile: conservative
  strategies:
    tool_output_compaction: true
    tool_schema_compaction: true
    context_pack_compression: light
    memory_compression: off
    cache_alignment: true
    output_policy: concise
  safety:
    protected_regions: strict
    fallback_on_validation_failure: true
    disable_for_audit_evidence: true
  measurement:
    emit_receipts: true
    emit_observability: true
    compare_baseline: true
```

**Profiles** (conceptual only — no runtime config files in this architecture slice):

| Profile | Intent |
|---------|--------|
| `off` | No optimization; measurement may still record baseline counts. |
| `measure_only` | Count and attribute tokens without mutating payloads. |
| `conservative` | Lossless/normalization, schema minimization, light dedup, cache alignment; no semantic compression. |
| `balanced` | Conservative plus light structural context compression and explicit output shaping. |
| `aggressive` | Broader pruning and lossy summarization where validators and regression gates pass. |
| `experimental` | Plugin or research strategies; never default in production proof paths. |

Effective policy is resolved by `UNIFIED_EXECUTION_RUNTIME` and may be downgraded by step risk, audit mode, or protected-region failures.

### 8.5 Plugin and extensibility model

Token optimization mechanisms, strategies, and algorithms behave like other platform extension points: **replaceable, policy-governed, observable, and contract-based**.

A developer with a proven third-party token optimizer should integrate through a shared platform contract and a thin adapter/plugin on their side — not by bypassing policy, validation, or telemetry.

**Plugin classes allowed:**

```text
built-in strategies
experimental strategies
application-specific strategies
provider-specific strategies
third-party optimization plugins
```

**Conceptual extension interfaces** (not implemented in this architecture slice):

```text
TokenOptimizationPlugin
TokenOptimizationStrategy
ContentClassifier
ProtectedRegionDetector
TokenEstimator
OptimizationPolicyResolver
OptimizationValidator
ReceiptBuilder
```

**Plugin contract responsibilities:**

- declare plugin id and version,
- declare supported source types,
- declare supported strategies,
- declare whether optimization is lossless/lossy/reversible,
- declare safety class,
- declare required validators,
- accept a `TokenOptimizationRequest`-like input,
- return a `TokenOptimizationResult`-like output,
- preserve or report protected regions,
- report baseline/optimized token counts when available,
- report fallback reason when optimization is rejected,
- produce or attach a receipt reference,
- emit no private telemetry outside approved platform paths.

**Plugin boundaries:**

- plugins must not bypass platform policy,
- plugins must not bypass protected-region validation,
- plugins must not mutate canonical tool contracts,
- plugins must not create private telemetry buses,
- plugins must not export raw prompts, raw documents, secrets, tool args, or raw RAG chunks,
- plugins must be observable through the Harness Observability Spine or approved domain-signal path,
- plugins must fallback safely when validation fails.

**Example third-party integration flow:**

```text
third-party optimizer
  → thin Intergrax adapter
  → shared TokenOptimizationPlugin contract
  → platform policy resolver
  → protected-region validator
  → receipt + observability
  → optimized payload or fallback
```

### 8.6 Benchmark and public claim model

Intergrax must **not** make global unsupported claims such as:

```text
saves 95% tokens everywhere
```

Public claims must be tied to measured workflows and validation status, for example:

```text
Up to 95% fewer tokens on measured tool-output-heavy workflows.
```

**Required claim fields:**

```text
workflow_id
workload description
baseline_tokens
optimized_tokens
saved_tokens
saved_ratio
model
provider
runtime profile
optimization profile
strategies applied
validation status
fallback status
quality/regression status
receipt references
known limitations
```

**Confidence levels:**

| Level | Use |
|-------|-----|
| `measured` | Baseline and optimized counts captured on the same workflow with receipts and validation — **only level allowed in public proof tables**. |
| `estimated` | Projected from partial categories or sampled steps; not for headline public claims. |
| `projected` | Model-based forecast without full workflow replay. |
| `not comparable` | Baseline and optimized runs differ in model, profile, or workload shape. |

Claims without `measured` confidence, passing validation, and receipt references must not appear in public proof artifacts.

### 8.7 First public proof mechanism selection

The first public Token Optimization proof should prefer safe and measurable mechanisms:

```text
tool output compaction
tool catalog/schema compaction
RAG/context-pack light compression
cache alignment where measurable
output policy only where comparison is explicit
```

Defer until protected-region validation, receipts, telemetry, and regression gates are in place:

```text
deeper semantic compression
persistent memory compression
aggressive lossy strategies
experimental third-party plugins without full validator coverage
```

This selection aligns with LKW proof workflows **LKW-TOK-W1**–**W4** defined in the feature plan; see [`../plan/TOKEN_OPTIMIZATION.md`](../plan/TOKEN_OPTIMIZATION.md) §LKW-PF6-0.

---

## 9. Protected region policy

Token Optimization must preserve exact text for:

- fenced code blocks,
- inline code,
- file paths,
- URLs,
- API names,
- class names,
- function names,
- commands,
- environment variables,
- identifiers,
- enum values,
- hashes,
- dates,
- version numbers,
- exact error strings,
- security warnings,
- irreversible action confirmations,
- legal/compliance text unless explicit lossy summarization is allowed.

Compression should be disabled or downgraded for:

- destructive operations,
- security warnings,
- legal/compliance analysis,
- migration steps where sequence ambiguity matters,
- strict structured output schemas,
- evidence-heavy audit output,
- citation-sensitive answers,
- user clarification requests,
- previous misunderstanding caused by compression.

Persistent compression rule:

```text
Never overwrite live source before validation.
Use staging output → validate → receipt → atomic replace → rollback metadata.
```

---

## 10. Context Engineering lifecycle extension

Token Optimization extends the Context Engineering lifecycle after ranking and budgeting, before final formatting/preflight:

```text
COLLECT
  → NORMALIZE
  → SCORE
  → FILTER
  → RANK
  → BUDGET
  → TOKEN_OPTIMIZE
  → FORMAT
  → TOKEN_PREFLIGHT
  → VALIDATE
  → EMIT
```

Rules:

- ranking happens before lossy compression,
- policy checks happen before and after optimization,
- token estimates are recalculated after optimization,
- validation can force fallback to original content,
- provenance records both original source and compression receipt,
- existing `ContextCompiler`, `ContextBudgetPolicy`, `DegradationLadder`, and adapter-token preflight are extended, not duplicated.

---

## 11. Tool schema optimization rules

`ToolSchemaOptimizer` may compress:

- tool `description` fields,
- natural-language examples,
- repeated prose in tool catalog presentation,
- LLM-facing planner hints that are not schema semantics.

It must preserve:

- tool names,
- parameter names,
- enum values,
- required fields,
- JSON schema semantics,
- command strings,
- exact error examples where required.

It must not compress by default:

- tool call payloads,
- tool result JSON,
- strict schema definitions.

The optimizer must produce a compact LLM-facing catalog view, not mutate the canonical `ToolContract` registry.

---

## 12. Observability requirements

Token Optimization emits through the Harness Observability Spine. It must not create a private telemetry channel.

Preferred implementation:

- typed diagnostic payloads for receipts and optimization summaries,
- domain signal / event-kind style telemetry where possible,
- metrics derived from the same optimization result data,
- compatibility with unified run journal.

Candidate event kinds / domain signals:

```text
TOKEN_OPTIMIZATION_STARTED
TOKEN_OPTIMIZATION_APPLIED
TOKEN_OPTIMIZATION_BYPASSED
TOKEN_OPTIMIZATION_FAILED
TOKEN_OPTIMIZATION_RECEIPT_CREATED
TOKEN_BUDGET_EXCEEDED
TOKEN_REGRESSION_DETECTED
```

Candidate counters:

```text
input_tokens_raw_total
input_tokens_after_dedup_total
input_tokens_after_compression_total
output_tokens_budgeted_total
output_tokens_actual_total
saved_tokens_total
saved_cost_estimate_total
compression_attempts_total
compression_failures_total
compression_fallbacks_total
token_budget_overflow_total
token_regression_failures_total
```

Savings must be attributable by:

- run,
- step,
- tenant,
- agent,
- model,
- provider,
- source type,
- compression strategy,
- output profile.

---

## 13. First-class implementation invariants

1. Do not build a second LLM token counter. Use `LLMAdapter.count_messages_tokens()` when an adapter is in scope.
2. Do not build a second context compiler. Extend the existing Context Engineering pipeline.
3. Do not mutate canonical tool contracts for compact schema presentation.
4. Do not compress tool call payloads by default.
5. Do not overwrite memory or documentation summaries without staging, validation, receipt, and rollback metadata.
6. Do not report token savings without quality/safety validation.
7. Do not treat output terseness as sufficient token optimization.
8. Do not apply adaptive compression automatically until telemetry and policy governance are in place.

---

## 14. Architecture adoption rule

Feature architecture coordinates cross-layer behavior. Domain architecture remains authoritative for domain-owned implementation details.

When Token Optimization requires concrete implementation in a domain:

1. Update this feature architecture if the cross-layer contract changes.
2. Update the affected `docs/architecture/<DOMAIN>.md` file.
3. Add implementation rows to the affected `docs/plan/<DOMAIN>.md` file.
4. Keep `docs/features/plan/TOKEN_OPTIMIZATION.md` as the coordination map.
