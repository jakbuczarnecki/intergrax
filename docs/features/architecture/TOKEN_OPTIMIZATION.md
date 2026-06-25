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

## 1. Purpose

Token Optimization is a multi-layer Intergrax capability that minimizes unnecessary token usage while preserving correctness, safety, provenance, auditability, and developer control.

It is not a prompt-only brevity mode. It is a runtime-controlled token economy capability spanning:

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
docs/features/plan/TOKEN_OPTIMIZATION.md
```

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
| `LLM_ADAPTERS` | Provides tokenizer-consistent counting, context window metadata, output budget metadata, and cost/latency signals. |
| `TOOLS` | Owns `ToolSchemaOptimizer` and compact tool catalog presentation. |
| `MEMORY` | Owns `MemorySummaryCompressor`, safe persistent compression, rollback metadata, and memory receipt storage. |
| `RAG` | Allows post-retrieval/post-ranking chunk compression where citations and grounding remain intact. |
| `OBSERVABILITY` | Owns token optimization events, counters, spans, and savings attribution. |
| `UNIFIED_EXECUTION_RUNTIME` | Resolves runtime token policy, compression level, output profile, and safety bypass rules. |
| `AGENT_CONTRACTS_AND_ASSEMBLY` | Allows agent-level hints for output profile and context compactness without manual prompt assembly. |
| `ADAPTIVE_HARNESS_INTELLIGENCE` | Later learns optimal budgets and compression strategies from telemetry. |

---

## 6. Core capability model

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

Planned runtime components:

```text
TokenOptimizer
OutputPolicyResolver
ToolSchemaOptimizer
MemorySummaryCompressor
ContextPackOptimizer
CompressionReceiptValidator
TokenOptimizationTelemetryEmitter
TokenRegressionBenchmarkRunner
```

---

## 7. Protected region policy

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

## 8. Context Engineering lifecycle extension

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
- provenance records both original source and compression receipt.

---

## 9. Tool schema optimization rules

`ToolSchemaOptimizer` may compress:

- tool `description` fields,
- natural-language examples,
- repeated prose in tool catalog presentation.

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

---

## 10. Observability requirements

Candidate events:

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

## 11. Architecture adoption rule

Feature architecture coordinates cross-layer behavior. Domain architecture remains authoritative for domain-owned implementation details.

When Token Optimization requires concrete implementation in a domain:

1. Update this feature architecture if the cross-layer contract changes.
2. Update the affected `docs/architecture/<DOMAIN>.md` file.
3. Add implementation rows to the affected `docs/plan/<DOMAIN>.md` file.
4. Keep `docs/features/plan/TOKEN_OPTIMIZATION.md` as the coordination map.
