<!--
© Artur Czarnecki. All rights reserved.
Intergrax framework – proprietary and confidential.
Use, modification, or distribution without written permission is prohibited.
-->

# Token Optimization

**Status:** Draft architecture proposal  
**Layer type:** Cross-cutting runtime capability / Context Engineering extension  
**Target maturity:** L3 production runtime, L4 adaptive optimization later  
**Primary owner:** `CONTEXT_ENGINEERING`  
**Secondary owners:** `LLM_ADAPTERS`, `TOOLS`, `MEMORY`, `RAG`, `OBSERVABILITY`, `UNIFIED_EXECUTION_RUNTIME`, `AGENT_CONTRACTS_AND_ASSEMBLY`, `ADAPTIVE_HARNESS_INTELLIGENCE`  
**Reference inspiration:** [`JuliusBrussee/caveman`](https://github.com/JuliusBrussee/caveman) — used as a market and mechanism reference, not as an implementation dependency.

---

## 1. Purpose

Token Optimization is the Intergrax runtime capability that minimizes unnecessary token usage while preserving correctness, safety, provenance, auditability, and developer control.

The goal is not to make agents merely "speak shorter". The goal is to make the whole execution path token-aware:

```text
user/task input
  → context collection
  → context ranking
  → context compression
  → tool schema injection
  → memory injection
  → model routing
  → output shaping
  → telemetry + receipt
```

Token Optimization must answer four questions for every model call:

1. Which tokens are necessary for this step?
2. Which tokens are redundant, stale, low-value, or unsafe to include?
3. Which tokens can be compressed without losing required semantics?
4. How much cost was saved, and did quality remain acceptable?

---

## 2. Strategic product rationale

Modern agent systems waste large amounts of tokens through:

- repeated system instructions,
- verbose tool descriptions,
- large MCP/tool schemas,
- unbounded conversation history,
- oversized RAG chunks,
- low-value memory injection,
- verbose agent narration,
- debug logs copied into prompts,
- repeated architectural documentation,
- subagent exploration loops,
- lack of per-step context budgeting.

Intergrax can turn token efficiency into a market advantage by making token usage:

- explicit,
- budgeted,
- policy-controlled,
- measurable,
- comparable across runs,
- safe by default,
- visible in observability dashboards.

The product claim should not be:

```text
Intergrax makes prompts shorter.
```

The product claim should be:

```text
Intergrax provides runtime-controlled token economy for governed AI agents.
```

---

## 3. Reference: Caveman mechanisms worth learning from

The `JuliusBrussee/caveman` project proves there is strong demand for token-efficient agent operation. Its core ideas are relevant to Intergrax, but Intergrax should implement them as architecture-level runtime capabilities rather than style-only prompt tricks.

Useful mechanisms:

| Caveman mechanism | Intergrax interpretation |
|-------------------|--------------------------|
| Terse output modes | Runtime `OutputPolicy` with named profiles |
| Memory file compression | Validated `MemorySummaryCompressor` / `DocumentCompressor` |
| MCP tool description shrinker | `ToolSchemaOptimizer` for tool catalogs and descriptions |
| Token savings stats | `TokenUsageTelemetry` and dashboard events |
| Safety exceptions for clarity | Policy-governed compression bypass rules |
| Preserve code, paths, URLs, identifiers | Protected-region validation |

Important distinction:

```text
Caveman = prompt-level brevity plugin.
Intergrax = architecture-level token economy engine.
```

Intergrax should not copy the "caveman speak" style. It should extract the underlying mechanisms and implement them as professional runtime policies.

---

## 4. Scope

### 4.1 In scope

Token Optimization covers:

- input token budgeting,
- context assembly efficiency,
- semantic and structural context compression,
- memory summary compression,
- tool description/schema compression,
- output verbosity control,
- token-aware model routing signals,
- token savings telemetry,
- compression receipts,
- regression benchmarks for tokens vs quality,
- policy-governed compression safety.

### 4.2 Out of scope

Token Optimization does not:

- alter model reasoning tokens directly,
- compress private chain-of-thought,
- mutate executable code,
- rewrite JSON schemas structurally unless explicitly safe,
- remove required audit evidence,
- replace RAG ranking,
- replace memory lifecycle management,
- replace LLM adapter token counting,
- replace model selection logic.

---

## 5. Design principles

| Principle | Meaning |
|----------|---------|
| Budget-first | Every LLM call has an explicit input and output budget. |
| Quality-preserving | Token savings are invalid if answer quality or safety drops below threshold. |
| Policy-governed | Compression may be disabled or relaxed by runtime policy. |
| Step-aware | Different execution steps use different context and output profiles. |
| Source-aware | RAG, memory, tools, history, attachments, and policy text compress differently. |
| Protected regions | Code, paths, URLs, API names, error strings, IDs, hashes, dates, and legal/security warnings are preserved. |
| Observable | Every optimization emits structured telemetry and can be audited. |
| Reversible where possible | Persistent compression writes backups and receipts. |
| Fail-safe | If validation fails, use original content or lower compression level. |
| No silent degradation | Dropped or compressed content must be represented in provenance or receipt data. |

---

## 6. Relationship to existing architecture

Token Optimization should be implemented as a cross-cutting capability anchored in `CONTEXT_ENGINEERING`.

Existing Context Engineering already owns:

- context collection,
- ranking,
- budgeting,
- formatting,
- validation,
- provenance,
- degradation steps.

Token Optimization extends this with:

- first-class compression strategies,
- output policy contracts,
- tool catalog optimization,
- persistent memory/document compression,
- token savings telemetry,
- token-efficiency regression gates.

### 6.1 Primary layer integrations

| Layer | Required integration |
|------|----------------------|
| `CONTEXT_ENGINEERING` | Add token optimization stage to context lifecycle and budget allocation. |
| `LLM_ADAPTERS` | Use adapter-native tokenizer / context window / cost metadata. |
| `TOOLS` | Optimize tool descriptions and tool catalog injection. |
| `MEMORY` | Compress long-term memory summaries and session summaries safely. |
| `RAG` | Compress or select retrieved chunks after ranking, not before source relevance is known. |
| `OBSERVABILITY` | Emit token optimization events, counters, spans, receipts. |
| `UNIFIED_EXECUTION_RUNTIME` | Enforce token policies through runtime profiles and safety gates. |
| `AGENT_CONTRACTS_AND_ASSEMBLY` | Allow agents to declare output mode and context compactness requirements. |
| `ADAPTIVE_HARNESS_INTELLIGENCE` | Later use telemetry to learn optimal budgets and compression strategies. |

---

## 7. Domain boundaries

| Concern | Owner | Token Optimization role |
|--------|-------|-------------------------|
| Model tokenizer and context window | `LLM_ADAPTERS` | Consume adapter metadata and token counters. |
| Context candidate collection | `CONTEXT_ENGINEERING` | Do not collect; optimize after collection and ranking. |
| Context ranking | `CONTEXT_ENGINEERING` / later `AHI` | Respect ranking; do not invent relevance. |
| Tool execution | `TOOLS` | Do not modify call payloads or execution results unless explicitly safe. |
| Tool catalog presentation | `TOOLS` + Token Optimization | Compress descriptions and examples. |
| Memory persistence | `MEMORY` | Provide validated compressed summaries, not raw mutation without receipt. |
| RAG retrieval | `RAG` | Compress selected chunks only after retrieval/ranking. |
| Policy decisions | `UNIFIED_EXECUTION_RUNTIME` | Decide when compression is forbidden or capped. |
| Telemetry | `OBSERVABILITY` | Emit token savings, risk, validation, quality metrics. |

Anti-patterns:

- Agent code manually slices prompts.
- Tool result JSON is compressed before parsing.
- Memory files are overwritten before validation.
- Output terseness is controlled only by prompt wording.
- Compression removes citations, paths, errors, IDs, or safety warnings.
- Token savings are reported without quality regression checks.

---

## 8. Capability model

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

### 8.1 OutputPolicy

Controls how much the agent says after the model has enough information.

Suggested profiles:

| Profile | Purpose |
|---------|---------|
| `minimal` | Status-only, ≤6 lines, no explanation unless required. |
| `terse` | Default operator update, concise but complete. |
| `standard` | Normal user-facing explanation. |
| `full` | Detailed answer, design discussion, or documentation. |
| `audit` | Full findings, evidence, risks, recommendations. |
| `machine_receipt` | Structured output intended for automation. |
| `debug_verbose` | Detailed logs and internal diagnostics when explicitly requested. |

OutputPolicy should be selected by:

- user instruction,
- agent contract,
- runtime profile,
- step kind,
- safety/compliance policy,
- available output budget.

### 8.2 CompressionPolicy

Controls whether and how content is compressed.

Suggested levels:

| Level | Meaning |
|------|---------|
| `none` | Preserve exact content. |
| `light` | Remove filler and redundant prose only. |
| `structural` | Compact headings, bullets, repeated phrasing, examples. |
| `semantic` | Summarize meaning while preserving required facts. |
| `aggressive` | Use only for low-risk, low-audit, non-legal/non-security content. |

### 8.3 ProtectedRegionPolicy

The optimizer must preserve exact text for:

- fenced code blocks,
- inline code,
- file paths,
- URLs,
- API names,
- class/function names,
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

### 8.4 Safety bypass rules

Compression should be disabled or downgraded for:

- destructive operations,
- security warnings,
- legal/compliance analysis,
- medical/financial high-stakes advice,
- migration steps where sequence ambiguity matters,
- user asks for clarification,
- previous compressed answer caused misunderstanding,
- structured outputs with strict schema,
- evidence/citation-heavy audit output.

---

## 9. Core runtime components

### 9.1 `TokenOptimizer`

Central service that applies policy-driven optimization.

Responsibilities:

- resolve effective token policy,
- choose compression strategy,
- apply protected-region parsing,
- run compression,
- validate result,
- produce receipt,
- emit telemetry.

Pseudo-contract:

```python
@dataclass(frozen=True, slots=True)
class TokenOptimizationRequest:
    run_id: str
    step_id: str
    tenant_id: str
    source_type: str
    content: str
    policy: TokenOptimizationPolicy
    model_id: str | None = None
    step_kind: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class TokenOptimizationResult:
    content: str
    receipt: CompressionReceipt
    telemetry: TokenOptimizationTelemetry
    validation: TokenOptimizationValidation
```

### 9.2 `OutputPolicyResolver`

Resolves output profile for a model call.

Inputs:

- user request,
- agent contract,
- runtime environment profile,
- task type,
- step kind,
- safety gates,
- output budget.

Output:

```python
@dataclass(frozen=True, slots=True)
class OutputPolicy:
    profile: Literal[
        "minimal",
        "terse",
        "standard",
        "full",
        "audit",
        "machine_receipt",
        "debug_verbose",
    ]
    max_output_tokens: int | None
    require_sections: tuple[str, ...] = ()
    forbid_sections: tuple[str, ...] = ()
    preserve_exact: tuple[str, ...] = ()
```

### 9.3 `ToolSchemaOptimizer`

Optimizes tool catalog presentation for LLM context.

Rules:

- may compress `description` fields,
- may compress natural-language examples,
- must preserve tool names,
- must preserve parameter names,
- must preserve enum values,
- must preserve required fields,
- must preserve JSON schema semantics,
- must not compress tool call payloads by default,
- must not compress tool results unless result type allows it.

Primary win:

```text
large tool catalog → compact runtime catalog → same tool selection quality with fewer tokens
```

### 9.4 `MemorySummaryCompressor`

Compresses memory summaries and persistent natural-language memory blocks.

Rules:

- no live overwrite before validation,
- write to staging target first,
- validate protected regions,
- validate semantic equivalence if lossy,
- store original hash,
- store compressed hash,
- store receipt,
- allow rollback.

### 9.5 `ContextPackOptimizer`

Optimizes assembled context after collection/ranking and before final formatting/preflight.

It should work on `ContextFragment` objects, not raw prompt strings.

Responsibilities:

- compress low-risk fragments,
- preserve mandatory fragments,
- reduce repeated policy/system text,
- merge equivalent fragments,
- compact RAG snippets only after ranking,
- attach compression metadata to provenance.

### 9.6 `CompressionReceipt`

Every persistent or runtime compression should be auditable.

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
    validation_status: Literal["passed", "failed", "bypassed"]
    quality_score: float | None = None
    fallback_used: bool = False
```

---

## 10. Context Engineering lifecycle update

Current conceptual lifecycle:

```text
COLLECT → NORMALIZE → SCORE → FILTER → RANK → BUDGET → COMPRESS → FORMAT → VALIDATE → EMIT
```

Token Optimization makes `COMPRESS` first-class and observable:

```text
COLLECT
  → NORMALIZE
  → SCORE
  → FILTER
  → RANK
  → BUDGET
  → TOKEN_OPTIMIZE
      - dedup repeated content
      - compress allowed fragments
      - preserve protected regions
      - produce receipts
      - update token estimates
  → FORMAT
  → TOKEN_PREFLIGHT
  → VALIDATE
  → EMIT
```

Important ordering:

- ranking happens before lossy compression,
- policy checks happen before and after optimization,
- token estimates are recalculated after optimization,
- validation can force fallback to original content,
- provenance records both original source and compression receipt.

---

## 11. Observability model

### 11.1 Events

Add events:

```text
TOKEN_OPTIMIZATION_STARTED
TOKEN_OPTIMIZATION_APPLIED
TOKEN_OPTIMIZATION_BYPASSED
TOKEN_OPTIMIZATION_FAILED
TOKEN_OPTIMIZATION_RECEIPT_CREATED
TOKEN_BUDGET_EXCEEDED
TOKEN_REGRESSION_DETECTED
```

### 11.2 Metrics

Track:

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

Recommended dimensions:

- tenant_id,
- run_id,
- agent_id,
- model_id,
- provider,
- step_kind,
- source_type,
- compression_strategy,
- output_profile.

---

## 12. Quality and regression gates

Token optimization must never be judged only by savings.

Required evaluation dimensions:

| Dimension | Required check |
|----------|----------------|
| Token savings | Before/after token count. |
| Semantic preservation | Meaning preserved above configured threshold. |
| Protected regions | Exact preservation of code, paths, IDs, URLs, errors. |
| Tool selection | Same or better tool selection accuracy. |
| RAG answer quality | Same citation coverage / answer correctness. |
| Runtime safety | No removal of warnings, confirmations, policy text. |
| Latency | Compression overhead does not exceed configured limit. |
| Cost | Compression cost must not exceed saved cost except where quality requires it. |

Regression gate examples:

```text
uv run python scripts/check_token_optimization_contracts.py
uv run python scripts/check_tool_schema_optimizer.py
uv run python scripts/check_compression_receipts.py
uv run pytest tests/unit/token_optimization/ -q
uv run pytest tests/unit/runtime/nexus/context/ -q
```

---

## 13. Implementation phases

### TOKEN-1 — Architecture and contracts

Goal: establish canonical architecture and contracts.

Tasks:

- Add this architecture doc or convert it into canonical layer doc.
- Add ADR for Token Optimization domain boundary.
- Add plan hub `docs/plan/TOKEN_OPTIMIZATION.md`.
- Add audit slice `docs/guides/audit_slices/TOKEN_OPTIMIZATION.md`.
- Add contract stubs for policies, receipts, telemetry.

Exit criteria:

- architecture, plan, ADR, audit slice aligned,
- no runtime behavior changed yet,
- Cursor can audit layer scope without reading full repo.

### TOKEN-2 — OutputPolicy runtime

Goal: replace prompt-only verbosity control with runtime policy.

Tasks:

- Add `OutputPolicy` contract.
- Add `OutputPolicyResolver`.
- Wire policy into LLM call path where max output tokens / response profile are available.
- Add minimal/terse/standard/full/audit profiles.
- Add safety bypass for irreversible/security/legal contexts.

Exit criteria:

- agent output verbosity can be controlled from runtime profile,
- default operator replies use terse mode,
- audit/full mode remains available explicitly.

### TOKEN-3 — ToolSchemaOptimizer

Goal: reduce recurring tool catalog token cost.

Tasks:

- Add optimizer for tool descriptions.
- Preserve schema semantics and parameter names.
- Add tests proving JSON schema does not change structurally.
- Add telemetry for tool catalog savings.
- Add opt-in profile for runtime compact catalogs.

Exit criteria:

- compact tool catalog generated safely,
- tool descriptions shorter,
- no tool call schema breakage,
- receipts/telemetry emitted.

### TOKEN-4 — ContextPackOptimizer

Goal: optimize selected context fragments before formatting.

Tasks:

- Add token optimization stage after rank/budget and before format.
- Attach compression receipt metadata to fragment provenance.
- Recalculate token estimates after compression.
- Add fallback to original fragments when validation fails.
- Add source-specific strategies for memory, RAG, tool output, history, policy text.

Exit criteria:

- assembled context can be compacted safely,
- mandatory/policy fragments preserved,
- total token count decreases in benchmark tasks,
- quality gate passes.

### TOKEN-5 — MemorySummaryCompressor

Goal: safely compress persistent natural-language memory and documentation context.

Tasks:

- Add staging write + validation + atomic commit flow.
- Add backup and rollback metadata.
- Add protected-region validator.
- Add semantic equivalence validator for lossy summaries.
- Add receipts stored with memory metadata.

Exit criteria:

- no live overwrite before validation,
- corrupted compression cannot replace source,
- receipt enables audit and rollback.

### TOKEN-6 — Telemetry, dashboards, benchmarks

Goal: make token savings measurable and product-visible.

Tasks:

- Add events and counters.
- Add token savings calculation per run/step/source.
- Add regression benchmark dataset.
- Add CI checks for token regression.
- Add documentation for interpreting savings vs quality.

Exit criteria:

- every optimized model call reports before/after token data,
- regression checks fail on uncontrolled token growth,
- savings can be aggregated per agent/model/tenant.

### TOKEN-7 — Adaptive optimization

Goal: use historical telemetry to choose optimal budgets and compression strategies.

Tasks:

- Feed token telemetry to `ADAPTIVE_HARNESS_INTELLIGENCE`.
- Learn preferred budgets by task/step/source/model.
- Route simple steps to compact profiles automatically.
- Escalate to full context when quality drops.

Exit criteria:

- runtime adapts token budgets based on observed quality/cost,
- optimization remains policy-governed,
- manual override remains available.

---

## 14. Required documentation updates after accepting this architecture

Cursor should update these documents in scoped PRs, not all at once.

### 14.1 Architecture docs

Update:

- `docs/architecture/CONTEXT_ENGINEERING.md`
- `docs/architecture/LLM_ADAPTERS.md`
- `docs/architecture/TOOLS.md`
- `docs/architecture/MEMORY.md`
- `docs/architecture/RAG.md`
- `docs/architecture/OBSERVABILITY.md`
- `docs/architecture/UNIFIED_EXECUTION_RUNTIME.md`
- `docs/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md`
- `docs/architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md`
- `docs/intergrax_runtime_architecture.md`
- `docs/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md`

### 14.2 Plan docs

Create/update:

- `docs/plan/TOKEN_OPTIMIZATION.md`
- `docs/plan/CONTEXT_ENGINEERING.md`
- `docs/plan/LLM_ADAPTERS.md`
- `docs/plan/TOOLS.md`
- `docs/plan/MEMORY.md`
- `docs/plan/RAG.md`
- `docs/plan/OBSERVABILITY.md`
- `docs/plan/UNIFIED_EXECUTION_RUNTIME.md`
- `docs/plan/AGENT_CONTRACTS_AND_ASSEMBLY.md`
- `docs/plan/ADAPTIVE_HARNESS_INTELLIGENCE.md`

### 14.3 ADRs

Create:

- `ADR-TOKEN-001` — Token Optimization domain boundary and ownership.
- `ADR-TOKEN-002` — Compression receipts and protected-region validation.
- `ADR-TOKEN-003` — Tool schema optimization safety model.

### 14.4 Tests and gates

Add future gates:

- `scripts/check_token_optimization_contracts.py`
- `scripts/check_compression_receipts.py`
- `scripts/check_tool_schema_optimizer.py`
- `scripts/check_output_policy_wiring.py`
- `tests/unit/token_optimization/`
- `tests/unit/runtime/nexus/context/test_context_pack_optimizer.py`
- `tests/unit/tools/test_tool_schema_optimizer.py`
- `tests/unit/memory/test_memory_summary_compressor.py`

---

## 15. Cursor implementation handoff

### 15.1 First Cursor task: architecture sync only

Recommended prompt:

```text
Pracujemy na repozytorium `jakbuczarnecki/intergrax`, branch `development`.

Cel sesji:
Na podstawie `docs/architecture/TOKEN_OPTIMIZATION.md` wykonaj wyłącznie architecture/plan sync bez implementacji runtime.

Zakres:
1. Przeczytaj `docs/architecture/TOKEN_OPTIMIZATION.md`.
2. Przeczytaj tylko read-scope blocks powiązanych warstw:
   - CONTEXT_ENGINEERING
   - LLM_ADAPTERS
   - TOOLS
   - MEMORY
   - OBSERVABILITY
   - UNIFIED_EXECUTION_RUNTIME
3. Zaproponuj minimalny zestaw zmian dokumentacyjnych wymagany, aby Token Optimization stał się formalnym elementem architektury Intergrax.
4. Nie implementuj kodu.
5. Nie czytaj całego repozytorium.
6. Przygotuj plan PR-ów, każdy PR ma mieć jeden cel i zamykać jeden zakres.

Wynik:
- lista plików do zmiany,
- kolejność PR-ów,
- ryzyka,
- czy `TOKEN_OPTIMIZATION` powinno być osobną warstwą, czy podwarstwą `CONTEXT_ENGINEERING`,
- propozycja ID zadań planistycznych TOKEN-1..TOKEN-7.
```

### 15.2 Second Cursor task: canonical docs

```text
Na podstawie zatwierdzonego planu z poprzedniego kroku:

1. Zaktualizuj `docs/architecture/CONTEXT_ENGINEERING.md`, aby zawierał formalne miejsce dla Token Optimization stage.
2. Utwórz `docs/plan/TOKEN_OPTIMIZATION.md` jako plan hub.
3. Dodaj ADR-TOKEN-001.
4. Nie implementuj kodu.
5. Zachowaj token budget rules: read only scoped sections, no repo-wide exploration.
6. Uruchom tylko dokumentacyjne/check scripts, jeżeli istnieją.
```

### 15.3 Third Cursor task: first implementation slice

```text
Zaimplementuj TOKEN-2 OutputPolicy runtime jako pierwszy mały slice.

Zakres:
- contracts only,
- resolver,
- tests,
- no ToolSchemaOptimizer yet,
- no MemorySummaryCompressor yet,
- no adaptive logic yet.

Wymagania:
- RuntimeState examples must always provide explicit `run_id`.
- Any new Python file must start with the Intergrax copyright header.
- Use `TraceLevel` enum for trace events where applicable.
- One focused PR.
```

---

## 16. Recommended final target state

Token Optimization is complete when Intergrax can show, for every agent run:

```text
Raw input tokens:           120,000
Selected context tokens:     32,000
Compressed context tokens:   21,000
Output budget:                2,000
Actual output:                  740
Saved tokens:                99,260
Quality score:                 0.94
Compression receipts:             7
Protected region failures:        0
Fallbacks used:                    1
```

This turns token efficiency from an internal engineering trick into a visible platform capability.

---

## 17. Decision recommendation

Recommendation:

```text
Accept Token Optimization as a cross-cutting Intergrax capability anchored in CONTEXT_ENGINEERING.
```

Do not implement it as a single prompt rule.

Do implement it as:

1. runtime policy,
2. context compiler extension,
3. tool catalog optimizer,
4. memory/document compressor,
5. telemetry and receipts layer,
6. regression-tested token economy.

This creates a stronger market advantage than Caveman-style terse output alone.
