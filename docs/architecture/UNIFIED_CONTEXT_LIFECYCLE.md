# Unified Context Lifecycle

**Status:** Correction delivered / ready for review (**CTX-UCL-ARCH-1-R1**)
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)
**Plan (1:1):** [`plan/UNIFIED_CONTEXT_LIFECYCLE.md`](../plan/UNIFIED_CONTEXT_LIFECYCLE.md)
**Related:** [`CONTEXT_ENGINEERING.md`](CONTEXT_ENGINEERING.md) · [`MEMORY.md`](MEMORY.md) · [`features/architecture/TOKEN_OPTIMIZATION.md`](../features/architecture/TOKEN_OPTIMIZATION.md) · [`NEXUS_EXECUTION_FLOW.md`](NEXUS_EXECUTION_FLOW.md)
**ADR:** [`ADR-UCL-001`](../adr/entries/2026-08-01/ADR-UCL-001.md) (UCL ownership, single-budget authority, versioned projections — Proposed / Ready for Review) · [`ADR-MEM-001`](../adr/entries/2026-06-08/ADR-MEM-001.md) (Context Compiler — superseded where UCL conflicts)
**Last architecture pass:** 2026-08-01 — **CTX-UCL-ARCH-1-R1** ownership and TOKEN-10E reconciliation

---

## 1. Status and ownership

| Item | Value |
|------|-------|
| Task | **CTX-UCL-ARCH-1-R1** — ownership, flow, and TOKEN-10E reconciliation (supersedes CTX-UCL-ARCH-1 delivery gaps) |
| Prior pass | **CTX-UCL-ARCH-1** — cross-domain architecture freeze (accepted elements preserved) |
| Runtime implementation | **Not started** |
| TOKEN-10E implementation | **Blocked** pending **CTX-UCL-CLOSEOUT-1** accepted/closed |
| Owning domains | **MEMORY** (durable ledger/revisions) · **CONTEXT_ENGINEERING** (single budget authority) · **TOKEN_OPTIMIZATION** (transformation executor) · **NEXUS** (lifecycle coordinator) · **APPLICATION_HOSTING** (config/auth/UX) |
| Supersedes | **TOKEN-10E-ARCH-1** as standalone compaction architecture — durable compaction now defined under UCL + TOKEN-10E |

---

## 2. Purpose

Platform audit identified **multiple independent context-reduction authorities** operating without a shared lifecycle model. Before **TOKEN-10E** (durable in-cache compaction) implementation begins, this document freezes:

1. **One durable source of truth** for conversation history (Memory/Session).
2. **One global input budget authority** (Context Engineering).
3. **One transformation executor** for optimization strategies (Token Optimization).
4. **One lifecycle coordinator** (Nexus) for ephemeral assembly and durable compaction.
5. **Two explicit modes:** `EPHEMERAL_ASSEMBLY` and `DURABLE_COMPACTION`.

This is **architecture and documentation only**. No runtime code, storage migration, or public API changes are in scope for CTX-UCL-ARCH-1.

---

## 3. Current-state audit

Audit performed by tracing call sites and wiring (2026-08-01). **Existence of code or configuration does not imply production hot-path activity.** **Documented mechanism count: 19** (table rows below; prior delivery report incorrectly cited 17).

| Component / file | Owning domain today | Modifies durable history | Model-facing only | Operates on | Preserves message IDs | Preserves roles | Preserves tool linkage | Protected-region validation | Receipt | Rollback | Concurrency protection | Hot path | Classification | Target decision |
|------------------|---------------------|--------------------------|-------------------|-------------|----------------------|-----------------|------------------------|----------------------------|---------|----------|------------------------|----------|----------------|-----------------|
| `ConversationalMemory._trim_if_needed` (`intergrax/memory/conversational_memory.py`) | Memory | **Yes** (FIFO drop) | No | messages | **No** (dropped) | Yes (remaining) | Not validated | No | No | No | `threading.RLock` on aggregate | Active when `max_messages` set | retention-only | **retain** as retention; classify explicitly; not token optimization |
| `InMemorySessionStorage` FIFO (`intergrax/runtime/nexus/session/in_memory_session_storage.py`) | Memory / Session | **Yes** | No | messages | **No** (dropped) | Yes | Not validated | No | No | No | per-session dict | Dev/test default path | retention-only | **retain** dev/test only; document as retention |
| `SqliteConversationalMemoryStore` max_messages (`intergrax/memory/stores/sqlite_conversational_memory_store.py`) | Memory | **Yes** | No | messages | Partial | Yes | Not validated | No | No | No | store-dependent | Active on SQLite conv path | retention-only | **adapt** — retention policy, not compaction |
| `SessionManager.append_message` docs (`session_manager.py`) | Memory | Delegates to storage | No | messages | Depends on store | Yes | Depends on store | No | No | No | storage-dependent | Active | canonical write path | **retain** — append-only contract target |
| `fragments_from_session_history` (`intergrax/context/providers/legacy_bridge.py`) | Context Engineering | No | **Yes** | strings (`role: text`) | **No** (synthetic `source_id`) | Partial (flattened) | **No** | No | No | No | None | **Active** — `builtin.session_history` on CE collect | legacy/compatibility | **replace** — structured `ConversationSnapshot` + refs |
| `builtin.session_history` max_entries (`intergrax/context/providers/builtin.py`) | Context Engineering | No | **Yes** | `messages[-N:]` → fragments | **No** | Partial | **No** | No | No | No | None | **Active** on CE graph/UAEP when handle wired | legacy slicing | **deprecate** pre-`ContextPlan` slicing |
| `DefaultContextFormatter` (`intergrax/context/formatter.py`) | Context Engineering | No | **Yes** | synthetic system strings | **No** | Lost in flatten | **No** | No | No | No | None | Active on CE format path | compatibility flattening | **adapt** — injection format only; not canonical history |
| `HistoryLayer` (`intergrax/runtime/nexus/context/engine_history_layer.py`) | Nexus (legacy) | No (ephemeral to `state.base_history`) | **Yes** | tokens/messages | Partial | Yes | Not validated | No | No | No | None | **Dormant** — constructed in `RuntimeContext.build()` but `build_base_history()` has **no production call sites** | legacy | **deprecate** → UCL `EPHEMERAL_ASSEMBLY` |
| `HistoryCompressionStrategy` (`intergrax/runtime/nexus/responses/response_schema.py`) | Nexus / Application | No | Intended model-facing | enum | N/A | N/A | N/A | No | No | No | None | **Dormant** with HistoryLayer; mapped from `legal_application` bridge | legacy config | **adapt** — map to `ContextOptimizationPolicy` |
| `ContextCompiler` + `DegradationLadder` (`context_compiler.py`, `degradation_ladder.py`) | Context Engineering | No | **Yes** | tokens/messages | Partial | Yes | Partial (index-based) | No | No | No | None | **Active** — ACP `compile_service`, `DefaultNexusContextEngine.assemble` | canonical budget path (partial) | **adapt** — sole global budget under UCL; trim rules frozen below |
| `TOKENIZER_HARD_TRIM` (`degradation_ladder.py`) | Context Engineering | No | **Yes** | tokens/strings | Partial | Yes | **No** | No | No | No | None | Active as last ladder step | legacy overflow behavior | **replace** — `trim_safe` segments only; fail-closed otherwise |
| `ContextManager` legacy char trim (`context_manager.py`) | Context Engineering | No | **Yes** | chars | Partial | Yes | Not validated | No | No | No | None | Active when engine not wired | legacy | **deprecate** after CE-3 full wiring |
| `verify_context_preflight` (`context_preflight.py`) | Context Engineering | No | Validates only | tokens | N/A | N/A | N/A | No | No | No | None | Active post-compile | canonical preflight | **retain** — final boundary before adapter |
| Token Optimization pipeline (`intergrax/runtime/token_optimization/pipeline.py`) | Token Optimization | No | **Yes** | strings (`TextArtifact`) | N/A | N/A | N/A | **Yes** (string) | **Yes** | Metadata only | None | Active when `CacheAwareTokenOptimizationRuntime.run()` invoked | canonical executor (text) | **retain** + extend `MessageSequenceArtifact` |
| `protected_regions.py` | Token Optimization | No | Validates | strings | N/A | N/A | N/A | **Yes** | Feeds receipt | No | None | Active in pipeline | canonical (text) | **retain** + structural validator for messages |
| `budget_aware_packing` / `context_pack.py` | Token Optimization | No | **Yes** | chars/fragments | N/A | N/A | N/A | Partial | Via pipeline | No | None | Active in pipeline layers | prototype | **retain** under CE budget allocation |
| `semantic_compression_enabled` + metadata (`context_runtime_bridge.py`, `applications/_shared/context_runtime_bridge.py`) | Application / Runtime wiring | No | Intended | config metadata | N/A | N/A | N/A | No | No | No | None | **Config-only** — no runtime consumer reads `semantic_compression.v1` | dormant / misleading | **adapt** — fail-fast or default-off (migration task) |
| `SessionMemoryConsolidationService._prepare_conversation_for_prompt` | User profile / LTM | No | **Yes** (prompt slice) | messages/chars | Partial | Yes | Not validated | No | No | No | None | Active on consolidation path | **separate concern** | **retain** — not conversation compaction |
| `ConversationalMemory.get_for_model(native_tools=True)` | Memory | No | **Yes** | messages | Yes | Filtered | Strips tool msgs | No | No | No | lock | Active when native_tools path used | model-presentation | **retain** — adapter presentation, not compaction |

### 3.1 Duplicate authorities (highest risk)

1. **Pre-`ContextPlan` history slicing** (`messages[-N:]`) vs **ContextCompiler global budget** vs **dormant HistoryLayer token compression**.
2. **Storage FIFO retention** presented as bounded memory but indistinguishable from optimization in ops.
3. **`semantic_compression_enabled`** in product defaults without runtime consumer.
4. **String flattening** (`role: text` fragments) blocks structural validation required for durable compaction.

---

## 4. Risk register

| ID | Risk | Severity | Mitigation (UCL) |
|----|------|----------|------------------|
| R1 | Silent durable loss via storage FIFO | High | Classify retention-only; separate from model-facing compaction |
| R2 | Competing global budgets (HistoryLayer + ContextCompiler) | High | CE sole budget authority; HistoryLayer deprecated |
| R3 | TOKEN-10E adds third compaction engine | High | UCL foundation before TOKEN-10E-1 |
| R4 | `semantic_compression_enabled` false advertising | Medium | Fail-fast or default-off migration |
| R5 | `TOKENIZER_HARD_TRIM` cuts protected content | High | `trim_safe` only + `MANDATORY_CONTEXT_EXCEEDS_MODEL_LIMIT` |
| R6 | No revision CAS → lost updates on durable compaction | High | CTX-UCL-2 activation contracts |
| R7 | String-only optimization cannot validate tool groups | High | CTX-UCL-4 `MessageSequenceArtifact` + structural validation |

---

## 5. Domain ownership

### 5.1 Memory / Session — durable source of truth

**Owns:** `ConversationLedger` (append-only raw messages and events); stable message IDs; sequence numbers; tool-call/tool-result relationships; attachments and message metadata; `SessionContextRevision` manifests; `ActiveContextRevisionPointer`; revision persistence contracts; revision activation through compare-and-swap; rollback execution; retention and archival lifecycle; optimistic concurrency.

**Does not own:** global prompt budget; optimization algorithms; strategy selection; model-facing context composition; application authorization or UX.

**Rule:** Context compression for the model must not be a hidden side effect of `append_message()`. Storage-level retention is not token optimization. Compaction does not rewrite or delete the raw `ConversationLedger`.

### 5.2 Context Engineering — single budget authority

**Owns:** one global input budget per model invocation; per-source budget allocation; segment classification (`mandatory`, `protected`, `compressible`, `droppable`, `trim_safe`); `ContextPlan`; final model-facing `ChatMessage[]`; final compilation; final model-facing integrity validation orchestration; token-window preflight (`verify_context_preflight`).

**Does not own:** durable conversation persistence; active revision pointer; optimization strategy implementation; application policy persistence.

**Rule:** No history provider or optimization layer may hold a second independent global budget.

### 5.3 Token Optimization — transformation executor

**Owns:** optimization strategy catalog; policy gates used by transformation execution; typed artifact executors; candidate construction; transformation pipeline composition; text protected-region validation; message-sequence structural validation contracts; measurements; receipts; safe reporting; cache attribution; cache-lineage transition calculation.

**Does not own:** `ConversationLedger`; `SessionContextRevision` persistence; `ActiveContextRevisionPointer`; revision activation; rollback execution; global prompt budget; application authorization; retention.

### 5.4 Nexus — lifecycle coordinator

**Owns:** snapshot acquisition; CE `ContextPlan` orchestration; resolved policy delivery; TOKEN-10D timing decision; optional Token Optimization invocation; validation sequencing; final CE compilation; final model-facing integrity check; preflight; adapter invocation; dispatch of durable activation request to Memory/Session.

**Does not own:** storage implementation; compression algorithms; tenant policy persistence; product UX.

### 5.5 Application host — configuration, authorization, UX

**Owns:** profile selection; tenant configuration; authorization; feature opt-in; human review UX; rollback UX; retention configuration; persistence adapter selection and wiring; product-specific presentation.

**Does not own:** a private session revision model; a private active revision pointer; a parallel compression engine; a parallel global budget; direct activation logic bypassing Memory/Session contracts.

**Normalization path:**

```text
ApplicationEnvironmentProfile
  → neutral RuntimeEnvironmentProfile
  → resolved ContextOptimizationPolicy
```

Canonical bridge: `intergrax/runtime/wiring/context_runtime_bridge.py`.
Compatibility shim: `intergrax/applications/_shared/context_runtime_bridge.py`.

### 5.6 Memory consolidation — separate concern

`SessionMemoryConsolidationService` and LTM consolidation remain **out of scope** for conversation context compaction. Distinct lifecycles:

| Concern | Owner |
|---------|-------|
| Conversation context compaction (ephemeral/durable) | UCL / CE + TO |
| User LTM extraction | Memory lifecycle |
| Session summary stored as memory | Memory lifecycle |
| Retention / archival | Memory / Session |
| Model-facing ephemeral reduction | CE + TO under Nexus |

---

## 6. Unified Context Lifecycle

### 6.1 Ephemeral assembly flow (`EPHEMERAL_ASSEMBLY` — canonical model call)

```text
Memory/Session:
  resolve ActiveContextRevisionPointer
        ↓
  load immutable ConversationSnapshot / structural references
        ↓
Nexus:
  start lifecycle coordination
        ↓
Context Engineering:
  collect sources
  resolve one global ContextPlan
        ↓
Nexus:
  resolve ContextOptimizationPolicy
  invoke TOKEN-10D timing gate when applicable
        ↓
Token Optimization:
  execute only approved transformations
  using the correct typed artifact executor
        ↓
candidate-level schema validation
candidate-level structural validation
candidate-level protected-region validation
candidate-level quality validation
candidate-level policy validation
        ↓
Context Engineering:
  final compilation into exact model-facing ChatMessage[]
        ↓
FINAL MODEL-FACING INTEGRITY VALIDATION
        ↓
verify_context_preflight / token-window validation
        ↓
exact-send materialization and hash/integrity verification
        ↓
LLM Adapter invocation
```

**Rules:** `EPHEMERAL_ASSEMBLY` never changes `ActiveContextRevisionPointer`; never creates a durable revision by default; no content transformation after final model-facing integrity validation.

### 6.2 Durable compaction flow (`DURABLE_COMPACTION` — TOKEN-10E scope)

```text
immutable baseline SessionContextRevision
        ↓
ContextPlan-selected MessageSequenceArtifact target
        ↓
TOKEN-10D timing/policy decision
        ↓
Token Optimization candidate construction
        ↓
candidate schema / structural / protected / quality validation
        ↓
new immutable SessionContextRevision manifest
        ↓
receipt + rollback metadata
        ↓
Memory/Session CAS:
  expected ActiveContextRevisionPointer == baseline revision
        ↓
activate candidate revision or return conflict
        ↓
cache-lineage transition
        ↓
safe audit event
```

**Rules:** no in-place mutation; no direct Application activation; no hidden retry after CAS conflict; no durable activation through Token Optimization itself.

---

## 7. Ephemeral assembly (`EPHEMERAL_ASSEMBLY`)

| Property | Rule |
|----------|------|
| Scope | Single model call |
| Active revision | Unchanged |
| Durable history | Not overwritten |
| Transforms | May select, skip, or temporarily compress approved segments |
| Output | Model-facing assembled context only |
| Legacy replacement | Replaces `HistoryLayer` responsibility on request path |

`HistoryLayer.build_base_history()` is **not** the canonical path. Ephemeral assembly flows through CE collection → `ContextPlan` → optional TO → CE compilation.

---

## 8. Durable compaction (`DURABLE_COMPACTION`)

| Property | Rule |
|----------|------|
| Baseline | Immutable revision |
| Candidate | Separate revision; no in-place mutation of active revision |
| Validation | Structural + protected + quality; receipt when policy requires |
| Activation | Compare-and-swap on active revision pointer |
| Ledger | Original append-only ledger preserved or referenced |
| Cache | Separate cache-lineage transition |
| Implementation | **CTX-UCL-1…6** then **TOKEN-10E-1…4** |

---

## 9. Canonical data model (contracts — not implemented)

### 9.0 Core durable concepts

| Concept | Definition |
|---------|------------|
| **`ConversationLedger`** | Append-only raw record of conversation messages and events. Never replaced by compaction. Subject only to explicit retention or archival policy. |
| **`SessionContextRevision`** | Immutable model-facing context projection manifest. References raw ledger ranges, compacted artifacts, pinned segments, and recent tail. Does not rewrite or delete the raw `ConversationLedger`. |
| **`ActiveContextRevisionPointer`** | Tenant/session-scoped pointer to the currently active model-facing projection. Updated through compare-and-swap. |
| **Rollback** | Changes `ActiveContextRevisionPointer` to an eligible prior `SessionContextRevision`. Does not erase, rewrite, or time-travel the `ConversationLedger`. |

### 9.1 `ConversationSnapshot`

| Field | Purpose |
|-------|---------|
| `tenant_id`, `session_id`, `revision_id`, `parent_revision_id` | Identity and lineage |
| `message or segment references` | Paginated structural view |
| `sequence range` | Ledger slice |
| `model_facing_hash`, `ledger_hash`, `active_prefix_hash` | Integrity |
| `cache_lineage_ref` | Cache attribution |
| `created_at` | Audit |

### 9.2 `ConversationMessageRef`

`entry_id`, `role`, `sequence_no`, `created_at`, `tool_call_id`, tool linkage, attachments, `metadata_hash`, `content_hash`.

### 9.3 `MessageGroup`

Atomic groups: user+assistant exchange; assistant tool call + tool results; pinned decision block; citation/evidence group; protected recent tail.

### 9.4 `ContextPlan`

`resolved_global_budget`, per-source allocation, required/protected/compressible/droppable/trim_safe segment IDs, target sizes, allowed strategy classes, final validation requirements.

### 9.5 `ContextOptimizationPolicy`

Normalized contract (not independent flags): `enabled`, `mode` (`EPHEMERAL` | `DURABLE`), `allow_lossy`, `allowed_targets`, `allowed_strategies`, `require_receipt`, `require_rollback_metadata`, `require_human_review`, recent-tail policy, protected-region policy, quality thresholds, cache policy, retention policy reference.

### 9.6 `OptimizationArtifact` (typed union)

| Variant | Executor |
|---------|----------|
| `TextArtifact` | `TextArtifactExecutor` — existing string pipeline (compatible) |
| `MessageSequenceArtifact` | `MessageSequenceArtifactExecutor` — TOKEN-10E / CTX-UCL-4 (conversation history; not string flattening) |
| `FragmentSetArtifact` | CE fragment sets |
| `ToolCatalogArtifact` | Tool schema optimization |
| `StructuredDataArtifact` | JSON/tabular payloads |

### 9.7 `OptimizationCandidate`

`operation_id`, `idempotency_key`, `baseline_revision_id`, `baseline_hash`, artifact type, candidate artifact, `policy_version`, strategy trace, validation results, measurement, receipt ref, rollback metadata ref, cache-lineage transition, status.

### 9.8 `SessionContextRevisionActivationRequest`

`tenant_id`, `session_id`, `expected_active_revision_id`, `candidate_revision_id`, `operation_id`, `idempotency_key`. Activation succeeds only when active revision matches baseline. Memory/Session owns activation execution; Application host authorizes and wires the persistence adapter.

**Compatibility:** provisional name `SessionRevisionActivationRequest` — superseded before implementation.

---

## 10. Policy normalization

### 10.1 `HistoryCompressionStrategy` compatibility map

| Legacy | UCL mapping |
|--------|-------------|
| `OFF` | Optimization policy disabled |
| `TRUNCATE_OLDEST` | CE selection/degradation on full message groups |
| `SUMMARIZE_OLDEST` | Structured `MessageSequence` optimization strategy |
| `HYBRID` | **Deprecated** — require explicit composed pipeline |

### 10.2 `semantic_compression_enabled`

Until real runtime wiring: production preset must not declare an active function that does not execute; configuration defaults off **or** strict mode fails at startup with `UNSUPPORTED_CONTEXT_OPTIMIZATION_CONFIGURATION`. Migration task — not CTX-UCL-ARCH-1 implementation.

---

## 11. Structural and protected validation

### 11.1 Candidate validation (Token Optimization boundary)

Runs immediately after Token Optimization candidate creation:

```text
schema validation
structural validation
protected-region validation
quality validation
policy validation
```

**Purpose:** determine whether the proposed transformed artifact is acceptable before CE final compilation.

Required for `MessageSequenceArtifact` (not satisfied by string substring validator alone):

- Message IDs preserved when element not replaced
- Roles, order preserved
- No orphaned tool results; no tool calls without required results
- Attachments, citation/evidence groups, atomic message groups preserved
- Pinned decisions and active unresolved commitments preserved
- Recent tail preserved per policy
- Tenant/session identity unchanged

Text `protected_regions.py` remains valid for `TextArtifact` only.

### 11.2 Final model-facing integrity validation (Context Engineering boundary)

Runs **after** final CE compilation and **before** token preflight / exact send:

```text
final message order
roles
stable message identity where applicable
tool-call/tool-result linkage
required citation/evidence groups
protected values
system/developer/platform blocks
current user turn
recent tail
exact-send message hash
tool-envelope consistency
no post-validation transformation
```

**Purpose:** prove that formatting, merging, and compilation did not invalidate candidate-level safety guarantees.

---

## 12. Final compilation and preflight

**Required ordering:**

```text
candidate validation
  → final CE compilation
  → final model-facing integrity validation
  → token-window preflight (verify_context_preflight)
  → exact-send materialization
  → adapter invocation
```

**Final validation rule:** After final model-facing integrity validation, no further content transformation is permitted before adapter invocation.

`verify_context_preflight` is a budget/window check and must not be described as a replacement for structural or protected-content integrity validation.

`TOKENIZER_HARD_TRIM` (or equivalent):

- May operate **only** on segments marked `trim_safe`
- Must not cut: system/developer/platform instructions, current user turn, tool-call/result groups, pinned decisions, exact errors, code blocks, citations, protected identifiers, active recent tail
- If mandatory content exceeds model limit → **fail closed** with `MANDATORY_CONTEXT_EXCEEDS_MODEL_LIMIT`

`verify_context_preflight` remains the last token check before adapter call.

---

## 13. Storage and revision architecture

### 13.1 Append-only conversation ledger (`ConversationLedger` — preferred)

```text
tenant_id
session_id
sequence_no
message_id
message payload/reference
```

Do not model durable history as one growing JSON blob rewritten on each append.

### 13.2 Session context revision manifest (`SessionContextRevision`)

New revision references existing raw ledger ranges and summary artifacts — does not copy full conversation and does not mutate the `ConversationLedger`.

### 13.3 Content-addressed artifacts

Summary and transformed artifacts carry stable hashes; deduplication enabled.

### 13.4 Optimistic concurrency

Revision activation via compare-and-swap. May use existing backend-neutral conditional-store capability — no vendor-specific dependency.

### 13.5 Idempotency

Every durable operation: `operation_id`, `idempotency_key`, baseline revision, `policy_version`, strategy version.

### 13.6 Pagination

Snapshot API supports ranges and pagination — no full multi-year history load per request.

### 13.7 Multi-tenant safety

All snapshot, candidate, activation, and rollback operations are tenant-scoped.

---

## 14. Concurrency and idempotency

| Operation | Mechanism |
|-----------|-----------|
| Durable activation | CAS on `expected_active_revision_id` |
| Duplicate operation | `IDEMPOTENCY_CONFLICT` |
| Stale baseline | `STALE_CONTEXT_REVISION` |
| Concurrent activation | `ACTIVATION_CONFLICT` |

Rollback execution owner: **Memory/Session** (with Application host UX).

---

## 15. Cache lineage

Explicit separation:

| Concept | Meaning |
|---------|---------|
| Content revision | Durable conversation state |
| Prompt prefix identity | Stable prefix hash for assembly |
| Cache lineage | Logical lineage for attribution |
| Provider cache observation | Reported evidence only |

**Forbidden claims:** mutating provider KV cache; deleting provider cache; knowing TTL without provider report; guaranteeing cache hit; equating content savings with cache savings.

---

## 16. Observability and safe reporting

Safe event model — **no raw conversation content** in standard events.

| Event | Allowed fields |
|-------|----------------|
| `snapshot_created` | IDs, revision, hashes, counts |
| `context_plan_resolved` | budget, segment counts, policy version |
| `optimization_requested` | operation_id, mode, strategy IDs |
| `candidate_generated` / `candidate_rejected` | status, reason codes, measurements |
| `validation_completed` | validation type, pass/fail, reason codes |
| `activation_succeeded` / `activation_conflict` | revision IDs, operation_id |
| `rollback_requested` / `rollback_completed` | revision IDs, status |
| `cache_lineage_changed` | lineage refs, hashes |

---

## 17. Failure semantics (fail-closed reason codes)

| Reason code | When |
|-------------|------|
| `UNSUPPORTED_CONTEXT_OPTIMIZATION_CONFIGURATION` | Config declares unsupported optimization |
| `MANDATORY_CONTEXT_EXCEEDS_MODEL_LIMIT` | Mandatory segments exceed model window |
| `STALE_CONTEXT_REVISION` | Baseline revision no longer active |
| `STRUCTURAL_VALIDATION_FAILED` | Message structure invariant violated |
| `PROTECTED_REGION_VALIDATION_FAILED` | Protected content altered |
| `QUALITY_VALIDATION_FAILED` | Quality threshold not met |
| `RECEIPT_REQUIRED` | Policy requires receipt; none produced |
| `ROLLBACK_METADATA_REQUIRED` | Policy requires rollback metadata |
| `REVIEW_REQUIRED` | Human review required |
| `ACTIVATION_CONFLICT` | CAS activation failed |
| `IDEMPOTENCY_CONFLICT` | Duplicate durable operation |
| `POLICY_BLOCKED` | Policy gate blocked transform |

**No hidden fallback** to another lossy strategy without explicit result entry and receipt.

---

## 18. Legacy migration

| Mechanism | Decision |
|-----------|----------|
| `HistoryLayer` | Legacy; frozen for new code; remove from canonical runtime construction after call-graph confirmation; compatibility path only |
| `HistoryCompressionStrategy` | Map to `ContextOptimizationPolicy` (§10.1) |
| `semantic_compression_enabled` | Fail-fast or default-off (separate migration) |
| Provider `messages[-N:]` slicing | Legacy — canonical provider delivers structural snapshot |
| Context formatter flattening | Injection format only — not canonical history for optimization |
| Storage FIFO trimming | Retention-only classification |
| Duplicate bridges | Single normalization direction (§5.5) |

---

## 19. Scalability model

See §13. Production model: append-only ledger, revision manifests, content-addressed artifacts, paginated snapshots, tenant-scoped operations, CAS activation. Long-running sessions do not require full in-memory history per request.

---

## 20. Security and multi-tenancy

- Every snapshot, candidate, activation, and rollback is `tenant_id`-scoped.
- Application host enforces authorization before durable activation opt-in.
- Safe events exclude raw prompts and thread content.
- Cross-tenant access to revision or ledger data is forbidden.

---

## 21. Explicit invariants

1. One durable ledger owner (Memory/Session).
2. One global input budget owner per model call (CE).
3. One optimization executor (TO) — no parallel application compression engines.
4. Nexus coordinates; does not implement algorithms or storage.
5. Ephemeral assembly never mutates active revision.
6. Durable compaction never mutates active revision in place.
7. Final CE output is immutable before adapter invocation.
8. LTM consolidation ≠ conversation compaction.
9. Retention ≠ token optimization.
10. Cache lineage ≠ content revision.

---

## 22. Rejected alternatives

1. Second universal compression engine beside CE.
2. Dual global budgets in HistoryLayer and ContextCompiler.
3. Storage-driven prompt optimization.
4. In-place active history overwrite.
5. LTM consolidation as conversation compaction.
6. String-only representation of full conversation for structural optimization.
7. Provider-specific durable compaction contracts.
8. Hidden fallbacks between lossy strategies.
9. Final hard trim after protected-region validation on non-`trim_safe` segments.
10. Application-specific parallel compression systems.
11. Separate UCL runtime independent of Nexus, CE, and TO.

---

## 23. Out of scope (CTX-UCL-ARCH-1)

No Python runtime, public exports, SessionStorage changes, ContextCompiler changes, HistoryLayer removal, application preset changes, fail-fast config implementation, revision storage, CAS implementation, `MessageSequenceArtifact` implementation, TOKEN-10E implementation, LKW integration, Slack integration, or live infrastructure.

---

## 24. Implementation decomposition

| ID | Scope | Status |
|----|-------|--------|
| **CTX-UCL-ARCH-1** | Cross-domain architecture freeze | **Correction delivered through CTX-UCL-ARCH-1-R1** |
| **CTX-UCL-ARCH-1-R1** | Ownership, flow, TOKEN-10E reconciliation, ADR-UCL-001 | **Ready for review** |
| **CTX-UCL-1** | Canonical policy and typed artifact contracts | Not started |
| **CTX-UCL-2** | `ConversationLedger`, snapshot, `SessionContextRevision`, activation contracts | Not started |
| **CTX-UCL-3** | Structured SessionHistoryProvider + single CE budget authority | Not started |
| **CTX-UCL-4** | `MessageSequenceArtifact` executor + structural validators | Not started |
| **CTX-UCL-5** | Ephemeral assembly integration on canonical runtime paths | Not started |
| **CTX-UCL-6** | Legacy HistoryLayer, profile, provider, bridge migration | Not started |
| **CTX-UCL-CLOSEOUT-1** | Cross-domain runtime and documentation closeout | Not started |

**Dependency gate (canonical):** `CTX-UCL-ARCH-1-R1` → accepted → `CTX-UCL-1` … `CTX-UCL-6` → `CTX-UCL-CLOSEOUT-1` accepted/closed → **TOKEN-10E-1** may begin.

**After CTX-UCL-CLOSEOUT-1:**

| ID | Scope | Status |
|----|-------|--------|
| **TOKEN-10E-1** | Durable compaction contracts over UCL | Blocked |
| **TOKEN-10E-2** | Durable candidate construction over `MessageSequenceArtifact` | Blocked |
| **TOKEN-10E-3** | Receipts, rollback metadata, validation | Blocked |
| **TOKEN-10E-4** | Revision activation + cache-lineage transition | Blocked |
| **TOKEN-10E-CLOSEOUT-1** | Public contract freeze | Blocked |

**TOKEN-10F/G/H:** Planned (proof harness, hard gates, public promotion).

---

## 25. User-visible meaning

- **Users** retain full conversation history subject to retention policy; compaction does not silently delete durable turns without explicit durable mode + policy.
- **Review UX** (when `require_human_review`) shows candidate summary before activation — application host responsibility.
- **Rollback** restores prior active revision when policy and metadata support it.
- **Ephemeral assembly** may shorten what the model sees for one turn without changing stored history.
