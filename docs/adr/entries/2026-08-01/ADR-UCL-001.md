# ADR-UCL-001 — Unified Context Lifecycle Ownership, Single-Budget Authority and Versioned Context Projections

| Field | Value |
|-------|-------|
| **Status** | Proposed / Ready for Review |
| **Date** | 2026-08-01 |
| **Deciders** | Platform architecture review |
| **Related** | [`UNIFIED_CONTEXT_LIFECYCLE.md`](../../architecture/UNIFIED_CONTEXT_LIFECYCLE.md) · [`plan/UNIFIED_CONTEXT_LIFECYCLE.md`](../../plan/UNIFIED_CONTEXT_LIFECYCLE.md) · [ADR-MEM-001](../2026-06-08/ADR-MEM-001.md) · [ADR-CTX-001](../2026-06-12/ADR-CTX-001.md) |

## Context

Platform audit identified multiple independent context-reduction authorities without a shared lifecycle model. **TOKEN-10E-ARCH-1** and early **TOKEN_OPTIMIZATION** section 8.10 text assigned durable persistence and activation to the application host and described a direct Application to Token Optimization runtime path. That conflicts with Memory/Session revision contracts, Context Engineering single-budget authority, and Nexus lifecycle coordination.

**CTX-UCL-ARCH-1-R1** reconciles these domains before runtime implementation (**CTX-UCL-1**) begins.

## Decision

Freeze **one canonical Unified Context Lifecycle (UCL)** architecture:

1. **Memory/Session** owns durable conversation ledger, immutable context projections, CAS activation, and rollback execution.
2. **Context Engineering** owns the single global input budget per model invocation, final compilation, and final model-facing integrity validation orchestration.
3. **Token Optimization** owns transformation executors, candidate validation contracts, receipts, and cache-lineage calculation — not revision persistence or activation.
4. **Nexus** coordinates ephemeral assembly and durable compaction without owning storage or algorithms.
5. **Application host** owns configuration, authorization, adapter wiring, and review/rollback UX — not a parallel revision model or direct activation bypassing Memory/Session.

**TOKEN-10E-1** MUST NOT begin until **CTX-UCL-CLOSEOUT-1** is accepted/closed.

## Domain ownership

| Domain | Owns | Does not own |
|--------|------|--------------|
| **Memory/Session** | ConversationLedger; raw messages/events; stable IDs; sequence numbers; tool relationships; attachments/metadata; SessionContextRevision manifests; ActiveContextRevisionPointer; revision persistence; CAS activation; rollback execution; retention/archival; optimistic concurrency | Global prompt budget; optimization algorithms; strategy selection; model-facing composition; application authorization/UX |
| **Context Engineering** | One global input budget; ContextPlan; segment classification; final ChatMessage[]; final compilation; final model-facing integrity validation orchestration; token-window preflight | Durable persistence; active revision pointer; optimization implementation; application policy persistence |
| **Token Optimization** | Strategy catalog; policy gates; typed artifact executors; candidate construction; pipeline composition; text protected-region validation; message-sequence structural validation contracts; measurements; receipts; safe reporting; cache attribution; cache-lineage transition calculation | ConversationLedger; SessionContextRevision persistence; ActiveContextRevisionPointer; revision activation; rollback execution; global budget; authorization; retention |
| **Nexus** | Snapshot acquisition; CE ContextPlan orchestration; resolved policy delivery; TOKEN-10D timing; optional TO invocation; validation sequencing; final CE compilation; final integrity check; preflight; adapter invocation; durable activation request dispatch to Memory/Session | Storage implementation; compression algorithms; tenant policy persistence; product UX |
| **Application host** | Profile selection; tenant configuration; authorization; feature opt-in; human review UX; rollback UX; retention configuration; persistence adapter selection/wiring; product presentation | Private session revision model; private active revision pointer; parallel compression engine; parallel global budget; direct activation bypassing Memory/Session |

## ConversationLedger vs SessionContextRevision

| Concept | Definition |
|---------|------------|
| **ConversationLedger** | Append-only raw record of conversation messages and events. Never replaced by compaction. Subject only to explicit retention or archival policy. |
| **SessionContextRevision** | Immutable model-facing context projection manifest. References raw ledger ranges, compacted artifacts, pinned segments, and recent tail. Does not rewrite or delete the raw ConversationLedger. |
| **ActiveContextRevisionPointer** | Tenant/session-scoped pointer to the currently active model-facing projection. Updated through compare-and-swap. |
| **Rollback** | Changes ActiveContextRevisionPointer to an eligible prior SessionContextRevision. Does not erase, rewrite, or time-travel the ConversationLedger. |

Preferred activation contract name: **SessionContextRevisionActivationRequest** (supersedes provisional SessionRevisionActivationRequest).

## EPHEMERAL_ASSEMBLY

Single model-call flow coordinated by Nexus:

Memory/Session: resolve ActiveContextRevisionPointer, load ConversationSnapshot
Nexus: start lifecycle coordination
CE: collect sources, resolve one global ContextPlan
Nexus: resolve ContextOptimizationPolicy, TOKEN-10D timing gate when applicable
TO: execute approved transformations via typed artifact executor
candidate validation (schema, structural, protected, quality, policy)
CE: final compilation into model-facing ChatMessage[]
FINAL MODEL-FACING INTEGRITY VALIDATION
verify_context_preflight / token-window validation
exact-send materialization and hash verification
LLM Adapter invocation

Rules: never changes ActiveContextRevisionPointer; never creates durable revision by default; no content transformation after final model-facing integrity validation.

## DURABLE_COMPACTION

immutable baseline SessionContextRevision
to ContextPlan-selected MessageSequenceArtifact target
to TOKEN-10D timing/policy decision
to TO candidate construction
to candidate schema/structural/protected/quality validation
to new immutable SessionContextRevision manifest
to receipt + rollback metadata
to Memory/Session CAS (expected ActiveContextRevisionPointer == baseline)
to activate candidate or return conflict
to cache-lineage transition
to safe audit event

Rules: no in-place mutation; no direct Application activation; no hidden retry after CAS conflict; no durable activation through Token Optimization itself.

## Single-budget authority

Context Engineering is the **sole** global input budget authority per model invocation. No history provider, HistoryLayer, or optimization layer may hold a second independent global budget.

## Typed artifact executor boundary

Token Optimization executor framework:
- TextArtifactExecutor: existing string-based pipeline (compatible)
- MessageSequenceArtifactExecutor: CTX-UCL-4 (conversation history compaction)
- FragmentSetArtifactExecutor
- ToolCatalogArtifactExecutor
- StructuredDataArtifactExecutor

TOKEN-10E conversation history compaction **requires** MessageSequenceArtifactExecutor. TOKEN-10E may not flatten the complete conversation to one string or use line-level deduplication as the structural history engine.

## Validation ordering

Two distinct levels:

1. **Candidate validation** (after TO candidate creation): schema, structural, protected-region, quality, policy.
2. **Final model-facing integrity validation** (after final CE compilation, before token preflight/exact send): message order, roles, stable IDs, tool linkage, citation/evidence groups, protected values, system/developer/platform blocks, current user turn, recent tail, exact-send hash, tool-envelope consistency.

verify_context_preflight is a budget/window check only; it does not replace structural or protected-content integrity validation.

## Concurrency and activation

- Durable activation: CAS on expected_active_revision_id via Memory/Session.
- Application host authorizes and presents UX; Memory/Session executes activation and rollback.

## Cache-lineage separation

Content revision, prompt prefix identity, cache lineage, and provider cache observation remain separate dimensions.

## Consequences

### Positive

- One lifecycle architecture eliminates competing ownership and duplicate budgets.
- Distinct ledger vs projection model enables durable compaction without silent history loss.
- TOKEN-10E integrates as a bounded TO contribution under UCL rather than a parallel lifecycle.

### Negative

- Requires CTX-UCL-1 through CTX-UCL-6 and closeout before TOKEN-10E-1.
- ADR-MEM-001 Context Compiler semantics are superseded where they conflict with UCL.

## Migration impact

Application-owned persistence/activation wording migrates to Memory/Session ownership with Application adapter wiring and UX.

## Rejected alternatives

1. Application-owned revision persistence and activation.
2. Token Optimization as platform owner of context versions and activation.
3. TOKEN-10E wrapping the existing string pipeline for full conversation history.
4. Final structural validation before final CE compilation.
5. TOKEN-10E-1 starting after CTX-UCL-1 only (without closeout).

## Relationship to ADR-MEM-001 and ADR-CTX-001

- **ADR-MEM-001** remains historical/canonical for existing ContextCompiler semantics but is superseded where it conflicts with UCL.
- **ADR-CTX-001** remains valid where not superseded by UCL.

## Compliance

- Tier boundaries preserved.
- Linked architecture and plan docs updated in CTX-UCL-ARCH-1-R1.

## Implementation notes

- Documentation and guardrail tests only in CTX-UCL-ARCH-1-R1.
- Next step after ADR acceptance: **CTX-UCL-1**.
