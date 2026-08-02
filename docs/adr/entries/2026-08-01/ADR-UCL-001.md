# ADR-UCL-001 — Unified Context Lifecycle Ownership, Single-Budget Authority and Versioned Context Projections

| Field | Value |
|-------|-------|
| **Status** | Proposed / Ready for Review |
| **Date** | 2026-08-01 |
| **Deciders** | Platform architecture review |
| **Related** | [UNIFIED_CONTEXT_LIFECYCLE.md](../../../architecture/UNIFIED_CONTEXT_LIFECYCLE.md) · [plan/UNIFIED_CONTEXT_LIFECYCLE.md](../../../plan/UNIFIED_CONTEXT_LIFECYCLE.md) · [ADR-MEM-001](../2026-06-08/ADR-MEM-001.md) · [ADR-CTX-001](../2026-06-12/ADR-CTX-001.md) |

## Context

Platform audit identified multiple independent context-reduction authorities without a shared lifecycle model. **TOKEN-10E-ARCH-1** and early **TOKEN_OPTIMIZATION** section 8.10 text assigned durable persistence and activation to the application host and described a direct Application to Token Optimization runtime path. That conflicts with Memory/Session revision contracts, Context Engineering single-budget authority, and Nexus lifecycle coordination.

**CTX-UCL-ARCH-1-R4** extends prior reconciliation passes (R1/R2/R3) with internal optimization model-call boundary, single-flight artifact creation, and concrete repository delivery ownership before runtime implementation (**CTX-UCL-1**) begins.

## Decision

Freeze **one canonical Unified Context Lifecycle (UCL)** architecture:

1. **Memory/Session** owns durable conversation ledger, immutable context projections, CAS activation, rollback execution, and the Reusable Optimization Artifact Catalog (`OptimizationArtifactRepository` contracts and `InMemoryOptimizationArtifactRepository` reference implementation in CTX-UCL-2).
2. **Context Engineering** owns the single global input budget per model invocation, source/target requirements, final compilation, and final model-facing integrity validation orchestration.
3. **Token Optimization** owns transformation executors, candidate validation contracts, receipts, and cache-lineage calculation — not revision persistence, artifact catalog lookup, or activation. Token Optimization remains artifact creator, not repository owner.
4. **Nexus** coordinates ephemeral assembly and durable compaction, including lookup-before-create orchestration and creation reservation coordination, without owning storage or algorithms.
5. **Application host** owns configuration, authorization, adapter wiring, and review/rollback UX — not a parallel revision model, application-local summary cache, or direct activation bypassing Memory/Session.

**TOKEN-10E-1** MUST NOT begin until **CTX-UCL-CLOSEOUT-1** is accepted/closed.

## Reusable optimization artifact lifecycle

UCL uses **reuse-before-create**.

Valid artifacts are content-addressed and looked up before transformation via ArtifactLookupKey (tenant/session scope, source range, source_content_hash, artifact type, strategy/policy/validation versions, compression target, lossiness profile).

Identical compatible source **must not** trigger repeated LLM summarization. A lookup hit produces REUSE_ARTIFACT with llm_transform_invoked = false.

**Decision outcomes:** NO_OP, SELECT_ONLY, REUSE_ARTIFACT, CREATE_ARTIFACT, POLICY_BLOCKED, FAIL_CLOSED — every canonical model call traverses the decision point; Token Optimization execution remains optional.

**Ownership:**

- **Memory/Session** — artifact persistence contract, lookup contract, single-flight creation reservation, catalog lifecycle, invalidation persistence; references from SessionContextRevision; `InMemoryOptimizationArtifactRepository` reference implementation (CTX-UCL-2).
- **Nexus** — orchestrates lookup-before-create and reservation coordination; maps lookup result to REUSE_ARTIFACT or CREATE_ARTIFACT; prevents transform execution on valid reuse hit or when reservation is ALREADY_IN_PROGRESS.
- **Token Optimization** — creates artifacts only on CREATE_ARTIFACT (or explicit approved refresh) through typed executors; internal summarizer invocations are INTERNAL_OPTIMIZATION_CALL.
- **Context Engineering** — supplies source ranges and target requirements; compiles final prompt using raw groups or reusable artifacts.
- **Application host** — tenant policy configuration, authorization, review UX, artifact provenance visibility where required; does not implement its own summary cache.

SessionContextRevision references artifacts by ID or content hash without rewriting the raw ConversationLedger. Activating a new revision or rolling back must not regenerate artifact content.

Ephemeral assembly may persist reusable artifacts to the catalog when policy permits without changing ActiveContextRevisionPointer.

## Internal optimization model-call boundary

Every **primary model call** (`PRIMARY_MODEL_CALL`) traverses the full UCL optimization decision point.

An **internal optimization call** (`INTERNAL_OPTIMIZATION_CALL`) does not recursively traverse the full UCL optimization lifecycle for the same optimization target. Internal optimization calls use a bounded internal assembly path with explicit budgets, protected inputs, preflight, and telemetry — but without history summarization or artifact creation recursion.

`OptimizationExecutionGuard` enforces: `PRIMARY_MODEL_CALL` at `optimization_depth == 0`; first `INTERNAL_OPTIMIZATION_CALL` at `optimization_depth == 1`; same `ArtifactLookupKey` already active in operation ancestry → fail closed (`OPTIMIZATION_RECURSION_BLOCKED`); depth exceeded → `OPTIMIZATION_DEPTH_EXCEEDED`.

Internal summarization calls receive only explicitly selected source material and summarization instructions — not the complete conversation, unrelated RAG context, or application tool catalog.

## Single-flight artifact creation

For one compatible `ArtifactLookupKey`, the platform **MUST** allow at most one active artifact-creation execution at a time.

Content-addressed storage deduplication alone is insufficient because it does not prevent duplicate LLM calls.

`ArtifactCreationReservation` coordinates creation via `try_acquire_creation_reservation`. Non-owner callers on `ALREADY_IN_PROGRESS` do not invoke Token Optimization or the summarizer. Successful store and reservation completion must be atomic or observably ordered.

Two concurrent canonical model calls with the same compatible key and no existing artifact must result in exactly one CREATE_ARTIFACT execution, exactly one summarizer invocation, one validated stored artifact, and the second caller later reusing the stored artifact or returning an explicit defer result.

`ArtifactCreationCoordinationStatus` describes reservation/concurrency state separately from `ContextOptimizationDecision`.

## Repository delivery ownership

| Task | Delivery |
|------|----------|
| **CTX-UCL-1** | Contracts only: `ModelCallExecutionScope`, `OptimizationExecutionGuard`, `ArtifactCreationReservation`, reason codes — no repository implementation |
| **CTX-UCL-2** | `OptimizationArtifactRepository` interface + `InMemoryOptimizationArtifactRepository` reference implementation with single-flight reservation and concurrency tests |
| **TOKEN-10E-4** | First durable production `OptimizationArtifactRepository` adapter and durable `SessionContextRevision` activation integration (implementation may live in Memory/Session packages; delivery coordinated by TOKEN-10E-4) |

TOKEN-10E reuses UCL repository and reservation contracts. TOKEN-10E must not create a second repository or reservation mechanism.

## Domain ownership

| Domain | Owns | Does not own |
|--------|------|--------------|
| **Memory/Session** | ConversationLedger; raw messages/events; stable IDs; sequence numbers; tool relationships; attachments/metadata; SessionContextRevision manifests; ActiveContextRevisionPointer; revision persistence; CAS activation; rollback execution; retention/archival; optimistic concurrency; OptimizationArtifactRepository contracts; InMemoryOptimizationArtifactRepository (CTX-UCL-2); Reusable Optimization Artifact Catalog persistence and lookup | Global prompt budget; optimization algorithms; strategy selection; model-facing composition; application authorization/UX; optimization reuse decision for current model call |
| **Context Engineering** | One global input budget; ContextPlan; segment classification; source/target requirements; final ChatMessage[]; final compilation; final model-facing integrity validation orchestration; token-window preflight; internal-call budget classification (CTX-UCL-3) | Durable persistence; active revision pointer; optimization implementation; application policy persistence; artifact catalog lookup |
| **Token Optimization** | Strategy catalog; policy gates; typed artifact executors; artifact creation on CREATE_ARTIFACT; candidate construction; pipeline composition; text protected-region validation; message-sequence structural validation contracts; measurements; receipts; safe reporting; cache attribution; cache-lineage transition calculation | ConversationLedger; SessionContextRevision persistence; ActiveContextRevisionPointer; revision activation; rollback execution; global budget; authorization; retention; reusable artifact repository |
| **Nexus** | Snapshot acquisition; CE ContextPlan orchestration; resolved policy delivery; canonical optimization decision point; ArtifactLookupKey construction; lookup-before-create orchestration; creation reservation coordination; TOKEN-10D timing; optional TO invocation on CREATE_ARTIFACT only; validation sequencing; final CE compilation; final integrity check; preflight; adapter invocation; durable activation request dispatch to Memory/Session | Storage implementation; compression algorithms; tenant policy persistence; product UX |
| **Application host** | Profile selection; tenant configuration; authorization; feature opt-in; human review UX; rollback UX; retention configuration; persistence adapter selection/wiring; product presentation | Private session revision model; private active revision pointer; parallel compression engine; parallel global budget; direct activation bypassing Memory/Session; application-local summary cache |

## ConversationLedger vs SessionContextRevision

| Concept | Definition |
|---------|------------|
| **ConversationLedger** | Append-only raw record of conversation messages and events. Never replaced by compaction. Subject only to explicit retention or archival policy. |
| **SessionContextRevision** | Immutable model-facing context projection manifest. References raw ledger ranges, compacted artifacts, pinned segments, and recent tail. Does not rewrite or delete the raw ConversationLedger. |
| **ActiveContextRevisionPointer** | Tenant/session-scoped pointer to the currently active model-facing projection. Updated through compare-and-swap. |
| **Rollback** | Changes ActiveContextRevisionPointer to an eligible prior SessionContextRevision. Reuses artifact references from the prior revision. Does not erase, rewrite, or time-travel the ConversationLedger. |

Preferred activation contract name: **SessionContextRevisionActivationRequest** (supersedes provisional SessionRevisionActivationRequest).

## EPHEMERAL_ASSEMBLY

Single model-call flow coordinated by Nexus (`PRIMARY_MODEL_CALL`; every canonical model call traverses the optimization decision point):

Memory/Session: resolve ActiveContextRevisionPointer, load ConversationSnapshot
CE: collect sources, resolve one global ContextPlan
Nexus: determine whether optimization is required
  → NO_OP / SELECT_ONLY when budget satisfied without transformation
  → construct ArtifactLookupKey when optimization required
OptimizationArtifactRepository: lookup compatible artifact
  → REUSE_ARTIFACT (no LLM summarizer) on hit
  → try_acquire_creation_reservation on miss
    → CREATE_ARTIFACT → INTERNAL_OPTIMIZATION_CALL → TO typed executor (TOKEN-10D RUN when applicable) on ACQUIRED
    → wait/defer on ALREADY_IN_PROGRESS (no summarizer for non-owner)
CE: final compilation into model-facing ChatMessage[]
FINAL MODEL-FACING INTEGRITY VALIDATION
verify_context_preflight / token-window validation
exact-send materialization and hash verification
primary LLM adapter invocation

Rules: never changes ActiveContextRevisionPointer; never creates durable revision by default; no content transformation after final model-facing integrity validation; reusable artifact may be persisted to catalog when policy permits; internal summarizer result is not the primary application response.

## DURABLE_COMPACTION

active SessionContextRevision (immutable baseline)
→ ContextPlan-selected source range
→ ArtifactLookupKey
→ catalog lookup → REUSE_ARTIFACT or CREATE_ARTIFACT (with reservation coordination on miss)
→ new immutable SessionContextRevision manifest referencing artifact_id/hash
→ receipt + rollback metadata
→ Memory/Session CAS (expected ActiveContextRevisionPointer == baseline)
→ activate candidate or return conflict (no summary regeneration on activation)
→ cache-lineage transition
→ safe audit event

Rules: no in-place mutation; no direct Application activation; no hidden retry after CAS conflict; no durable activation through Token Optimization itself; rollback reuses prior artifact references.

## Single-budget authority

Context Engineering is the **sole** global input budget authority per model invocation. No history provider, HistoryLayer, or optimization layer may hold a second independent global budget.

## Typed artifact executor boundary

Token Optimization executor framework:
- TextArtifactExecutor: existing string-based pipeline (compatible)
- MessageSequenceArtifactExecutor: CTX-UCL-4 (conversation history compaction; invoked only on CREATE_ARTIFACT; INTERNAL_OPTIMIZATION_CALL for LLM summarizer)
- FragmentSetArtifactExecutor
- ToolCatalogArtifactExecutor
- StructuredDataArtifactExecutor

TOKEN-10E conversation history compaction **requires** MessageSequenceArtifactExecutor. TOKEN-10E may not flatten the complete conversation to one string or use line-level deduplication as the structural history engine.

## Validation ordering

Three distinct stages:

1. **Creation-time candidate validation** (after TO candidate creation on CREATE_ARTIFACT): schema, structural, protected-region, quality, policy.
2. **Reuse-time compatibility and integrity validation** (on REUSE_ARTIFACT): lightweight eligibility check without rerunning full transformation.
3. **Final model-facing integrity validation** (after final CE compilation, before token preflight/exact send): message order, roles, stable IDs, tool linkage, citation/evidence groups, protected values, system/developer/platform blocks, current user turn, recent tail, exact-send hash, tool-envelope consistency.

verify_context_preflight is a budget/window check only; it does not replace structural or protected-content integrity validation.

## Concurrency and activation

- Durable activation: CAS on expected_active_revision_id via Memory/Session.
- Application host authorizes and presents UX; Memory/Session executes activation and rollback.
- Activation operates on SessionContextRevision artifact references; must not regenerate artifact content.
- Same-key artifact creation: single-flight via ArtifactCreationReservation; content addressing does not replace creation coordination.

## Cache-lineage separation

Content revision, prompt prefix identity, cache lineage, provider cache observation, and Reusable Optimization Artifact Catalog remain separate dimensions. The artifact catalog is platform-owned persisted content, not provider KV cache.

## Consequences

### Positive

- One lifecycle architecture eliminates competing ownership and duplicate budgets.
- Distinct ledger vs projection model enables durable compaction without silent history loss.
- TOKEN-10E integrates as a bounded TO contribution under UCL rather than a parallel lifecycle.
- Lower latency and model cost through reuse-before-create.
- Stable summaries and improved determinism.
- Better prompt-prefix stability and traceable artifact lineage.
- Prevents infinite optimization recursion via execution-scope boundary.
- Prevents duplicate concurrent LLM cost via single-flight creation.
- Makes the roadmap executable with concrete repository delivery tasks.
- Supports deterministic concurrency tests.
- Allows runtime proof before production storage via InMemoryOptimizationArtifactRepository.

### Negative

- Requires CTX-UCL-1 through CTX-UCL-6 and closeout before TOKEN-10E-1.
- ADR-MEM-001 Context Compiler semantics are superseded where they conflict with UCL.
- Requires compatibility identity (ArtifactLookupKey), invalidation rules, artifact lifecycle storage, and version migrations.
- May retain multiple compression levels per source range.
- Requires execution-scope propagation through model invocation paths.
- Requires reservation/lease state and timeout/recovery semantics.
- Adds repository concurrency tests and careful process-crash handling.

## Migration impact

Application-owned persistence/activation wording migrates to Memory/Session ownership with Application adapter wiring and UX. Legacy HistoryLayer summarization migrates to UCL lookup-before-create path with reservation coordination; HistoryLayer must not keep its own summary cache.

## Rejected alternatives

1. Application-owned revision persistence and activation.
2. Token Optimization as platform owner of context versions and activation.
3. TOKEN-10E wrapping the existing string pipeline for full conversation history.
4. Final structural validation before final CE compilation.
5. TOKEN-10E-1 starting after CTX-UCL-1 only (without closeout).
6. Regenerate summary on every model call.
7. Application-local summary cache.
8. Token Optimization-owned artifact persistence.
9. Reuse based only on source range without source_content_hash.
10. Reuse without policy and validation version checks.
11. Run full UCL recursively for summarizer calls.
12. Rely only on content-addressed deduplication to prevent duplicate LLM calls.
13. Allow duplicate creation and deduplicate after LLM execution.
14. Application-local mutex for summary creation coordination.
15. Token Optimization-owned repository.
16. Ambiguous CTX-UCL-2+ repository delivery without concrete task assignment.

## Relationship to ADR-MEM-001 and ADR-CTX-001

- **ADR-MEM-001** remains historical/canonical for existing ContextCompiler semantics but is superseded where it conflicts with UCL.
- **ADR-CTX-001** remains valid where not superseded by UCL.

## Compliance

- Tier boundaries preserved.
- Linked architecture and plan docs updated in CTX-UCL-ARCH-1-R4.

## Implementation notes

- Documentation and guardrail tests only in CTX-UCL-ARCH-1-R4.
- Runtime implementation has not started.
- Next step after ADR acceptance: **CTX-UCL-1**.