# ADR-MP-006: Principal-scoped ContextView ownership and contract boundary

| Field | Value |
|-------|-------|
| **Status** | Accepted — architecture and ownership gate only; MP-5B+ runtime contracts and implementation **NOT STARTED** |
| **Date** | 2026-09-17 |
| **Deciders** | Intergrax platform architecture (MP-5A ownership freeze) |
| **Related** | [`architecture/COLLABORATIVE_WORK.md`](../../../../architecture/COLLABORATIVE_WORK.md) · [`plan/COLLABORATIVE_WORK.md`](../../../../maintainers/plans/COLLABORATIVE_WORK.md) · [`capabilities/architecture/MULTIPLAYER_AI.md`](../../../../capabilities/architecture/MULTIPLAYER_AI.md) · [ADR-MP-001](../2026-08-11/ADR-MP-001.md) · [ADR-MP-002](../2026-08-11/ADR-MP-002.md) · [ADR-MP-004](../2026-09-07/ADR-MP-004.md) |

## Context

Multiplayer AI MP-5 requires a **principal-scoped ContextView**: a policy-governed, read-oriented composition of what context categories a collaborative actor may see under a resolved collaborative scope — without introducing a second Memory system, RAG system, UCL lifecycle, Context Engineering pipeline, or Token Optimization authority.

MP-0 provisionally listed `UNIFIED_CONTEXT_LIFECYCLE`, `CONTEXT_ENGINEERING`, `MEMORY`, and `RAG` as likely MP-5 owners pending bounded verification. Repository canon already states:

- **MP-INV-15…18:** Memory is not collaborative source of truth; private memory is not automatically shared; context is principal-specific; external agents receive least context.
- **COLLABORATIVE_WORK** scopes Principal-scoped ContextView (MP-5) from MP-1 effective authority.
- **UCL** owns durable/ephemeral context lifecycle, revisions, and optimization artifact catalog — not principal visibility policy.
- **Context Engineering** owns model-facing assembly, global input budget, truncation, and compaction orchestration for a model invocation — not collaborative visibility classes.
- **Memory** owns durable memory records, session/conversation stores, and memory retrieval semantics — not principal-scoped view composition policy.
- **RAG / Knowledge** owns retrieval, indexing, reranking, and knowledge provenance — not membership-aware eligibility policy.
- **Token Optimization** owns transformation executors and strategy catalog — not MP-5 implementation identity.

Adjacent read models (`SharedContextView`, `DecisionContextView`, `MemoryView`) serve other domains and **must not** substitute for MP-5 Principal-scoped ContextView.

Split ownership of the same semantic concept (e.g. UCL “being” ContextView, or Memory “being” the view store) would violate PLATFORM-INV-003 and enable cross-tenant or cross-workspace leakage via implicit merge.

## Alternatives considered

### 1. UNIFIED_CONTEXT_LIFECYCLE as primary owner

Rejected. UCL answers how durable conversation context is versioned, compacted, and reused. ContextView answers **who** may see **which eligible sources** under **which collaborative scope**. MP-5 **consumes** UCL public contracts; it does not own lifecycle or mutate UCL storage internals.

### 2. CONTEXT_ENGINEERING as primary owner

Rejected. CE answers what enters the **model-facing** context for one execution step under budget. MP-5 answers principal/workspace/work-item **visibility and eligibility** before optional CE handoff. CE remains budget and assembly authority.

### 3. MEMORY as primary owner

Rejected. Memory owns stores and recall semantics. MP-5 defines **eligibility** of memory for a principal scope (or delegates selection to a typed port); it does not own memory records or create `ContextView` storage.

### 4. RAG / Knowledge as primary owner

Rejected. RAG owns retrieval implementation and knowledge provenance. MP-5 may constrain **eligible knowledge scope**; it does not implement retrievers or indexes.

### 5. Dedicated neutral “platform context” domain (separate from Collaborative Work)

Rejected for MP-5A. Principal-scoped visibility is a **collaborative work-plane primitive** (same plane as WorkItem and WorkArtifact). Contracts and runtime modules remain under **Collaborative Work** / Multiplayer MP-5 capability coordination — without importing Nexus or application ownership.

### 6. COLLABORATIVE_WORK extends MP-1…MP-3 plane (Accepted)

Accepted. MP-5 extends the collaborative work plane with Principal-scoped ContextView semantics and composition policy, reusing MP-1 `EffectiveAuthorityRequest` / membership / delegation resolution and tenant + workspace (+ optional work-item) scoping patterns.

## Decision

### 1. Single semantic owner

**COLLABORATIVE_WORK** (Multiplayer AI **MP-5** capability) is the single canonical owner of:

| Owned by Collaborative Work (MP-5) | Not owned (reuse only) |
|--------------------------------------|-------------------------|
| Principal-scoped **ContextView** semantics | UCL lifecycle and `OptimizationArtifact` identity |
| **ContextView policy** (who / which categories / under which collaborative scope) | CE global input budget and model-facing `ChatMessage[]` assembly |
| **ContextView composition policy** (eligible sources, ordering intent, fail-closed omission) | Memory store lifecycle and recall implementation |
| **Visibility classes** (private-to-principal, workspace-shared, work-item, delegated-visible, platform-visible) as policy concepts | RAG retrieval, indexing, reranking |
| **Reference-first** `ContextView` entries pointing at domain-owned refs | Token Optimization executors (`TOKEN-10E-*`) |
| **External-consumer projection** seam (least-context subset) | `SharedContextView`, `DecisionContextView`, LKW conversation context |
| Deterministic policy before any LLM consumption | Nexus orchestration internals |

**Hard invariant:** `ContextView ≠ storage`. No `ContextViewDatabase`, `ContextViewMemoryStore`, or `ContextViewVectorDatabase` in MP-5.

### 2. ContextView definition

**ContextView** is:

- principal-scoped, workspace-scoped (and optionally work-item- or operation-scoped),
- policy-governed,
- composed (logical projection over existing sources),
- read-oriented,
- auditable (provenance via reused CE / evidence contracts — no new Evidence system),
- provider-neutral.

**ContextView** is **not**:

- a shared mutable memory dump,
- a raw Memory query result,
- a RAG search result page,
- a UCL artifact or revision,
- an LLM prompt or CE `AssembledContext`,
- an application-specific payload,
- authorization proof from bare `principal_id` / `workspace_id` / `tenant_id` strings alone.

### 3. Security and isolation model

Preferred flow:

```text
resolve principal + membership + effective authority
  → determine eligible source categories and scopes
  → retrieve / reference via domain public contracts
  → compose ContextView (fail-closed on ambiguity)
```

- **Cross-tenant:** never compose tenant A + tenant B in one view.
- **Cross-workspace:** workspace A context does not appear in workspace B view without explicit federation (out of MP-5 scope).
- **Delegation:** cannot amplify visibility beyond effective delegated authority.
- **Least-context:** minimum necessary for the current operation — not “everything the principal can access.”
- **Private ≠ shared ≠ work-item ≠ external-agent:** no implicit merge of sources.
- **LLM** is not a security boundary; visibility is deterministic policy.

### 4. Public contract boundary (MP-5B+ — names frozen at gate level)

MP-5A freezes **concepts**; concrete typed contracts ship in **MP-5B** in `intergrax/contracts/` (Collaborative Work namespace), with **no** `runtime.nexus` dependency on the public contract surface.

| Concept | Owner | Purpose | Replacement seam |
|---------|-------|---------|------------------|
| Authority / scope inputs | Collaborative Work (MP-1) | `EffectiveAuthorityRequest`, membership, delegation | Existing MP-1 ports |
| ContextView request | MP-5 (CW) | Operation-scoped view intent + scope | Injectable request builder |
| ContextView scope | MP-5 (CW) | Typed collaborative scope (tenant, workspace, optional work-item) | Policy strategies |
| ContextView result | MP-5 (CW) | Immutable composed projection | Composer implementations |
| ContextView entry / reference | MP-5 (CW) | Reference-first admission to Memory / Knowledge / UCL / CE artifacts | Per-source eligibility adapters |
| Composition policy | MP-5 (CW) | Eligibility, visibility, ordering rules | Replaceable policy port |
| Composer | MP-5 (CW) | Orchestrates adapters without owning stores | Replaceable composer port |
| UCL lifecycle read | UCL | Eligible revisions / artifacts under scope | UCL public ports only |
| Memory recall eligibility | Memory | Scoped recall under MP-5 eligibility | Memory public ports only |
| Knowledge retrieval eligibility | RAG | Scoped retrieve under MP-5 eligibility | RAG public ports only |
| Model-facing assembly (optional downstream) | CE | Budget + format for model call | CE pipeline (consumer of admitted candidates) |

### 5. Anti-substitution (normative)

- `UCL ≠ Principal-scoped ContextView`
- `Memory ≠ Principal-scoped ContextView`
- `RAG result ≠ Principal-scoped ContextView`
- `Context Engineering ≠ Principal-scoped ContextView`
- `Token Optimization ≠ MP-5 implementation`
- `LKW conversation context ≠ platform Principal-scoped ContextView`
- `SharedContextView ≠ Principal-scoped ContextView`
- `DecisionContextView ≠ Principal-scoped ContextView`

### 6. Dependency direction

```text
MP-5 ContextView (Collaborative Work contracts)
  → public contracts of UCL, Context Engineering, Memory, RAG
```

Never:

```text
Memory / RAG / UCL → Multiplayer ContextView concrete implementation
```

as a required reverse dependency without an explicitly approved contract.

### 7. Domain placement

- **Architecture SSOT:** [`COLLABORATIVE_WORK.md`](../../../../architecture/COLLABORATIVE_WORK.md) § Principal-scoped ContextView.
- **Capability coordination:** [`MULTIPLAYER_AI.md`](../../../../capabilities/architecture/MULTIPLAYER_AI.md) MP-5 slices.
- **Future runtime (MP-5E+):** `intergrax/collaborative_work/` composition modules; **not** `applications/`, **not** `runtime/nexus` as contract owner.

### 8. MP-5 implementation decomposition (frozen at gate)

| Slice | Purpose | Status after MP-5A |
|-------|---------|----------------------|
| MP-5A | Ownership, contracts architecture, ADR, docs sync | **APPROVED / CLOSED** |
| MP-5B | Core Principal-scoped ContextView typed contracts | **NEXT** |
| MP-5C | Principal-scope visibility policy | PLANNED |
| MP-5D | Source composition ports (UCL / Memory / Knowledge / CE handoff) | PLANNED |
| MP-5E | Default composition implementation | PLANNED |
| MP-5F | Source adapters / integration | PLANNED |
| MP-5G | E2E / isolation qualification | PLANNED |
| MP-5H | Final MP-5 enterprise certification | PLANNED |

## Consequences

- MP-5B may add contracts to Collaborative Work namespace; must remain extra-forbid, typed, and serialization-safe.
- Domain plans for UCL, CE, Memory, and RAG remain authoritative for their mechanisms; MP-5 rows reference consumption boundaries only.
- Regression gates may assert frozen ownership markers and anti-substitution text in SSOT docs.
- Independent audit of this ADR and doc sync is required before implementation sign-off.
