# ADR-MP-004: WorkArtifact collaborative ownership and version authority

| Field | Value |
|-------|-------|
| **Status** | Accepted — architecture and ownership gate only; MP-3 runtime implementation **NOT STARTED** |
| **Date** | 2026-09-07 |
| **Deciders** | Intergrax platform architecture (MP-3 ownership freeze) |
| **Related** | [`architecture/COLLABORATIVE_WORK.md`](../../../../architecture/COLLABORATIVE_WORK.md) · [`plan/COLLABORATIVE_WORK.md`](../../../../maintainers/plans/COLLABORATIVE_WORK.md) · [`capabilities/architecture/MULTIPLAYER_AI.md`](../../../../capabilities/architecture/MULTIPLAYER_AI.md) · [ADR-MP-001](../2026-08-11/ADR-MP-001.md) · [ADR-MP-002](../2026-08-11/ADR-MP-002.md) · [ADR-MP-003](../2026-09-06/ADR-MP-003.md) |

## Context

Multiplayer AI MP-3 requires platform-owned durable collaborative outputs with explicit versioning, publication semantics, and provenance — distinct from context-runtime optimization, memory state, execution state, evidence records, application payloads, or channel adapter artifacts.

MP-0 provisionally listed `UNIFIED_CONTEXT_LIFECYCLE`, `PROOF_RECEIPTS`, and `MEMORY` as likely MP-3 owners pending bounded verification. Repository canon already states:

- **MP-INV-10:** `WorkArtifact != UCL OptimizationArtifact`.
- **COLLABORATIVE_WORK** ownership boundary lists WorkArtifact and WorkArtifactVersion as collaborative primitives on the same work plane as WorkItem (MP-2).
- **MP-3 extension boundary (ADR-MP-003):** WorkArtifact / WorkArtifactVersion — not WorkItem payload.
- **UCL** owns `OptimizationArtifactRepository`, `ReusableOptimizationArtifact`, and context compaction/catalog semantics — not collaborative publication authority.
- **Memory** owns session/durable memory semantics — not collaborative artifact identity or current-version pointer.
- **Proof Receipts** owns evidence/attestation — not authoritative collaborative artifact content or version lifecycle.

Split ownership would violate PLATFORM-INV-003 (single ownership), conflate optimization cache entries with durable collaborative outputs, and allow silent substitution (receipt identity for artifact identity, memory promotion for publication, UCL artifact reuse for WorkArtifactVersion).

## Alternatives considered

### 1. UNIFIED_CONTEXT_LIFECYCLE as primary owner

Rejected. UCL owns context lifecycle, optimization artifacts, compaction, and reuse-before-create catalog semantics. `OptimizationArtifact` is a context-runtime optimization/cache primitive — not an authoritative collaborative output version. A published `WorkArtifactVersion` may later be **consumed** as context input without transferring ownership to UCL.

### 2. MEMORY as primary owner

Rejected. Memory owns conversation/session durable state and retrieval/indexing. Memory may index WorkArtifact references and make published content retrievable; memory promotion is not artifact publication and does not own WorkArtifact identity, version history, or current-version pointer semantics.

### 3. PROOF_RECEIPTS as primary owner

Rejected. Proof Receipts owns proof/evidence records and attestation/verification semantics. Evidence may prove that an artifact version was produced or published; `ProofReceipt != WorkArtifactVersion`. Artifact publication must not require replacing artifact identity with receipt identity.

### 4. Split ownership across UCL / Memory / Proof Receipts

Rejected. Would preserve ambiguous version authority, encourage silent last-write-wins across subsystems, and prevent a single collaborative publication contract.

### 5. COLLABORATIVE_WORK extends MP-2 plane (Accepted)

Accepted. MP-2 established WorkItem and Assignment on the collaborative work plane. MP-3 extends the same domain with WorkArtifact and WorkArtifactVersion as first-class collaborative primitives, reusing MP-1 authority, MP-2 repository/concurrency/idempotency patterns, and neutral `ExecutionProvenanceRef` where execution produced a version.

## Decision

1. **COLLABORATIVE_WORK** is the single canonical owner of MP-3 WorkArtifact collaborative semantics:

   | Owned by Collaborative Work | Not owned (reuse only) |
   |-----------------------------|-------------------------|
   | WorkArtifact identity | UCL OptimizationArtifact identity |
   | WorkArtifact lifecycle semantics (aggregate/reference) | Context compaction / optimization catalog |
   | WorkArtifactVersion identity | OptimizationArtifact revision/cache entry |
   | Artifact/version association | Memory record identity |
   | Authoritative current-version pointer semantics | ProofReceipt identity |
   | Collaborative publication semantics | Execution result object identity |
   | Tenant/workspace/work-item scoping for artifacts | WorkItem mutable payload embedding |
   | Collaborative authority requirements for publication | LKW attachment/message/file identity |
   | Optimistic concurrency for current-version changes | Channel adapter mappings |
   | Idempotent publication semantics | |
   | Collaborative lineage relationships (artifact-level) | |

2. **Hard anti-substitution invariants:**

   - `WorkArtifact != UCL OptimizationArtifact`
   - `WorkArtifactVersion != OptimizationArtifact` revision/cache entry
   - `WorkArtifact != ProofReceipt`; `WorkArtifactVersion != ProofReceipt`
   - `WorkArtifact != Memory record`; `WorkArtifactVersion != Memory record`
   - `WorkArtifact != Execution result object`; `WorkArtifactVersion != Execution result object`
   - `WorkArtifact != WorkItem payload` (no mutable artifact bodies embedded in WorkItem)
   - `WorkArtifact != LKW attachment/message/file` (adapter mappings only)

3. **WorkItem relationship:**

   ```text
   WorkItem → zero..N WorkArtifact
   WorkArtifact → one..N WorkArtifactVersion
   ```

   - Each WorkArtifact belongs to one Collaborative Work `tenant_id + workspace_id` scope and references its owning WorkItem.
   - WorkItem lifecycle and artifact version lifecycle remain distinct.
   - Completing a WorkItem does **not** delete artifact versions.
   - Deleting/ending execution does **not** delete artifact versions.

4. **WorkArtifactVersion immutability:**

   - `WorkArtifactVersion` is append-only / immutable — no in-place content mutation.
   - Corrections produce a new version.
   - The `WorkArtifact` aggregate may update its current-version pointer under CAS; historical versions remain independently addressable and are not rewritten to simulate rollback.

5. **Current-version authority:**

   - A `WorkArtifact` has an explicit `current_version_id` (or equivalent pointer/reference).
   - `current_version_id` must refer to a `WorkArtifactVersion` belonging to the same tenant, workspace, and work artifact.
   - Publishing a new version: creates a new immutable `WorkArtifactVersion`; may update `WorkArtifact.current_version` reference; must use optimistic concurrency; stale current-version update fails explicitly — no silent last-write-wins.
   - Do not derive authority from human-readable version labels (`v1`, `1.0.0`); canonical identity is `WorkArtifactVersionId`.

6. **Content vs metadata separation:**

   - Do not model `WorkArtifactVersion` as `dict[str, Any]` or an arbitrary payload bag in architecture.
   - Separate: (A) collaborative artifact/version identity + lineage; (B) content representation/storage reference.
   - Preferred pattern: `WorkArtifactVersion` → `ArtifactContentRef` / typed content descriptor → external/document/blob/object storage adapter.
   - Collaborative Work owns artifact/version **metadata** and authoritative version semantics; it does **not** require raw binary/blob storage in the Collaborative Work database.
   - No vendor-specific storage contract is frozen (no S3/filesystem/Mongo-specific canon).

7. **Authority reuse (MP-1):**

   - Publication and version creation record authoritative collaborative actor identity via MP-1 Principal / membership / delegation / effective authority.
   - No `ArtifactUser`, artifact-specific principal hierarchy, or second ACL engine.
   - Prefer explicit provenance field names when runtime is designed: `created_by_principal_id`, `published_by_principal_id` — not ambiguous generic `owner_id` for domain ownership semantics.

8. **Execution provenance (optional, neutral reuse):**

   - Reuse `ExecutionProvenanceRef` for optional production provenance when an execution produced a version.
   - `created/published by Principal` is **mandatory** collaborative provenance.
   - Execution provenance is **optional** — human-created collaborative artifact versions are valid.
   - Do not make WorkArtifact a runtime-owned object.

9. **Reuse-only collaborator boundaries:**

   | Domain | Role for MP-3 |
   |--------|----------------|
   | UNIFIED_CONTEXT_LIFECYCLE | Context lifecycle, optimization artifacts — may consume published versions as input |
   | MEMORY | May index references, retrieve derived content — not version authority |
   | PROOF_RECEIPTS | May attest production/publication — not artifact content authority |
   | UNIFIED_EXECUTION_RUNTIME / NEXUS | Neutral execution identities via `ExecutionProvenanceRef` when applicable |
   | LKW / channel adapters | Consumer; adapter ID mappings only — not canonical WorkArtifact identity |
   | DocumentStore / blob providers | Content storage implementation capability — not collaborative version semantics |

10. **MP-4 / MP-6 non-leakage:**

    - Do not encode approval, decision, review status, or HITL inside WorkArtifact lifecycle (MP-4 owns Decision).
    - Do not implement Collaborative Activity, activity feeds, or generic provenance graph (MP-6 owns activity projection). MP-3 may expose stable IDs/timestamps/actor/provenance refs for MP-6 consumption later.

11. **Persistence reuse intent (implementation later):**

    - Extend the same Collaborative Work repository port pattern: in-memory reference → SQLite local durable → PostgreSQL production-qualified.
    - Reuse MP-1 authority gate, revision/CAS semantics, idempotency semantics, tenant/workspace isolation.
    - Do not duplicate persistence framework or introduce speculative `ArtifactPlugin` / `ArtifactProviderRegistry` / lifecycle plugin registries during this gate.

## Consequences

### Positive

- Single semantic owner for collaborative artifact identity, version history, and current-version pointer.
- Clear separation from UCL optimization cache, memory, and evidence planes.
- MP-3 builds on proven MP-1/MP-2 authority, concurrency, and persistence patterns.
- Content storage remains pluggable without leaking provider keys into authoritative collaborative identity.

### Negative

- Collaborative Work domain scope grows; implementation waves (MP-3A…MP-3H) required before MP-3 closure.
- Content-storage reference boundary must be designed without premature provider lock-in.

## Compliance

- PLATFORM-INV-001 / PLATFORM-INV-003: single domain ownership preserved.
- MP-INV-10, MP-INV-19, MP-INV-20 honored.
- CW extension boundaries for MP-3, MP-4, MP-6 preserved.
- Linked architecture, plan, and feature docs updated in MP-3 ownership freeze task.

## Implementation notes

- Implementation **must not** begin until MP-3 architecture/contract roadmap decomposition is accepted.
- First implementation slice: **MP-3A** — WorkArtifact / WorkArtifactVersion contracts + invariants.
- Verification: `python scripts/maintenance/check_harness_adr.py`; `python scripts/docs/check_docs_domain_pairs.py`; documentation link integrity; `git diff --check`.
