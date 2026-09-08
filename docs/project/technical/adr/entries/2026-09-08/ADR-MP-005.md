# ADR-MP-005: Decision / Approval / Governance collaborative ownership and contract freeze

| Field | Value |
|-------|-------|
| **Status** | Accepted — architecture and ownership gate only; MP-4 runtime implementation **NOT STARTED** |
| **Date** | 2026-09-08 |
| **Deciders** | Intergrax platform architecture (MP-4A ownership freeze) |
| **Related** | [`architecture/DECISION_APPROVAL_GOVERNANCE.md`](../../../../architecture/DECISION_APPROVAL_GOVERNANCE.md) · [`plan/DECISION_APPROVAL_GOVERNANCE.md`](../../../../maintainers/plans/DECISION_APPROVAL_GOVERNANCE.md) · [`capabilities/architecture/MULTIPLAYER_AI.md`](../../../../capabilities/architecture/MULTIPLAYER_AI.md) · [`architecture/COLLABORATIVE_WORK.md`](../../../../architecture/COLLABORATIVE_WORK.md) · [ADR-MP-001](../2026-08-11/ADR-MP-001.md) · [ADR-MP-002](../2026-08-11/ADR-MP-002.md) · [ADR-MP-003](../2026-09-06/ADR-MP-003.md) · [ADR-MP-004](../2026-09-07/ADR-MP-004.md) |

## Context

Multiplayer AI MP-4 requires platform-owned collaborative **Decision**, **Approval**, and **Governance** semantics — distinct from Collaborative Work artifacts (MP-3), Unified Execution runtime state (UER/Nexus), Governed Execution enforcement machinery, evidence records, or channel adapter presentation.

Prior Multiplayer canon established provisional MP-4 intent and invariants (MP-INV-09 Decision ≠ HITL; MP-INV-23 approval/evidence ≠ execution authorization) but left ownership as **`OWNERSHIP_TO_CONFIRM_BEFORE_IMPLEMENTATION`**. MP-3 (ADR-MP-004) explicitly deferred Decision, approval status, and HITL encoding outside WorkArtifact lifecycle.

Without an ownership freeze, implementers could:

- encode approval as `WorkArtifactVersion.status = APPROVED` (artifact lifecycle substitution),
- encode human gates as `ExecutionState.WAITING_FOR_HUMAN` (runtime state substitution),
- duplicate Principal/ACL subsystems for approvers,
- import Nexus orchestration or WorkItem lifecycle into Decision services,
- conflate MP-4 Governance with GOVERNED_EXECUTION execution-boundary enforcement.

Split or ambiguous ownership would violate PLATFORM-INV-003 (single ownership) and MP anti-substitution invariants.

## Alternatives considered

### 1. COLLABORATIVE_WORK as owner of Decision + Approval + Governance

Rejected. Collaborative Work owns durable collaborative **work** and **artifact** semantics (WorkItem, WorkArtifact). Decision records **what choice was made**; Approval records **human/policy gate actions**; Governance records **collaborative control posture and evidence linkage**. Folding these into WorkItem/WorkArtifact lifecycle would recreate the forbidden approval-status-on-artifact pattern ADR-MP-004 rejected.

### 2. UNIFIED_EXECUTION_RUNTIME / Nexus as owner

Rejected. UER owns Task/Run/Attempt/Execution lifecycle, scheduling, and orchestration. Decision and Approval are durable collaborative primitives that may exist **without** an active execution. HITL pause/resume remains execution-boundary machinery (Governed Execution / Reliability); MP-4 may **bridge** to HITL but must not own or substitute execution state.

### 3. GOVERNED_EXECUTION as owner of all MP-4 semantics

Rejected. Governed Execution owns execution-centric policy evaluation, enforcement outcomes, canonical HITL, and governed continuation. MP-4 Governance Domain owns **collaborative** governance posture for decision/approval workflows (controls, evidence linkage for multiplayer decisions) — consumption of Governed Execution, not duplication of execution-boundary enforcement.

### 4. Three-domain model under MP-4 plane (Accepted)

Accepted. **Decision Domain**, **Approval Domain**, and **Governance Domain (collaborative)** share the MP-4 architecture/plan hub and contract namespace direction, with frozen dependency rules and explicit non-ownership of Collaborative Work artifacts and UER runtime state.

## Decision

1. **MP-4 semantic ownership** (single architecture/plan hub; three semantic domains):

   | Capability | Owner |
   |------------|-------|
   | WorkItem / Assignment / shared-work lifecycle | Collaborative Work (MP-2) |
   | WorkArtifact / WorkArtifactVersion | Collaborative Work (MP-3) |
   | **Decision** identity, lifecycle, outcome | **Decision Domain (MP-4)** |
   | **Approval** request, lifecycle, human actions | **Approval Domain (MP-4)** |
   | **Collaborative governance** policies, controls, evidence linkage | **Governance Domain (MP-4)** |
   | Task / Run / Attempt / Execution lifecycle | Unified Execution Runtime |
   | Execution-boundary policy + canonical HITL | Governed Execution (reuse) |
   | Proof / attestation records | Evidence / Proof Receipts (reference) |
   | Principal / membership / delegation | Collaborative Work MP-1 (reuse) |

2. **Hard anti-substitution invariants:**

   - `Decision != WorkArtifact` — Decision records what choice was made; artifact records collaborative output existence.
   - `Approval != WorkArtifactVersion.status` — Approval references `WorkArtifactVersionId`; forbidden: `APPROVED` on artifact version lifecycle.
   - `Approval != ExecutionState.WAITING_FOR_HUMAN` — human interaction is Approval Domain process; HITL pause is execution machinery; bridge only.
   - `Decision != HITL` (MP-INV-09).
   - `Approval/evidence != execution authorization` (MP-INV-23).
   - `Governance evidence != ProofReceipt` — linkage only.

3. **Dependency direction (contracts-first):**

   ```text
   intergrax/contracts/  → decision / approval / governance contracts (future slices)
   collaborative_work    → reference-only (ids, Principal, ExecutionProvenanceRef)
   execution runtime     → execution lifecycle + optional HITL bridge consumer
   ```

   **Forbidden production imports** in MP-4 contracts/services: `intergrax.runtime.nexus.*`, GraphExecutor, TaskState ownership, WorkItem lifecycle mutation, ProofReceipt ownership, Memory lifecycle, UCL ownership, new Principal/ACL subsystems.

4. **Existing contract reuse (mandatory before new types):**

   | Reuse | Owner |
   |-------|-------|
   | `CollaborativePrincipal`, membership, delegation | MP-1 |
   | `EffectiveAuthorityRequest` / `EffectiveAuthorityDecision`, `PolicyDecision` | MP-1 / Governed Execution |
   | `ExecutionProvenanceRef` | Tier-0 contracts |
   | Evidence reference handles | Evidence subsystem |

   **Do not create:** `NewPrincipal`, `NewUserIdentity`, `NewExecutionIdentity`, new ACL system.

5. **Reference model (conceptual):**

   ```text
   Approval → references WorkArtifactVersionId and/or DecisionId
   Decision → may reference WorkItemId, WorkArtifactVersionId, ExecutionProvenanceRef (optional)
   Governance evidence → references platform evidence; does not own ProofReceipt
   ```

6. **Non-goals (MP-4 program until separately gated):**

   - No runtime orchestration or workflow engine in MP-4 architecture scope
   - No WorkArtifact lifecycle ownership
   - No execution lifecycle ownership
   - No ACL duplication
   - No persistence implementation in MP-4A

7. **Implementation decomposition:** MP-4A (this gate) → MP-4B Decision contracts → MP-4C Approval/HITL contracts → MP-4D Authority → MP-4E Persistence boundary → MP-4F Evidence → MP-4G Qualification → MP-4H Closure. See [`plan/DECISION_APPROVAL_GOVERNANCE.md`](../../../../maintainers/plans/DECISION_APPROVAL_GOVERNANCE.md).

## Consequences

### Positive

- Single accepted ownership gate before MP-4B contract work.
- Clear separation from MP-3 artifacts, UER runtime, and Governed Execution enforcement.
- Mandatory reuse of MP-1 identity/authority and Tier-0 execution provenance references.
- Anti-substitution rules are auditable via documentation and grep gates.

### Negative

- Three semantic domains under one MP-4 hub require disciplined slice gates (MP-4B…H).
- HITL bridge semantics must be designed without importing Nexus into MP-4 contracts.

## Compliance

- PLATFORM-INV-001 / PLATFORM-INV-003: single domain ownership preserved per capability row.
- MP-INV-09, MP-INV-23, ADR-MP-004 MP-4 non-leakage honored.
- Collaborative Work and UER hubs updated with MP-4 boundary references.
- Linked architecture, plan, feature, and ADR index updated in MP-4A task.

## Implementation notes

- Implementation **must not** begin until MP-4A gate closes and MP-4B slice opens.
- MP-4A delivers documentation and ADR only — **no Python production code**.
- Verification: `git diff --check`; `python scripts/docs/check_docs_domain_pairs.py`; `python scripts/maintenance/check_harness_adr.py`; leakage greps documented in architecture hub.
