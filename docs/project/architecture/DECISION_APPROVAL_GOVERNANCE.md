<!--
© Artur Czarnecki. All rights reserved.
Intergrax is source-available under the Intergrax Evaluation and Collaboration License 1.0.
See LICENSE for permitted evaluation, collaboration, and contribution use.
-->

# Decision / Approval / Governance — Collaborative Semantics (MP-4)

**Status:** **MP-4A — APPROVED / CLOSED** (ownership + contracts freeze) · **MP-4B — READY_FOR_REVIEW** · MP-4C…MP-4H **NOT STARTED**
**ADR:** [ADR-MP-005](../technical/adr/entries/2026-09-08/ADR-MP-005.md)
**Feature coordination:** [`MULTIPLAYER_AI`](../capabilities/architecture/MULTIPLAYER_AI.md) · [`COLLABORATIVE_WORK`](COLLABORATIVE_WORK.md) (reference-only boundary)
**Plan (1:1):** [`plan/DECISION_APPROVAL_GOVERNANCE.md`](../maintainers/plans/DECISION_APPROVAL_GOVERNANCE.md)

---

## Purpose

Freeze semantic ownership, domain boundaries, contract direction, and anti-substitution rules for Multiplayer **Decision**, **Approval**, and **Governance** primitives before any MP-4 runtime, persistence, or workflow-engine implementation.

This hub does **not** implement contracts, repositories, services, orchestration, or storage.

---

## Semantic ownership (frozen)

Three distinct collaborative domains on the Multiplayer plane. Each owns its lifecycle and authoritative state; none substitutes for Collaborative Work artifacts or Unified Execution runtime state.

```text
Decision Domain
    |
    +-- Decision identity
    +-- Decision lifecycle
    +-- Decision outcome

Approval Domain
    |
    +-- approval requests
    +-- approval lifecycle
    +-- human actions (HITL as domain process — not runtime state substitution)

Governance Domain (collaborative)
    |
    +-- policies (collaborative governance posture for decisions/approvals)
    +-- controls (decision/approval gate requirements)
    +-- governance evidence (linkage to platform evidence — not evidence ownership)
```

| Capability | Owner | Notes |
|------------|-------|-------|
| WorkItem / Assignment / shared-work lifecycle | Collaborative Work (MP-2) | Reference only |
| WorkArtifact / WorkArtifactVersion | Collaborative Work (MP-3) | Reference only |
| **Decision** identity, lifecycle, outcome | **Decision Domain (MP-4)** | What choice was made |
| **Approval** request, lifecycle, human actions | **Approval Domain (MP-4)** | Who approved/rejected and when |
| **Collaborative governance** posture, controls, evidence linkage | **Governance Domain (MP-4)** | Not GOVERNED_EXECUTION enforcement |
| Task / Run / Attempt / Execution lifecycle | Unified Execution Runtime | Execution lifecycle only |
| Policy evaluation at execution boundary | Governed Execution | Reuse — not re-owned |
| Proof / attestation records | Evidence / Proof Receipts | Reference only |
| Principal / membership / delegation | Collaborative Work (MP-1) | Reuse — not re-owned |

**GOVERNED_EXECUTION vs MP-4 Governance:** [`GOVERNED_EXECUTION`](GOVERNED_EXECUTION.md) owns execution-centric policy enforcement, HITL pause/resume machinery at the execution boundary, and governed continuation. MP-4 **Governance Domain** owns collaborative governance semantics for decision/approval workflows (which policies apply to a decision context, control requirements, governance evidence linkage). MP-4 **consumes** Governed Execution outcomes; it does **not** duplicate execution-boundary enforcement or own `TaskState` / pause machinery.

---

## Absolute boundaries

### Collaborative Work — remains owner (no MP-4 leakage)

Collaborative Work **continues to own**:

- WorkItem, WorkArtifact, WorkArtifactVersion
- Execution linkage references (`ExecutionProvenanceRef` on WorkItem links / artifact versions)
- Content references (`ArtifactContentRef`)

Collaborative Work **must not add**:

- Decision, Approval, or HumanReviewState fields on WorkItem / WorkArtifact / WorkArtifactVersion
- GovernanceStatus on artifact lifecycle
- `DRAFT` / `REVIEW` / `APPROVED` / `ARCHIVED` status on WorkArtifactVersion (MP-4 owns approval semantics separately)

### Unified Execution Runtime / Nexus — remains owner (no MP-4 leakage)

UER **continues to own**:

- Task, Run, Attempt, Execution identity and lifecycle
- Scheduling, orchestration, retries

UER **must not add**:

- Decision lifecycle, Approval lifecycle, or Governance workflow as execution-state substitutes
- `ExecutionState.WAITING_FOR_HUMAN` (or equivalent) as approval-workflow replacement — HITL pause remains execution-boundary machinery; Approval is a separate domain process that may **bridge** to HITL

---

## Anti-substitution rules (frozen)

| Forbidden | Correct model |
|-----------|---------------|
| Decision encoded as WorkArtifact body or WorkItem state | Decision is a distinct primitive; may **reference** WorkItem / WorkArtifactVersion |
| `WorkArtifactVersion.status = APPROVED` | Approval references `WorkArtifactVersionId`; approval lifecycle is Approval Domain |
| HumanReviewState on WorkItem / Assignment | Approval request + approval lifecycle in Approval Domain |
| `ExecutionState.WAITING_FOR_HUMAN` substituting approval workflow | Approval Domain process; optional **bridge** to existing Nexus HITL pause/resume only |
| Decision outcome substituting TaskState / RunState | Decision outcome references execution via neutral `ExecutionProvenanceRef` when applicable |
| Governance evidence record substituting ProofReceipt | Governance evidence **references** platform evidence; Proof Receipts remains owner |
| Approval/evidence alone authorizing execution | MP-INV-23 — authorization remains policy + Governed Execution path |

**Normative invariants (Multiplayer register):** MP-INV-09 (Decision ≠ HITL), MP-INV-23 (approval/evidence ≠ execution authorization).

---

## Dependency direction (frozen)

```text
intergrax/contracts/
    |
    +-- decision contracts        (MP-4B — `intergrax/contracts/decision.py`)
    +-- approval contracts        (MP-4C — future)
    +-- governance contracts      (MP-4D+ — future; posture/evidence linkage)

collaborative_work (MP-1…MP-3)
    |
    +-- references only (WorkItemId, WorkArtifactVersionId, Principal, ExecutionProvenanceRef)

execution runtime (UER / Nexus)
    |
    +-- execution lifecycle only; optional HITL bridge consumer

governed_execution
    |
    +-- policy evaluation + HITL machinery — reuse, not import into MP-4 contracts
```

**Forbidden production import directions (MP-4 contracts and services, when implemented):**

| Forbidden import / coupling | Reason |
|----------------------------|--------|
| `intergrax.runtime.nexus.*` | No Nexus orchestration ownership in MP-4 |
| `GraphExecutor`, Nexus task graph internals | Orchestration is UER |
| `TaskState`, Run lifecycle mutation from MP-4 | Execution state is UER |
| WorkItem lifecycle ownership from MP-4 | Collaborative Work owns WorkItem |
| ProofReceipt ownership / mutation | Evidence subsystem owns receipts |
| Memory lifecycle ownership | Memory domain |
| UCL / OptimizationArtifact ownership | UCL domain |
| New Principal / ACL subsystem | Reuse MP-1 authority |

**Allowed reference pattern:** neutral Tier-0 contracts (`ExecutionProvenanceRef`, `CollaborativePrincipal`, evidence reference handles) via `intergrax/contracts/*` only — no runtime module imports.

---

## Existing contract reuse audit (frozen)

Before any new MP-4 types, implementation **must reuse** existing platform contracts.

| Need | Reuse (owner) | Do not create |
|------|---------------|---------------|
| Actor identity | `CollaborativePrincipal`, MP-1 membership/delegation | `NewPrincipal`, `NewUserIdentity`, artifact-specific user types |
| Authorization evaluation | `EffectiveAuthorityRequest` / `EffectiveAuthorityDecision`, `PolicyDecision` / `PolicyAction` | Second ACL engine, `ArtifactApprover` hierarchy |
| Execution linkage | `ExecutionProvenanceRef` (Tier-0) | `NewExecutionIdentity`, embedding Task/Run in Decision identity |
| Evidence linkage | Existing evidence / proof reference patterns | ProofReceipt as Decision identity; mutable evidence on Decision |
| Workspace scope | `tenant_id` + `workspace_id` from Collaborative Work conventions | Parallel tenancy model |

**Audit conclusion (MP-4A):** no new identity, ACL, or execution-identity subsystem required. MP-4B…MP-4F introduce **decision/approval/governance-specific** contracts only where reuse is insufficient — each slice must cite reuse table compliance in its gate.

---

## Cross-domain reference model (conceptual — not implementation)

```text
WorkItem (CW)
  → may reference zero..N Decision (Decision Domain)
  → may reference WorkArtifactVersion (CW) independently of Decision

Decision (Decision Domain)
  → records what choice was made + outcome
  → may reference WorkArtifactVersionId (not own artifact lifecycle)
  → may reference ExecutionProvenanceRef (optional)

Approval (Approval Domain)
  → approval request + lifecycle + human actions
  → references WorkArtifactVersionId and/or DecisionId (not substitute artifact state)
  → may bridge to Governed Execution HITL for execution pause/resume

Governance evidence (Governance Domain)
  → collaborative governance posture + control satisfaction
  → references platform evidence handles — does not own ProofReceipt
```

---

## Non-goals (MP-4 program)

Explicitly **out of scope** for MP-4 architecture and all slices until separately gated:

- Runtime orchestration or workflow engine inside MP-4
- WorkArtifact / WorkArtifactVersion lifecycle ownership
- Task / Run / Attempt / Execution lifecycle ownership
- ACL or Principal subsystem duplication
- Persistence implementation (MP-4E)
- Slack / channel adapter ownership of Decision semantics
- Relabeling Nexus HITL rows as MP-4 implementation

---

## Implementation roadmap (decomposition frozen)

Full slice rows: [`plan/DECISION_APPROVAL_GOVERNANCE.md`](../maintainers/plans/DECISION_APPROVAL_GOVERNANCE.md).

| Slice | Scope | Status |
|-------|-------|--------|
| **MP-4A** | Ownership + contracts freeze (this gate) | **APPROVED / CLOSED** |
| **MP-4B** | Decision contracts | **READY_FOR_REVIEW** |
| MP-4C | Approval / HITL contracts | NOT STARTED |
| MP-4D | Authority integration | NOT STARTED |
| MP-4E | Persistence boundary | NOT STARTED |
| MP-4F | Evidence / provenance integration | NOT STARTED |
| MP-4G | Qualification | NOT STARTED |
| MP-4H | Final closure audit | NOT STARTED |

---

## Related documents

| Document | Role |
|----------|------|
| [`COLLABORATIVE_WORK.md`](COLLABORATIVE_WORK.md) | Collaborative work plane; MP-4 non-leakage boundary |
| [`GOVERNED_EXECUTION.md`](GOVERNED_EXECUTION.md) | Execution-boundary policy + canonical HITL |
| [`RELIABILITY_FAILURE_AND_HITL.md`](RELIABILITY_FAILURE_AND_HITL.md) | HITL reliability semantics |
| [`UNIFIED_EXECUTION_RUNTIME.md`](UNIFIED_EXECUTION_RUNTIME.md) | Execution identity boundary |
| [`MULTIPLAYER_AI.md`](../capabilities/architecture/MULTIPLAYER_AI.md) | Multi-layer feature coordination |
