<!--
© Artur Czarnecki. All rights reserved.
Intergrax is source-available under the Intergrax Evaluation and Collaboration License 1.0.
See LICENSE for permitted evaluation, collaboration, and contribution use.
-->

# Decision / Approval / Governance — Multiplayer integration (MP-4 / MP-4R)

**Status:** **MP-4R0 — CURRENT** (core rebase & supersession gate) · legacy **MP-4A** `SUPERSEDED_BY_MP4R0` · **MP-4B** `FROZEN_PENDING_CONVERGENCE` · **MP-4C** `FROZEN_PENDING_CONVERGENCE` · **MP-4D** `FROZEN_PENDING_AUTHORITY_REBASE` · legacy MP-4E…MP-4H **cancelled/replaced** by MP-4R1…MP-4R8
**ADR:** [ADR-MP-009](../technical/adr/entries/2026-09-15/ADR-MP-009.md) (authoritative after MP-4R0) · [ADR-MP-005](../technical/adr/entries/2026-09-08/ADR-MP-005.md) (MP-4A historical; ownership table superseded)
**Feature coordination:** [`MULTIPLAYER_AI`](../capabilities/architecture/MULTIPLAYER_AI.md) · [`COLLABORATIVE_WORK`](COLLABORATIVE_WORK.md)
**Plan (1:1):** [`plan/DECISION_APPROVAL_GOVERNANCE.md`](../maintainers/plans/DECISION_APPROVAL_GOVERNANCE.md)

---

## Purpose

Define how **Multiplayer** integrates with canonical platform authorities for Decision, human authorization (Governance/HITL), Execution continuation, Evidence, and Diagnostics — **without** creating parallel lifecycle or truth sources.

MP-4R0 performs ownership rebase, legacy inventory, supersession, and architecture gates. It does **not** delete legacy MP-4B–D production modules.

---

## Canonical ownership (frozen at MP-4R0)

| Capability | Canonical owner | Multiplayer role |
|------------|-----------------|------------------|
| Decision identity / version | Decision System (`decision_identity`, …) | References / bindings only (MP-4R4+) |
| Decision lifecycle / resolution / finalization | Decision System + Execution host | None — no second lifecycle |
| Decision human review semantics | Decision System + canonical HITL contracts | Bridge via platform contracts only |
| Authorization / `REQUIRE_HUMAN` | Governance / HITL | Consume — do not re-own |
| Human authorization evidence | Governance / HITL + Evidence contracts | Link — do not own facts |
| Pause / wait / resume | Execution Engine | None |
| Public continuation boundary | `ExecutionContinuationPort` | Integrate via port (MP-4R3) |
| Execution identity / lifecycle | Execution Engine | `ExecutionProvenanceRef` references only |
| Orchestration | Nexus **internal** to Execution Engine | **No public Nexus dependency** |
| Principal / membership / delegation | MP-1 (Collaborative Work) | Reuse |
| WorkItem / Assignment | Multiplayer Collaborative Work (MP-2) | Owner |
| WorkArtifact / WorkArtifactVersion | Multiplayer Collaborative Work (MP-3) | Owner |
| Evidence facts | Observability / Evidence Plane | Emit/link canonical facts (MP-4R5) |
| Historical factual reconstruction | Shared Evidence Plane reconstruction | None |
| Diagnostic interpretation | Central Diagnostics | Read/integration via canonical contracts |
| Multiplayer Decision integration | Multiplayer plane | **Collaborative binding / projection only** |

**Hard invariants:**

```text
Multiplayer MUST NOT own a second Decision lifecycle.
Multiplayer MUST NOT own a second Approval/HITL authority.
Multiplayer MUST NOT own Execution lifecycle.
Multiplayer MUST NOT expose Nexus.
Multiplayer MUST NOT own evidence truth.
Multiplayer MUST NOT own diagnostic interpretation.
```

---

## Target integration model

```text
                PLATFORM CORE

      Canonical Decision System
                │
        Governance / HITL
                │
      ExecutionContinuationPort
                │
         Execution Engine
                │
           Evidence Plane
                │
           Diagnostics

                ↑
       typed references / bindings

          MULTIPLAYER PLANE
                │
       WorkItem / Assignment
                │
 WorkArtifact / WorkArtifactVersion
```

---

## Legacy MP-4 inventory (frozen — no deletion in MP-4R0)

| Component | Location | Classification (MP-4R0) |
|-----------|----------|-------------------------|
| MP-4B Decision contracts | `intergrax/contracts/decision.py` | `REPLACE_WITH_CANONICAL` → `decision_identity` / DS-CORE; `REMOVE_AFTER_CALLER_PROOF` |
| MP-4C Approval contracts | `intergrax/contracts/approval.py` | `REPLACE_WITH_CANONICAL` → `decision_human_review` + Governance/HITL; `REMOVE_AFTER_CALLER_PROOF` |
| MP-4D Approval service | `intergrax/approval/` | `MIGRATE` authority path to Governance/HITL + MP-1 `CollaborativeWorkEnforcementGate`; not a second HITL runtime |
| Contract / service tests | `tests/unit/contracts/test_decision_contracts.py`, `test_approval_*`, `tests/unit/approval/*` | `KEEP` until MP-4R6 caller proof |
| Architecture gates MP-4B–D | `tests/unit/contracts/test_*_architecture_gates.py`, `tests/unit/approval/test_approval_authority_architecture_gates.py` | `KEEP` — quarantine legacy surface |
| MP-4R0 collaborative gates | `tests/unit/runtime/architecture/test_mp4r0_multiplayer_rebase_architecture_gates.py` | `KEEP` — protect new boundaries |

**Caller audit (production):** `intergrax/contracts/decision.py` consumers are **only** legacy MP-4C (`approval.py`) and unit tests — **no** runtime Decision System consumer. Canonical Decision uses `intergrax/contracts/decision_identity.py` and related DS-CORE modules.

**Unique collaborative capability audit:** workspace-scoped `Decision` aggregate in MP-4B does **not** justify a second Decision authority — collaborative association is a **binding** concern (MP-4R4), not lifecycle ownership.

---

## Anti-substitution rules (retained — canonical owners)

| Forbidden | Correct model |
|-----------|---------------|
| Decision encoded as WorkArtifact body or WorkItem state | Canonical Decision + optional WorkItem binding |
| `WorkArtifactVersion.status = APPROVED` | Human review / governance outcomes via canonical contracts |
| `ExecutionState` substituting approval workflow | Governance/HITL + `ExecutionContinuationPort` |
| Decision outcome substituting TaskState / RunState | `ExecutionProvenanceRef` when correlated |
| Governance evidence substituting ProofReceipt | Evidence Plane owns facts; linkage only |
| Approval/evidence alone authorizing execution | MP-INV-23 — Governed Execution path |

**Normative:** MP-INV-09 (Decision ≠ HITL), MP-INV-23 (approval/evidence ≠ execution authorization).

---

## Contract-first integration (frozen)

```text
Multiplayer
   ↓
stable platform contract / port
   ↓
configured platform implementation
```

**Forbidden for new Multiplayer production modules:**

- `intergrax.runtime.nexus.*`, `GraphExecutor`, `NexusLoop`, `NexusIntakeRunner`
- New `DecisionId` / `DecisionLifecycle*` / `DecisionRuntime` / `MultiplayerDecisionEngine` (outside legacy quarantine)
- `MultiplayerHitlEngine`, `ApprovalRuntime`, `HumanReviewRuntime`
- `EvidenceStore`, `ExecutionReconstructor`, `DiagnosticEngine`, `ProblemLifecycleEngine` as Multiplayer-owned authorities

**Reuse (mandatory):** `ExecutionContinuationPort`, Decision System contracts, Governance/HITL contracts, Evidence Plane contracts, Diagnostic read surfaces, MP-1 authority.

---

## MP-4R roadmap (replaces MP-4E…MP-4H)

| Slice | Status |
|-------|--------|
| **MP-4R0** — Core rebase & supersession gate | **CURRENT** |
| MP-4R1 — Decision contract convergence | NOT STARTED |
| MP-4R2 — Human review / Approval convergence | NOT STARTED |
| MP-4R3 — Execution continuation integration | NOT STARTED |
| MP-4R4 — Collaborative decision binding | NOT STARTED |
| MP-4R5 — Evidence Plane adoption | NOT STARTED |
| MP-4R6 — Legacy removal & migration | NOT STARTED |
| MP-4R7 — Enterprise integration qualification | NOT STARTED |
| MP-4R8 — Final closure audit | NOT STARTED |

Detail: [`plan/DECISION_APPROVAL_GOVERNANCE.md`](../maintainers/plans/DECISION_APPROVAL_GOVERNANCE.md).

---

## Legacy slice status (historical)

| Stary slice | Status |
|-------------|--------|
| MP-4A | `SUPERSEDED_BY_MP4R0` |
| MP-4B | `FROZEN_PENDING_CONVERGENCE` |
| MP-4C | `FROZEN_PENDING_CONVERGENCE` |
| MP-4D | `FROZEN_PENDING_AUTHORITY_REBASE` |
| MP-4E | `CANCELLED_BEFORE_START` |
| MP-4F | `CANCELLED_REPLACED_BY_EVIDENCE_ADOPTION` |
| MP-4G | `CANCELLED_IN_OLD_FORM` |
| MP-4H | `REPLACED_BY_MP4R8` |

---

## Related documents

| Document | Role |
|----------|------|
| [`DECISION_SYSTEM.md`](DECISION_SYSTEM.md) | Canonical Decision authority |
| [`GOVERNED_EXECUTION.md`](GOVERNED_EXECUTION.md) | Governance / HITL |
| [`UNIFIED_EXECUTION_ARCHITECTURE.md`](UNIFIED_EXECUTION_ARCHITECTURE.md) | Execution + internal Nexus |
| [`OBSERVABILITY.md`](OBSERVABILITY.md) | Evidence Plane |
| [`DIAGNOSTICS.md`](DIAGNOSTICS.md) | Diagnostic interpretation |
| [`COLLABORATIVE_WORK.md`](COLLABORATIVE_WORK.md) | MP-1…MP-3 ownership |
