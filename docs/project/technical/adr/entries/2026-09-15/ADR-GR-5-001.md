# ADR-GR-5-001: Canonical Execution HITL Continuation Ownership

| Field | Value |
|-------|-------|
| **Status** | Accepted (architecture) |
| **Date** | 2026-09-15 |
| **Task** | GR-5-ADR1 |
| **Audit HEAD** | `9336beff5ec72c747b440e63f3fb2dddc0b4bf8d` (`development`) |
| **Deciders** | Platform architecture / Governed Execution rebase |
| **Related** | [`UNIFIED_EXECUTION_RUNTIME.md`](../../../architecture/UNIFIED_EXECUTION_RUNTIME.md) · [`GOVERNED_EXECUTION.md`](../../../architecture/GOVERNED_EXECUTION.md) · [`GOVERNANCE_ARCHITECTURE_REBASE_GAP_LEDGER.md`](../../../maintainers/qualification/GOVERNANCE_ARCHITECTURE_REBASE_GAP_LEDGER.md) · [ADR-GOVERNED-CONTINUATION-001](../2026-07-20/ADR-GOVERNED-CONTINUATION-001.md) |

## 1. Context

GR-5 audit on `development` at the listed HEAD established **`MISSING_CANONICAL_EXECUTION_HITL_LIFECYCLE_CONTRACT`**: governed continuation **authorization** (four-ID grants, one-shot consume, stale mismatch fail-closed) is materially present, but **pause / wait / human resolution / resume** of the **same** Execution is still orchestrated primarily through **Task lifecycle**, **Task governance projection**, and **Nexus intake/orchestration** paths without a single typed Execution Engine lifecycle contract.

Prior working notes sometimes mislabeled **Nexus as an external peer** of the Execution Engine. That topology is **rejected**. Nexus is an **internal orchestration subsystem** of the Execution Engine. The defect is not Nexus exists, but **dual lifecycle authority** when Task/Nexus-visible state can diverge from canonical Execution lifecycle truth.

Canonical architecture already assigns HITL lifecycle consequences to **Unified Execution Runtime (UER)**:

```text
RUNNING → WAITING_FOR_HUMAN / PAUSED → RESUMED
```

([`UNIFIED_EXECUTION_RUNTIME.md`](../../../architecture/UNIFIED_EXECUTION_RUNTIME.md) § HITL; UER-INV-006 pause/resume preserves identity.)

**Governance** owns whether `REQUIRE_HUMAN` applies, exact blocked scope, and whether human evidence authorizes continuation. **Governance does not** own execution scheduling, lifecycle state machine mechanics, or resume transport.

### 1.1 Accepted audit facts (verified or consistent at HEAD)

| Fact | Status at HEAD |
|------|----------------|
| `GovernedContinuationRequest` carries TaskId, RunId, AttemptId, ExecutionId | Yes (`intergrax/contracts/governed_continuation.py`) |
| `GovernedContinuationApprovalGrant` exact-scope, one-shot, stale mismatch fail-closed | Yes (GR-1 / PG-FIX-C mechanism tests) |
| Duplicate resolution guarded (`HumanPauseCoordinator.resolve_human_response`) | Yes (`intergrax/runtime/human/pause.py`) |
| `WAITING_FOR_HUMAN` driven via `TaskLifecycle` + checkpoint/Nexus flow | Yes (`meaningful_side_effect_authorization.py`, Nexus runners) |
| Dedicated canonical Execution HITL continuation **port** in `intergrax/contracts` | **No** |
| `ExecutionLifecyclePort` (ERL) | Yes — **recovery handoff only** (`apply_recovery_lifecycle_intent`), not HITL suspend/resume |

### 1.2 Corrected Execution Engine topology

```text
EXECUTION ENGINE
│
├── canonical Execution lifecycle (authority target)
├── canonical execution identity (TaskId, RunId, AttemptId, ExecutionId)
├── ExecutionRuntime / Execution boundary
├── Nexus (INTERNAL — orchestration, planning, graph, HITL orchestration)
├── HITL integration (HumanRequest, resolution evidence)
├── checkpoint / continuation persistence adapters
└── execution strategies (INFERENCE, AGENTIC, ORCHESTRATION)
```

**Forbidden:** Nexus as independent execution truth parallel to canonical Execution lifecycle.

**Allowed:** Nexus implements orchestration and may **internally** re-enter graph/planning after a canonical resume transaction for the **same** four IDs.

## 2. Problem statement

External and cross-layer consumers lack one **platform-level, hard-typed** contract to suspend the exact Execution, inspect pending continuation, apply human resolution, resume without GR-2 root admission, reject stale/duplicate resolutions, and restore after process restart.

Today, **`NexusIntakeRunner`** + **`HumanPauseCoordinator`** + **`TaskState.WAITING_FOR_HUMAN`** implement much of the behavior as orchestration-path machinery, which risks a **de facto second lifecycle authority** unless unified under Execution Engine contracts.

## 3. Central questions — answers

**Canonical contract:** **`ExecutionContinuationPort`** at `intergrax/contracts`, implemented inside Execution Engine. Not `HumanPauseCoordinator`; not raw Nexus runner APIs.

**Terminology:** Reuse UER **`PAUSE_REQUESTED` / `PAUSED` / `WAITING_FOR_HUMAN` / `RESUMED`**. Port verbs may map suspend/resume to those facts.

**`ExecutionInterrupt`:** signal only. **`AgentExecutionResult(NEEDS_INPUT)`:** projection only.

## 4. Alternatives

| Option | Verdict |
|--------|---------|
| **A — `ExecutionContinuationPort`** | **Chosen** |
| **B — Extend `ExecutionLifecyclePort`** | **Rejected for HITL** (ERL recovery-only) |
| **C — Task/Nexus already canonical** | **Rejected** (GOV-GAP-003 OPEN) |

**Selected design:** **`INTRODUCE_CANONICAL_EXECUTION_CONTINUATION_CONTRACT`**.

## 5. Decision (summary)

Single Execution Engine lifecycle authority; Nexus internal orchestration only; Governance/HITL → `ExecutionContinuationPort` → implementation (Nexus, checkpoint, Task projection). Preserve governed continuation contracts. Same four IDs on pause/resume; resume is not root admission. Reject maps to existing semantics (today `TaskState.FAILED` on Nexus rejection path). Injectable pending-continuation persistence; checkpoint under canonical state.

## 6. Architecture diagram

```text
APPLICATION / GOVERNANCE / HITL UI
        │
        ▼
ExecutionContinuationPort
        │
        ▼
EXECUTION ENGINE
        ├── canonical Execution lifecycle
        ├── Nexus (internal)
        ├── checkpoint
        ├── HITL adapters
        └── Task projection
```

## 7. Authority table

| Concern | Canonical owner |
|---------|-----------------|
| execution identity | Execution Engine |
| lifecycle state | Execution Engine |
| orchestration implementation | Nexus (inside Execution Engine) |
| policy REQUIRE_HUMAN | Governance |
| human request | HITL subsystem |
| approval evidence | HITL/Governance contracts |
| continuation authorization | Governance/grant validation |
| persistence | injected port |
| Task state | projection |
| ExecutionInterrupt | signal |
| checkpoint | durability adapter |

## 8. Component classification (HEAD)

| Component | Category |
|-----------|----------|
| ExecutionRuntime / Execution | EXECUTION CORE |
| GovernedContinuation* / Grant | CONTRACT |
| ExecutionLifecyclePort (ERL) | CONTRACT (recovery handoff) |
| HumanPauseCoordinator | EXECUTION/HITL TASK PROJECTION |
| TaskState.WAITING_FOR_HUMAN | EXECUTION PROJECTION |
| NexusIntakeRunner | EXECUTION/NEXUS INTERNAL (resume spine today; port in GR-5-R4) |
| NexusHitlRunner / planning runners | EXECUTION/NEXUS INTERNAL |
| MeaningfulSideEffectAuthorizationBoundary | GOVERNANCE COMPONENT |
| Task governance as sole pause truth | LEGACY DUPLICATE AUTHORITY |

## 9. Contract shape (GR-5-R1)

Typed four IDs; `request_human_pause`, `get_pending`, `apply_resolution`, `resume` — see full narrative in gap ledger cross-link. No Nexus/vendor/UI types on public surface.

## 10–12. Checkpoint, pluginability, future gates

Canonical state → injectable persistence → Task projection. Platform-fixed: identity invariants and lifecycle semantics. Future: external layers must not bypass port to call Nexus lifecycle internals.

## 13. GR-5-R* roadmap

| ID | Goal |
|----|------|
| GR-5-R1 | Typed `ExecutionContinuationPort` + DTOs |
| GR-5-R2 | UER owns pause/resume transitions |
| GR-5-R3 | Task/HumanPauseCoordinator projection alignment |
| GR-5-R4 | Nexus internal HITL integration via port |
| GR-5-R5 | Restart + exact identity qualification |

## 14–18. Regression, risks, consequences, gap status

PG-FIX-C grant **VERIFIED**; lifecycle ownership **OPEN**. GR-5-R1 **NEXT**. GR-6 **BLOCKED** on GR-5.

---

*Architecture decision only — implementation requires independent code audit on GitHub.*
