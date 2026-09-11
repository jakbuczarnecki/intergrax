# NPSC-5D/R2 — Final Qualification, Cross-Layer Physical Delegation Governance & Freeze

**Status:** `FROZEN / PASS`

**Verdict:** **PASS** (final freeze certification)

**Date:** 2026-09-09

**Branch:** `development`

**Task:** NPSC-5D/R2 Final — Physical Delegation Governance Qualification & Freeze

---

## Purpose

NPSC-5D/R2 freezes the **post-selection, pre-acquisition physical delegation governance contract**. It is **not** R3 HITL implementation, pause/resume, selection redesign, or trust redesign.

R2 answers: **whether the exact selected specialist identity may proceed to acquisition** after Agent Distribution selection and before `TaskScopedAgentService.acquire` / AC-3 trust admission.

```text
R1 stage-1 ALLOW
    ↓
TaskCapabilityResolver / resolved requirement path
    ↓
AgentDiscoveryStrategy
    ↓
CapabilityMatcher
    ↓
AgentSelectionStrategy
    ↓
require_selected_identity
    ↓
PhysicalDelegationGovernancePort
    ↓
ALLOW / DENY / REQUIRE_HUMAN
    ↓
(if ALLOW) build acquisition plan
    ↓
TaskScopedAgentService.acquire
    ↓
canonical AC-3 trust/admission
    ↓
SpecialistInvocationPort
    ↓
ChildExecutionPort
```

`REQUIRE_HUMAN` preserves exact `PhysicalDelegationGovernedContinuation` — no re-selection, no lease, no child Execution.

---

## Baseline / provenance

| Label | SHA |
| ----- | --- |
| `origin/development` baseline (task start) | `5a768aacc2c65d56d63bd06a4c6027c9e2491f57` |
| NPSC-5D/R1 FREEZE | `d90c4c67010c00d4e24be021a87a123c6de1cb25` |
| R2 implementation | `d61fe0b25194b8a229842c79fa884ecf721d711a` |
| R2 fan-out hardening | `555ea245176eb26995074fb50254ce4807de710a` |
| R2-H1 continuation preservation | `74aff023983c126d0bfd65a0e6c79a194f931353` |

**Unrelated commits after R2-H1** (do not block R2 freeze):

```text
VPI storage-bootstrap operator + integration tests only
```

No shared NPSC / Governance / DelegatedSubtask / Fan-out / AC-3 seam changed after R2-H1.

---

## Implementation chain (frozen)

| Artifact | Module | Role |
| -------- | ------ | ---- |
| `PhysicalDelegationGovernanceRequest` | `intergrax/contracts/physical_delegation_governance.py` | Typed post-selection physical delegation facts |
| `PhysicalDelegationGovernedContinuation` | `intergrax/contracts/physical_delegation_governance.py` | Immutable governed continuation (R2-H1) |
| `PhysicalDelegationGovernancePort` | contracts | Public evaluator boundary |
| `PhysicalDelegationGovernanceBoundary` | `intergrax/runtime/governance/physical_delegation_governance.py` | Fail-closed admission |
| `build_physical_delegation_governance_request` | `intergrax/agent_distribution/physical_delegation_governance_adapter.py` | Caller adapter |
| `DelegatedSubtaskService` | `intergrax/agent_distribution/delegated_subtasks.py` | Canonical evaluation location |
| `GovernanceRequiresHumanError` / `FanOutItemFailure` | Agent Distribution + Execution adapter | Continuation propagation (R2-H1) |

---

## Ownership table

| System | Owns |
| ------ | ---- |
| **Decision System** | WHAT work is required |
| **Agent Distribution** | WHO (discovery, selection, lease lifecycle via `TaskScopedAgentService`) |
| **Governance (R2)** | WHETHER exact selected delegation may proceed |
| **AC-3** | Package/trust admission for runtime |
| **Execution** | Child lifecycle + authority |
| **Nexus** | HOW / WHEN fan-out scheduling |
| **HITL (R3)** | Human decision owner for governed continuation |

**Never:**

- R2 ranks, selects, or substitutes specialists
- R2 acquires leases or invokes child Execution
- R2 ALLOW implies AC-3 ALLOW
- R1 DENY reaches R2 evaluation
- Continuation evidence authorizes future execution by itself

---

## Gate position (frozen)

```text
selection < R2 governance < acquisition
```

No pre-selection governance. No post-acquisition governance.

**Evaluation point:** `GovernanceEvaluationPoint.MULTI_AGENT_DELEGATION` (distinct from R1 `MULTI_AGENT_COORDINATION`).

---

## Semantics (frozen)

### ALLOW

May proceed to normal acquisition / trust / execution only. Does not grant package trust, lease authority, child authority expansion, budget expansion, or permission expansion.

### DENY

Zero side effects: no acquisition plan side effects beyond evaluation, no `TaskScopedAgentService.acquire`, no lease, no delegate, no child Execution, no release.

### REQUIRE_HUMAN

Exact selected physical identity known → `PhysicalDelegationGovernedContinuation`. No lease, delegate, child, or re-selection.

### AC-3 independence

R2 ALLOW does not imply AC-3 ALLOW. R2 ALLOW + AC-3 DENY → no executable child.

### Fan-out (NPSC-5B compatible)

- `GOVERNANCE_REQUIRES_HUMAN` → typed continuation required on `FanOutItemFailure`
- All other failure codes → continuation forbidden
- Mixed A ALLOW / B DENY|REQUIRE_HUMAN / C ALLOW preserves 3 outputs, request order, sibling isolation
- No new `WAITING` / `PAUSED` / `REQUIRES_HUMAN` fan-out statuses

---

## Continuation contract (R2-H1)

Location: `intergrax/contracts/physical_delegation_governance.py`

Preserves: `delegation_id`, `task_scope_id`, `application_id`, `application_environment_id`, `selected_identity`, `capability_requirement`, `governance_result`, governance evidence.

Validation enforces: `governance_result.permitted == False`, `requires_governed_continuation == True`, evidence fields match continuation identity.

---

## Test matrix (final qualification)

| # | Scenario | Result |
| - | -------- | ------ |
| 1 | SINGLE R1 ALLOW + R2 ALLOW | PASS |
| 2 | SINGLE R1 ALLOW + R2 DENY | PASS |
| 3 | SINGLE R2 REQUIRE_HUMAN | PASS |
| 4 | Exact selected identity preserved | PASS |
| 5 | No fallback candidate after DENY | PASS |
| 6 | R2 DENY before acquire | PASS |
| 7 | R2 REQUIRE_HUMAN before acquire | PASS |
| 8 | FAN_OUT ALLOW / DENY / ALLOW | PASS |
| 9 | FAN_OUT ALLOW / REQUIRE_HUMAN / ALLOW | PASS |
| 10 | R1 DENY short-circuits R2 (0 evaluations) | PASS |
| 11 | AC-3 deny after R2 allow | PASS |
| 12 | Decision-backed path | PASS |
| 13 | Deterministic producer path | PASS |

**Final qualification test:** `tests/unit/runtime/architecture/test_npsc5d_r2_final_qualification.py`

Focused boundary suites: `test_physical_delegation_governance_boundary.py`, `test_physical_delegation_governance.py`, `test_delegated_subtasks.py`, `test_bounded_multi_agent_fanout.py`.

---

## Regression matrix

| Suite | Status |
| ----- | ------ |
| NPSC-5A coordination / delegation | PASS |
| NPSC-5B fan-out / fan-in | PASS |
| NPSC-5C decision projection / E2E | PASS |
| NPSC-5D/R1 final qualification | PASS |
| TaskScopedAgentService / AC-3 | PASS |
| Execution child authority | PASS |
| Runtime policy / governance | PASS |

---

## Deferred to R3

| Item | Scope |
| ---- | ----- |
| Human request creation | R3 |
| Approval persistence | R3 |
| Approval decision | R3 |
| Resume / child Execution continuation | R3 |
| Full pause/resume E2E | R3 |

**R3 invariant:** human approval must bind exact `PhysicalDelegationGovernedContinuation` — no re-selection.

---

## Formal verdict

```text
NPSC-5D/R1 = FROZEN / PASS
NPSC-5D/R2 = FROZEN / PASS
NPSC-5D     = ACTIVE (R3 remains)
```

Production code unchanged in this qualification task.

---

## Freeze statement

> **NPSC-5D/R2 freezes exactly one invariant: after Agent Distribution selection, each concrete physical delegation must pass Governance before acquisition; DENY ends it without side effects, REQUIRE_HUMAN preserves exact continuation identity, and ALLOW still does not replace AC-3, lease ownership, or Execution authority.**

R2 frozen ≠ NPSC-5D frozen. R3 remains active under NPSC-5D.

---

## Next task

**NPSC-5D/R3 — Canonical HITL Governed Continuation**
