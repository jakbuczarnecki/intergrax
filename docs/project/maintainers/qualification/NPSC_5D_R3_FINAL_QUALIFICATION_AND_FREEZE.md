# NPSC-5D/R3 — Final Qualification, Canonical HITL Governed Continuation & Freeze

**Status:** `FROZEN / PASS`

**Verdict:** **PASS** (final freeze certification)

**Date:** 2026-09-09

**Branch:** `development`

**Task:** NPSC-5D/R3 Final — Canonical HITL Governed Continuation Qualification & Freeze

---

## Purpose

NPSC-5D/R3 freezes the **canonical human-governed continuation contract** for:

```text
SINGLE + FAN_OUT exact slot continuation
```

R3 is **not** new HITL runtime, new Nexus scheduling, recovery/checkpoint, or NPSC-5E durability.

R3 answers: after R2 `REQUIRE_HUMAN`, how does canonical HITL pause → exact approval grant → policy-fresh resume → AC-3 → child execution proceed **without re-selection**, and how does Nexus resume **only the exact blocked fan-out slot** while preserving successful siblings.

---

## Baseline / provenance

| Label | SHA |
| ----- | --- |
| `origin/development` baseline (task start) | `738c05708ba8ab947886b750687dd6d96dfea479` |
| Session `START_HEAD` | `5e2fef428329608e2437c8d5a1da87ac0b6848c7` |
| NPSC-5D/R1 FREEZE | `d90c4c67010c00d4e24be021a87a123c6de1cb25` |
| NPSC-5D/R2 FREEZE | `7fef5f942cc2697f080edc14521e23792331a7a5` |
| NPSC-5D/R3 BASE IMPLEMENTATION | `36c2be6a03411e117250c56c4519d759682b8032` |
| NPSC-5D/R3-H1 | `39e923494bac55f89858471072fd809c311923a8` |
| NPSC-5D/R3 Final | `<this commit>` |

**H1 drift gate:** no R3 seam changes between R3-H1 and `origin/development` at task start. Parallel commits after H1 were VPI-only (storage-bootstrap, diagnostics docs) — not NPSC/R3 contract drift.

**Unrelated commits excluded from R3 provenance:** VPI storage-bootstrap, diagnostics lineage docs.

---

## Canonical SINGLE topology

```text
R1 ALLOW
  → Agent Distribution selects S1
  → R2 REQUIRE_HUMAN
  → PhysicalDelegationGovernedContinuation(S1)
  → GovernedContinuationRequest projection
  → ExecutionInterrupt + HumanRequest
  → HumanPauseCoordinator
  → HumanApprovalResolution (APPROVE)
  → PhysicalDelegationContinuationApprovalGrant
  → continue_governed_delegation
  → current policy evaluation
  → consume matching grant (at-most-once)
  → TaskScopedAgentService.acquire
  → AC-3 trust admission
  → ChildExecutionPort
  → release
  → DelegatedSubtaskResult
```

**Resume provenance:** `DelegatedSelectionProvenanceKind.PRESERVED_GOVERNED_CONTINUATION` with `selection_decision is None`.

---

## Canonical FAN_OUT continuation topology

```text
FanOutRequest(A,B,C)
  → canonical Nexus topology submit
  → A SUCCESS / B GOVERNANCE_REQUIRES_HUMAN / C SUCCESS
  → human approves exact B continuation
  → OrchestrationTopologyContinuationPort.continue_slot
  → FanOutCoordinationSlotExecutor.continue_slot
  → continue_governed_coordination
  → continue_governed_delegation (B only)
  → aggregate: A old result, B resumed result, C old result
```

**Forbidden on resume path:** whole fan-out rerun, whole topology resubmit, sibling rerun, synthetic selection.

---

## Ownership table

| System | Owns |
| ------ | ---- |
| **Decision** | WHAT work is required |
| **Agent Distribution** | WHO + exact physical delegation continuation semantics |
| **Governance** | WHETHER (policy freshness on resume) |
| **HITL** | Human decision / approval identity |
| **Execution** | Child lifecycle + authority |
| **Nexus** | Topology + scheduling + exact slot continuation |
| **TaskScopedAgentService** | Acquisition + lease |
| **AC-3** | Package trust / runtime admission |

---

## Truthful selection provenance

| Path | `selection_provenance_kind` | `selection_decision` |
| ---- | --------------------------- | -------------------- |
| Initial selection | `SELECTED` | required |
| Governed resume | `PRESERVED_GOVERNED_CONTINUATION` | `None` |

Contract rejects `SELECTED + None` and `PRESERVED_GOVERNED_CONTINUATION + decision`.

---

## Canonical HITL lifecycle (reused, not duplicated)

```text
GovernedContinuationRequest
ExecutionInterrupt
HumanRequest
HumanPauseCoordinator
HumanApprovalResolution
PhysicalDelegationContinuationApprovalGrant
```

No new R3 HITL engine. `GovernedContinuationApprovalGrant` (side-effect scope) ≠ `PhysicalDelegationContinuationApprovalGrant` (physical delegation).

---

## Approval grant binding

`PhysicalDelegationContinuationApprovalGrant` binds:

```text
continuation_digest
delegation_id
task_scope_id
run_id
selected_identity
capability_requirement
governance_request_digest
pause_id
human_request_id
+ policy provenance fields
```

Grant is at-most-once: consume before acquisition, cleared on use.

---

## Replay protection (fail-closed)

Blocked replays proven in final qualification:

```text
wrong pause_id
wrong human_request_id
wrong run_id
wrong delegation_id
wrong selected_identity
wrong capability_requirement
wrong governance_request_digest
duplicate grant consumption
wrong topology execution
wrong slot continuation
```

---

## Policy freshness

After human `APPROVE` and before acquisition:

- current physical delegation policy is evaluated;
- `APPROVE` + current policy `DENY` → `DENY` (no acquire, no child);
- stale/incompatible governance requirement/digest → fail-closed;
- exact matching consumed grant may satisfy current `REQUIRE_HUMAN` per R3 semantics.

---

## AC-3 independence

```text
human APPROVE ≠ AC-3 ALLOW
```

`APPROVE` + AC-3 `DENY` → no child execution.

---

## Nexus exact slot continuation

Public seam: `OrchestrationTopologyContinuationPort`

Implementation: `CanonicalOrchestrationTopologySubmissionPort.continue_slot`

Nexus owns exact slot scheduling/context. Agent Distribution owns physical delegation resume semantics. Nexus core does not import physical delegation identity types.

---

## Sibling preservation

For `A SUCCESS / B REQUIRE_HUMAN / C SUCCESS`:

- before approval: child counts `A=1, B=0, C=1`;
- after successful B resume: `A=1, B=1, C=1`;
- A/C results unchanged; cardinality and request order preserved.

---

## Authority monotonicity

Human approval does not expand authority, budget, principal, or tenant. Child authority remains `child <= parent`.

---

## Test matrix (final qualification)

| # | Scenario | Result |
| - | -------- | ------ |
| 1 | Architecture gates (no synthetic selection / discovery / matcher / selector on resume) | PASS |
| 2 | Architecture gates (no whole fan-out / topology resubmit) | PASS |
| 3 | Architecture gates (Nexus core neutral; no new scheduler/runtime) | PASS |
| 4 | Physical grant ≠ side-effect grant | PASS |
| 5 | Result contract invariant | PASS |
| 6 | SINGLE HITL E2E | PASS |
| 7 | SINGLE replay matrix | PASS |
| 8 | SINGLE ESCALATE / REJECT | PASS |
| 9 | Policy freshness + stale requirement | PASS |
| 10 | FAN_OUT HITL E2E + sibling counters | PASS |
| 11 | FAN_OUT wrong topology / wrong slot / duplicate resume | PASS |
| 12 | FAN_OUT REJECT / AC-3 deny preserve siblings | PASS |
| 13 | Multiple blocked slot identities independent | PASS |
| 14 | One approval ≠ whole fan-out | PASS |
| 15 | Decision-backed path uses same R3 mechanics | PASS |

**Final qualification test:** `tests/unit/runtime/architecture/test_npsc5d_r3_final_qualification.py`

**Focused R3 suites:** `test_npsc5d_r3_governed_continuation.py`, `test_physical_delegation_governance_boundary.py`, `test_g5b_hitl_resolution.py`, `test_g5c2b1_governed_continuation_grant.py`.

---

## Regression matrix

| Suite | Status |
| ----- | ------ |
| NPSC-5A coordination / delegation | PASS |
| NPSC-5B fan-out / fan-in | PASS |
| NPSC-5C decision projection / E2E | PASS |
| NPSC-5D/R1 final qualification | PASS |
| NPSC-5D/R2 final qualification | PASS |
| NPSC-5D/R3 governed continuation | PASS |
| HITL / governed continuation grant | PASS |
| Nexus topology submission / fan-out adapter | PASS |
| TaskScopedAgentService / AC-3 | PASS |
| Physical delegation governance | PASS |

---

## Deferred scope (NPSC-5E / NPSC-5F)

```text
cross-process durable slot continuation
recovery / checkpoint restore
restart after process failure
durable continuation replay
extended observability / evidence replay
```

---

## Formal verdict

```text
NPSC-5D/R1 = FROZEN / PASS
NPSC-5D/R2 = FROZEN / PASS
NPSC-5D/R3 = FROZEN / PASS
NPSC-5D     = READY FOR FINAL QUALIFICATION
```

**Production code changed in this task:** NO

---

## Freeze statement

> **R3 freezes canonical human-governed continuation: approval is valid only for the exact blocked physical delegation and exact fan-out slot; resume is not a new selection, does not rerun siblings, remains subject to current policy, AC-3, and Execution authority.**

R3 frozen ≠ NPSC-5D frozen.

---

## Next task

**NPSC-5D Final — Multi-Agent Governance Final Qualification & Freeze**
