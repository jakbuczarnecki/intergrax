# NPSC-5E/R3 Final — Child & Fan-Out Partial Recovery Qualification and Freeze

**Status:** `FROZEN / PASS`

**Verdict:** **PASS** (final freeze certification)

**Date:** 2026-09-10

**Branch:** `development`

**Task:** NPSC-5E/R3 Final — Child & Fan-Out Partial Recovery Qualification and Freeze

---

## Purpose

Formal freeze of the canonical enterprise **partial recovery** contract for child execution and fan-out topology:

```text
exact failed recoverable slot recovery
+ successful sibling preservation
+ deterministic fan-in
+ cross-process continuation
+ R2 checkpoint revision CAS
+ R1 retry interoperability
+ current authority/governance
+ truthful lineage
+ canonical Nexus scheduling
+ canonical ChildExecutionPort
```

**Production code changed in this task:** `NO`

---

## Provenance

| Label | SHA |
| ----- | --- |
| NPSC-5D Final | `a4a1faca01cd5004e372f235132184a84aa5a6bd` |
| R1 Final | `76603ed266f9f54106bf4718fe8886180a826351` |
| R2 Final | `c030d1d4752513bc11ec2597de4e760ae551a978` |
| R3 implementation | `465742eba5d4162061797baafbbb3f90f414fcdd` |
| Final task `START_HEAD` | `465742eba5d4162061797baafbbb3f90f414fcdd` |
| Final task `START_ORIGIN` | `465742eba5d4162061797baafbbb3f90f414fcdd` |

**Drift vs R3 implementation (`465742e..origin/development`):** `NONE`

---

## Qualification gate

`tests/unit/runtime/architecture/test_npsc5e_r3_final_child_fanout_partial_recovery_qualification.py`

Composes R3 implementation gate, R1 Final, R2 Final, P0A, DG_001, NPSC-5A/B/C, NPSC-5D Final, HITL R3, attempt/child/terminal/cancellation/checkpoint/long-running/fan-out suites plus cross-layer partial-recovery E2E freeze scenarios.

---

## Ownership

| Concern | Owner |
| --- | --- |
| Recovery intent / policy | `PartialRecoveryRequest`, `evaluate_slot_recovery_policy` |
| Topology durable state | `TopologyRecoverySnapshot` (`RuntimeCheckpoint.topology_recovery`) |
| Fan-out partial recovery | `FanOutPartialRecoveryService` |
| Slot scheduling | `OrchestrationTopologyContinuationPort.recover_failed_slot` → Nexus |
| HITL continuation | `continue_slot` (NPSC-5D/R3 frozen) |
| Child execution | `ChildExecutionPort` |
| Checkpoint revision CAS | `TaskCheckpointPersistence` (R2 frozen) |
| Attempt retry | `ExecutionAttemptRetryService` (R1 frozen) |
| Lineage | `ExecutionLineagePersistence` |
| Fan-in projection | `BoundedMultiAgentFanOutService` (NPSC-5B frozen) |

---

## Recovery unit

```text
exact topology slot
or exact child Execution
≠ entire fan-out replay
```

---

## Typed contracts

| Contract | Role |
| --- | --- |
| `PartialRecoveryRequest` | identity-bound recovery intent |
| `SlotRecoveryDisposition` | durable per-slot state |
| `SlotRecoveryPolicyAction` | RECOVER / PRESERVE_FAILURE / WAIT / CANCEL |
| `evaluate_slot_recovery_policy` | narrow policy seam — no schedule/mint/execute |
| `TopologyRecoverySnapshot` | optional `runtime_checkpoint.v2` field |

---

## Slot state semantics

| Disposition | Automatic recovery |
| --- | --- |
| `SUCCEEDED` | NO (`PRESERVE_FAILURE` — terminal, no mutation) |
| `FAILED` / `INTERRUPTED` / `RECOVERING` | eligible when policy RECOVER |
| `WAITING_FOR_HUMAN` | NO — canonical `continue_slot` only |
| `UNKNOWN_UNSAFE` | NO — fail closed |
| cancelled / parent cancelled | NO |
| governance / authority / trust deny | NO |
| permanent / budget / deadline | NO |

---

## Successful sibling preservation

Successful siblings are never rescheduled because another slot failed. Results remain byte/semantic-equivalent to original canonical output. Lineage for successful siblings remains unchanged.

---

## Fan-in order / cardinality

Original request order and cardinality preserved. Recovered output replaces only its slot's failure position. Partial failure remains representable.

---

## R1 interoperability

R3 decides **which** slot may recover. R1 decides **whether** an execution attempt may retry. Transient child failure uses `ExecutionAttemptRetryService` → `AttemptLifecycleService`. R3 cannot reset R1 retry budget.

---

## R2 checkpoint / revision interoperability

Recovery persistence uses `TaskCheckpoint.revision` + `expected_revision` CAS. No second recovery revision authority. Stale recovery writers blocked. Cross-process partial recovery validated (process A checkpoint → process B recovery → revision N+1).

`RuntimeCheckpoint v2` remains backward-compatible; old v2 without `topology_recovery` parses with `topology_recovery=None`. Unknown checkpoint versions fail closed.

---

## Cross-process recovery

Mandatory scenario: partial fan-out → durable checkpoint revision N → process boundary → load N → recover exact failed slot → persist N+1. No process-local state required.

---

## Concurrency ownership

Same slot: exactly one effective recovery execution (idempotent correlation cache + post-success noop). Different failed slots: bounded concurrent recovery through canonical Nexus only. No local recovery scheduler.

---

## Authority / governance freshness

Current authority, governance and trust dominate historical checkpoint state. Authority narrowing preserved. Child authority ≤ parent authority. Historical checkpoint policy does not authorize recovery.

---

## AC-3

Current trust remains authoritative. Trust deny (`TRUST_DENIED`) blocks automatic recovery via policy.

---

## HITL

`WAITING_FOR_HUMAN` → `continue_slot` only. Never `recover_failed_slot`.

---

## Cancellation / terminal

Parent cancellation dominates recovery. Terminal incompatible states block recovery. Cancellation during recovery propagates through canonical Execution cancellation.

---

## Lineage

Recovered slot produces truthful lineage binding parent/root, topology, slot, recovery predecessor and new attempt/execution relationship. No synthetic lineage. Sealed attempts are not reopened. Successful sibling lineage unchanged.

---

## Side-effect safety

Unknown external effect → no blind replay. Crash during recovery fail-closed where state uncertain. Duplicate recovery idempotent. Unknown commit retry does not duplicate successful recovery.

---

## Regression matrix

| Suite | Result |
| --- | --- |
| R1 Final | PASS |
| R2 Final | PASS |
| R3 implementation gate | PASS |
| P0A | PASS |
| DG_001 | PASS |
| NPSC-5A | PASS |
| NPSC-5B Final | PASS |
| NPSC-5C | PASS |
| NPSC-5D Final | PASS |
| HITL R3 | PASS |
| Attempt lifecycle | PASS |
| Child execution | PASS |
| Terminal | PASS |
| Cancellation | PASS (minus documented pre-existing) |
| Checkpoint store | PASS |
| Long-running | PASS |
| Fan-out | PASS |

**Unexpected skips:** `0`

---

## Known pre-existing failures

| Test | Baseline | Classification |
| ---- | -------- | -------------- |
| `test_p0c5_cancellation_continuity.py::test_terminal_cancellation_survives_process_restart` | R2 Final | Fixture calls `persist_checkpoint` with default `TaskState.CREATED`; R2 `assert_checkpoint_persistable` rejects before terminal path. Terminal cancellation durability proven by other p0c5 tests (excluded from matrix). **Does not contradict R3 freeze.** |
| `test_partial_results.py::test_build_task_progress_view_aggregates_checkpoints` | pre-R2 | `human_request_expires_at` aggregation in progress view; unrelated to partial recovery gate. |

**New failures:** `0`

---

## Static quality

| Tool | Scope | Result |
| ---- | ----- | ------ |
| ruff | R3 production surface + final qualification test | PASS |
| pyright | R3 production surface + final qualification test | PASS |

**New static errors:** `0`

---

## Deferred scope

```text
full historical replay engine
evidence replay
time-travel execution
global recovery journal
arbitrary topology rewinding
```

Belongs primarily to NPSC-5F or future scope.

---

## Final verdict

```text
NPSC-5E/R3: FROZEN / PASS
NPSC-5E: ACTIVE (awaiting NPSC-5E Final)
```

---

## Freeze statement

> Canonical partial recovery in Intergrax operates on one exact failed or interrupted recoverable topology slot. Successful sibling executions, outputs and lineage remain immutable and are never replayed solely because another slot failed. Recovery preserves original topology identity and fan-in ordering, reuses frozen R2 durable checkpoint revision CAS, delegates attempt retry exclusively to frozen R1, refreshes current authority, governance and trust, schedules exclusively through Nexus, and executes children exclusively through ChildExecutionPort. HITL continuation, cancellation, permanent failures and unknown side-effect states remain distinct fail-closed paths.

**Next:** NPSC-5E Final — Recovery Plane Final Qualification and Freeze
