# NPSC-5E/R1 — Final Canonical Retry & Attempt Qualification and Freeze

**Status:** `FROZEN / PASS`

**Verdict:** **PASS** (final freeze certification)

**Date:** 2026-09-09

**Branch:** `development`

**Task:** NPSC-5E/R1 Final — Canonical Retry & Attempt Qualification and Freeze

**Final freeze task START SHA:** `0dda823a2682551d0c8047fc6be288731a640ed2`

---

## Purpose

NPSC-5E/R1 freezes the **canonical bounded execution-attempt retry contract** within the existing Execution Engine. It is **not** checkpoint/resume (R2), partial fan-out recovery (R3), transport retry, step/node retry, agent invocation retry, or topology recovery.

R1 answers: after typed failure classification and gate evaluation, may the run transition to **exactly one** successor attempt under the **same RunId**, with truthful lineage and without authority, budget, deadline, governance, AC-3, or re-selection bypass.

```text
failure
  → ExecutionFailureClassification / ExecutionFailureKind
  → evaluate_execution_retry_eligibility
  → cancellation / terminal / deadline / budget gates
  → ExecutionAttemptRetryService.transition_for_retry
  → AttemptLifecycleService.transition_to_next_attempt
  → exactly one new AttemptId
  → same RunId
  → lineage RETRY_SUPERSEDED (when persistence wired)
  → canonical NexusGraphRunner path
```

---

## Provenance

| Label | SHA |
| ----- | --- |
| R1 implementation | `0dda823a2682551d0c8047fc6be288731a640ed2` |
| NPSC-5D FROZEN | `a4a1faca01cd5004e372f235132184a84aa5a6bd` |
| NPSC-5E lineage baseline | `a72c9b568c61e28180756059ae48a99fb56eaa19` |
| P0A qualification | `7a2fcc6415f8d56234977e98f8c7f3329b5e8e96` |
| Final freeze task start (`HEAD` == `origin/development`) | `0dda823a2682551d0c8047fc6be288731a640ed2` |

**Drift gate:** `git diff --name-only 0dda823..origin/development` → empty at task start. **No production code changed** in this freeze task.

---

## Implementation SHA

Commit: `0dda823a2682551d0c8047fc6be288731a640ed2`

Message: `feat(execution): add canonical bounded retry attempt semantics`

| Artifact | Module | Role |
| -------- | ------ | ---- |
| `ExecutionFailureKind`, `ExecutionRetryAction`, eligibility models | `intergrax/contracts/execution_retry.py` | Typed retry contracts |
| `ExecutionAttemptRetryService` | `intergrax/runtime/execution/retry/service.py` | Eligibility + bounded orchestration |
| `evaluate_execution_retry_eligibility` | `intergrax/runtime/execution/retry/policy.py` | Canonical eligibility seam |
| `compute_backoff_delay` | `intergrax/runtime/execution/retry/backoff.py` | Bounded backoff |
| `AttemptLifecycleService.transition_to_next_attempt` | `intergrax/runtime/execution/attempt_lifecycle/service.py` | Sole AttemptId transition authority |
| `NexusGraphRunner._transition_attempt_for_retry` | `intergrax/runtime/nexus/orchestration/graph_runner.py` | Orchestration wiring |

---

## Retry taxonomy (frozen)

| Retry kind | Owner | New AttemptId? |
| --- | --- | ---: |
| Transport retry | Worker transport | NO |
| Step/node retry | Nexus step policy / `StepRetryBudget` | NO |
| Agent invocation retry | `RetryEngine` | NO |
| **Execution attempt retry** | **`ExecutionAttemptRetryService` + `AttemptLifecycleService`** | **YES** |
| Topology recovery | Nexus / `LongRunningCoordinator` | depends (not R1) |

Transport retry ≠ step retry ≠ agent retry ≠ execution attempt retry ≠ topology recovery.

---

## Ownership

| Concern | Owner |
| --- | --- |
| Retry eligibility + orchestration | `ExecutionAttemptRetryService` |
| Attempt transition / AttemptId mint | `AttemptLifecycleService` (sole authority) |
| Lineage seal | `ExecutionLineagePersistence` via `seal_lineage_attempt` (records; does not decide) |
| Orchestration wiring | `NexusGraphRunner._transition_attempt_for_retry` |
| Run/attempt budget | `ResiliencePolicy.max_attempts` (max **total** attempts) |

No second retry runtime, scheduler, or attempt store.

---

## Failure classification

`ExecutionFailureClassification` / `ExecutionFailureKind` are **execution-attempt retry projections**, not a universal platform error ontology.

| Kind | Retry |
| --- | --- |
| `RETRYABLE_TRANSIENT`, `RETRYABLE_TIMEOUT` | eligible if all gates pass |
| `NON_RETRYABLE_PERMANENT` | blocked |
| `GOVERNANCE_DENIED`, `AUTHORITY_DENIED`, `TRUST_DENIED` | blocked |
| `CONTRACT_ERROR` | blocked |
| `UNKNOWN` | fail closed |
| `UNKNOWN_UNSAFE` / unknown side-effect without idempotency | no blind retry |
| `CANCELLED` | cancel |
| terminal success/cancel/deny | blocked |
| timeout | typed policy decides (not automatic) |

Decision contract: `FAIL` | `RETRY` | `CANCEL` only. No checkpoint/resume in R1.

---

## Eligibility

Canonical seam: `evaluate_execution_retry_eligibility`.

Mandatory gates: cancellation precedence, terminal precedence, max attempts (off-by-one proven), global deadline (`now + backoff >= deadline` → FAIL), unknown side-effect idempotency.

---

## Attempt identity

```text
RunId      = preserved
AttemptId  = new (sole mint: AttemptLifecycleService via mint_retry_attempt_id)
generation = canonical total attempt position
N → N+1 only (no skip, no fork, no N+2)
```

Static gate: `mint_retry_attempt_id` only in identity authority + attempt lifecycle service.

---

## Retry budget

`ResiliencePolicy.max_attempts` = maximum **total** execution attempts (e.g. `3` → attempts 1, 2, 3).

Distinct and not collapsed:

- `StepRetryBudget.max_retries`
- agent `RetryPolicy.max_retries`
- transport `RetryPolicy.max_retries`
- `RetryCoordinator.max_run_retries`

Retry does **not** reset execution budget, cost/token budget, deadline, or expand authority.

---

## Deadline

Retry cannot extend the original global deadline. Backoff deadline gate enforced in eligibility.

---

## Cancellation

Hard invariant: **cancellation > retry**. Cancel before retry and cancel during backoff → no new AttemptId.

---

## Backoff

Canonical: `compute_backoff_delay` with bounded `NONE`, `FIXED`, `EXPONENTIAL`, `JITTERED`; `max_delay_seconds` cap; provider `retry-after` bounded by max delay; jitter cannot exceed configured maximum.

---

## Authority / governance / trust

| Deny | Retry |
| --- | --- |
| governance deny | blocked |
| authority deny | blocked |
| AC-3 / trust deny | blocked |

`REQUEST_HUMAN` / HITL governed continuation ≠ execution attempt retry.

---

## Lineage

```text
attempt N → RETRY_SUPERSEDED → attempt N+1
```

Ordering: retry decision → attempt transition → lineage seal → active attempt rebind (when identity bound). Transition failure does not record false successor state.

---

## Idempotency / concurrency

| Scenario | Result |
| --- | --- |
| duplicate retry request | idempotent (second returns None) |
| concurrent retry from same predecessor | exactly one successor (CAS-safe) |
| attempt fork | blocked |
| attempt skip | blocked |

---

## No re-selection

Execution retry path does not invoke `AgentDiscoveryStrategy`, `CapabilityMatcher`, or `AgentSelectionStrategy`. Final qualification: **additional selector calls = 0** in `_transition_attempt_for_retry`.

Failover (S1 → S2) ≠ ordinary R1 retry.

---

## HITL / transport / step / agent distinction

| Path | Mints run AttemptId? |
| --- | ---: |
| HITL governed continuation | NO |
| worker redelivery (transport) | NO |
| node/step retry | NO |
| `RetryEngine` agent retry | NO |

---

## Test matrix (final qualification)

Gate modules:

- `tests/unit/runtime/architecture/test_npsc5e_r1_execution_retry_attempt_semantics.py` (R1 implementation)
- `tests/unit/runtime/architecture/test_npsc5e_r1_final_retry_attempt_qualification.py` (final cross-layer freeze)

| # | Scenario | Result |
| - | -------- | ------ |
| 1 | transient fail → retry → success path | PASS |
| 2 | permanent failure | PASS (no attempt2) |
| 3 | max attempts exhaustion / off-by-one | PASS |
| 4 | cancelled | PASS |
| 5 | deadline backoff gate | PASS |
| 6 | governance deny | PASS |
| 7 | authority deny | PASS |
| 8 | AC-3 / trust deny | PASS |
| 9 | unknown fail closed | PASS |
| 10 | unknown unsafe side-effect | PASS |
| 11 | duplicate retry | PASS |
| 12 | concurrent retry | PASS |
| 13 | attempt fork blocked | PASS |
| 14 | attempt skip blocked | PASS |
| 15 | no re-selection (static) | PASS |
| 16 | HITL distinct | PASS |
| 17 | transport / step / agent distinct | PASS |
| 18 | all `ExecutionFailureKind` handled | PASS |
| 19 | all `BackoffKind` bounded | PASS |
| 20 | terminal matrix | PASS |

**Session totals:** R1 implementation + final qualification = **99 passed**.

---

## Regression matrix

| Suite | Status |
| ----- | ------ |
| NPSC-5A coordination / delegation | PASS (30) |
| NPSC-5B fan-out / fan-in | (included above) |
| NPSC-5C decision / intent | (included above) |
| NPSC-5D R1/R2/R3 + final governance | PASS (84) |
| P0A lineage baseline | PASS (included in 78) |
| Attempt lifecycle | PASS |
| Lineage suites | PASS |
| Cancellation | PASS (47) |
| Terminal | PASS (included) |
| HITL R3 governed continuation | PASS (included in 5D R3) |
| Graph runner resilience | **PRE-EXISTING ONLY** (3 Task validation failures) |
| Budget ticks | **PRE-EXISTING ONLY** (5 identity binding failures) |
| Policy trace component | **PRE-EXISTING ONLY** (1 identity failure) |
| Checkpoint restore merge | **PRE-EXISTING ONLY** (1 resumable-state failure) |
| Timeout semantics | PASS (covered in R1 eligibility tests) |

---

## Known pre-existing failures

Unchanged by R1 implementation commit (`0dda823` did not modify these test files):

| Test | Failure |
| ---- | ------- |
| `test_graph_runner_resilience` (3 tests) | `Task` pydantic validation |
| `test_budget_ticks` (5 tests) | `active execution identity required` |
| `test_policy_trace_component::test_policy_trace_component_is_recorded` | `active execution identity required` |
| `test_checkpoint_store::test_restore_if_resuming_merges_snapshot` | checkpoint not resumable |

**NEW R1-related regressions: 0**

---

## Static quality

| Check | Result |
| ----- | ------ |
| Ruff (R1 production + tests) | PASS |
| Pyright (R1 production + tests) | PASS |
| NEW static errors | 0 |
| Authoritative retry path reflection | none |
| Second retry runtime names | none |

---

## Formal verdict

```text
NPSC-5E/R1 = FROZEN / PASS
NPSC-5E     = ACTIVE
```

Production code unchanged in this qualification task. Single canonical execution-attempt retry path proven with sole `AttemptLifecycleService` transition authority.

---

## Freeze statement

> Execution-attempt retry is a bounded canonical transition within the existing Execution Engine. `AttemptLifecycleService` is the sole authority for successor AttemptId creation. Retry preserves RunId and execution lineage, cannot expand authority, reset budget, extend deadline, bypass governance or AC-3, re-select a specialist, or create an alternate runtime or scheduler.

R1 frozen ≠ NPSC-5E frozen. R2 and R3 remain active under NPSC-5E.

---

## Deferred to R2

```text
checkpoint identity cross-validation
stale checkpoint handling
unknown checkpoint version handling
durable restore ownership
cross-process restore hardening
```

---

## Deferred to R3

```text
partial fan-out retry
exact failed-slot recovery
sibling preservation under failure recovery
child partial recovery
```

---

## Next task

**NPSC-5E/R2 — Checkpoint & Durable Resume Hardening**
