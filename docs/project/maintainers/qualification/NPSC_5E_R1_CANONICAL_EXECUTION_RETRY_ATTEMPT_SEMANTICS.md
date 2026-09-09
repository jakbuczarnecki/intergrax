# NPSC-5E/R1 — Canonical Execution Retry & Attempt Semantics

> **Status:** PASS (implementation qualification; not frozen)

## Scope

Canonical bounded execution-attempt retry: same logical run, new AttemptId, truthful lineage, no re-selection, no alternate runtime.

## Retry taxonomy

| Retry kind | Owner | New AttemptId? | Scope |
|---|---|---:|---|
| Transport retry | Worker transport | NO | delivery |
| Step/node retry | Nexus step policy | NO | node |
| Agent invocation retry | RetryEngine | NO | call |
| **Execution attempt retry** | **ExecutionAttemptRetryService + AttemptLifecycleService** | **YES** | run attempt |
| Topology recovery | Nexus | depends | orchestration |

## Ownership

| Concern | Owner |
|---|---|
| Attempt identity transition | `AttemptLifecycleService` |
| Failure classification projection | `runtime/execution/retry/classification` |
| Retry eligibility | `evaluate_execution_retry_eligibility` |
| Backoff | `compute_backoff_delay` |
| Run/attempt budget | `ResiliencePolicy.max_attempts` |
| Global deadline check | `ExecutionRetryEligibilityRequest` |
| Cancellation | `CancellationCoordinator` + eligibility |
| Orchestration wiring | `NexusGraphRunner._transition_attempt_for_retry` |

## Qualification

Gate module: `tests/unit/runtime/architecture/test_npsc5e_r1_execution_retry_attempt_semantics.py`

Covers: transient/permanent retry, budget off-by-one, cancellation, deadline, lineage RETRY_SUPERSEDED, CAS concurrency, no direct mint outside authority, static architecture gates.

## Baselines

```text
NPSC-5D FROZEN: a4a1faca01cd5004e372f235132184a84aa5a6bd
NPSC-5E LINEAGE: a72c9b568c61e28180756059ae48a99fb56eaa19
P0A: 7a2fcc6415f8d56234977e98f8c7f3329b5e8e96
```

## Next

NPSC-5E/R1 Final — freeze, or NPSC-5E/R2 checkpoint hardening.
