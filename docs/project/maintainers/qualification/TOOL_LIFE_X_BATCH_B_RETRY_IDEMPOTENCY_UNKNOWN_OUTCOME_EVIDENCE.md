# TOOL-LIFE-X Batch B — Retry, Idempotency & Unknown-Outcome Evidence

| Field | Value |
| --- | --- |
| **Task** | `TOOL-LIFE-X-BATCH-B` |
| **Parent** | `TOOL-LIFE-X` (§3.0.5) — **not CLOSED** |
| **Predecessor** | Batch A — **READY FOR INDEPENDENT AUDIT** @ `e5db3b29346000b2f1f2e8e2716aec7ba53297f2` (65/65 L2) |
| **START_HEAD** | `e5db3b29346000b2f1f2e8e2716aec7ba53297f2` |
| **Branch** | `development` |
| **Disposition** | **`TOOL-LIFE-X BATCH B — READY FOR INDEPENDENT AUDIT`** (pending publication commit SHA) |

**Non-closure:** Does not close `TOOL-LIFE-X`, does not promote global `FRZ-*` PASS, does not rerun Batch A 65-test suite (no shared admission semantics changed).

## Canonical owner inventory (closed-world)

| Concern | Owner |
| --- | --- |
| Tool invocation execution | `RuntimeToolInvoker` |
| Retry policy / attempt budget | `RuntimeToolInvoker._execute_with_policy` + `tool_operation_retry_classification.effective_max_retry_attempts` |
| Global execution deadline | UER / `deadline_scope` + `_remaining_parent_execution_seconds` |
| Operation safety classification | `ToolOperationRetryClass` + `classify_tool_operation_retry_class` (contract-only) |
| Idempotency identity / ledger | `IdempotencyPreEffectCoordinator` + `IdempotencyStore` |
| External-effect authorization | Governance ports + `require_meaningful_side_effect_authorization` |
| Provider invocation | `ToolExecutor` / registry handlers |
| Outcome certainty | `ToolEffectCertainty` + `classify_idempotency_outcome` |
| Cancellation / backoff races | `CancellationCoordinator` + cooperative delay + attempt re-auth |
| Retry evidence | Tool trace steps `tool_invocation_retry` / `tool_invocation_error` |

## Batch B scope delivered

- **B1** Typed `ToolOperationRetryClass` (no HTTP/name inference).
- **B2** Centralized bounded attempts; parent deadline gates inter-attempt retry; no timeout auto-retry on `FuturesTimeoutError`.
- **B3** `IDEMPOTENT` / deduplicated paths require idempotency key before `max_attempts` > 1; ledger operation identity unchanged.
- **B4** `UNCERTAIN` timeout + ledger `InvocationUncertaintyError` on blind replay (existing + L1).
- **B5** `_require_current_attempt_authorization` on every attempt including post-backoff.
- **B6** Cancellation/backoff/scope revocation negatives in Batch B + P0-SAFETY-8 tests.

## TL1–TL15 (Batch B reconciliation)

| ID | Classification | Owner | Evidence | Notes |
| --- | --- | --- | --- | --- |
| TL1 | ALREADY PROVEN (Batch A) | Boundary + invoker | Batch A L2 | Unchanged |
| TL2 | ALREADY PROVEN (Batch A) | Invoker deadline | Batch A L2 | Unchanged |
| TL3 | ALREADY PROVEN (Batch A scoped) | Cancel/backoff | Batch A + B L2 | Real-infra deferred parent |
| TL4 | **ALREADY PROVEN (Batch B L2)** | Classification + invoker | L2 57/57 | TL4 closure this batch |
| TL5 | **ALREADY PROVEN (Batch B L2)** | `ToolEffectCertainty` | L1 + side-effect suite | UNKNOWN replay blocked |
| TL6 | **ALREADY PROVEN (Batch B L2)** | Pre-effect coordinator | L2 | Key/tenant/identity negatives |
| TL7 | OPERATOR DECISION | Parent | — | Batch C |
| TL8 | ALREADY PROVEN (Batch A) | Boundary | Batch A | Unchanged |
| TL9 | ALREADY PROVEN (Batch A scoped) | Stale guard | Batch A | Provider matrix parent |
| TL10–TL13 | BLOCKED | Parent batches | — | D/E |
| TL14 | ALREADY PROVEN | Architecture | Prior U5 | Unchanged |
| TL15 | BLOCKED | Parent | — | E |

## Race / failure matrix (Batch B — selected rows)

| # | Scenario | Retry decision | Evidence |
| --- | --- | --- | --- |
| 1–2 | Success / transient | Policy + class | P0-SAFETY-8 |
| 3 | Non-retryable side effect | `max_attempts` → 1 | `test_batch_b_l1_non_idempotent_*` |
| 4–5 | Budget / deadline exhausted | Fail closed | Batch B + P0-SAFETY-8 |
| 6–8 | Cancel / backoff / scope revoke | No second attempt | Batch B + P0-SAFETY-8 |
| 8 | Lost ack (timeout) | No retry; UNCERTAIN ledger | L1 unknown-outcome |
| 9–10 | Duplicate / concurrent claim | Ledger block | side-effect safety + Batch B tenant |
| 11–13 | Key / tenant / payload | Store conflict / isolation | Batch B tests |
| 17–18 | Late terminal / cancel race | No resurrection (scoped) | Batch A ownership (unchanged) |
| 19 | Store unavailable | Out of scope | Batch C parent |

## Implementation

| Module | Change |
| --- | --- |
| `intergrax/runtime/tools/tool_operation_retry_classification.py` | **NEW** — classification + `effective_max_retry_attempts` |
| `intergrax/runtime/nexus/tools/invoker.py` | Delegate attempt budget to classification (request-aware) |
| `tests/unit/runtime/nexus/tools/test_tool_life_x_batch_b_retry_idempotency_unknown_outcome.py` | **NEW** — Batch B qualification |
| `tests/unit/runtime/security/test_p0_safety_8_retry_redelivery_authorization.py` | Governance identity fixture (Batch A/B coupling) |
| `tests/unit/runtime/tools/test_tools_side_effect_safety.py` | Governance identity + threading scope fixes |

## Validation

**L1** (4 tests, `-k batch_b_l1`): **4/4 passed** — log `.tmp/session/TOOL-LIFE-X-BATCH-B/l1-batch-b.log`

**L2** (consolidated):

```text
uv run pytest -p no:xdist \
  tests/unit/runtime/nexus/tools/test_tool_life_x_batch_b_retry_idempotency_unknown_outcome.py \
  tests/unit/runtime/security/test_p0_safety_8_retry_redelivery_authorization.py \
  tests/unit/runtime/tools/test_tools_side_effect_safety.py -q
```

| Metric | Value |
| --- | --- |
| **Passed** | **57** |
| **Failed** | **0** |
| **Log** | `.tmp/session/TOOL-LIFE-X-BATCH-B/l2-batch-b.log` |

## Scoped FRZ / ECAP

| ID | Batch B advance |
| --- | --- |
| FRZ-REL-03 | Side-effect retry safety + UNKNOWN — scoped tests, not global PASS |
| FRZ-REL-04 | Bounded retry + re-auth — scoped |
| FRZ-REL-06 | Idempotency coordinator path — scoped |
| FRZ-GOV-02 | Attempt-scoped governance evidence with active identity in qual tests |
| FRZ-BND-03 | No new retry/idempotency owner |

## Remaining parent obligations

- Batch C: STALLED/ABANDONED, worker takeover, recovery breadth.
- Batches D–E: provider real-infra + parent TL15 closure.
- Batch A 65-test L2 **not** re-run (dependency boundary unchanged).

## Publication

Record **execution SHA** after operator commit/push. Batch A evidence @ `e5db3b29346000b2f1f2e8e2716aec7ba53297f2` preserved.
