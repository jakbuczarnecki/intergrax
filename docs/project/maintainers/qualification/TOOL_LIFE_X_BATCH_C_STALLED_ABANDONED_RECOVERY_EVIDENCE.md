# TOOL-LIFE-X Batch C — Stalled / Abandoned / Recovery Evidence

| Field | Value |
| --- | --- |
| **Task** | `TOOL-LIFE-X-BATCH-C` |
| **Parent** | `TOOL-LIFE-X` (§3.0.5) — **not CLOSED** |
| **START_HEAD** | `beb46a447fe146eb4404e6ddb460fd425fb5fe2e` |
| **Remediation** | Independent audit C-F01–C-F04 (ownership + fencing + recovery authority) |
| **Authority decision** | **Path B** — no canonical certified owner-loss evidence port; RUNNING→UNKNOWN recovery **fail closed** |
| **Disposition** | **`TOOL-LIFE-X BATCH C — READY FOR INDEPENDENT AUDIT`** (publication SHA below) |

**Non-closure:** Does not close `TOOL-LIFE-X`, does not promote global `FRZ-*` PASS, does not start Batch D/E. Distributed worker-loss certification remains parent L3.

## Audit root causes (C-F01–C-F04)

| ID | Root cause | Remediation |
| --- | --- | --- |
| **C-F01** | `RuntimeToolInvoker` swept foreign `RUNNING` records via token inequality | Removed automatic `reconcile_orphaned_tool_external_operations` from invoke path; `reconcile_orphaned_running_operations` is explicit no-op |
| **C-F02** | `ensure_terminal_recorded` and idempotent terminal return bypassed owner fencing; `finalize_operation_intent` without owner predicate | Owner-matched terminal + `finalize_operation_intent_if_owner_matches`; cancellation observe passes owner |
| **C-F03** | Recovery sweep treated `different_owner ≠ dead_owner` | Recovery gate no longer sweeps by token; certified API name only (see C-F04) |
| **C-F04** | `recover_running_operation_after_certified_owner_loss` transitioned RUNNING→UNKNOWN without certified owner-loss authority | **Path B:** API raises `ExternalOperationCertifiedOwnerLossRequiredError`; durable record unchanged; no fabricated authority |

## Authority Path B — evidence

| Candidate | Scope | Applicable to RUNNING→UNKNOWN? |
| --- | --- | --- |
| `PreEffectSuspendedWorkRecoveryAuthority` | Pre-effect idempotency ledger (W2) | **No** — explicit pre-effect suspended-work only |
| `ProcessLocalExternalOperationOwner` token / revision | Process-local fencing | **No** — mismatch ≠ owner death |
| `reconcile_orphaned_running_operations` | Orphan sweep | **No** — no-op (C-F01) |
| Parent L3 distributed certification | Future composition | **Not wired** — obligation remains |

**Before (C-F04):** `recover_running_operation_after_certified_owner_loss(store, operation_id, expected_revision)` → CAS to UNKNOWN+TERMINATING when physical RUNNING.

**After:** Same signature; when physical RUNNING → `ExternalOperationCertifiedOwnerLossRequiredError` (no CAS). Non-RUNNING → idempotent return of loaded record.

## Canonical owner / caller inventory

| Caller | Authority | Proof | Transition | Rejection |
| --- | --- | --- | --- | --- |
| `recover_running_operation_after_certified_owner_loss` | **None (Path B)** | N/A | **Denied** RUNNING→UNKNOWN | `ExternalOperationCertifiedOwnerLossRequiredError` |
| `RuntimeToolInvoker` | Invoker + store | N/A | No recovery sweep | Foreign RUNNING preserved |
| `prepare_external_operations_for_recovery` | Recovery admission gate | Load physical state | None | `ExternalOperationRecoveryBlockedError` on RUNNING/UNKNOWN |
| `ToolExternalOperationAttempt` terminals | Owner-matched CAS | `owner.token` | SUCCEEDED/FAILED/UNKNOWN/CANCELLED | `ExternalOperationFencedCommitRejectedError` |
| `reconcile_orphaned_*` | None | N/A | No writes | Returns `()` |

## Terminal writers (tool path) — fencing status

| Writer | Fencing |
| --- | --- |
| `ToolExternalOperationAttempt._mark_terminal` | `mark_physical_terminal_if_owner_matches` + fenced finalize |
| `ensure_terminal_recorded` | Owner-matched UNKNOWN + fenced finalize |
| `complete_cancellation_after_termination` | `mark_observed_cancellation_terminal(..., owner=...)` |
| `mark_physical_terminal` (raw) | LLM/legacy paths unchanged; tool path avoids direct use |
| `reconcile_orphaned_*` | No terminal writes (no-op) |
| `recover_running_operation_after_certified_owner_loss` | **No writes** (Path B) |

## TL1–TL15 (post C-F04)

| ID | Classification | Owner / contract | Evidence SHA | Min test | FRZ |
| --- | --- | --- | --- | --- | --- |
| TL1 | ALREADY PROVEN | Batch A | prior Batch A | — | — |
| TL2 | ALREADY PROVEN | Batch A | prior Batch A | — | — |
| TL3 | ALREADY PROVEN (scoped) | Cancellation + observe owner | remediation SHA | W4-C case 1–2 | FRZ-REL-04 scoped |
| TL4 | ALREADY PROVEN | Batch B | prior Batch B | — | — |
| TL5 | VALIDATION REQUIRED | UNKNOWN non-replay | L2 @ publication | `test_batch_c_l1_unknown_outcome_blocks_blind_replay_after_abandon` | FRZ-REL scoped |
| TL6 | ALREADY PROVEN | Idempotency | Batch B subset L2 | Batch B tests in L2 | FRZ-REL-03/07 scoped |
| TL7 | VALIDATION REQUIRED | STALLED = deadline only | L2 @ publication | `test_batch_c_l1_stalled_classification_*` | FRZ-REL-01 scoped |
| TL8 | **VALIDATION REQUIRED (scoped)** | No RUNNING→UNKNOWN without L3 cert | **publication commit** | `test_batch_c_l1_owner_loss_recovery_denied_*`, W4-C case 3 | FRZ-REL-05, FRZ-REC scoped |
| TL9 | ALREADY PROVEN + extended | Fenced terminal / stale finalize | remediation SHA | Batch C fencing tests | FRZ-REL-05 scoped |
| TL10–TL13 | BLOCKED | Batch D–E | — | — | — |
| TL14 | ALREADY PROVEN | Single invoker owner | remediation SHA | `test_batch_c_l1_invoker_preserves_foreign_running_*` | FRZ-BND-03 scoped |
| TL15 | BLOCKED | Parent rollup | — | — | — |

## Historical evidence (reuse scope)

| Artifact | SHA | Note |
| --- | --- | --- |
| Batch C L2 43/43 | pre-`beb46a44` remediation | **Not** evidence for C-F04 closure |
| Batch C L1 9/9 | `a5f813a2…` | **Not** safety evidence post C-F04 |
| C-F01–C-F03 fixes | `beb46a44` ancestry | Still valid; extended by C-F04 |

## Validation (C-F04 remediation)

### L1

```text
uv run pytest -p no:xdist tests/unit/runtime/nexus/tools/test_tool_life_x_batch_c_stalled_abandoned_recovery.py -q
```

| Metric | Value |
| --- | --- |
| **Passed** | **12** |
| **Failed** | **0** |
| **Execution HEAD** | *(publication commit)* |
| **Log** | `.tmp/session/TOOL-LIFE-X-BATCH-C/l1-batch-c-cf04.log` |

### L2 (consolidated)

```text
uv run pytest -p no:xdist \
  tests/unit/runtime/nexus/tools/test_tool_life_x_batch_c_stalled_abandoned_recovery.py \
  tests/unit/runtime/architecture/test_enterprise_scale_resilience_w4_c_distributed_cancellation_qualification.py \
  tests/unit/runtime/nexus/tools/test_tool_life_x_batch_b_retry_idempotency_unknown_outcome.py \
  tests/unit/runtime/nexus/tools/test_tool_life_x_batch_b_real_governance_authorization.py -q
```

| Metric | Value |
| --- | --- |
| **Passed** | **44** |
| **Failed** | **0** |
| **Execution HEAD** | *(publication commit)* |
| **Log** | `.tmp/session/TOOL-LIFE-X-BATCH-C/l2-batch-c-cf04.log` |

## Scoped FRZ / ECAP

| ID | Finding |
| --- | --- |
| FRZ-BND-03 | No second lifecycle owner; invoker does not infer death; recovery API does not mutate without authority |
| FRZ-REL-03/05 | ABANDONED + fenced terminals (scoped) |
| FRZ-REC | UNKNOWN admission fail-closed; **no** uncertified RUNNING→UNKNOWN |
| ECAP-P4 | Global FRZ PASS remains **0** |

## Remaining parent / real-infrastructure blockers

- **Parent L3:** Wire canonical certified owner-loss evidence before re-enabling `recover_running_operation_after_certified_owner_loss` CAS path (distributed worker-loss / lease / heartbeat qualification).
- Heartbeat-based STALLED operator certification (TL7).
- Batch D/E provider matrices (TL10–TL13).

---

**TOOL-LIFE-X BATCH C — READY FOR INDEPENDENT AUDIT**

Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.
