# TOOL-LIFE-X Batch C — Stalled / Abandoned / Recovery Evidence

| Field | Value |
| --- | --- |
| **Task** | `TOOL-LIFE-X-BATCH-C` |
| **Parent** | `TOOL-LIFE-X` (§3.0.5) — **not CLOSED** |
| **START_HEAD** | `a5f813a2c6a2195b09d7879359d5fd6eace6bf22` |
| **Remediation** | Independent audit C-F01 / C-F02 / C-F03 (ownership + fencing + recovery) |
| **Disposition** | **`TOOL-LIFE-X BATCH C — READY FOR INDEPENDENT AUDIT`** (publication SHA below) |

**Non-closure:** Does not close `TOOL-LIFE-X`, does not promote global `FRZ-*` PASS, does not start Batch D/E.

## Audit root causes (C-F01–C-F03)

| ID | Root cause | Remediation |
| --- | --- | --- |
| **C-F01** | `RuntimeToolInvoker` swept foreign `RUNNING` records via token inequality | Removed automatic `reconcile_orphaned_tool_external_operations` from invoke path; `reconcile_orphaned_running_operations` is explicit no-op |
| **C-F02** | `ensure_terminal_recorded` and idempotent terminal return bypassed owner fencing; `finalize_operation_intent` without owner predicate | Owner-matched terminal + `finalize_operation_intent_if_owner_matches`; cancellation observe passes owner |
| **C-F03** | Recovery sweep treated `different_owner ≠ dead_owner` | Certified owner-loss API `recover_running_operation_after_certified_owner_loss`; recovery gate no longer sweeps by token |

## Canonical owner inventory

| Concern | Owner |
| --- | --- |
| Tool invocation | `RuntimeToolInvoker` |
| Durable external-op record | `ExternalOperationStateStore` |
| Physical attempt | `ToolExternalOperationAttempt` + `external_operation_ownership` |
| Atomic revision | `compare_and_set` |
| Certified RUNNING→UNKNOWN | `recover_running_operation_after_certified_owner_loss` (caller must certify loss) |
| Recovery admission | `prepare_external_operations_for_recovery` (fail-closed on RUNNING/UNKNOWN) |
| Idempotency / pre-effect | `IdempotencyPreEffectCoordinator` |

**Trusted recovery authority:** Composition / recovery orchestration that can certify owner loss (parent L3); **not** process-local token mismatch.

## Terminal writers (tool path) — fencing status

| Writer | Fencing |
| --- | --- |
| `ToolExternalOperationAttempt._mark_terminal` | `mark_physical_terminal_if_owner_matches` + fenced finalize |
| `ensure_terminal_recorded` | Owner-matched UNKNOWN + fenced finalize |
| `complete_cancellation_after_termination` | `mark_observed_cancellation_terminal(..., owner=...)` |
| `mark_physical_terminal` (raw) | LLM/legacy paths unchanged; tool path avoids direct use |
| `reconcile_orphaned_*` | No terminal writes (no-op) |

**Linearization:** Owner predicate evaluated before CAS; CAS on `expected_revision` remains the atomic boundary; stale revision → idempotent terminal only when owner matches.

## TL1–TL15 (post-remediation)

| ID | Classification | Notes | FRZ |
| --- | --- | --- | --- |
| TL1–TL2 | ALREADY PROVEN | Batch A @ prior SHA | — |
| TL3 | ALREADY PROVEN (scoped) | Cancellation | FRZ-REL-04 scoped |
| TL4–TL6 | ALREADY PROVEN | Batch B | FRZ-REL-03/06/07 scoped |
| TL7 | VALIDATION REQUIRED | Deadline STALLED only; no heartbeat | FRZ-REL-01 scoped |
| TL8 | **VALIDATION REQUIRED (scoped)** | Certified owner-loss only; distributed liveness → parent L3 | FRZ-REL-05, FRZ-REC scoped |
| TL9 | **ALREADY PROVEN + extended** | Fenced terminal + stale finalize | FRZ-REL-05 scoped |
| TL10–TL13 | BLOCKED | Batches D–E | — |
| TL14 | ALREADY PROVEN | Single invoker owner | FRZ-BND-03 scoped |
| TL15 | BLOCKED | Parent rollup | — |

## Historical evidence (superseded for safety claims)

| Artifact | SHA | Note |
| --- | --- | --- |
| Batch C L1 9/9 | `a5f813a2…` (pre-remediation) | **Not** evidence for C-F01–C-F03 closure |
| Batch C L2 41/41 | `a5f813a2…` | Historical only |

## Validation (remediation)

### L1 (critical safety — included in L2 batch file)

```text
uv run pytest -p no:xdist tests/unit/runtime/nexus/tools/test_tool_life_x_batch_c_stalled_abandoned_recovery.py -q
```

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
| **Passed** | **43** |
| **Failed** | **0** |
| **Execution HEAD** | *(publication commit)* |
| **Log** | `.tmp/session/TOOL-LIFE-X-BATCH-C/l2-batch-c-remediation.log` |

## Scoped FRZ / ECAP

| ID | Finding |
| --- | --- |
| FRZ-BND-03 | No second lifecycle owner; invoker does not infer death |
| FRZ-REL-03/05 | ABANDONED + fenced terminals (scoped) |
| FRZ-REC | UNKNOWN admission fail-closed; certified transition only |
| ECAP-P4 | Global FRZ PASS remains **0** |

## Remaining parent / real-infrastructure blockers

- Distributed worker-loss certification without heartbeat/lease (parent TOOL-LIFE-X L3).
- Heartbeat-based STALLED operator certification (TL7).
- Batch D/E provider matrices (TL10–TL13).

---

**TOOL-LIFE-X BATCH C — READY FOR INDEPENDENT AUDIT**

Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.
