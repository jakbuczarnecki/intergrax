# TOOL-LIFE-X Batch A — Canonical Deadline Hierarchy Evidence

| Field | Value |
| --- | --- |
| **Task** | `TOOL-LIFE-X-BATCH-A-CANONICAL-DEADLINE-HIERARCHY` |
| **Parent** | `TOOL-LIFE-X` (§3.0.5) — **not CLOSED** |
| **Mode** | Change-delta (ECAP-P1/P2/P4/P6) |
| **Disposition** | **`TOOL-LIFE-X BATCH A — READY FOR INDEPENDENT AUDIT`** |

**Non-closure:** Does not close `TOOL-LIFE-X`, does not promote global `FRZ-*` PASS, does not substitute `HARNESS-W4` / `DIST-SCALE-X`.

## Invariant evidence (scoped)

| Invariant | Owner | Contract | Test / gate | FRZ | Status |
| --- | --- | --- | --- | --- | --- |
| Parent execution deadline bounds tool physical attempt timeout | `RuntimeToolInvoker` | `resolve_active_provider_timeout_seconds` + `ToolContract.timeout_ms` | `test_tool_life_x_batch_a_deadline_hierarchy.py` | FRZ-REL-01 | VALIDATION REQUIRED (Batch A wave) |
| Protected-work admission blocks pre-effect when expired | `CanonicalHardProtectedWorkAdmission` | `ExecutionProtectedWorkAdmissionPort` | `test_tool_protected_work_admission.py`, Batch A expired-before | FRZ-REL-01 | VALIDATION REQUIRED |
| Queue admission wait consumes remaining parent budget | `DependencyAttemptExecutionBoundary` + invoker | `acquire_timeout_seconds` projection | Batch A queue-wait test | FRZ-REL-07 | VALIDATION REQUIRED |
| Retry backoff consumes same operation budget | `RuntimeToolInvoker._execute_with_policy` | capped `cooperative_delay_seconds` | Batch A backoff-bounded test | FRZ-REL-07 | VALIDATION REQUIRED |
| Single sanctioned invoker path | Architecture gates | U5 zero-bypass | existing U5 tests (unchanged) | FRZ-BND-03, FRZ-EXE-02 | ALREADY PROVEN (scoped) |

**Execution SHA:** record at commit (see git `FINAL_COMMIT` in terminal report).

## TL1–TL15 (Batch A reconciliation)

See parent P0 matrix; Batch A advances **TL2** primary evidence only. **TL15** remains **BLOCKED** until parent criteria close.

## Reused evidence

- P0 pre-audit @ `f5c9330973f7af7ea1d19fbbd2afd13942faac43` (baseline only).
- HARNESS-02 Q09/Q11 patterns @ historical SHAs — acceptance scope unchanged.
