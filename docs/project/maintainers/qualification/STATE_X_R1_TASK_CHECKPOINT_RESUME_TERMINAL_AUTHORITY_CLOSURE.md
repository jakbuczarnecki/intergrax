# STATE-X-R1 — Task Checkpoint / Resume / Terminal Authority Closure

**Parent:** STATE-X — Persistence, State & Recovery Certification

**Baseline:** STATE-X-P0 = ACCEPTED

**START_HEAD:** `c094d2dbdc9fd89755cddaf557332151696e4f01`

**Status:** READY FOR AUDIT (pending independent GitHub SHA audit)

---

## Blockers disposition

| ID | Disposition |
|---|---|
| SX-B01 | Closed — explicit `ExecutionTerminalPersistenceCapability` narrowing at Nexus composition |
| SX-B02 | Closed — canonical snapshot parser + admission binding |
| R1-B03 | Closed — required `task_id`/`tenant_id` presence before `Task.model_validate` |

---

## Production delta (max 3)

| File | Change |
|---|---|
| `intergrax/runtime/nexus/nexus_loop.py` | Terminal capability narrowing for `wire_execution_terminal_store` |
| `intergrax/runtime/long_running/models.py` | `task_snapshot: JsonObject` |
| `intergrax/runtime/long_running/checkpoint_resume_validation.py` | Canonical `_parse_checkpoint_snapshot_task`, admission order |

---

## Ownership matrix

| Concern | Semantic owner |
|---|---|
| Task checkpoint | Long-running checkpoint subsystem |
| Terminal fact | Execution terminal subsystem |
| Current execution authority | Current admission/execution authority |
| Historical authority bound | Checkpoint historical evidence |
| Tenant/task identity on resume | Current target + checkpoint/snapshot consistency |
| Recovery coordination | Coordination only (SX-F15) |

---

## Snapshot admission (executable evidence)

| Case | Result |
|---|---|
| empty `{}` | REJECT_MALFORMED |
| missing `task_id` | REJECT_MALFORMED |
| missing `tenant_id` | REJECT_MALFORMED |
| malformed Task JSON | REJECT_MALFORMED |
| snapshot task ≠ checkpoint | REJECT_IDENTITY |
| snapshot tenant ≠ checkpoint | REJECT_TENANT |
| valid snapshot, `execution_authority=None` | ALLOW (historical unknown) |
| valid snapshot, scoped historical authority | narrow via `C ∩ H` |

Mechanical proof: `tests/qualification/state_x/test_state_x_r1_checkpoint_resume_terminal.py`

---

## FRZ evidence (no PASS promotion)

Applicable criteria mapped to R1 tests and historical replay; disposition remains **certification evidence only** until independent audit.

---

**Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.**
