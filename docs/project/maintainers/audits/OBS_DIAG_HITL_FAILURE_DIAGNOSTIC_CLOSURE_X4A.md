# OBS-DIAG-X4A — HITL Resume Failure Diagnostic Closure

**Verdict:** `PASS — HITL RESUME FAILURE DURABLE DIAGNOSTICS QUALIFIED` (verified on clean `development` worktree @ `8f2887887d32728b2a7210f2d91eaeea641cd40d` with X4A patch).

## Gap closed

OBS-DIAG-X4 proved real OS-process HITL pause/resume and human-governed rejection terminal state, but did **not** assert durable `Problem` + fresh read on a **diagnostic-worthy** resumed failure path. X4A closes that gap without treating operator `REJECT` as a diagnostic anomaly.

## Semantic classification

| Path | Governed outcome? | Diagnostic Problem? |
| --- | --- | --- |
| Operator `REJECT` after valid pause | **YES** — `TaskState.FAILED`, `TASK_FAILED` runtime event | **NO** — expected policy terminal; `problem_count == 0` |
| Approved resume + lab post-terminal violation (`inject_violation_on_resume`) | Task completes; `RETRY_SCHEDULED` appended after `TASK_COMPLETED` on canonical `RuntimeEventBus` | **YES** — same mechanism as Kafka X4 failure proof |

The violation hook is **not** a direct `DiagnosticOrchestrator` bypass: it subscribes to persisted terminal completion on the Nexus loop and appends an anomaly event that flows through the production terminal diagnostic port (identical to `build_diagnostic_nexus_loop(inject_violation=True)` used elsewhere in OBS spine qualification).

## Proof architecture

```text
Process A → HITL pause → durable checkpoint + continuation export → exit
Process B → fresh runtime → restore continuation → approve → resume
         → TASK_COMPLETED → inject_violation_on_resume → RETRY_SCHEDULED
         → terminal diagnostics → durable Problem → exit
Fresh reader (parent) → new SQLite adapters + DiagnosticReadService → same Problem IDs
```

## Tests

| Test | Role |
| --- | --- |
| `test_hitl_cross_process_human_rejection_terminal_behavior` | Renamed from misleading `...failure_problem`; asserts FAILED + no Problem |
| `test_hitl_cross_process_resume_injected_violation_creates_durable_problem` | **X4A mandatory proof** — Problem + fresh read + identity continuity |

Module: `tests/integration/runtime/test_obs_universal_spine_cross_process_x4_e2e.py`

## Harness notes

- `HitlProcessResult.terminal_event_type` — typed subprocess evidence (no log parsing).
- `testing_support/cross_process_spine/runtime_terminal_read.py` — lightweight terminal read shared by HITL CLI and Kafka stack (avoids heavy Kafka/Echo imports in HITL subprocess).

## Honest history

X4 originally left the HITL resume failure → durable Problem assertion unproven; X4A adds the proof without retroactively claiming X4 had it.

## Out of scope

External HITL vendor, provider outage/recovery matrices — OBS-DIAG-X5+.
