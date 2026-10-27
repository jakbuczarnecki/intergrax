# DIST-SCALE-X — WL-7 physical soak deferral

| Field | Value |
| --- | --- |
| **Record ID** | `DIST-SCALE-X-WL7-PHYSICAL-DEFERRAL-2026-10-27` |
| **Operator intent** | Complete **implementation + L9 documentary audit** before **14400 s** physical soak |
| **Envelope** | [DIST_SCALE_X_P6_ENVELOPE_RECORD.md](DIST_SCALE_X_P6_ENVELOPE_RECORD.md) — duration **14400 s** unchanged |
| **Harness evidence** | `wl07-soak-41ad330f8ef2` @ EXECUTION_SHA `9ac74ca…`, `soak_complete_envelope=0`, 120 s, 0 errors |

## Effect on exit criteria

| Criterion | Until full soak |
| --- | --- |
| L9 row **11** (soak M) | **BLOCKED** / **VALIDATION REQUIRED** |
| L9 row **16** (blockers) | **BLOCKED** (row 11 open) |
| Parent **CLOSED** | **FORBIDDEN** until soak + G-06 PASS + G-07 |

## Aborted partial physical attempt (2026-10-27)

| Field | Value |
| --- | --- |
| **Reason** | `OPERATOR_PAUSE` |
| **Elapsed** | ~3135 s of 14400 s (~21.7%) |
| **Outcome** | Process terminated; **no** P6 manifest; `soak_complete_envelope` **not** achieved |
| **Session log** | `.tmp/session/dist-scale-x-wl7-4h/` (`run.log`, `PAUSE_RECORD.md`) |

Deferral and L9 rows **11** / **16** remain **unchanged** until a **full** 14400 s successful run.

## Failed resume physical attempt (2026-10-27, session `dist-scale-x-wl7-4h-resume`)

| Field | Value |
| --- | --- |
| **Command** | `DIST_SCALE_X_WL7_DURATION_SECONDS=14400` — `wl_07_soak_mixed_load.py` |
| **EXECUTION_SHA** | `06e57d4532b4231440c358bf1e3fa8301f040817` |
| **Wall elapsed** | ~**13974 s** (~**97%** of 14400 s envelope) |
| **Terminal** | `exit_code=1`, status **aborted** (Cursor shell); ended 2026-10-27T18:22:30Z |
| **Outcome** | **No** P6 manifest under `.tmp/proof/dist-scale-x/` — `write_p6_manifest` runs only after the soak loop completes |
| **`soak_complete_envelope`** | **Not achieved** (run did not reach loop end; not an assert-failure path — that would still emit a manifest) |
| **Session evidence** | `.tmp/session/dist-scale-x-wl7-4h-resume/` (`run.log`, `poll.log`, `OUTCOME.md`) |
| **Infra at start** | Docker PG + RabbitMQ reported ready |

**Operator action:** schedule a **fresh** uninterrupted **14400 s** run (new session dir; do not treat this partial wall time as envelope evidence). Prefer `python -u` / unbuffered pytest or post-run manifest check so completion is auditable without relying on `run.log` growth during the loop.

## Not a SAC amendment

Deferred **execution** only; mandatory **14400 s** remains in P6 envelope until operator lock or completion run.
