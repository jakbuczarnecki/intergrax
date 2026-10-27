# DIST-SCALE-X — Dimension G Evidence (P4-A)

| Field | Value |
| --- | --- |
| **START_HEAD** | `50838f077a70d35abb527fc4a499416f5971f1b0` |
| **IMPLEMENTATION_SHA** | `f0c49fd3cff8c00fef096e4db96f7ddcf07c9845` |
| **EXECUTION_SHA** | `f0c49fd3cff8c00fef096e4db96f7ddcf07c9845` |
| **run_id** | `p4-a-20261027-094853` |
| **Envelope** | `DIST-SCALE-X-ENVELOPE-P4-v1` |
| **Disposition** | **DIST-SCALE-X P4-A — READY FOR INDEPENDENT AUDIT** (scoped) |

## Harness

- `p4_staged_persistence.py` — production ports through F01→F03→F07→F09 boundaries.
- `p4_crash_worker.py` — reach boundary, write `state.json`, sync marker, await kill.
- `p4_pg_state_inspection.py` — independent-process PostgreSQL assertions.
- `p4_physical_crash.py` — SIGKILL + `assert_boundary_snapshot` per boundary.

## Executed scenarios @ EXECUTION_SHA

| ID | Status | Observation |
| --- | --- | --- |
| **G-01…G-06** staged + kill + PG inspect | **PASS** | `p4_physical_crash.py` @ `p4-a-20261027-094853` batch |
| **H** SIGKILL @ G-06 + recovery | **PASS** | `p4_physical_restart.py` in same batch |

## Residual

None for dimension **G** @ `EXECUTION_SHA` on qual PostgreSQL used for this run.
