# DIST-SCALE-X — Dimension G Evidence (P4-A)

| Field | Value |
| --- | --- |
| **START_HEAD** | `50838f077a70d35abb527fc4a499416f5971f1b0` |
| **Envelope** | `DIST-SCALE-X-ENVELOPE-P4-v1` |

## Harness

- `p4_staged_persistence.py` — production ports through F01→F03→F07→F09 boundaries.
- `p4_crash_worker.py` — reach boundary, write `state.json`, sync marker, await kill.
- `p4_pg_state_inspection.py` — independent-process PostgreSQL assertions.
- `p4_physical_crash.py` — SIGKILL + `assert_boundary_snapshot` per boundary.

## Executed scenarios (requires live qual PG)

| ID | Status | Observation |
| --- | --- | --- |
| **G-01…G-06** staged + kill + PG inspect | **BLOCKED** | Not executed @ final differential (PG handshake false) |
| **H** SIGKILL @ G-06 + recovery | **BLOCKED** | `p4_physical_restart.py` — pending live PG |

## Residual

Authoritative matrix PASS requires `p4_physical_crash.py` + H test @ committed **IMPLEMENTATION_SHA** with **EXECUTION_SHA** in provenance.
