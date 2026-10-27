# DIST-SCALE-X — Dimension G Evidence (P4-A)

| Field | Value |
| --- | --- |
| **START_HEAD** | `50838f077a70d35abb527fc4a499416f5971f1b0` |
| **Envelope** | `DIST-SCALE-X-ENVELOPE-P4-v1` |

## Harness

- `p4_crash_worker.py` — hard kill at sync boundaries (`G-01`…`G-06`).
- `p4_physical_crash.py` — parametrized boundary reachability + `proc.kill()`.

## Executed scenarios (requires live qual PG)

| ID | Status | Observation |
| --- | --- | --- |
| **G-01…G-06** sync + kill | **READY** | Passed in corrective dev session @ psycopg handshake; re-run when PG qual infra up |
| **G-02 F01-only path** | **READY** | Dedicated F01 persist before `after_f01` sync (avoids full F01–F09 latency) |

## Residual

Re-execute `p4_physical_crash.py` @ corrective `IMPLEMENTATION_SHA` after infra preflight. Post-kill durable assertions beyond sync markers remain **ENVELOPE_ONLY** observation.
