# DIST-SCALE-X — Dimension H Evidence (P4-A)

| Field | Value |
| --- | --- |
| **START_HEAD** | `50838f077a70d35abb527fc4a499416f5971f1b0` |
| **Envelope** | `DIST-SCALE-X-ENVELOPE-P4-v1` |
| **Recovery cap** | 120s (envelope) |

## Harness

- `p4_restart_worker.py` — subprocess write then reload.
- `test_p4_h_durable_restart_all_families` — asserts F01 rev 1, F03 present, F09 present after new process.

## Executed scenarios (requires live qual PG + `ensure_qual_pg_schema`)

| ID | Status |
| --- | --- |
| **H-01** full family reload | **READY** — passed manual + pytest when PG available |

## Residual

Re-run @ corrective `IMPLEMENTATION_SHA` when qual PostgreSQL handshake succeeds.
