# DIST-SCALE-X — Dimension H Evidence (P4-A)

| Field | Value |
| --- | --- |
| **START_HEAD** | `50838f077a70d35abb527fc4a499416f5971f1b0` |
| **Envelope** | `DIST-SCALE-X-ENVELOPE-P4-v1` |
| **Recovery cap** | 120s (envelope) |

## Harness

- `p4_h_crash_recovery_worker.py` — write to G-06, SIGKILL, recover in new PID (revision 1→2).
- `test_p4_h_sigkill_then_recovery_continuation` — reload F01/F03/F07/F09 + sanctioned continuation.

## Executed scenarios (requires live qual PG + `ensure_qual_pg_schema`)

| ID | Status |
| --- | --- |
| **H-01** SIGKILL + recovery | **BLOCKED** — not executed @ final differential (PG handshake false) |

## Residual

Re-run @ corrective `IMPLEMENTATION_SHA` when qual PostgreSQL handshake succeeds.
