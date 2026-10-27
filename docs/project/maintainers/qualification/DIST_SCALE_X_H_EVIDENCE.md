# DIST-SCALE-X — Dimension H Evidence (P4-A)

| Field | Value |
| --- | --- |
| **START_HEAD** | `50838f077a70d35abb527fc4a499416f5971f1b0` |
| **IMPLEMENTATION_SHA** | `f0c49fd3cff8c00fef096e4db96f7ddcf07c9845` |
| **EXECUTION_SHA** | `f0c49fd3cff8c00fef096e4db96f7ddcf07c9845` |
| **run_id** | `p4-a-20261027-094853` |
| **Envelope** | `DIST-SCALE-X-ENVELOPE-P4-v1` |
| **Recovery cap** | 120s (envelope) |
| **Disposition** | **DIST-SCALE-X P4-A — CLOSED / INDEPENDENTLY ACCEPTED (SCOPED)** — [DIST_SCALE_X_P4_FORMAL_CLOSURE_RECONCILIATION.md](DIST_SCALE_X_P4_FORMAL_CLOSURE_RECONCILIATION.md) |

## Harness

- `p4_h_crash_recovery_worker.py` — write to G-06, SIGKILL, recover in new PID (revision 1→2).
- `test_p4_h_sigkill_then_recovery_continuation` — reload F01/F03/F07/F09 + sanctioned continuation.

## Executed scenarios @ EXECUTION_SHA

| ID | Status |
| --- | --- |
| **H-01** SIGKILL + recovery | **PASS** — `test_p4_h_sigkill_then_recovery_continuation` in P4 consolidated batch |

## Residual

None for dimension **H** @ `EXECUTION_SHA` on qual PostgreSQL used for this run.
