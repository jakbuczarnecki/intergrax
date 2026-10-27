# DIST-SCALE-X — Dimension F Evidence (P4-A)

| Field | Value |
| --- | --- |
| **START_HEAD** | `50838f077a70d35abb527fc4a499416f5971f1b0` |
| **IMPLEMENTATION_SHA** | `f0c49fd3cff8c00fef096e4db96f7ddcf07c9845` |
| **EXECUTION_SHA** | `f0c49fd3cff8c00fef096e4db96f7ddcf07c9845` |
| **run_id** | `p4-a-20261027-094853` |
| **Envelope** | `DIST-SCALE-X-ENVELOPE-P4-v1` (`ENVELOPE_ONLY` / `NOT_LOCKED`) |
| **Disposition** | **DIST-SCALE-X P4-A — CLOSED / INDEPENDENTLY ACCEPTED (SCOPED)** — [DIST_SCALE_X_P4_FORMAL_CLOSURE_RECONCILIATION.md](DIST_SCALE_X_P4_FORMAL_CLOSURE_RECONCILIATION.md) |

## Executed scenarios

| Scenario | Status | Notes |
| --- | --- | --- |
| **F-PARTIAL** (WL-4 HTTP cite) | **CITED** | Historical @ P3 `e6020efc3f299afc9ff776fd8a1e4a4ac1ac913c` |
| **F-EVENT** | **PASS** | `test_p4_f_event_injected_append_outage_fail_closed` — injected F07 append outage; F01 partial durable, F03/F07/F09 absent |
| **F-STORAGE** | **PASS** | `test_p4_f_storage_pg_pause_surfaces_write_error` — writes under `docker pause` on `intergrax-postgresql`; persistence errors surfaced |

## Residual

None for dimension **F** @ `EXECUTION_SHA` on qual PostgreSQL + Docker host used for this run.
