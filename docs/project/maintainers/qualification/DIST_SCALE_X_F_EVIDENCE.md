# DIST-SCALE-X — Dimension F Evidence (P4-A)

| Field | Value |
| --- | --- |
| **START_HEAD** | `50838f077a70d35abb527fc4a499416f5971f1b0` |
| **IMPLEMENTATION_SHA (initial)** | `04329842a0ffc3f62eb928dbf85a49919cf5fb10` |
| **Corrective batch** | See `DIST_SCALE_X_P4_PHYSICAL_PROVENANCE.md` — `IMPLEMENTATION_SHA` after Protocol 03→04 correction |
| **Envelope** | `DIST-SCALE-X-ENVELOPE-P4-v1` (`ENVELOPE_ONLY` / `NOT_LOCKED`) |

## Executed scenarios

| Scenario | Status | Notes |
| --- | --- | --- |
| **F-PARTIAL** (WL-4 HTTP cite) | **CITED** | Historical @ P3 `e6020efc3f299afc9ff776fd8a1e4a4ac1ac913c` |
| **F-EVENT** | **IMPLEMENTED / PG-pending** | `p4_physical_faults.py::test_p4_f_event_append_outage_fail_closed` — qual-local `append` patch (narrow seam, not production API) |
| **F-STORAGE** | **PG-pending** | `test_p4_f_storage_pg_pause_surfaces_write_error` skips when `docker pause` does not surface errors; requires Docker + qual PG container |

## Residual

Dimension **F** storage-wide outage evidence **not** closed on this workstation when Docker Desktop is stopped.
