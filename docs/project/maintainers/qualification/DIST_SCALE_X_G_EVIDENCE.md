# DIST-SCALE-X — Dimension G Evidence (P4-A)

| Field | Value |
| --- | --- |
| **START_HEAD** | `50838f077a70d35abb527fc4a499416f5971f1b0` |
| **Envelope** | `DIST-SCALE-X-ENVELOPE-P4-v1` |

## Executed scenarios

| ID | Status | Observation |
| --- | --- | --- |
| **G-02 class** (stale F01 CAS across connections) | **EXECUTED** | `tests/qualification/dist_scale_x/p4_physical_restart.py::test_p4_g_cross_process_cas_conflict` |
| **G-01..G-06** (process kill at commit boundaries) | **NOT EXECUTED** | No instrumented kill harness for F07/F09/G-05 effect boundaries in this batch |

## Residual

Full crash-consistency matrix **not** closed for P4-A.
