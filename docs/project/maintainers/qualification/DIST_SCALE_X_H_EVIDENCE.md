# DIST-SCALE-X — Dimension H Evidence (P4-A)

| Field | Value |
| --- | --- |
| **START_HEAD** | `50838f077a70d35abb527fc4a499416f5971f1b0` |
| **Envelope** | `DIST-SCALE-X-ENVELOPE-P4-v1` |
| **Recovery cap** | 120s (envelope; not stress-tested at full duration) |

## Executed scenarios

| ID | Status | Observation |
| --- | --- | --- |
| **H-01** (new PID reload F01) | **PARTIAL** | `p4_restart_worker.py` subprocess write + parent reload via `test_p4_h_durable_restart_reload_checkpoint` |
| **H full** (F01/F03/F07/F09 reload + resume) | **NOT EXECUTED** | No end-to-end recovery worker replay in this batch |

## Residual

Durable restart evidence is **partial** — PG M only; not full family reload.
