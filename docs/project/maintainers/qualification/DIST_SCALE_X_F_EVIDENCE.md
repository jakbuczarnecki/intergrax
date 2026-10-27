# DIST-SCALE-X — Dimension F Evidence (P4-A)

| Field | Value |
| --- | --- |
| **START_HEAD** | `50838f077a70d35abb527fc4a499416f5971f1b0` |
| **Envelope** | `DIST-SCALE-X-ENVELOPE-P4-v1` (`ENVELOPE_ONLY` / `NOT_LOCKED`) |
| **Scope** | WL-5 persistence-heavy partial-failure class |

## Executed scenarios

| Scenario | Status | Notes |
| --- | --- | --- |
| **F-PARTIAL** (WL-4 HTTP cite) | **CITED** | Historical @ P3 `e6020efc3f299afc9ff776fd8a1e4a4ac1ac913c` — not replayed |
| **F-STORAGE** (PG pause / reject during F01/F03) | **NOT EXECUTED** | Bounded `docker pause` not run in this batch — qualification-only container policy |
| **F-EVENT** (F07 append outage) | **NOT EXECUTED** | No sanctioned narrow event-only fault seam exercised |

## Residual

Dimension **F** physical matrix for P4-A is **incomplete**. Independent audit must not treat F as PASS.
