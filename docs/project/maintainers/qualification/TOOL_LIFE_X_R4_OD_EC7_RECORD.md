# OD-EC7 — DEADLINE_ONLY WITH VERIFIED SCOPE

| Field | Value |
| --- | --- |
| **Decision** | `OPERATOR APPROVED — OD-EC7 DEADLINE_ONLY WITH VERIFIED SCOPE` |
| **Date** | 2026-10-25 (R4 continuation wave) |

## Accepted scope

Bounded deadline-based stalled / abandoned detection on the canonical **RuntimeToolInvoker / Execution** hierarchy per Batch C evidence and F4-01 audit recommendation.

## Explicit exclusions

- Physical provider or worker termination guaranteed by timeout alone.
- Universal liveness / heartbeat invention.
- Gate A certified owner-loss recovery (G-A09 positive).
- TL8 scenario 24 positive until OD-TL8-IMPL + Gate A L3.

## Qualification basis

- [`TOOL_LIFE_X_R3_1_F4_01_TOOL_CLASS_DEADLINE_AUDIT.md`](TOOL_LIFE_X_R3_1_F4_01_TOOL_CLASS_DEADLINE_AUDIT.md)
- L2: `tests/unit/runtime/nexus/tools/test_tool_life_x_batch_c_stalled_abandoned_recovery.py`

## EC7 parent row

**PASS (scoped)** for F7 reconciliation when Batch C L2 green @ implementation SHA and this record is cited. Not equivalent to EC8/24 recovery PASS.
