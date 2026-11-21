# PLAN T-IGX-051-C — T-AF2-RW-TRUST scaffold (ARCH-FREEZE development tor)

Status: **draft scaffold** (2026-11-21). Source: `MEP-AF2_v2.1.md` row T-AF2-RW-TRUST; ticket id **T-IGX-051-C** in WP-12 tables.

## 1. Purpose

Remediate trust-boundary findings before RW / G2 on train; on `development`, establish ticket SSOT so parallel work in `af-dev-rw-trust` can proceed after plan approval.

## 2. Prerequisites

- `SCOPE` (train) / RC-1 admission on `development` documentation track.
- Operator decisions on trust-class model where PDC marks **Operator decision** (EX-D2-01/02/07).

## 3. Indicative write set

| Finding cluster | Primary paths |
|-----------------|---------------|
| EX-D2-01 | `intergrax/llm/messages.py`, `graph_assembly.py`, context fragment consumers |
| EX-D2-02 | `intergrax/context/` egress and sensitivity defaults |
| EX-D2-05 | `intergrax/context/planner.py` |
| EX-D2-07 | `intergrax/llm/` orphan `tool_call_id` handling |
| EX-L-05 / D2-R004 | RAG trust tests under `tests/unit/rag/` where root cause is product-bound |

## 4. Test contract (placeholder)

- Red-first: injection cannot place untrusted content in system role (fixture TBD).
- STRICT mode refuses orphan tool correlation ids (TBD node ids).
- Regression list to be copied from PDC / closed-world census in plan revision 2.

## 5. Out of scope

Same as ticket.

## 6. Next steps

Architect session fills node ids, regression command, and digest; operator approval; then implementation in `af-dev-rw-trust`.