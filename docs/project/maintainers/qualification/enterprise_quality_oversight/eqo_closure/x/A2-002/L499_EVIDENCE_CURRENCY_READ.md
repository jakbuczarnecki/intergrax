# EQO-X-02 — L499 evidence currency read (DOC_PIN ledger)

**Task:** `EQO-X-02` · **Findings:** EQOA-A2-001 (HIGH), EQOA-A2-002 (BLOCKER)  
**Objective:** ledger read @ `DOC_PIN` for ARCH-FREEZE entry condition L499 — no owner enum assignment.

## Pins

| Pin | SHA |
|-----|-----|
| DOC_PIN | `8cee3c5a913f234e8fb65053b6ab4f34ed78249b` |
| TRAIN_PIN | `28c27cb896d02e0185b9fa02047411fa152f6469` |

## Matrix extract (`EQO_A_REM_MASTER_FINDING_MATRIX.jsonl`)

| Finding | Severity | `current_status` | `effective_arch_freeze_impact` |
|---------|----------|------------------|--------------------------------|
| EQOA-A2-001 | HIGH | `BLOCKED` | `BLOCKS_ARCH_FREEZE` |
| EQOA-A2-002 | BLOCKER | `BLOCKED` | `BLOCKS_ARCH_FREEZE` |

## L499 linkage (`EQO_INTEGRATED_REQUALIFICATION_PLAN.md`)

Closure waits **PT-Q08/Q09** (`WAIT_FOR_INTEGRATION`) post-FCS; pre-FCS EQO records DOC_PIN ledger currency only.

## Interim disposition

`OWNER_DECISION` route per `EQO_B2_DISPOSITION.jsonl` — both IDs **`BLOCKED`** until operator ARCH-FREEZE / TOOL-LIFE-X packet addresses independent exact-SHA evidence rule. EQO does not name persons or edit roadmap.

**Audit:** session E pending (no PASS).

**Recorded:** 2026-11-21 · lane `eqo/wave-4/x` · `Ref: governance/tickets/T-IGX-046-F.md`

## Wave-8 read pass (`EQO-X-02`)

**Land:** `569254840` (PR #698). Ledger rows for EQOA-A2-001 / A2-002 remain **`BLOCKED`** / `BLOCKS_ARCH_FREEZE`; no register edit. AUD-1 did not PASS this cluster (WB02 **FAIL** documents tenant drift adjacent band).

**Recorded:** 2026-11-21 · lane `eqo/wave-8/x`