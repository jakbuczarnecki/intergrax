# EQO-W4-04 — REQUALIFY batch (full B-2 queue)

**Parent:** `EQO-W4-04` · **Task:** link five `REQUALIFY` disposition rows to Campaign C layer evidence  
**Scope:** planning memo only — no product requalification, no matrix CLS, no FRZ edits.

## Pins

| Pin | SHA |
|-----|-----|
| DOC_PIN | `8cee3c5a913f234e8fb65053b6ab4f34ed78249b` |
| TRAIN_PIN | `28c27cb896d02e0185b9fa02047411fa152f6469` |

## Batch queue (`EQO_B2_DISPOSITION.jsonl`)

| Finding ID | Severity | Owner | Suggested `layer_id` (`EQO_C_DRIFT.jsonl`) | `drift_class` @ pins |
|------------|----------|-------|---------------------------------------------|----------------------|
| EQOA-A0-008 | MEDIUM | PDC-X (T-IGX-041-J EBH-7 re-pin) | `PDC-X` | `FULL_REQUAL_REQUIRED` |
| EQOA-A2-035 | MEDIUM | PDC-X | `PDC-X` | `FULL_REQUAL_REQUIRED` |
| EQOA-A3-013 | MEDIUM | TENANT-X / ARCH-FREEZE | `TENANT-X` | `NO_PRODUCT_DELTA` (mechanical); dossier drift open |
| EQOA-A3-030 | MEDIUM | ARCH-FREEZE | `GOV-X` / programme layers TBD | `FULL_REQUAL_REQUIRED` on `GOV-X` |
| EQOA-A5-011 | MEDIUM | TENANT-X stage owner / ARCH-FREEZE | `TENANT-X` | `NO_PRODUCT_DELTA` (mechanical); PT-Q07 post-FCS |

## Steward action

For each row: cite `EQO_C_HISTORICAL_CERTIFICATION_REQUALIFICATION_MATRIX.jsonl` + matching `EQO_C_DRIFT.jsonl` line; record **BLOCKED** until owner PT node runs (EQO does not execute PT-Q* jobs).

**Recorded:** 2026-11-21 · lane `eqo/wave-4/w4` · `Ref: governance/tickets/T-IGX-046-F.md`