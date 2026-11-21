# EQO-A remediation — closure gap report

Generated: 2026-11-20T17:57:06Z

## Count reconciliation

| Source | Count |
|--------|------:|
| FINDINGS.jsonl rows | 240 |
| OPEN | 239 |
| MERGED | 1 |

## BLOCKS semantics

- **Impact BLOCKS (dossier, open):** 8
- **Severity BLOCKER (open):** 2
- **Merged impact BLOCKER (historical):** EQOA-A2-025

Do not equate 9 dossier BLOCKS rows with 2 active severity BLOCKERS.

## EDER

- `reconciliation.json` **rows_total** 1069 (initial ingest equation)
- **register.jsonl** 1073 = 1069 + **EDER-N-0001..0004** (`post_ingest_rows` in reconciliation.json)
- Steward action **EQO-E-01**: document only; no second register.

## Train vs development

All coded A-REM merges through #677 are **MERGED_TO_TRAIN** only until FCS. FRZ checklist blob unchanged by train code until PT-D10 (Operator).
