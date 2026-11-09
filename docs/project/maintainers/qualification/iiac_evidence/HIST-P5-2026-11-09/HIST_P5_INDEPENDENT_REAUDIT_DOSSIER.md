# HIST P5 independent IIAC re-audit dossier

| Field | Value |
|---|---|
| Auditor session | `IIAC-HIST-P5-REAUDIT-2026-11-09` |
| Assurance | `process_enforced` |
| Audited implementation SHA | `3c91694c645ff0d97d104adf0c5a564b9a4c5b3c` |
| Audit execution SHA | `0946b11576d2ee617321fb099ed794f8b21bf5ba` |
| Operator decision | `IIAC-OD-HIST-P5-2026-11-09` |

## Objective

Correct ACT-N10 / N-03 failure: prior HIST P5 activation attempt bound `corrective_implementation_sha` `2a6545f1…` while advancing `audit_execution_sha` to a post-merge tree, causing `verify_iiac_role_separation.py --strict-product-diff` to detect twelve auditor product write-surface paths.

## Method

Read-only verification at `0946b11576d2ee617321fb099ed794f8b21bf5ba`. Confirmed zero product-path delta between audited merge SHA `3c91694c645ff0d97d104adf0c5a564b9a4c5b3c` and auditor publication SHA. Replayed governance unit tests cited in `test_matrix.txt`. Restored immutable `IIAC-GOV-1` certification manifest bytes from pre-HIST publication.

## Scope

HIST C1–C4 product paths (T-IGX-030-B/C, 032-A, 033-A). Excludes T-IGX-021-B/C. Does not certify FRZ promotions or campaign closure.

## Verdict

**PASS** — activation re-binding may proceed when Operator two-phase publication completes and `verify_iiac_policy_activation --require-in-force` exits 0.

## Lineage

Session B governance report digest `c405c65faf53243a0511b0b9584a9f3abda4db08c09ee9b9266c86fe9f6b7a4e` (ACT-01 dossier under `IIAC-GOV-1/certification/`) remains the governance audit artifact; this dossier is the HIST P5 activation supplement.
