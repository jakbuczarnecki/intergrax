# DUP-X — Protocol 03 Independent Audit Pass Record

| Field | Value |
| --- | --- |
| **Record type** | ECAP §2.B **B6** — Independent exact-SHA acceptance |
| **Status** | **PENDING — NOT ISSUED** (template for external auditor) |
| **Verdict** | *(auditor completes)* |
| **INDEPENDENT_AUDIT_BASELINE_SHA** | *(set @ audit start — recommend HEAD after Block II Commit G)* |
| **QUALIFICATION_EXECUTION_SHA** | *(B5 replay SHA — see manifest)* |
| **EVIDENCE_PUBLICATION_SHA** | *(Block II evidence commits)* |
| **Auditor identity** | Independent auditor — Protocol DUP-X-03 B1–B6 |
| **Date** | *(auditor)* |

## Scoped acceptance (explicit — to complete on PASS)

PASS applies to **DUP-X Cross-Architecture Duplicate & Bypass Certification** @ pinned SHA only. **Not** global **FRZ PASS** · **not** **PROD-Q** · **not** **ARCH-FREEZE** · **not** automatic **ROADMAP-REPLAY-X** closure.

## B6 checklist (auditor)

| Item | Auditor disposition |
| --- | --- |
| B1 Scope — 15 families · DUP-01..14 · CAND-01..15 | |
| B2 SHA ladder vs `git rev-parse` | |
| B3 Factory-first · deltas · harness projection | |
| B4 COMPAT-R3 · tools_base | |
| B5 `tests/qualification/dup_x/` replay @ baseline | |
| **NPSC-4.2 exclusion** ([`DUP_X_NPSC_4_2_DUPX_EXCLUSION_RECORD.md`](DUP_X_NPSC_4_2_DUPX_EXCLUSION_RECORD.md)) | **ACCEPT / REJECT** |

## Agent forbidden

Cursor/implementer must **not** set Status to ISSUED PASS or execute **OD-DUPX-08**.
