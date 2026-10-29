# DUP-X — Enterprise Findings Register

| Field | Value |
| --- | --- |
| **AUDITED_HEAD** | `8245b7471379414dc24385d1e770ccb56a7b0e2d` |
| **Python SSOT** | `tests/qualification/dup_x/_findings_register.py` |
| **Prefix** | `DUPX-F-*` (confirmed) · `DUPX-GAP-*` (evidence gaps) |

## Summary

| Metric | Count |
| --- | ---: |
| Open gaps (`DUPX-GAP-*`) | **2** |
| Unresolved DUPLICATE/BLOCKER (`DUPX-F-*`) | **0** |
| RATIFIED bypass fence (COMPAT-R3) | **1** |
| Pre-audit candidates classified | **15 / 15** |

## Findings

| ID | Severity | Class | Kind | Status | Production reachability | Remediation |
| --- | --- | --- | --- | --- | --- | --- |
| DUPX-GAP-CLASS-D | Gap | DUP-10 | GAP | **OPEN** | unknown — Wave B L3 | Class D audit + OD-DUPX-01 |
| DUPX-GAP-L3-TIER3-DEDUPE | Gap | DUP-02 | GAP | **OPEN** | per-app L3 | Tier-3 dedupe disposition |
| DUPX-F-COMPAT-R3-FENCE | Major | DUP-08, DUP-09 | BYPASS | **RATIFIED** | lab-only | Qual fence tests |

## Candidate disposition (DUPX-CAND-01..15)

| ID | Classes | Status | Disposition |
| --- | --- | --- | --- |
| DUPX-CAND-01 | DUP-01, DUP-09 | N/A WITH EVIDENCE | Legacy tools_base — no prod imports |
| DUPX-CAND-02 | DUP-03, DUP-13 | RATIFIED | registry_v2 projection — see delta dossier |
| DUPX-CAND-03 | DUP-10 | OPEN | Class D — linked to DUPX-GAP-CLASS-D |
| DUPX-CAND-04 | DUP-03, DUP-11, DUP-14 | RATIFIED | Harness projection — delta dossier |
| DUPX-CAND-05 | DUP-04, DUP-11 | N/A WITH EVIDENCE | Resolver pipeline stages |
| DUPX-CAND-06 | DUP-01, DUP-05 | RATIFIED | context dedup wrapper — delta dossier |
| DUPX-CAND-07 | DUP-05 | N/A WITH EVIDENCE | Distinct dedup semantics |
| DUPX-CAND-08 | DUP-14 | PASS | CONFIG-X ∪ EBH-5 hints |
| DUPX-CAND-09 | DUP-10, DUP-12 | OPEN | Observability export routing review |
| DUPX-CAND-10 | DUP-03, DUP-10 | N/A WITH EVIDENCE | Tenant factory registry bounded |
| DUPX-CAND-11 | DUP-07, DUP-12 | N/A WITH EVIDENCE | Diagnostics — evidence ports |
| DUPX-CAND-12 | DUP-02 | OPEN | Tier-3 dedupe — DUPX-GAP-L3 |
| DUPX-CAND-13 | DUP-01 | RATIFIED | Ephemeral store — delta dossier |
| DUPX-CAND-14 | DUP-05, DUP-10 | N/A WITH EVIDENCE | SQLite annex replay |
| DUPX-CAND-15 | DUP-08, DUP-09 | RATIFIED | COMPAT-R3 lab-only |

**Agent forbidden:** forge Protocol 03 PASS · OD-DUPX-08 · declare DUP-X CLOSED.
