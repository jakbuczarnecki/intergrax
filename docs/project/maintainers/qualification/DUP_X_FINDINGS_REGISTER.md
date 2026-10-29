# DUP-X — Enterprise Findings Register

| Field | Value |
| --- | --- |
| **AUDITED_HEAD** | `8245b7471379414dc24385d1e770ccb56a7b0e2d` |
| **WAVE1_IMPLEMENTATION_HEAD** | `2325460ff7f1f0290f95dae03145e30e20e0d991` |
| **Python SSOT** | `tests/qualification/dup_x/_findings_register.py` |
| **Prefix** | `DUPX-F-*` (confirmed) · `DUPX-GAP-*` (evidence gaps) |

## Summary

| Metric | Count |
| --- | ---: |
| Open gaps (`DUPX-GAP-*`) | **0** (Wave 1 closed with evidence) |
| Unresolved DUPLICATE/BLOCKER (`DUPX-F-*`) | **0** |
| RATIFIED bypass fence (COMPAT-R3) | **1** |
| Pre-audit candidates classified | **15 / 15** |

## Findings

| ID | Severity | Class | Kind | Status | Production reachability | Remediation |
| --- | --- | --- | --- | --- | --- | --- |
| DUPX-GAP-CLASS-D | Gap | DUP-10 | GAP | **PASS** | factory + config canonical | Wave 1 caller matrix — OD-DUPX-01 still OPEN |
| DUPX-GAP-L3-TIER3-DEDUPE | Gap | DUP-02 | GAP | **PASS** | LKW Slack only | Tier-3 disposition doc |
| DUPX-F-COMPAT-R3-FENCE | Major | DUP-08, DUP-09 | BYPASS | **RATIFIED** | lab-only | Qual fence tests |

## Candidate disposition (DUPX-CAND-01..15)

| ID | Classes | Status | Disposition |
| --- | --- | --- | --- |
| DUPX-CAND-01 | DUP-01, DUP-09 | N/A WITH EVIDENCE | Legacy tools_base — no prod imports |
| DUPX-CAND-02 | DUP-03, DUP-13 | RATIFIED | registry_v2 projection — see delta dossier |
| DUPX-CAND-03 | DUP-10 | RATIFIED | CANONICAL Class D — caller matrix |
| DUPX-CAND-04 | DUP-03, DUP-11, DUP-14 | RATIFIED | Harness projection — delta dossier |
| DUPX-CAND-05 | DUP-04, DUP-11 | N/A WITH EVIDENCE | Resolver pipeline stages |
| DUPX-CAND-06 | DUP-01, DUP-05 | RATIFIED | context dedup wrapper — delta dossier |
| DUPX-CAND-07 | DUP-05 | N/A WITH EVIDENCE | Distinct dedup semantics |
| DUPX-CAND-08 | DUP-14 | PASS | CONFIG-X ∪ EBH-5 hints |
| DUPX-CAND-09 | DUP-10, DUP-12 | RATIFIED | DISTINCT observability export routing — delta dossier |
| DUPX-CAND-10 | DUP-03, DUP-10 | N/A WITH EVIDENCE | Tenant factory registry bounded |
| DUPX-CAND-11 | DUP-07, DUP-12 | N/A WITH EVIDENCE | Diagnostics — evidence ports |
| DUPX-CAND-12 | DUP-02 | N/A WITH EVIDENCE | LKW Slack DISTINCT — Tier-3 disposition |
| DUPX-CAND-13 | DUP-01 | RATIFIED | Ephemeral store — delta dossier |
| DUPX-CAND-14 | DUP-05, DUP-10 | N/A WITH EVIDENCE | SQLite annex replay |
| DUPX-CAND-15 | DUP-08, DUP-09 | RATIFIED | COMPAT-R3 lab-only |

## NPSC-4.2 exclusion (DUP-X scope)

| Field | Value |
| --- | --- |
| **Gate** | `tests/unit/runtime/architecture/test_npsc4_2_residual_compatibility_gate.py` |
| **DUP-X mapping** | Architecture allowlist drift — **not** a DUP-01..14 invariant |
| **DUP invariants** | Duplicate/bypass census + factory-first Class D + COMPAT-R3 fence remain L2 primary |
| **Wave 1 action** | Document only — **no** NPSC allowlist change |
| **Exclusion record** | [`DUP_X_NPSC_4_2_DUPX_EXCLUSION_RECORD.md`](DUP_X_NPSC_4_2_DUPX_EXCLUSION_RECORD.md) — 10 `_orchestration_backend` modules · **excluded from DUP-X mandatory** |

**Agent forbidden:** forge Protocol 03 PASS · OD-DUPX-08 · declare DUP-X CLOSED.
