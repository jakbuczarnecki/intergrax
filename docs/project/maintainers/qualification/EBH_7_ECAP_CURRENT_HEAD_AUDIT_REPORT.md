# EBH-7 — ECAP current HEAD audit report (C1–C5)

| Counter | Structural @ materialization | Behavioral qualification |
| --- | --- | ---: |
| ECAP-C1 unclassified mechanisms (685 SSOT) | 0 | INTEGRITY OK |
| ECAP-C2 unauthorized bypass (dup_x subset) | scoped gates execute | **QUALIFIED** (W4 DUP-X subset @ HEAD) |
| ECAP-C3 NPSC global allowlist | gate executes PASS | QUALIFIED (PROHIBITED unresolved **0** @ EBH-6 scope — revalidated @ final closure) |
| ECAP-C4 tenant E2E (26) | registry MATERIALIZED | **26/26 EXACT_PROOF** |
| ECAP-C5 cross-domain (34) | registry MATERIALIZED | **34/34 EXACT_PROOF** |

**Semantic SSOT:** `tests/qualification/ebh_7/_semantic_disposition_registry.py` — no auto-EXACT from registered callable alone.

**RPL-H @ HEAD:** **56 QUALIFIED · 0 PARTIAL · 0 BLOCKED** ([`_rpl_h_semantic_evidence.py`](../../../tests/qualification/ebh_7/_rpl_h_semantic_evidence.py)).

**B6 readiness (agent):** `EBH-7 — READY FOR INDEPENDENT B6` (60/60 EXACT, RPL semantic green, L3 EBH-7 without B6 xfails).

**Explicit non-claims:** OD-EBH7-08 **OPEN**; B6 **OPEN** (external); global FRZ PASS not claimed; stage **not** CLOSED until operator OD-08.

**Disposition SHA:** see `EBH7_SEMANTIC_DISPOSITION_SHA` in disposition registry (updated @ final closure commit).

**TEST_EXECUTION_SHA:** see `CLOSURE_SHA` in [`EBH_7_SHA_LADDER.md`](EBH_7_SHA_LADDER.md).
