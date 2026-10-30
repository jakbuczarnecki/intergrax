# EBH-7 — ECAP current HEAD audit report (C1–C5)

| Counter | Structural @ materialization | Behavioral qualification |
| --- | --- | ---: |
| ECAP-C1 unclassified mechanisms (685 SSOT) | 0 | INTEGRITY OK |
| ECAP-C2 unauthorized bypass (dup_x subset) | scoped gates execute | **QUALIFIED** (W4 DUP-X subset @ HEAD) |
| ECAP-C3 NPSC global allowlist | gate executes PASS | QUALIFIED |
| ECAP-C4 tenant E2E (26) | registry MATERIALIZED | **14/26 EXACT_PROOF** (12 OPEN — fail-closed disposition) |
| ECAP-C5 cross-domain (34) | registry MATERIALIZED | **16/34 EXACT_PROOF** (18 OPEN) |

**Semantic SSOT:** `tests/qualification/ebh_7/_semantic_disposition_registry.py` — no auto-EXACT from registered callable alone.

**RPL-H:** 56/56 mechanical REPLACEABLE @ matrix; **51× PARTIAL** semantic review (Package C) + **5× QUALIFIED** legacy behavioral.

**B6 readiness:** `EBH-7 — SEMANTIC PROOF INTEGRITY PARTIAL / EXACT BLOCKERS` (30/60 EXACT).

**Explicit non-claims:** OD-EBH7-01 OPEN; OD-EBH7-08 OPEN; B6 OPEN; global FRZ PASS not claimed; stage **not** CLOSED.

**Disposition baseline SHA (pre-publication commit):** `260d8c108fd7ab67dbb02928bfeb1eb6cbce057c`
