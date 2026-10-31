# EBH-7 — ECAP current HEAD audit report (C1–C5)

| Counter | Structural @ materialization | Behavioral qualification |
| --- | --- | ---: |
| ECAP-C1 unclassified mechanisms (685 SSOT) | 0 | INTEGRITY OK |
| ECAP-C2 unauthorized bypass (dup_x subset) | scoped gates execute | **QUALIFIED** (W4 DUP-X subset @ HEAD) |
| ECAP-C3 NPSC global allowlist | gate executes PASS | QUALIFIED |
| ECAP-C4 tenant E2E (26) | registry MATERIALIZED | **18/26 EXACT_PROOF** (8 OPEN — fail-closed disposition) |
| ECAP-C5 cross-domain (34) | registry MATERIALIZED | **16/34 EXACT_PROOF** (18 OPEN) |

**Semantic SSOT:** `tests/qualification/ebh_7/_semantic_disposition_registry.py` — no auto-EXACT from registered callable alone.

**RPL-H @ HEAD:** **6 QUALIFIED · 29 PARTIAL · 21 BLOCKED** ([`_rpl_h_semantic_review.py`](../../../tests/qualification/ebh_7/_rpl_h_semantic_review.py)).

**B6 readiness:** `EBH-7 — SEMANTIC PROOF INTEGRITY PARTIAL / EXACT BLOCKERS` (34/60 EXACT).

**Explicit non-claims:** OD-EBH7-08 OPEN; B6 OPEN; global FRZ PASS not claimed; stage **not** CLOSED.

**Disposition baseline SHA:** `54da93b1b3aa1756d10a44834fabb83938092063` (Batch D D1); audit HEAD `8b7661af15641a6a22d4eaa9127a5372d9968bfd`
