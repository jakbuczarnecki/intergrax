# EBH-7 — Independent audit corrective findings

**Trigger:** B6 BLOCKED — automatic `proof_module` → EXACT_PROOF promotion @ closure wave (`f913944ff`).

**Correction @ semantic disposition SSOT:**

| Finding | Resolution |
| --- | --- |
| Auto-EXACT in `_semantic_proof_binding_audit.py` | Removed; `_semantic_disposition_registry.py` authoritative |
| ECAP 26/26 + 34/34 false PASS | Restored **14+16 EXACT**, **12+18 OPEN** |
| Platform proof delegations | Documented PARTIAL with assertion inspection notes |
| W6/W7 pending==0 gates | Replaced with open-register counts + B6 XFAIL exact coverage |
| RPL 56/56 without semantic review | `_rpl_h_semantic_review.py` — 51× PARTIAL Package C |

**Remaining OPEN for B6:** 30 mandatory EXACT blockers; RPL semantic; OD-EBH7-01; B6; OD-EBH7-08.

**Not claimed:** B6 PASS, EBH-7 CLOSED, global FRZ PASS.
