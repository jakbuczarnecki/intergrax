# EBH-7 — ECAP current HEAD audit report (C1–C5)

| Counter | Structural @ materialization | Behavioral qualification |
| --- | --- | ---: |
| ECAP-C1 unclassified mechanisms (685 SSOT) | 0 | INTEGRITY OK |
| ECAP-C2 unauthorized bypass (dup_x subset) | replay counters @ prior stages | PARTIAL — not full platform replay |
| ECAP-C3 NPSC global allowlist | gate executes PASS | QUALIFIED |
| ECAP-C4 tenant E2E (26) | registry MATERIALIZED | **14 EXACT_PROOF / 12 OPEN** (semantic audit) |
| ECAP-C5 cross-domain (34) | registry MATERIALIZED | **16 EXACT_PROOF / 18 OPEN** (semantic audit) |

**Explicit non-claims:** 51× RPL-H NOT YET; OD-EBH7-08 OPEN; B6 OPEN; global FRZ PASS not claimed.

**Planning baseline SHA:** `5df7018aa71f8107b07531c50507eada2cc76f86` (IMPLEMENTATION_SHA updates after operator OD-EBH7-01).
