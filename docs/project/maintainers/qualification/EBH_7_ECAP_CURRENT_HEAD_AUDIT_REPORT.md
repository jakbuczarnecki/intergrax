# EBH-7 — ECAP current HEAD audit report (C1–C5)

| Counter | Structural @ materialization | Behavioral qualification |
| --- | --- | ---: |
| ECAP-C1 unclassified mechanisms (685 SSOT) | 0 | INTEGRITY OK |
| ECAP-C2 unauthorized bypass (dup_x subset) | scoped gates execute | **QUALIFIED** (W4 DUP-X subset @ HEAD) |
| ECAP-C3 NPSC global allowlist | gate executes PASS | QUALIFIED |
| ECAP-C4 tenant E2E (26) | registry MATERIALIZED | **26/26 EXACT_PROOF** |
| ECAP-C5 cross-domain (34) | registry MATERIALIZED | **34/34 EXACT_PROOF** |

**Explicit non-claims:** OD-EBH7-01 OPEN; OD-EBH7-08 OPEN; B6 OPEN; global FRZ PASS not claimed; stage **not** CLOSED until independent B6 + operator OD-EBH7-08.

**RPL-H:** 56/56 REPLACEABLE QUALIFIED @ matrix (Package C + legacy five).

**Planning baseline SHA:** `5df7018aa71f8107b07531c50507eada2cc76f86`

**TEST_EXECUTION_SHA:** `f913944ff608c97a4a4145aae6c2bbd610004f64`
