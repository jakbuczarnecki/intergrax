# EBH-7 — Stage certification readiness (reconciled)

| Wave | Materialization | Behavioral qualification | Notes |
| --- | --- | --- | --- |
| W0 entry SSOT + docs | MATERIALIZED | INTEGRITY OK | OD-EBH7-01 OPEN |
| W1 mechanism registry 685 | MATERIALIZED | INTEGRITY OK | unclassified=0 |
| W2 planning matrices | MATERIALIZED | INTEGRITY OK | |
| W3 NPSC allowlist | MATERIALIZED | QUALIFIED | B4-aligned gate executes |
| W4 gov/dup reuse | MATERIALIZED | PARTIAL | Structural import checks only |
| W5 VBN + RPL-H | MATERIALIZED | **OPEN** | 51/56 RPL NOT YET |
| W6 tenant E2E | MATERIALIZED | **OPEN** | 14/26 proofs execute; 12 pending |
| W7 cross-domain | MATERIALIZED | **OPEN** | 18/34 proofs execute; 16 pending |
| W8 ECAP + L3 | MATERIALIZED | PARTIAL | ECAP narrative; L3 reuse subset |
| W9 dossier | MATERIALIZED | INTEGRITY OK | `.tmp/proof/ebh-7-materialization/` |
| W10 Protocol 03 | MATERIALIZED | **OPEN** | B6 OPEN |
| OD-EBH7-08 | — | **OPEN** | Not operator-approved |
| Independent acceptance | — | **OPEN** | Not claimed |

**Not claimed:** EBH-7 CLOSED, B6 PASS, global FRZ PASS, ARCH-FREEZE, production admission.

**Program status:** IN PROGRESS — **READY FOR INDEPENDENT AUDIT** blocked until behavioral OPEN items close.
