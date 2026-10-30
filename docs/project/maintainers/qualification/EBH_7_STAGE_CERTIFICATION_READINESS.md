# EBH-7 — Stage certification readiness (closure wave)

| Wave | Materialization | Behavioral qualification | Notes |
| --- | --- | --- | --- |
| W0 entry SSOT + docs | MATERIALIZED | INTEGRITY OK | OD-EBH7-01 OPEN |
| W1 mechanism registry 685 | MATERIALIZED | INTEGRITY OK | unclassified=0 |
| W2 planning matrices | MATERIALIZED | INTEGRITY OK | |
| W3 NPSC allowlist | MATERIALIZED | QUALIFIED | B4-aligned gate executes |
| W4 gov/dup reuse | MATERIALIZED | QUALIFIED | Scoped DUP-X gates execute |
| W5 VBN + RPL-H | MATERIALIZED | **QUALIFIED** | 56/56 @ matrix + Package C |
| W6 tenant E2E | MATERIALIZED | **QUALIFIED** | 26/26 EXACT_PROOF |
| W7 cross-domain | MATERIALIZED | **QUALIFIED** | 34/34 EXACT_PROOF |
| W8 ECAP + L3 | MATERIALIZED | QUALIFIED | `tests/qualification/ebh_7/` green |
| W9 dossier | MATERIALIZED | INTEGRITY OK | refresh @ publication SHA |
| W10 Protocol 03 | MATERIALIZED | **OPEN** | B6 OPEN — independent only |
| OD-EBH7-08 | — | **OPEN** | Not operator-approved |
| Independent acceptance | — | **READY FOR AUDIT** | Pending B6 + OD-EBH7-08 |

**Not claimed:** EBH-7 CLOSED, B6 PASS, global FRZ PASS, ARCH-FREEZE, production admission.

**Program status:** Behavioral qualification complete @ HEAD — **READY FOR INDEPENDENT AUDIT** (B6 + OD-EBH7-08 remain operator/external).
