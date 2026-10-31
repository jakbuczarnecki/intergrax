# EBH-7 — C0 baseline audit record

| Field | Value |
| --- | --- |
| Program | EBH-7 Comprehensive Platform Enterprise Architecture Certification |
| Planning baseline SHA | `5df7018aa71f8107b07531c50507eada2cc76f86` |
| Parent program | ROADMAP-REPLAY-X — CLOSED (SCOPED) — OD-REPLAYX-08 |
| Branch | `development` |
| REQUIREMENTS_TOTAL | 466 |
| TASKS_TOTAL | 92 |
| MECHANISM_UNIQUE_COUNT | 685 historical (dirty local tree) → **238** over the tracked tree (B6-F10-P03-DIRTY-TREE) |
| Tenant E2E scenarios | 26 |
| Cross-domain scenarios | 34 |
| RPL-H rows (Annex D) | 56 (51 NOT YET + 5 QUALIFIED) |
| NPSC backlog | 10 modules (Annex G) — gate refined @ W3 |
| Gap register | 11 (Annex H) |
| Global FRZ PASS | **Not claimed** (out of scope) |
| OD-EBH7-01 | OPEN — operator entry gate |

## Operational blockers (pre-IMPLEMENTATION_SHA)

- Untracked local dirs: `exports/`, `platform_proofs/test_*` — must not enter IMPLEMENTATION_SHA without operator decision.
- GAP-TEN-UNK-05: five `unknown_role` tenant surfaces — W6 closure target.

## SSOT mirrors

Pytest SSOT under `tests/qualification/ebh_7/` mirrors annexes A–B (466 requirements, 92 tasks, 685 mechanisms).
