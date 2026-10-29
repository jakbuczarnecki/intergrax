# EBH-6 — Independent Replay Checklist (E05–E08)

| Field | Value |
| --- | --- |
| **Baseline SHA** | `202239f726dd5e3fe23bf1a68cafa61c1eb9fca7` |
| **Export SSOT** | [`_p03_handoff_export.py`](../../../tests/qualification/ebh_6/_p03_handoff_export.py) |

Implementer attestation is **CANDIDATE-RESOLVED** / **PENDING** only. **RESOLVED** and family dispositions require Protocol 03 (E10).

## E05 — 16 architecture layers

| Task | Layer | Implementer status | Independent disposition | Checklist |
| --- | --- | --- | --- | --- |
| E05-LAY-01 | contracts | CANDIDATE-RESOLVED | **PENDING** | closed-world · L3 · A–J |
| E05-LAY-02 | semantic_core | CANDIDATE-RESOLVED | **PENDING** | + Q2-D2 FINDING |
| E05-LAY-03 | application_services | CANDIDATE-RESOLVED | **PENDING** | |
| E05-LAY-04 | composition_roots | CANDIDATE-RESOLVED | **PENDING** | |
| E05-LAY-05 | governance | CANDIDATE-RESOLVED | **PENDING** | |
| E05-LAY-06 | execution | CANDIDATE-RESOLVED | **PENDING** | |
| E05-LAY-07 | integrations | CANDIDATE-RESOLVED | **PENDING** | |
| E05-LAY-08 | state_persistence | CANDIDATE-RESOLVED | **PENDING** | |
| E05-LAY-09 | reliability_recovery | CANDIDATE-RESOLVED | **PENDING** | |
| E05-LAY-10 | observability_diagnostics | CANDIDATE-RESOLVED | **PENDING** | |
| E05-LAY-11 | events_traceability | CANDIDATE-RESOLVED | **PENDING** | |
| E05-LAY-12 | agent_distribution | CANDIDATE-RESOLVED | **PENDING** | |
| E05-LAY-13 | rag_memory | CANDIDATE-RESOLVED | **PENDING** | |
| E05-LAY-14 | runtime_hosting | CANDIDATE-RESOLVED | **PENDING** | |
| E05-LAY-15 | tools_skills_capabilities | CANDIDATE-RESOLVED | **PENDING** | |
| E05-LAY-16 | background_workers_infra_adapters | CANDIDATE-RESOLVED | **PENDING** | |

## E06 — 23 families (`_family_catalog.py`)

Families: **BND, OWN, CTR, TYP, PLG, RPL, EXE, GOV, STA, REC, CMP, CFG, SEC, REL, CTL, OBS, TRC, PRD, REG, HRN, TEN, DOC, VBN**

| Task | Family | Freshness (catalog) | Independent disposition |
| --- | --- | --- | --- |
| E06-FAM-01..23 | (each ID above) | HISTORICAL or CURRENT per row | **23 × PENDING** |

Gate sync: `test_ebh6_family_catalog_covers_applicable_themes` + matrix integrity test.

## E07 — 8 L9 blockers

| Task | Blocker ID | Family | Replay bundle |
| --- | --- | --- | --- |
| E07-BLK-01 | EBH6-BLK-L9-OWN-01 | OWN | `l9_blocker_replay` in P03 export |
| E07-BLK-02 | EBH6-BLK-L9-EXE-01 | EXE | |
| E07-BLK-03 | EBH6-BLK-L9-GOV-01 | GOV | |
| E07-BLK-04 | EBH6-BLK-L9-REC-01 | REC | |
| E07-BLK-05 | EBH6-BLK-L9-SEC-01 | SEC | |
| E07-BLK-06 | EBH6-BLK-L9-REL-01 | REL | |
| E07-BLK-07 | EBH6-BLK-L9-HRN-01 | HRN | |
| E07-BLK-08 | EBH6-BLK-L9-TEN-01 | TEN | |

**Rule:** nodeid existence ≠ replay PASS — auditor must execute @ **TEST_EXECUTION_SHA**.

## E08 — VBN / FRZ / CONFIG-X / register

| Task | Criterion | Mechanical @ baseline | Independent |
| --- | --- | --- | --- |
| E08-REG-01 | Register complete · unclassified 0 | PASS | **PENDING** |
| E08-PRO-01 | PROHIBITED unresolved 0 | PASS | **PENDING** |
| E08-CFG-01 | CONFIG-X 54/54 | PASS | **PENDING** |
| E08-VBN-01 | CURRENT_HEAD VBN audit | Implementer doc @ SHA | **PENDING** |
| E08-FRZ-01 | Scoped FRZ mapping | Closure plan RC-01 | **PENDING** |

**DONE:** all rows independently signed in Protocol 03 report (E10).
