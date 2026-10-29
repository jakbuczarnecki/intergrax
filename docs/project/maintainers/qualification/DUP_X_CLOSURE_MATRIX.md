# DUP-X — Unified Closure Matrix (template)

| OBLIGATION | OWNER | EVIDENCE | TEST | STATUS | AUDITOR | CLOSURE |
| --- | --- | --- | --- | --- | --- | --- |
| §3.0.3 unresolved DUPLICATE/BLOCKER | Implementer register | `DUP_X_FINDINGS_REGISTER.md` | `test_dup_x_section_13_measure_unresolved_duplicate_blockers` | **PASS** (0 open) | pending | OPEN |
| §3.0.3 shadow authority | Implementer register | `_findings_register.py` | `test_dup_x_section_13_measure_shadow_authority` | **PASS** (0) | pending | OPEN |
| §3.0.3 bypass (prod) | COMPAT-R3 fence | EBH6-DEBT-BND-COMPAT-R3 | `test_dup_x_compat_r3_lab_paths_not_imported_from_sanctioned_roots` | **RATIFIED** | pending | OPEN |
| 15 responsibility families | Wave A inventory | `DUP_X_CLOSED_WORLD_INVENTORY.md` | `test_dup_x_fifteen_responsibility_families` | **PASS** | pending | OPEN |
| DUP-01..14 coverage | Class matrix | `DUP_X_CLASS_COVERAGE_MATRIX.md` | `test_dup_x_fourteen_dup_classes_catalog` | **PASS** | pending | OPEN |
| DUPX-CAND-01..15 | Findings register | `_findings_register.py` | `test_dup_x_candidates_fifteen_rows_no_orphans` | **PASS** | pending | OPEN |
| Retained parallel delta proof | Delta dossiers | `DUP_X_RESPONSIBILITY_DELTA_*.md` | `test_dup_x_retained_parallel_delta_dossiers_present` | **PASS** | pending | OPEN |
| CONFIG-X 54/54 replay | CONFIG-X | EBH-6 reconciliation | `test_dup_x_exit_config_x_reconciled` | **PASS** | pending | OPEN |
| DUP-14 qual SSOT | EBH-5 hints | `_config_x_discovery.py` | `test_dup_x_exit_composition_hints_normative_in_config_x` | **PASS** | pending | OPEN |
| OD-DUPX-02 plan audit | Operator | Final parent closure master plan | — | **APPROVED — 2026-10-29** | operator | Block I–II authorized |
| OD-DUPX-01 Wave B prod | Operator | — | — | **NOT TRIGGERED** | operator | OPEN |
| Protocol DUP-X-03 | Independent auditor | `DUP_X_PROTOCOL_03_INDEPENDENT_AUDIT_PACKAGE.md` | `.tmp/proof/dup-x-final-b6-replay/` · pin `a7d83925b…` | **READY FOR EXTERNAL B6** — **44 passed** | external | B6 **OPEN** |
| OD-DUPX-08 closure | Operator | PASS record | — | **OPEN** | operator | **BLOCKED until B6 PASS** |
| DUPX-GAP-CLASS-D | Implementer | Gap register + caller matrix | class D gates + Wave 1 matrix | **PASS** | pending | OPEN |
| DUPX-GAP-L3-TIER3-DEDUPE | Implementer | Tier-3 disposition doc | L3 per app + parent closure gates | **PASS** | pending | OPEN |

**STATUS legend:** OPEN · IN_PROGRESS · PASS · N/A WITH EVIDENCE · BLOCKED · GAP

## STOP recovery (no replan)

| STOP | Recovery |
| --- | --- |
| S-01 B6 REJECT B4/NPSC | OD-DUPX-01 · re-qual · re-B6 |
| S-02 B6 REJECT scope | Inventory/delta fix · re-B6 |
| S-03 Qual fail | Minimal fix · new QUAL SHA |
| S-04 No OD-DUPX-08 | Stays NOT CLOSED |
| S-05 Class C prod | OD-03..07 first |

**Global stage:** `DUP-X — NOT CLOSED` (roadmap §5 only after OD-DUPX-08 post-B6)
