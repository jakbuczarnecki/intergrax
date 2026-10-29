# EBH-6 — T2 Evidence Independent Review (T2-AUD)

| Field | Value |
| --- | --- |
| **Authority** | [EBH_6_FINAL_BLOCKER_CLOSURE_PLAN.md](EBH_6_FINAL_BLOCKER_CLOSURE_PLAN.md) §3.B — invariant ↔ scope ↔ behavioral/structural gate |
| **Mechanical SSOT** | [`tests/qualification/ebh_6/_t2_evidence_bundles.py`](../../../tests/qualification/ebh_6/_t2_evidence_bundles.py) |
| **Gate test** | `test_ebh6_t2_bundle_review_complete` |
| **Disposition** | **PARTIAL** — **2 × INSUFFICIENT** (`contracts`×F, `application_services`×F); **OD-EBH6-Q2D2 OPEN** |

Replaces the false **T2=131** census caused by `scanner_effective()` returning `True` for all A–J classes. Tier census now derives from **bundle implementer review** (`ACCEPTED` → T2, `INSUFFICIENT` → T1).

## Bundle census (@ CERTIFICATION_HANDOFF_SHA)

| Implementer review | Bundles (unique) | Applicable pairs (133) |
| --- | ---: | ---: |
| **ACCEPTED** | 18 | 131 |
| **INSUFFICIENT** | 1 (`contracts_typing`) | 2 |
| **FINDING track** | `q2d2_finding` (ACCEPTED bundle, T3 outcome) | 1 |

**Tier @ HEAD (honest):** **T1=2**, **T2=129**, **T3=2** (`execution`×B PASS, `semantic_core`×F FINDING).

## Mechanism bundles

| Bundle ID | Primary behavioral gates (nodeids) | Review | Scope |
| --- | --- | --- | --- |
| `integration_factory` | EBH-5 factory/reachability, RPL-G | ACCEPTED | Integration materialization slice |
| `sanctioned_composition` | `test_ebh6_exit_sanctioned_composition_roots_non_empty` | ACCEPTED | TRACE-X registry |
| `execution_bypass` | EE-A1 P0 inventory | ACCEPTED | Execution bypass |
| `nexus_reachability` | Sanctioned roots + L3 primary gates | ACCEPTED | Nexus L3 index |
| `governance_boundary` | NPSC42 H1 + register | ACCEPTED | Governance non-execution |
| `persistence_sqlite` | SQLite annex gate | ACCEPTED | Annex accountability |
| `observability_authority` | **W5-B2 invoke** `test_production_composition_creates_bounded_sink` | ACCEPTED | Bounded sink wiring |
| `recovery_plane` | NPSC5e | ACCEPTED | Recovery admission |
| `agent_lifecycle` | Lifecycle architecture gate | ACCEPTED | Agent distribution |
| `rag_bootstrap` | EBH-5 RAG graph reachability | ACCEPTED | No shadow registry |
| `config_hosting` | CONFIG-X + EBH-5 inventory | ACCEPTED | Structural mapping |
| `contracts_typing` | Register ratified debt only | **INSUFFICIENT** | **Not** FRZ-TYP behavioral proof |
| `platform_vendor_scan` | Prohibited/unclassified + inventory | ACCEPTED | Static A/D scan |
| `replaceability_j` | **RPL-G invoke** + VBN matrix doc | ACCEPTED | Replaceability slice |
| `q2d2_finding` | Q2-D2 B3 gate + register | ACCEPTED (FINDING outcome) | Operator OD-EBH6-Q2D2 |
| `platform_fail_closed` / `platform_hardcoding` / `platform_typing_scan` / `l3_reachability` | Class gates + L3 **primary_gate** nodeids | ACCEPTED | Residual class coverage |

## INSUFFICIENT pairs (mandatory T1 @ HEAD)

| Layer | Class | Bundle | Remediation |
| --- | --- | --- | --- |
| `contracts` | F | `contracts_typing` | FRZ-TYP behavioral gate or operator NA rationale |
| `application_services` | F | `contracts_typing` | Same |

## Layer register impact (T-LAY-RECON)

**14 × CANDIDATE-RESOLVED**, **2 × L9_EVIDENCE_SUBMITTED** (`contracts`, `application_services`) — zero T1 on all applicable classes required for CANDIDATE-RESOLVED.

## Stage closure (operator / auditor — not implementer PASS)

| Step | Owner | Gate |
| --- | --- | --- |
| **E1** | Operator | **OD-EBH6-Q2D2** path A or B (+ OD-EBH6-01 if remediation) |
| **E2** | Implementer + auditor | Pre-cert PASS @ SHA — **0 mandatory INSUFFICIENT** |
| **E3** | Independent auditor | Protocol 03 ECAP B1–B6 |
| **E4** | Operator | **OD-EBH6-08** — EBH-6 CLOSED |

## References

- [EBH_6_SCOPE_RECONCILIATION_DOSSIER.md](EBH_6_SCOPE_RECONCILIATION_DOSSIER.md)
- [EBH_6_STAGE_CERTIFICATION_READINESS.md](EBH_6_STAGE_CERTIFICATION_READINESS.md)
- [EBH_6_L9_INDEPENDENT_AUDIT_HANDOFF.md](EBH_6_L9_INDEPENDENT_AUDIT_HANDOFF.md)
