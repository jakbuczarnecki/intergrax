# EBH-6 — T2 Evidence Independent Review (T2-AUD)

| Field | Value |
| --- | --- |
| **Authority** | [EBH_6_FINAL_BLOCKER_CLOSURE_PLAN.md](EBH_6_FINAL_BLOCKER_CLOSURE_PLAN.md) §3.B — invariant ↔ scope ↔ behavioral/structural gate |
| **Mechanical SSOT** | [`tests/qualification/ebh_6/_t2_evidence_bundles.py`](../../../tests/qualification/ebh_6/_t2_evidence_bundles.py) |
| **Gate test** | `test_ebh6_t2_bundle_review_complete` |
| **Disposition** | **PARTIAL** — **0 × INSUFFICIENT** @ contracts_typing repair; **OD-EBH6-Q2D2 OPEN** (`semantic_core`×F FINDING unchanged) |

Replaces the false **T2=131** census caused by `scanner_effective()` returning `True` for all A–J classes. Tier census derives from **bundle implementer review** (`ACCEPTED` → T2, `INSUFFICIENT` → T1).

## Bundle census (@ CERTIFICATION_HANDOFF_SHA)

| Implementer review | Bundles (unique) | Applicable pairs (133) |
| --- | ---: | ---: |
| **ACCEPTED** | 19 | 131 |
| **INSUFFICIENT** | 0 | 0 |
| **FINDING track** | `q2d2_finding` (ACCEPTED bundle, T3 outcome) | 1 |

**Tier @ HEAD (honest):** **T1=0**, **T2=131**, **T3=2** (`execution`×B PASS, `semantic_core`×F FINDING).

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
| `contracts_typing` | EBH-6 CSP + app _shared seam gate (+ EBH-2I CSP reuse) | **ACCEPTED** | Closed-world contracts surface; direct Tier-3 wiring `_shared` seams; **EBH6-DEBT-APP-TYP-01..11** registered |
| `platform_vendor_scan` | Prohibited/unclassified + inventory | ACCEPTED | Static A/D scan |
| `replaceability_j` | **RPL-G invoke** + VBN matrix doc | ACCEPTED | Replaceability slice |
| `q2d2_finding` | Q2-D2 B3 gate + register | ACCEPTED (FINDING outcome) | Operator OD-EBH6-Q2D2 |
| `platform_fail_closed` / `platform_hardcoding` / `platform_typing_scan` / `l3_reachability` | Class gates + L3 **primary_gate** nodeids | ACCEPTED | Residual class coverage |

## contracts_typing — production scope & exclusions

| Layer | Production-effective scope | Legitimate exclusion |
| --- | --- | --- |
| `contracts` | `discover_contracts_surface_paths()` (853 modules) | EBH-2I supplemental surfaces under `intergrax/applications/contracts/` — outside EBH-6 contracts layer closed world |
| `application_services` | Direct imports from Tier-3 `applications/**/*_wiring.py` + non-harness `_shared/*_wiring.py` seeds (158 modules) | Transitive `_shared` helpers not on wiring import surface; harness-only `_shared/*_wiring.py` seeds excluded |

**Registered typing debt (qual SSOT):** `APPLICATION_SERVICES_TYPING_DEBT` in [`_contracts_typing_qualification.py`](../../../tests/qualification/ebh_6/_contracts_typing_qualification.py) — **11** rows; remediation **OD-EBH6-01**. Gate proves **no unregistered** FRZ-TYP-02/03 public seam regressions on scoped surface.

## Layer register impact (T-LAY-RECON)

**16 × CANDIDATE-RESOLVED** @ B3 group outcomes (including `contracts`, `application_services`) — pending Protocol 03 acceptance. **Not** EBH-6 CLOSED.

## Stage closure (operator / auditor — not implementer PASS)

| Step | Owner | Gate |
| --- | --- | --- |
| **E1** | Operator | **OD-EBH6-Q2D2** path A or B (+ OD-EBH6-01 if remediation) |
| **E2** | Implementer + auditor | Pre-cert PASS @ SHA — **0 mandatory INSUFFICIENT** (**met** for T2-AUD; Q2-D2 remains) |
| **E3** | Independent auditor | Protocol 03 ECAP B1–B6 |
| **E4** | Operator | **OD-EBH6-08** — EBH-6 CLOSED |

## Independent pre-cert (E01)

Worksheet: [EBH_6_PRE_CERT_INDEPENDENT_AUDIT_WORKSHEET.md](EBH_6_PRE_CERT_INDEPENDENT_AUDIT_WORKSHEET.md) — **PENDING** auditor **E01-PRE-01**.

## References

- [EBH_6_SCOPE_RECONCILIATION_DOSSIER.md](EBH_6_SCOPE_RECONCILIATION_DOSSIER.md)
- [EBH_6_STAGE_CERTIFICATION_READINESS.md](EBH_6_STAGE_CERTIFICATION_READINESS.md)
- [EBH_6_L9_INDEPENDENT_AUDIT_HANDOFF.md](EBH_6_L9_INDEPENDENT_AUDIT_HANDOFF.md)
