# EBH-6 — Technical Pre-Cert Evidence Dossier (C1)

| Field | Value |
| --- | --- |
| **Workstream** | C1 — TECHNICAL PRE-CERT EVIDENCE DOSSIER |
| **QUALIFICATION_HEAD_SHA** | 05e3363c5382532fd4d5ed4990b610f342ef5834 |
| **CERTIFICATION_HANDOFF_SUBSTANCE_SHA** | c852e7705f318b5d50a771596d97193f68a00e67 |
| **Independent pre-cert (E01-PRE-01)** | **PENDING** — agent must not PASS |
| **Mechanical SSOT** | [_pre_cert_evidence_export.py](../../../tests/qualification/ebh_6/_pre_cert_evidence_export.py) |

## Metadata

```json
{
  "qualification_head_sha": "05e3363c5382532fd4d5ed4990b610f342ef5834",
  "certification_handoff_substance_sha": "c852e7705f318b5d50a771596d97193f68a00e67",
  "pair_count": 133,
  "bundle_count": 19,
  "independent_disposition": "PENDING — agent must not issue E01-PRE-01 PASS"
}
```

## 19 bundle replay matrix (implementer — auditor **PENDING**)

| Bundle | Review | Pairs | Invariant | Primary gates | Scope limit | Independent |
| --- | --- | ---: | --- | --- | --- | --- |
| `agent_lifecycle` | ACCEPTED | 2 | Agent distribution lifecycle + HARDENING-3 bound… | tests/integration/agent_distribution/test_canonical_agent_li… | Boundary/lifecycle subset — not all agen… | **PENDING** |
| `config_hosting` | ACCEPTED | 15 | CONFIG-X structural reconciliation @ HEAD… | tests/qualification/ebh_6/test_ebh_6_config_x_head_reconcili… | Mapping completeness — not host profile … | **PENDING** |
| `contracts_typing` | ACCEPTED | 2 | FRZ-TYP public contract seams… | tests/qualification/ebh_6/test_ebh_6_contracts_typing_qualif… | EBH-2I CSP on closed-world contracts sur… | **PENDING** |
| `execution_bypass` | ACCEPTED | 3 | EE-A1 execution bypass inventory… | tests/unit/runtime/architecture/test_ee_a1_execution_engine_… | EE-A1 certified surface — not whole-plat… | **PENDING** |
| `governance_boundary` | ACCEPTED | 10 | NPSC42 governance non-execution identity… | tests/unit/runtime/architecture/test_npsc42_h1_governance_bo… | NPSC42 frozen subset — observability G u… | **PENDING** |
| `integration_factory` | ACCEPTED | 7 | FRZ-CTR integration factory + EBH-5 reachability… | tests/qualification/ebh_5/test_ebh_5_reachability_gates.py::… | Integration factory + inventory — not li… | **PENDING** |
| `l3_reachability` | ACCEPTED | 7 | L3 mechanism primary_gate nodeids per layer… | tests/qualification/ebh_6/test_ebh_6_mechanical_exit_gates.p… | Primary_gate pytest nodeids from `_l3_re… | **PENDING** |
| `nexus_reachability` | ACCEPTED | 1 | L3 nexus call chains + sanctioned roots… | tests/qualification/ebh_6/test_ebh_6_mechanical_exit_gates.p… | Mechanism-level L3 index — primary_gate … | **PENDING** |
| `observability_authority` | ACCEPTED | 1 | W5-B2 bounded observability sink wiring… | tests/unit/runtime/events/test_enterprise_scale_resilience_w… | Behavioral W5-B2 invoke — not global OBS… | **PENDING** |
| `persistence_sqlite` | ACCEPTED | 8 | SQLite annex accountability… | tests/qualification/ebh_6/test_ebh_6_mechanical_exit_gates.p… | Annex accountability — not all shadow-st… | **PENDING** |
| `platform_fail_closed` | ACCEPTED | 12 | Fail-closed materialization markers… | tests/qualification/ebh_5/test_ebh_5_reachability_gates.py::… | Static fail-closed markers — not shadow … | **PENDING** |
| `platform_hardcoding` | ACCEPTED | 9 | Class-B literal scan + register… | tests/qualification/ebh_6/test_ebh_6_hardcoding_register_int… | Sample literal scan — not exhaustive bac… | **PENDING** |
| `platform_typing_scan` | ACCEPTED | 13 | Protocol/TYPE_CHECKING seam scan… | tests/qualification/ebh_6/test_ebh_6_hardcoding_register_int… | Static typing seam markers — not FRZ-TYP… | **PENDING** |
| `platform_vendor_scan` | ACCEPTED | 23 | Register prohibited/unclassified + layer closed-… | tests/qualification/ebh_6/test_ebh_6_hardcoding_register_int… | Inventory + static vendor scan — not liv… | **PENDING** |
| `q2d2_finding` | ACCEPTED | 1 | EBH6-DEBT-Q2-D2 / FRZ-TYP-03… | tests/qualification/ebh_6/test_ebh_6_b3_exit_gates.py::test_… | FINDING only — OD-EBH6-Q2D2 operator OPE… | **PENDING** |
| `rag_bootstrap` | ACCEPTED | 3 | T-RAG-01 graph bootstrap / no shadow registry… | tests/qualification/ebh_5/test_ebh_5_reachability_gates.py::… | RAG graph bootstrap gates — not full mem… | **PENDING** |
| `recovery_plane` | ACCEPTED | 1 | NPSC5e recovery plane qualification… | tests/unit/runtime/architecture/test_npsc5e_final_recovery_p… | Sample L3 recovery admission — not full … | **PENDING** |
| `replaceability_j` | ACCEPTED | 14 | RPL-G CONFIG-X blockers + VBN matrix… | tests/qualification/ebh_5/test_ebh_5_rpl_g_regression_gates.… | RPL-G behavioral invoke + matrix cite — … | **PENDING** |
| `sanctioned_composition` | ACCEPTED | 1 | TRACE-X sanctioned composition sites… | tests/qualification/ebh_6/test_ebh_6_mechanical_exit_gates.p… | Registry presence — not per-root wiring … | **PENDING** |

## 133× A–J pair index (implementer — independent **PENDING**)

| Layer | Class | Tier | Outcome | Bundle | Gate nodeid | Independent |
| --- | --- | --- | --- | --- | --- | --- |
| `contracts` | A | T2 | PARTIAL_CANDIDATE | `platform_vendor_scan` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `contracts` | F | T2 | PARTIAL_CANDIDATE | `contracts_typing` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `contracts` | H | T2 | PARTIAL_CANDIDATE | `l3_reachability` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `contracts` | J | T2 | PARTIAL_CANDIDATE | `replaceability_j` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `semantic_core` | A | T2 | PARTIAL_CANDIDATE | `platform_vendor_scan` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `semantic_core` | B | T2 | PARTIAL_CANDIDATE | `execution_bypass` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `semantic_core` | C | T2 | PARTIAL_CANDIDATE | `platform_fail_closed` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `semantic_core` | F | T3 | FINDING | `q2d2_finding` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `semantic_core` | H | T2 | PARTIAL_CANDIDATE | `nexus_reachability` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `semantic_core` | I | T2 | PARTIAL_CANDIDATE | `config_hosting` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `semantic_core` | J | T2 | PARTIAL_CANDIDATE | `replaceability_j` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `application_services` | A | T2 | PARTIAL_CANDIDATE | `platform_vendor_scan` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `application_services` | B | T2 | PARTIAL_CANDIDATE | `platform_hardcoding` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `application_services` | C | T2 | PARTIAL_CANDIDATE | `platform_fail_closed` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `application_services` | D | T2 | PARTIAL_CANDIDATE | `platform_vendor_scan` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `application_services` | F | T2 | PARTIAL_CANDIDATE | `contracts_typing` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `application_services` | H | T2 | PARTIAL_CANDIDATE | `l3_reachability` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `application_services` | I | T2 | PARTIAL_CANDIDATE | `config_hosting` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `application_services` | J | T2 | PARTIAL_CANDIDATE | `replaceability_j` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `composition_roots` | A | T2 | PARTIAL_CANDIDATE | `platform_vendor_scan` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `composition_roots` | B | T2 | PARTIAL_CANDIDATE | `platform_hardcoding` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `composition_roots` | C | T2 | PARTIAL_CANDIDATE | `integration_factory` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `composition_roots` | D | T2 | PARTIAL_CANDIDATE | `platform_vendor_scan` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `composition_roots` | F | T2 | PARTIAL_CANDIDATE | `platform_typing_scan` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `composition_roots` | H | T2 | PARTIAL_CANDIDATE | `sanctioned_composition` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `composition_roots` | I | T2 | PARTIAL_CANDIDATE | `config_hosting` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `composition_roots` | J | T2 | PARTIAL_CANDIDATE | `replaceability_j` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `governance` | C | T2 | PARTIAL_CANDIDATE | `platform_fail_closed` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `governance` | D | T2 | PARTIAL_CANDIDATE | `platform_vendor_scan` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `governance` | E | T2 | PARTIAL_CANDIDATE | `persistence_sqlite` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `governance` | F | T2 | PARTIAL_CANDIDATE | `platform_typing_scan` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `governance` | G | T2 | PARTIAL_CANDIDATE | `governance_boundary` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `governance` | H | T2 | PARTIAL_CANDIDATE | `governance_boundary` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `governance` | I | T2 | PARTIAL_CANDIDATE | `config_hosting` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `execution` | A | T2 | PARTIAL_CANDIDATE | `platform_vendor_scan` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `execution` | B | T3 | PASS_CANDIDATE | `execution_bypass` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `execution` | C | T2 | PARTIAL_CANDIDATE | `platform_fail_closed` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `execution` | F | T2 | PARTIAL_CANDIDATE | `platform_typing_scan` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `execution` | G | T2 | PARTIAL_CANDIDATE | `governance_boundary` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `execution` | H | T2 | PARTIAL_CANDIDATE | `execution_bypass` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `execution` | I | T2 | PARTIAL_CANDIDATE | `config_hosting` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `execution` | J | T2 | PARTIAL_CANDIDATE | `replaceability_j` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `integrations` | A | T2 | PARTIAL_CANDIDATE | `integration_factory` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `integrations` | B | T2 | PARTIAL_CANDIDATE | `integration_factory` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `integrations` | C | T2 | PARTIAL_CANDIDATE | `integration_factory` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `integrations` | D | T2 | PARTIAL_CANDIDATE | `platform_vendor_scan` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `integrations` | E | T2 | PARTIAL_CANDIDATE | `persistence_sqlite` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `integrations` | F | T2 | PARTIAL_CANDIDATE | `platform_typing_scan` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `integrations` | G | T2 | PARTIAL_CANDIDATE | `governance_boundary` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `integrations` | H | T2 | PARTIAL_CANDIDATE | `integration_factory` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `integrations` | I | T2 | PARTIAL_CANDIDATE | `integration_factory` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `integrations` | J | T2 | PARTIAL_CANDIDATE | `integration_factory` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `state_persistence` | A | T2 | PARTIAL_CANDIDATE | `platform_vendor_scan` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `state_persistence` | B | T2 | PARTIAL_CANDIDATE | `platform_hardcoding` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `state_persistence` | C | T2 | PARTIAL_CANDIDATE | `platform_fail_closed` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `state_persistence` | D | T2 | PARTIAL_CANDIDATE | `platform_vendor_scan` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `state_persistence` | E | T2 | PARTIAL_CANDIDATE | `persistence_sqlite` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `state_persistence` | F | T2 | PARTIAL_CANDIDATE | `platform_typing_scan` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `state_persistence` | H | T2 | PARTIAL_CANDIDATE | `persistence_sqlite` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `state_persistence` | I | T2 | PARTIAL_CANDIDATE | `config_hosting` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `state_persistence` | J | T2 | PARTIAL_CANDIDATE | `replaceability_j` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `reliability_recovery` | A | T2 | PARTIAL_CANDIDATE | `platform_vendor_scan` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `reliability_recovery` | B | T2 | PARTIAL_CANDIDATE | `platform_hardcoding` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `reliability_recovery` | C | T2 | PARTIAL_CANDIDATE | `platform_fail_closed` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `reliability_recovery` | E | T2 | PARTIAL_CANDIDATE | `persistence_sqlite` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `reliability_recovery` | F | T2 | PARTIAL_CANDIDATE | `platform_typing_scan` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `reliability_recovery` | G | T2 | PARTIAL_CANDIDATE | `governance_boundary` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `reliability_recovery` | H | T2 | PARTIAL_CANDIDATE | `recovery_plane` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `reliability_recovery` | I | T2 | PARTIAL_CANDIDATE | `config_hosting` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `reliability_recovery` | J | T2 | PARTIAL_CANDIDATE | `replaceability_j` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `observability_diagnostics` | C | T2 | PARTIAL_CANDIDATE | `platform_fail_closed` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `observability_diagnostics` | D | T2 | PARTIAL_CANDIDATE | `platform_vendor_scan` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `observability_diagnostics` | F | T2 | PARTIAL_CANDIDATE | `platform_typing_scan` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `observability_diagnostics` | G | T2 | PARTIAL_CANDIDATE | `observability_authority` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `observability_diagnostics` | H | T2 | PARTIAL_CANDIDATE | `l3_reachability` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `observability_diagnostics` | I | T2 | PARTIAL_CANDIDATE | `config_hosting` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `observability_diagnostics` | J | T2 | PARTIAL_CANDIDATE | `replaceability_j` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `events_traceability` | A | T2 | PARTIAL_CANDIDATE | `platform_vendor_scan` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `events_traceability` | B | T2 | PARTIAL_CANDIDATE | `platform_hardcoding` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `events_traceability` | C | T2 | PARTIAL_CANDIDATE | `platform_fail_closed` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `events_traceability` | D | T2 | PARTIAL_CANDIDATE | `platform_vendor_scan` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `events_traceability` | E | T2 | PARTIAL_CANDIDATE | `persistence_sqlite` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `events_traceability` | F | T2 | PARTIAL_CANDIDATE | `platform_typing_scan` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `events_traceability` | G | T2 | PARTIAL_CANDIDATE | `governance_boundary` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `events_traceability` | H | T2 | PARTIAL_CANDIDATE | `l3_reachability` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `events_traceability` | I | T2 | PARTIAL_CANDIDATE | `config_hosting` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `events_traceability` | J | T2 | PARTIAL_CANDIDATE | `replaceability_j` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `agent_distribution` | A | T2 | PARTIAL_CANDIDATE | `platform_vendor_scan` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `agent_distribution` | B | T2 | PARTIAL_CANDIDATE | `agent_lifecycle` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `agent_distribution` | C | T2 | PARTIAL_CANDIDATE | `platform_fail_closed` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `agent_distribution` | D | T2 | PARTIAL_CANDIDATE | `platform_vendor_scan` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `agent_distribution` | F | T2 | PARTIAL_CANDIDATE | `platform_typing_scan` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `agent_distribution` | H | T2 | PARTIAL_CANDIDATE | `agent_lifecycle` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `agent_distribution` | I | T2 | PARTIAL_CANDIDATE | `config_hosting` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `agent_distribution` | J | T2 | PARTIAL_CANDIDATE | `replaceability_j` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `rag_memory` | A | T2 | PARTIAL_CANDIDATE | `rag_bootstrap` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `rag_memory` | B | T2 | PARTIAL_CANDIDATE | `platform_hardcoding` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `rag_memory` | C | T2 | PARTIAL_CANDIDATE | `platform_fail_closed` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `rag_memory` | D | T2 | PARTIAL_CANDIDATE | `rag_bootstrap` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `rag_memory` | E | T2 | PARTIAL_CANDIDATE | `persistence_sqlite` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `rag_memory` | F | T2 | PARTIAL_CANDIDATE | `platform_typing_scan` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `rag_memory` | G | T2 | PARTIAL_CANDIDATE | `governance_boundary` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `rag_memory` | H | T2 | PARTIAL_CANDIDATE | `rag_bootstrap` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `rag_memory` | I | T2 | PARTIAL_CANDIDATE | `config_hosting` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `rag_memory` | J | T2 | PARTIAL_CANDIDATE | `replaceability_j` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `runtime_hosting` | A | T2 | PARTIAL_CANDIDATE | `platform_vendor_scan` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `runtime_hosting` | B | T2 | PARTIAL_CANDIDATE | `platform_hardcoding` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `runtime_hosting` | C | T2 | PARTIAL_CANDIDATE | `config_hosting` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `runtime_hosting` | D | T2 | PARTIAL_CANDIDATE | `platform_vendor_scan` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `runtime_hosting` | F | T2 | PARTIAL_CANDIDATE | `platform_typing_scan` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `runtime_hosting` | G | T2 | PARTIAL_CANDIDATE | `governance_boundary` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `runtime_hosting` | H | T2 | PARTIAL_CANDIDATE | `l3_reachability` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `runtime_hosting` | I | T2 | PARTIAL_CANDIDATE | `config_hosting` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `runtime_hosting` | J | T2 | PARTIAL_CANDIDATE | `replaceability_j` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `tools_skills_capabilities` | A | T2 | PARTIAL_CANDIDATE | `platform_vendor_scan` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `tools_skills_capabilities` | B | T2 | PARTIAL_CANDIDATE | `platform_hardcoding` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `tools_skills_capabilities` | C | T2 | PARTIAL_CANDIDATE | `platform_fail_closed` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `tools_skills_capabilities` | D | T2 | PARTIAL_CANDIDATE | `platform_vendor_scan` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `tools_skills_capabilities` | F | T2 | PARTIAL_CANDIDATE | `platform_typing_scan` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `tools_skills_capabilities` | G | T2 | PARTIAL_CANDIDATE | `governance_boundary` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `tools_skills_capabilities` | H | T2 | PARTIAL_CANDIDATE | `l3_reachability` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `tools_skills_capabilities` | I | T2 | PARTIAL_CANDIDATE | `config_hosting` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `tools_skills_capabilities` | J | T2 | PARTIAL_CANDIDATE | `replaceability_j` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `background_workers_infra_adapters` | A | T2 | PARTIAL_CANDIDATE | `platform_vendor_scan` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `background_workers_infra_adapters` | B | T2 | PARTIAL_CANDIDATE | `platform_hardcoding` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `background_workers_infra_adapters` | C | T2 | PARTIAL_CANDIDATE | `platform_fail_closed` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `background_workers_infra_adapters` | D | T2 | PARTIAL_CANDIDATE | `platform_vendor_scan` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `background_workers_infra_adapters` | E | T2 | PARTIAL_CANDIDATE | `persistence_sqlite` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `background_workers_infra_adapters` | F | T2 | PARTIAL_CANDIDATE | `platform_typing_scan` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `background_workers_infra_adapters` | G | T2 | PARTIAL_CANDIDATE | `governance_boundary` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `background_workers_infra_adapters` | H | T2 | PARTIAL_CANDIDATE | `l3_reachability` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `background_workers_infra_adapters` | I | T2 | PARTIAL_CANDIDATE | `config_hosting` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
| `background_workers_infra_adapters` | J | T2 | PARTIAL_CANDIDATE | `replaceability_j` | `tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py::test_ebh6_aj_class_gate[` | **PENDING** |
