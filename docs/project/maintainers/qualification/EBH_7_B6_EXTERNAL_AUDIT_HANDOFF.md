# EBH-7 — External independent B6 audit handoff

**Agent status:** G01–G09 **PASS** (0 OPEN_FINDING). B6-F10 computed certification (C1–C5 over closed normative sets) bound to the L3 profile and to a post-run JUnit execution attestation. **Not** B6 PASS · **Not** OD-EBH7-08.

## Checkout and replay

```powershell
git fetch origin development
git checkout 496016c0cf6cc5cf1f9a29d885677605dd0415d3
uv sync --frozen --extra dev-ci
uv run pytest tests/qualification/ebh_6/test_ebh_6_hardcoding_register_integrity.py::test_ebh6_prohibited_unresolved_zero tests/qualification/ebh_5/test_ebh_5_behavioral_package_c_v2.py tests/qualification/ebh_5/test_ebh_5_dual_slug_rpl_e_operational_gates.py tests/qualification/ebh_5/test_ebh_5_factory_rpl_one_pass.py tests/qualification/config_x/test_config_x_final_reconciliation_gates.py tests/qualification/dup_x/test_dup_x_closed_world_integrity.py tests/qualification/dup_x/test_dup_x_mechanical_exit_gates.py tests/qualification/dup_x/test_dup_x_parent_closure_mechanical_gates.py tests/qualification/ebh_5/test_ebh_5_l2_integrity_gates.py tests/qualification/ebh_6/test_ebh_6_architecture_family_matrix_integrity.py tests/qualification/ebh_6/test_ebh_6_config_x_head_reconciliation.py tests/unit/runtime/architecture/test_ee_a1_execution_engine_ownership_certification_gate.py tests/unit/runtime/architecture/test_hardening_3_layer_boundary_gate.py tests/unit/runtime/nexus/tools/test_runtime_tool_invoker_cross_tenant.py tests/unit/runtime/nexus/tools/test_runtime_tool_invoker_dependency_admission.py tests/unit/runtime/nexus/tools/test_runtime_tool_invoker_policy.py tests/unit/runtime/nexus/tools/test_tool_runtime_scope_enforcement.py tests/unit/runtime/security/test_tenant_scope.py tests/qualification/control_planes/ctrl_x/test_ctrl_x_r1_security_composition.py::test_r1_sec_01_canonical_composition_rejects_external_fail_open tests/qualification/control_planes/ctrl_x/test_ctrl_x_r2_middleware_composition.py::test_r2_mw_05_fail_open_rejected_before_attach tests/qualification/ebh_5/test_ebh_5_mechanical_exit_gates.py::test_ebh5_exit_inventory_complete tests/qualification/ebh_5/test_ebh_5_rpl_g_regression_gates.py::test_ebh5_rpl_g_config_x_active_blockers_remain_empty tests/qualification/ebh_5/test_ebh_5_rpl_g_regression_gates.py::test_ebh5_rpl_g_tier0_no_agents_import_applications tests/qualification/ebh_6/test_ebh_6_contracts_typing_qualification.py::test_ebh6_application_services_production_seam_typing_passes tests/qualification/ebh_6/test_ebh_6_hardcoding_register_integrity.py::test_ebh6_findings_unclassified_zero tests/qualification/ebh_6/test_ebh_6_hardcoding_register_integrity.py::test_ebh6_layer_coverage_findings_complete tests/qualification/ebh_6/test_ebh_6_l3_and_family_evidence_integrity.py::test_aj_evidence_required_matches_t2_bundle_gaps tests/qualification/ebh_6/test_ebh_6_mechanical_exit_gates.py::test_ebh6_exit_findings_register_complete tests/qualification/ebh_6/test_ebh_6_mechanical_exit_gates.py::test_ebh6_exit_mandatory_docs_present tests/qualification/ebh_6/test_ebh_6_mechanical_exit_gates.py::test_ebh6_exit_sanctioned_composition_roots_non_empty tests/qualification/ebh_6/test_ebh_6_mechanical_exit_gates.py::test_ebh6_exit_sqlite_annex_accountability tests/qualification/ebh_6/test_ebh_6_q2d2_dual_path.py::test_ebh6_q2d2_path_b_remediation_proves_registered_violation_removed tests/qualification/ebh_6/test_ebh_6_scope_reconciliation_integrity.py::test_ebh6_scope_dossier_content_controls tests/unit/architecture/test_ebh_2i_final_rescan_gate.py::test_ebh_2i_contract_surface_purity_gate_passes tests/unit/runtime/architecture/test_ee_b1_2_capacity_architecture_gate.py::test_ee_b1_2_capacity_contract_no_identity_mint tests/unit/runtime/architecture/test_ee_final_arch_owner_uniqueness.py::test_ee_final_arch_owner_matrix_documented tests/unit/runtime/architecture/test_npsc42_h1_governance_boundary_freeze.py::test_npsc42_h1_execution_does_not_host_policy_engine tests/unit/runtime/architecture/test_npsc42_h1_governance_boundary_freeze.py::test_npsc42_h1_governance_planes_do_not_mint_execution_identity tests/unit/runtime/architecture/test_npsc5e_final_recovery_plane_qualification_and_freeze.py::test_no_recovery_god_object_in_production tests/unit/runtime/architecture/test_w3_c4_decision_durable_recovery_canonical_wiring.py::test_decision_durable_recovery_has_no_production_bypass tests/unit/runtime/events/test_enterprise_scale_resilience_w5_b2_composition_wiring.py::test_production_composition_creates_bounded_sink tests/unit/runtime/events/test_enterprise_scale_resilience_w5_b2_composition_wiring.py::test_published_event_reaches_bounded_sink tests/unit/runtime/nexus/test_authoritative_decision_exposure_propagation.py::test_b13_ambiguous_candidates_fail_closed tests/qualification/ebh_7/ -q --tb=line
```

After evidence publication @ HEAD, verify `git merge-base --is-ancestor 496016c0cf6cc5cf1f9a29d885677605dd0415d3 HEAD`.

## GitHub Actions — independent L3 replay (B5)

Profile **`ebh7_p3_l3`** on workflow **`Regression gate`** (`unit-tests.yml`):

```powershell
gh workflow run unit-tests.yml --ref development -f profile=ebh7_p3_l3
```

- Default replay checkout: dispatch **`github.sha`** (B5 workflow @ `3c745c2c0…`; no pre-pytest digest reconcile).
- **`TEST_EXECUTION_SHA`**: `496016c0c…` (B5 historical `4eb7eb1a…`) · executable identity = full-tree diff + allowlist (not `tests/qualification/` only).
- **`EVIDENCE_PUBLICATION_SHA`**: MANIFEST pin (B5 historical `96069c09c…`).
- **F10 execution attestation:** workflow step `B6-F10 execution attestation` (`scripts/qualification/attest_ebh7_f10_execution.py`) fails the run when any PROFILE_BOUND F10 node is missing from or failed in JUnit.
- Prior run **`41212701986`** superseded (reconcile-before-verify) — see [`EBH_7_B5_INTEGRITY_RECORD.md`](EBH_7_B5_INTEGRITY_RECORD.md).
- **ECAP F10:** [`EBH_7_B6_F10_ECAP_NORMATIVE_MATRIX.md`](EBH_7_B6_F10_ECAP_NORMATIVE_MATRIX.md) · §14 report · F10/F11 gates @ replay HEAD.
- Record: [`EBH_7_P03_GITHUB_ACTIONS_REPLAY_RECORD.md`](ebh_7_p03_independent_replay/EBH_7_P03_GITHUB_ACTIONS_REPLAY_RECORD.md).

Push-triggered **`ci-smoke`** on `development` is **not** L3; recent **cancelled** runs (~10m) are smoke **timeout**, not L3 failure.

## SHA roles (Protocol 03)

| Role | SHA |
| --- | --- |
| TEST_EXECUTION_SHA | `496016c0cf6cc5cf1f9a29d885677605dd0415d3` |
| EBH7_B5_TEST_EXECUTION_SHA (historical) | `4eb7eb1a89f82215d5356aa7d000198e432777b0` |
| EBH7_B6_CORRECTIVE_IMPLEMENTATION_SHA (ancestor) | `8afff322cb2525f0106c2bbccf2d44f682d3e0df` |
| EBH7_CLOSURE_IMPLEMENTATION_SHA (lineage) | `d4937889fc0b07515649dbdddf602df2a4415489` |
| EVIDENCE_PUBLICATION_SHA | MANIFEST `EVIDENCE_PUBLICATION_SHA` (profile authority) |
| EBH7_B5_EVIDENCE_PUBLICATION_SHA (historical) | `96069c09c2ca5a7a3a9586db336716c9430ca1d7` |

## Primary artifacts

| Artifact | Purpose |
| --- | --- |
| [`EBH_7_PRE_CERT_INDEPENDENT_AUDIT_WORKSHEET.md`](EBH_7_PRE_CERT_INDEPENDENT_AUDIT_WORKSHEET.md) | Protocol 03 B2 checklist |
| [`EBH_7_CLOSURE_EVIDENCE_AUDIT_MATRIX.md`](ebh_7_p03_independent_replay/EBH_7_CLOSURE_EVIDENCE_AUDIT_MATRIX.md) | G01–G09 PASS / N/A / OPEN (SSOT) |
| [`B6_EVIDENCE_CONSISTENCY_MATRIX.md`](ebh_7_p03_independent_replay/B6_EVIDENCE_CONSISTENCY_MATRIX.md) | Counts + non-claims |
| [`EBH_7_INDEPENDENT_B6_AUDIT_DOSSIER.md`](EBH_7_INDEPENDENT_B6_AUDIT_DOSSIER.md) | Dossier scope |
| [`ebh_7_p03_independent_replay/ARTIFACT_DIGESTS.json`](ebh_7_p03_independent_replay/ARTIFACT_DIGESTS.json) | SHA256 ladder |

## Executable gates (sample)

```powershell
uv run pytest tests/qualification/ebh_7/test_ebh_7_closure_evidence_audit_gates.py tests/qualification/ebh_7/test_ebh_7_certification_ssot_gates.py -q
```

## B6 outcome record

- **PASS:** create [`EBH_7_PROTOCOL_03_INDEPENDENT_AUDIT_PASS_RECORD.md`](EBH_7_PROTOCOL_03_INDEPENDENT_AUDIT_PASS_RECORD.md); set `INDEPENDENT_ACCEPTANCE_BASELINE_SHA` in MANIFEST.
- **FAIL:** register findings in `tests/qualification/ebh_7/_gate_findings_register.py` — do **not** agent-self-issue PASS.

## OD-EBH7-08

Operator-only after B6 PASS: [`EBH_7_OD_EBH7_08_CLOSURE_PROCEDURE.md`](EBH_7_OD_EBH7_08_CLOSURE_PROCEDURE.md).
