# EBH-7 — Semantic proof-binding audit (SSOT mirror)

**Source SHA (working tree):** e2467a88ed042e2d97150568f9f6b25ba85cb6b1 + completion-wave deltas (uncommitted).

**Authoritative module:** 	ests/qualification/ebh_7/_semantic_proof_binding_audit.py

**Summary:** TEN-E2E 14 EXACT / 12 OPEN; XDM 16 EXACT / 18 OPEN (includes PARTIAL+STRUCTURAL).

| SCENARIO_ID | SEMANTIC_MATCH | EXISTING_PROOF | NOTES |
| --- | --- | --- | --- |
| EBH7-TEN-E2E-01 | EXACT_PROOF | tests.qualification.tenant_x.test_tenant_x_r1_fast::test_tx_r1_q09_adversarial_root_admission_missing_tenant | TENANT-X adversarial @ HEAD (explicit proof node) |
| EBH7-TEN-E2E-02 | EXACT_PROOF | tests.qualification.tenant_x.test_tenant_x_r1_fast::test_tx_r1_q08_adversarial_runtime_request_tenant_mismatch | TENANT-X adversarial @ HEAD (explicit proof node) |
| EBH7-TEN-E2E-03 | PARTIAL_PROOF | tests.qualification.tenant_x.test_tenant_x_r1_fast::test_tx_r1_q08_adversarial_runtime_request_tenant_mismatch | Gov→Exec cross-admission not isolated from R1 runtime mismatch probe |
| EBH7-TEN-E2E-04 | BLOCKED | MISSING | Prior credential-broker binding removed — architecture gap |
| EBH7-TEN-E2E-05 | PARTIAL_PROOF | tests.qualification.tenant_x.test_tenant_x_r2_fast::test_tx_r2_q05_adversarial_cross_tenant_credential_broker | Integration factory boundary approximated via credential broker |
| EBH7-TEN-E2E-06 | EXACT_PROOF | tests.qualification.tenant_x.test_tenant_x_r3_fast::test_tx_r3_q05_adversarial_checkpoint_recovery_cross_tenant | TENANT-X adversarial @ HEAD (explicit proof node) |
| EBH7-TEN-E2E-07 | EXACT_PROOF | tests.qualification.tenant_x.test_tenant_x_r3_fast::test_tx_r3_q07_adversarial_durable_load_cross_tenant | TENANT-X adversarial @ HEAD (explicit proof node) |
| EBH7-TEN-E2E-08 | EXACT_PROOF | tests.qualification.tenant_x.test_tenant_x_r2_fast::test_tx_r2_q07_adversarial_vector_scope_cross_tenant | TENANT-X adversarial @ HEAD (explicit proof node) |
| EBH7-TEN-E2E-09 | EXACT_PROOF | tests.qualification.tenant_x.test_tenant_x_r4_fast::test_tx_r4_q08_adversarial_vendor_async_queue | TENANT-X adversarial @ HEAD (explicit proof node) |
| EBH7-TEN-E2E-10 | EXACT_PROOF | tests.qualification.ebh_7.tenant_platform_e2e.test_ebh7_platform_tenant_boundary_proofs::test_ebh7_ten_e2e_10_distributed_queue_cross_tenant_fail_closed | TENANT-X adversarial @ HEAD (explicit proof node) |
| EBH7-TEN-E2E-11 | EXACT_PROOF | tests.qualification.ebh_7.tenant_platform_e2e.test_ebh7_platform_tenant_boundary_proofs::test_ebh7_ten_e2e_11_trace_attribution_cross_tenant_rejected | TENANT-X adversarial @ HEAD (explicit proof node) |
| EBH7-TEN-E2E-12 | MISSING_PROOF | MISSING | No executable proof node registered |
| EBH7-TEN-E2E-13 | PARTIAL_PROOF | tests.qualification.ebh_7.tenant_platform_e2e.test_ebh7_platform_tenant_boundary_proofs::test_ebh7_ten_e2e_13_plugin_compat_cross_tenant_rejected | Plugin host activation path not directly exercised |
| EBH7-TEN-E2E-14 | MISSING_PROOF | MISSING | No executable proof node registered |
| EBH7-TEN-E2E-15 | EXACT_PROOF | tests.qualification.tenant_x.test_tenant_x_r3_fast::test_tx_r3_q09_retry_resume_preserves_tenant | TENANT-X adversarial @ HEAD (explicit proof node) |
| EBH7-TEN-E2E-16 | EXACT_PROOF | tests.qualification.tenant_x.test_tenant_x_r2_fast::test_tx_r2_q06_adversarial_missing_tenant_credential_scope | TENANT-X adversarial @ HEAD (explicit proof node) |
| EBH7-TEN-E2E-17 | MISSING_PROOF | MISSING | No executable proof node registered |
| EBH7-TEN-E2E-18 | MISSING_PROOF | MISSING | No executable proof node registered |
| EBH7-TEN-E2E-19 | MISSING_PROOF | MISSING | No executable proof node registered |
| EBH7-TEN-E2E-20 | EXACT_PROOF | tests.qualification.ebh_7.tenant_platform_e2e.test_ebh7_platform_tenant_boundary_proofs::test_ebh7_ten_e2e_20_compat_assess_cross_tenant_rejected | Matches COMPAT→Tenant boundary |
| EBH7-TEN-E2E-21 | MISSING_PROOF | MISSING | No executable proof node registered |
| EBH7-TEN-E2E-22 | MISSING_PROOF | MISSING | No executable proof node registered |
| EBH7-TEN-E2E-23 | MISSING_PROOF | MISSING | No executable proof node registered |
| EBH7-TEN-E2E-24 | EXACT_PROOF | tests.qualification.tenant_x.test_tenant_x_r4_fast::test_tx_r4_q05_adversarial_context_llm_cross_tenant | TENANT-X adversarial @ HEAD (explicit proof node) |
| EBH7-TEN-E2E-25 | EXACT_PROOF | tests.qualification.tenant_x.test_tenant_x_r1_fast::test_tx_r1_q10_adversarial_functional_evidence_missing_tenant | TENANT-X adversarial @ HEAD (explicit proof node) |
| EBH7-TEN-E2E-26 | EXACT_PROOF | tests.qualification.tenant_x.test_tenant_x_r4_fast::test_tx_r4_q10_closure_derived_security_metrics | TENANT-X adversarial @ HEAD (explicit proof node) |
| EBH7-XDM-001 | EXACT_PROOF | tests.qualification.tenant_x.test_tenant_x_r1_fast::test_tx_r1_q08_adversarial_runtime_request_tenant_mismatch | tenant_x R1 |
| EBH7-XDM-002 | EXACT_PROOF | tests.qualification.tenant_x.test_tenant_x_r1_fast::test_tx_r1_q08_adversarial_runtime_request_tenant_mismatch | tenant_x R1 |
| EBH7-XDM-003 | EXACT_PROOF | tests.qualification.tenant_x.test_tenant_x_r3_fast::test_tx_r3_q05_adversarial_checkpoint_recovery_cross_tenant | tenant_x R3 |
| EBH7-XDM-004 | EXACT_PROOF | tests.qualification.tenant_x.test_tenant_x_r3_fast::test_tx_r3_q06_adversarial_missing_tenant_no_global_recovery | tenant_x R3 |
| EBH7-XDM-005 | EXACT_PROOF | tests.qualification.tenant_x.test_tenant_x_r4_fast::test_tx_r4_q09_adversarial_functional_evidence_tenant | tenant_x R4 |
| EBH7-XDM-006 | EXACT_PROOF | tests.qualification.tenant_x.test_tenant_x_r4_fast::test_tx_r4_q08_adversarial_vendor_async_queue | tenant_x R4 |
| EBH7-XDM-007 | MISSING_PROOF | MISSING | No executable proof node registered |
| EBH7-XDM-008 | EXACT_PROOF | tests.qualification.ebh_6.test_ebh_6_config_x_head_reconciliation::test_ebh6_config_x_all_concerns_mapped_at_head | ebh_6 config_x |
| EBH7-XDM-009 | EXACT_PROOF | tests.qualification.tenant_x.test_tenant_x_r2_fast::test_tx_r2_q06_adversarial_missing_tenant_credential_scope | tenant_x R2 |
| EBH7-XDM-010 | EXACT_PROOF | tests.qualification.ebh_7.cross_domain_e2e.test_ebh7_cross_domain_platform_proofs::test_ebh7_xdm_010_backend_persistence_sqlite_behavioral | Replaces structural touchpoint inventory-only binding |
| EBH7-XDM-011 | EXACT_PROOF | tests.qualification.tenant_x.test_tenant_x_r3_fast::test_tx_r3_q08_adversarial_partial_recovery_tenant | tenant_x R3 |
| EBH7-XDM-012 | MISSING_PROOF | MISSING | No executable proof node registered |
| EBH7-XDM-013 | EXACT_PROOF | tests.qualification.tenant_x.test_tenant_x_r3_fast::test_tx_r3_q09_retry_resume_preserves_tenant | tenant_x R3 |
| EBH7-XDM-014 | MISSING_PROOF | MISSING | No executable proof node registered |
| EBH7-XDM-015 | MISSING_PROOF | MISSING | No executable proof node registered |
| EBH7-XDM-016 | STRUCTURAL_ONLY | tests.qualification.ebh_6.test_ebh_6_mechanical_exit_gates::test_ebh6_exit_findings_register_complete | Hardcoding register integrity — not VBN factory runtime proof |
| EBH7-XDM-017 | MISSING_PROOF | MISSING | No executable proof node registered |
| EBH7-XDM-018 | PARTIAL_PROOF | tests.qualification.tenant_x.test_tenant_x_r2_fast::test_tx_r2_q05_adversarial_cross_tenant_credential_broker | Plugin×tenant approximated — not plugin host activation path |
| EBH7-XDM-019 | MISSING_PROOF | MISSING | No executable proof node registered |
| EBH7-XDM-020 | EXACT_PROOF | tests.qualification.dup_x.test_dup_x_b4_npsc_orchestration_backend_negative_boundary::test_b4_orchestration_backend_access_import_confined_to_execution_owner_zone | dup_x B4 |
| EBH7-XDM-021 | EXACT_PROOF | tests.qualification.tenant_x.test_tenant_x_r2_fast::test_tx_r2_q07_adversarial_vector_scope_cross_tenant | tenant_x R2 |
| EBH7-XDM-022 | MISSING_PROOF | MISSING | No executable proof node registered |
| EBH7-XDM-023 | MISSING_PROOF | MISSING | No executable proof node registered |
| EBH7-XDM-024 | MISSING_PROOF | MISSING | No executable proof node registered |
| EBH7-XDM-025 | MISSING_PROOF | MISSING | No executable proof node registered |
| EBH7-XDM-026 | MISSING_PROOF | MISSING | No executable proof node registered |
| EBH7-XDM-027 | MISSING_PROOF | MISSING | No executable proof node registered |
| EBH7-XDM-028 | MISSING_PROOF | MISSING | No executable proof node registered |
| EBH7-XDM-029 | MISSING_PROOF | MISSING | No executable proof node registered |
| EBH7-XDM-030 | MISSING_PROOF | MISSING | No executable proof node registered |
| EBH7-XDM-031 | MISSING_PROOF | MISSING | No executable proof node registered |
| EBH7-XDM-032 | EXACT_PROOF | tests.qualification.tenant_x.test_tenant_x_r1_fast::test_tx_r1_q12_adversarial_tool_scope_same_tenant | tenant_x R1 |
| EBH7-XDM-033 | EXACT_PROOF | tests.qualification.dup_x.test_dup_x_mechanical_exit_gates::test_dup_x_section_13_measure_open_bypass | dup_x exit |
| EBH7-XDM-034 | EXACT_PROOF | tests.qualification.dup_x.test_dup_x_b4_npsc_orchestration_backend_negative_boundary::test_b4_materialize_host_orchestration_backend_callers_allowlisted | npsc gate |
