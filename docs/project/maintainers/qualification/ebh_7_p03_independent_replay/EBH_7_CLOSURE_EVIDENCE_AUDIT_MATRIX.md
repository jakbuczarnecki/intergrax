# EBH-7 — Closure evidence audit matrix (G01–G09)

**TEST_EXECUTION_SHA:** `83136ef7be70ff33d0a81b978c154ac8528930a2`

**OPEN_FINDING count:** 0 (must be 0 before independent B6)

**Authority:** `tests/qualification/ebh_7/_ebh7_closure_evidence_audit.py`

| G | Subject | Status | Evidence | Notes |
| --- | --- | --- | --- | --- |
| SUM | RPL-H 56/56 QUALIFIED | PASS | {<RplSemanticVerdict.QUALIFIED: 'QUALIFIED'>: 56, <RplSemanticVerdict.PARTIAL: 'PARTIAL'>: 0, <Rp... | 0 B6-open |
| SUM | TEN 26 EXACT | PASS | 26/0 | semantic_exact_counts |
| SUM | XDM 34 EXACT | PASS | 34/0 | semantic_exact_counts |
| SUM | RPL-F registry | PASS | errors=0 | ok |
| G01 | integration.relational_store | N/A_WITH_EVIDENCE | RPL-E=RPL-C: IntegrationProfile selects distinct registered catalog providers @ resolve_from_prof... | N/A — WITH EVIDENCE: relational_store active operational failure @ persistenc... |
| G01 | integration.document_store | N/A_WITH_EVIDENCE | RPL-E=RPL-C: IntegrationProfile selects distinct registered catalog providers @ resolve_from_prof... | N/A — WITH EVIDENCE: document_store — profile selection + unregistered fail-c... |
| G01 | integration.message_bus | N/A_WITH_EVIDENCE | RPL-E=RPL-C: IntegrationProfile selects distinct registered catalog providers @ resolve_from_prof... | N/A — WITH EVIDENCE: message_bus — profile selection + unregistered fail-clos... |
| G01 | integration.object_storage | N/A_WITH_EVIDENCE | RPL-E=RPL-C: IntegrationProfile selects distinct registered catalog providers @ resolve_from_prof... | N/A — WITH EVIDENCE: object_storage — profile selection + unregistered fail-c... |
| G01 | integration.graph_store | N/A_WITH_EVIDENCE | RPL-E=RPL-C: IntegrationProfile selects distinct registered catalog providers @ resolve_from_prof... | N/A — WITH EVIDENCE: graph_store — profile selection + unregistered fail-clos... |
| G01 | integration.notification_channel | N/A_WITH_EVIDENCE | RPL-E=RPL-C: IntegrationProfile selects distinct registered catalog providers @ resolve_from_prof... | N/A — WITH EVIDENCE: notification_channel — profile selection + unregistered ... |
| G01 | integration.collaboration_suite | N/A_WITH_EVIDENCE | RPL-E=RPL-C: IntegrationProfile selects distinct registered catalog providers @ resolve_from_prof... | N/A — WITH EVIDENCE: collaboration_suite — profile selection + unregistered f... |
| G01 | integration.issue_tracker | N/A_WITH_EVIDENCE | RPL-E=RPL-C: IntegrationProfile selects distinct registered catalog providers @ resolve_from_prof... | N/A — WITH EVIDENCE: issue_tracker — profile selection + unregistered fail-cl... |
| G01 | integration.wiki_knowledge | N/A_WITH_EVIDENCE | RPL-E=RPL-C: IntegrationProfile selects distinct registered catalog providers @ resolve_from_prof... | N/A — WITH EVIDENCE: wiki_knowledge — profile selection + unregistered fail-c... |
| G01 | integration.browser_automation | N/A_WITH_EVIDENCE | RPL-E=RPL-C: IntegrationProfile selects distinct registered catalog providers @ resolve_from_prof... | N/A — WITH EVIDENCE: browser_automation — profile selection + unregistered fa... |
| G01 | integration.cloud_platform | N/A_WITH_EVIDENCE | RPL-E=RPL-C: IntegrationProfile selects distinct registered catalog providers @ resolve_from_prof... | N/A — WITH EVIDENCE: cloud_platform — profile selection + unregistered fail-c... |
| G01 | integration.secrets_store | N/A_WITH_EVIDENCE | RPL-E=RPL-C: IntegrationProfile selects distinct registered catalog providers @ resolve_from_prof... | N/A — WITH EVIDENCE: secrets_store — profile selection + unregistered fail-cl... |
| G01 | integration.sandbox_host | N/A_WITH_EVIDENCE | RPL-E=RPL-C: IntegrationProfile selects distinct registered catalog providers @ resolve_from_prof... | N/A — WITH EVIDENCE: sandbox_host — profile selection + unregistered fail-clo... |
| G01 | integration.speech_provider | N/A_WITH_EVIDENCE | RPL-E=RPL-C: IntegrationProfile selects distinct registered catalog providers @ resolve_from_prof... | N/A — WITH EVIDENCE: speech_provider — profile selection + unregistered fail-... |
| G01 | integration.embedding_provider | N/A_WITH_EVIDENCE | RPL-E=RPL-C: IntegrationProfile selects distinct registered catalog providers @ resolve_from_prof... | N/A — WITH EVIDENCE: embedding_provider — profile selection + unregistered fa... |
| G01 | integration.vector_store | N/A_WITH_EVIDENCE | RPL-E=RPL-C: IntegrationProfile selects distinct registered catalog providers @ resolve_from_prof... | N/A — WITH EVIDENCE: vector_store — profile selection + unregistered fail-clo... |
| G01 | integration.conversation_channel | N/A_WITH_EVIDENCE | RPL-E=RPL-C: IntegrationProfile selects distinct registered catalog providers @ resolve_from_prof... | N/A — WITH EVIDENCE: conversation_channel — profile selection + unregistered ... |
| G01 | integration.workflow_orchestrator | N/A_WITH_EVIDENCE | RPL-E=RPL-C: IntegrationProfile selects distinct registered catalog providers @ resolve_from_prof... | N/A — WITH EVIDENCE: workflow_orchestrator — profile selection + unregistered... |
| G01 | integration.ci_cd | N/A_WITH_EVIDENCE | RPL-E=RPL-C: IntegrationProfile selects distinct registered catalog providers @ resolve_from_prof... | N/A — WITH EVIDENCE: ci_cd — profile selection + unregistered fail-closed @ P... |
| G01 | integration.feature_flag | N/A_WITH_EVIDENCE | RPL-E=RPL-C: IntegrationProfile selects distinct registered catalog providers @ resolve_from_prof... | N/A — WITH EVIDENCE: feature_flag — profile selection + unregistered fail-clo... |
| G01 | integration.crm | N/A_WITH_EVIDENCE | RPL-E=RPL-C: IntegrationProfile selects distinct registered catalog providers @ resolve_from_prof... | N/A — WITH EVIDENCE: crm — profile selection + unregistered fail-closed @ Pac... |
| G01 | integration.observability_backend | N/A_WITH_EVIDENCE | RPL-E=RPL-C: IntegrationProfile selects distinct registered catalog providers @ resolve_from_prof... | N/A — WITH EVIDENCE: observability_backend — profile selection + unregistered... |
| G01 | integration.security_scanner | N/A_WITH_EVIDENCE | RPL-E=RPL-C: IntegrationProfile selects distinct registered catalog providers @ resolve_from_prof... | N/A — WITH EVIDENCE: security_scanner — profile selection + unregistered fail... |
| G01 | integration.identity_provider | N/A_WITH_EVIDENCE | RPL-E=RPL-C: IntegrationProfile selects distinct registered catalog providers @ resolve_from_prof... | N/A — WITH EVIDENCE: identity_provider — profile selection + unregistered fai... |
| G01 | integration.document_parser | N/A_WITH_EVIDENCE | RPL-E=RPL-C: IntegrationProfile selects distinct registered catalog providers @ resolve_from_prof... | N/A — WITH EVIDENCE: document_parser — profile selection + unregistered fail-... |
| G01 | integration.rerank_provider | N/A_WITH_EVIDENCE | RPL-E=RPL-C: IntegrationProfile selects distinct registered catalog providers @ resolve_from_prof... | N/A — WITH EVIDENCE: rerank_provider — profile selection + unregistered fail-... |
| G01 | integration.search_provider | N/A_WITH_EVIDENCE | RPL-E=RPL-C: IntegrationProfile selects distinct registered catalog providers @ resolve_from_prof... | N/A — WITH EVIDENCE: search_provider — profile selection + unregistered fail-... |
| G01 | integration.llm_guardrail | N/A_WITH_EVIDENCE | RPL-E=RPL-C: IntegrationProfile selects distinct registered catalog providers @ resolve_from_prof... | N/A — WITH EVIDENCE: llm_guardrail — profile selection + unregistered fail-cl... |
| G02 | integration.key_value_cache | PASS | pos=tests/qualification/ebh_5/test_ebh_5_behavioral_proofs.py::test_ebh5_behavioral_kv_external_p... | EBH7-W2-RPL-LEGACY |
| G02 | integration.external_work | PASS | pos=tests/qualification/ebh_5/test_ebh_5_factory_rpl_one_pass.py::test_ebh5_factory_rpl_positive[... | EBH7-ONE-PASS-FACTORY |
| G02 | llm.chat_completion | PASS | pos=tests/qualification/ebh_5/test_ebh_5_factory_rpl_one_pass.py::test_ebh5_factory_rpl_positive[... | EBH7-ONE-PASS-FACTORY |
| G02 | llm.embedding | PASS | pos=tests/qualification/ebh_5/test_ebh_5_factory_rpl_one_pass.py::test_ebh5_factory_rpl_positive[... | EBH7-ONE-PASS-FACTORY |
| G02 | llm.rerank | PASS | pos=tests/qualification/ebh_5/test_ebh_5_factory_rpl_one_pass.py::test_ebh5_factory_rpl_positive[... | EBH7-ONE-PASS-FACTORY |
| G02 | llm.fallback_policy | PASS | pos=tests/qualification/ebh_5/test_ebh_5_factory_rpl_one_pass.py::test_ebh5_factory_rpl_positive[... | EBH7-ONE-PASS-FACTORY |
| G02 | llm.model_routing_qualifier | PASS | pos=tests/qualification/ebh_5/test_ebh_5_factory_rpl_one_pass.py::test_ebh5_factory_rpl_positive[... | EBH7-ONE-PASS-FACTORY |
| G02 | persistence.relational_sqlite_lab | PASS | pos=tests/qualification/ebh_5/test_ebh_5_behavioral_proofs.py::test_ebh5_behavioral_relational_st... | EBH7-W2-RPL-LEGACY |
| G02 | int_config.configure_existing | PASS | pos=tests/qualification/ebh_5/test_ebh_5_behavioral_proofs.py::test_ebh5_behavioral_int_config_sq... | EBH7-W2-RPL-LEGACY |
| G02 | execution.integration_bound_resolution | PASS | pos=tests/qualification/ebh_5/test_ebh_5_factory_rpl_one_pass.py::test_ebh5_factory_rpl_positive[... | EBH7-ONE-PASS-FACTORY |
| G02 | observability.otlp_transport | PASS | pos=tests/qualification/ebh_5/test_ebh_5_behavioral_proofs.py::test_ebh5_behavioral_otlp_transpor... | EBH7-W2-RPL-LEGACY |
| G02 | plugin.integration_catalog | PASS | pos=tests/qualification/ebh_5/test_ebh_5_factory_rpl_one_pass.py::test_ebh5_factory_rpl_positive[... | EBH7-ONE-PASS-FACTORY |
| G02 | plugin.llm_adapter_registry | PASS | pos=tests/qualification/ebh_5/test_ebh_5_factory_rpl_one_pass.py::test_ebh5_factory_rpl_positive[... | EBH7-ONE-PASS-FACTORY |
| G02 | tool.marketplace_activation | PASS | pos=tests/qualification/ebh_5/test_ebh_5_factory_rpl_one_pass.py::test_ebh5_factory_rpl_positive[... | EBH7-ONE-PASS-FACTORY |
| G02 | agent.effective_profile | PASS | pos=tests/qualification/ebh_5/test_ebh_5_factory_rpl_one_pass.py::test_ebh5_factory_rpl_positive[... | EBH7-ONE-PASS-FACTORY |
| G02 | speech.provider | PASS | pos=tests/qualification/ebh_5/test_ebh_5_factory_rpl_one_pass.py::test_ebh5_factory_rpl_positive[... | EBH7-ONE-PASS-FACTORY |
| G02 | env.integration_slug | PASS | pos=tests/qualification/ebh_5/test_ebh_5_factory_rpl_one_pass.py::test_ebh5_factory_rpl_positive[... | EBH7-ONE-PASS-FACTORY |
| G02 | tokenizer.selection | PASS | pos=tests/qualification/ebh_5/test_ebh_5_behavioral_proofs.py::test_ebh5_behavioral_tokenizer_sel... | EBH7-W2-RPL-LEGACY |
| G02 | tool.observability_backend | PASS | pos=tests/qualification/ebh_5/test_ebh_5_factory_rpl_one_pass.py::test_ebh5_factory_rpl_positive[... | EBH7-ONE-PASS-FACTORY |
| G02 | distributed.sqlite_kv | PASS | pos=tests/qualification/ebh_5/test_ebh_5_factory_rpl_one_pass.py::test_ebh5_factory_rpl_positive[... | EBH7-ONE-PASS-FACTORY |
| G03 | application.composition_root | PASS | pos=tests/qualification/ebh_7/test_ebh_7_terminal_rpl_gates.py::test_ebh7_internal_rpl_positive[a... | INTERNAL_RATIFIED |
| G03 | integration.billing_meter | PASS | pos=tests/qualification/ebh_7/test_ebh_7_terminal_rpl_gates.py::test_ebh7_fixed_substrate_rpl_pos... | FIXED_SUBSTRATE_RATIFIED |
| G03 | integration.managed_retrieval | PASS | pos=tests/qualification/ebh_7/test_ebh_7_terminal_rpl_gates.py::test_ebh7_fixed_substrate_rpl_pos... | FIXED_SUBSTRATE_RATIFIED |
| G03 | integration.ml_inference_host | PASS | pos=tests/qualification/ebh_7/test_ebh_7_terminal_rpl_gates.py::test_ebh7_fixed_substrate_rpl_pos... | FIXED_SUBSTRATE_RATIFIED |
| G03 | integration.model_serving_runtime | PASS | pos=tests/qualification/ebh_7/test_ebh_7_terminal_rpl_gates.py::test_ebh7_fixed_substrate_rpl_pos... | FIXED_SUBSTRATE_RATIFIED |
| G03 | integration.vision_serving | PASS | pos=tests/qualification/ebh_7/test_ebh_7_terminal_rpl_gates.py::test_ebh7_fixed_substrate_rpl_pos... | FIXED_SUBSTRATE_RATIFIED |
| G03 | memory.sqlite.conversational | PASS | pos=tests/qualification/ebh_7/test_ebh_7_terminal_rpl_gates.py::test_ebh7_fixed_substrate_rpl_pos... | FIXED_SUBSTRATE_RATIFIED |
| G03 | prodq.colocated_runtime_adapter_isinstance | PASS | pos=tests/qualification/ebh_7/test_ebh_7_terminal_rpl_gates.py::test_ebh7_internal_rpl_positive[p... | INTERNAL_RATIFIED |
| G04 | EBH7-TEN-E2E-01 | PASS | tests.qualification.tenant_x.test_tenant_x_r1_fast::test_tx_r1_q09_adversarial_root_admission_mis... | FAIL-CLOSED |
| G04 | EBH7-TEN-E2E-02 | PASS | tests.qualification.tenant_x.test_tenant_x_r1_fast::test_tx_r1_q08_adversarial_runtime_request_te... | DENY |
| G04 | EBH7-TEN-E2E-03 | PASS | tests.qualification.ebh_7.tenant_platform_e2e.test_ebh7_platform_tenant_boundary_proofs::test_ebh... | DENY |
| G04 | EBH7-TEN-E2E-04 | PASS | tests.qualification.ebh_7.tenant_platform_e2e.test_ebh7_platform_tenant_boundary_proofs::test_ebh... | DENY |
| G04 | EBH7-TEN-E2E-05 | PASS | tests.qualification.ebh_7.tenant_platform_e2e.test_ebh7_platform_tenant_boundary_proofs::test_ebh... | REJECT |
| G04 | EBH7-TEN-E2E-06 | PASS | tests.qualification.tenant_x.test_tenant_x_r3_fast::test_tx_r3_q05_adversarial_checkpoint_recover... | REJECT |
| G04 | EBH7-TEN-E2E-07 | PASS | tests.qualification.tenant_x.test_tenant_x_r3_fast::test_tx_r3_q07_adversarial_durable_load_cross... | DENY |
| G04 | EBH7-TEN-E2E-08 | PASS | tests.qualification.tenant_x.test_tenant_x_r2_fast::test_tx_r2_q07_adversarial_vector_scope_cross... | DENY |
| G04 | EBH7-TEN-E2E-09 | PASS | tests.qualification.tenant_x.test_tenant_x_r4_fast::test_tx_r4_q08_adversarial_vendor_async_queue | DENY |
| G04 | EBH7-TEN-E2E-10 | PASS | tests.qualification.ebh_7.tenant_platform_e2e.test_ebh7_platform_tenant_boundary_proofs::test_ebh... | DENY |
| G04 | EBH7-TEN-E2E-11 | PASS | tests.qualification.ebh_7.tenant_platform_e2e.test_ebh7_platform_tenant_boundary_proofs::test_ebh... | REJECT |
| G04 | EBH7-TEN-E2E-12 | PASS | tests.qualification.ebh_7.tenant_platform_e2e.test_ebh7_platform_tenant_boundary_proofs::test_ebh... | DENY |
| G04 | EBH7-TEN-E2E-13 | PASS | tests.qualification.ebh_7.tenant_platform_e2e.test_ebh7_platform_tenant_boundary_proofs::test_ebh... | DENY |
| G04 | EBH7-TEN-E2E-14 | PASS | tests.qualification.ebh_7.tenant_platform_e2e.test_ebh7_platform_tenant_boundary_proofs::test_ebh... | PRESERVE |
| G04 | EBH7-TEN-E2E-15 | PASS | tests.qualification.tenant_x.test_tenant_x_r3_fast::test_tx_r3_q09_retry_resume_preserves_tenant | DENY |
| G04 | EBH7-TEN-E2E-16 | PASS | tests.qualification.tenant_x.test_tenant_x_r2_fast::test_tx_r2_q06_adversarial_missing_tenant_cre... | REJECT |
| G04 | EBH7-TEN-E2E-17 | PASS | tests.qualification.ebh_7.tenant_platform_e2e.test_ebh7_platform_tenant_boundary_proofs::test_ebh... | ISOLATE |
| G04 | EBH7-TEN-E2E-18 | PASS | tests.qualification.ebh_7.tenant_platform_e2e.test_ebh7_platform_tenant_boundary_proofs::test_ebh... | DENY |
| G04 | EBH7-TEN-E2E-19 | PASS | tests.qualification.ebh_7.tenant_platform_e2e.test_ebh7_platform_tenant_boundary_proofs::test_ebh... | DENY |
| G04 | EBH7-TEN-E2E-20 | PASS | tests.qualification.ebh_7.tenant_platform_e2e.test_ebh7_platform_tenant_boundary_proofs::test_ebh... | REJECT |
| G04 | EBH7-TEN-E2E-21 | PASS | tests.qualification.ebh_7.tenant_platform_e2e.test_ebh7_platform_tenant_boundary_proofs::test_ebh... | DENY |
| G04 | EBH7-TEN-E2E-22 | PASS | tests.qualification.ebh_7.tenant_platform_e2e.test_ebh7_platform_tenant_boundary_proofs::test_ebh... | DENY |
| G04 | EBH7-TEN-E2E-23 | PASS | tests.qualification.ebh_7.tenant_platform_e2e.test_ebh7_platform_tenant_boundary_proofs::test_ebh... | DENY |
| G04 | EBH7-TEN-E2E-24 | PASS | tests.qualification.tenant_x.test_tenant_x_r4_fast::test_tx_r4_q05_adversarial_context_llm_cross_... | DENY |
| G04 | EBH7-TEN-E2E-25 | PASS | tests.qualification.tenant_x.test_tenant_x_r1_fast::test_tx_r1_q10_adversarial_functional_evidenc... | DENY |
| G04 | EBH7-TEN-E2E-26 | PASS | tests.qualification.tenant_x.test_tenant_x_r4_fast::test_tx_r4_q10_closure_derived_security_metrics | DENY |
| G05 | EBH7-XDM-001 | PASS | tests.qualification.tenant_x.test_tenant_x_r1_fast::test_tx_r1_q08_adversarial_runtime_request_te... | DENY |
| G05 | EBH7-XDM-002 | PASS | tests.qualification.tenant_x.test_tenant_x_r1_fast::test_tx_r1_q08_adversarial_runtime_request_te... | DENY |
| G05 | EBH7-XDM-003 | PASS | tests.qualification.tenant_x.test_tenant_x_r3_fast::test_tx_r3_q05_adversarial_checkpoint_recover... | REJECT |
| G05 | EBH7-XDM-004 | PASS | tests.qualification.tenant_x.test_tenant_x_r3_fast::test_tx_r3_q06_adversarial_missing_tenant_no_... | DENY |
| G05 | EBH7-XDM-005 | PASS | tests.qualification.tenant_x.test_tenant_x_r4_fast::test_tx_r4_q09_adversarial_functional_evidenc... | REJECT |
| G05 | EBH7-XDM-006 | PASS | tests.qualification.tenant_x.test_tenant_x_r4_fast::test_tx_r4_q08_adversarial_vendor_async_queue | DENY |
| G05 | EBH7-XDM-007 | PASS | tests.qualification.ebh_7.cross_domain_e2e.test_ebh7_cross_domain_platform_proofs::test_ebh7_xdm_... | DENY |
| G05 | EBH7-XDM-008 | PASS | tests.qualification.ebh_6.test_ebh_6_config_x_head_reconciliation::test_ebh6_config_x_all_concern... | REJECT |
| G05 | EBH7-XDM-009 | PASS | tests.qualification.tenant_x.test_tenant_x_r2_fast::test_tx_r2_q06_adversarial_missing_tenant_cre... | DENY |
| G05 | EBH7-XDM-010 | PASS | tests.qualification.ebh_7.cross_domain_e2e.test_ebh7_cross_domain_platform_proofs::test_ebh7_xdm_... | REJECT |
| G05 | EBH7-XDM-011 | PASS | tests.qualification.tenant_x.test_tenant_x_r3_fast::test_tx_r3_q08_adversarial_partial_recovery_t... | REJECT |
| G05 | EBH7-XDM-012 | PASS | tests.qualification.ebh_7.cross_domain_e2e.test_ebh7_cross_domain_platform_proofs::test_ebh7_xdm_... | FAIL |
| G05 | EBH7-XDM-013 | PASS | tests.qualification.tenant_x.test_tenant_x_r3_fast::test_tx_r3_q09_retry_resume_preserves_tenant | REJECT |
| G05 | EBH7-XDM-014 | PASS | tests.qualification.ebh_7.cross_domain_e2e.test_ebh7_cross_domain_platform_proofs::test_ebh7_xdm_... | REJECT |
| G05 | EBH7-XDM-015 | PASS | tests.qualification.ebh_7.cross_domain_e2e.test_ebh7_cross_domain_platform_proofs::test_ebh7_xdm_... | REJECT |
| G05 | EBH7-XDM-016 | PASS | tests.qualification.ebh_7.cross_domain_e2e.test_ebh7_cross_domain_platform_proofs::test_ebh7_xdm_... | REJECT |
| G05 | EBH7-XDM-017 | PASS | tests.qualification.ebh_7.cross_domain_e2e.test_ebh7_cross_domain_platform_proofs::test_ebh7_xdm_... | REJECT |
| G05 | EBH7-XDM-018 | PASS | tests.qualification.ebh_7.cross_domain_e2e.test_ebh7_cross_domain_platform_proofs::test_ebh7_xdm_... | DENY |
| G05 | EBH7-XDM-019 | PASS | tests.qualification.ebh_7.cross_domain_e2e.test_ebh7_cross_domain_platform_proofs::test_ebh7_xdm_... | DENY |
| G05 | EBH7-XDM-020 | PASS | tests.qualification.dup_x.test_dup_x_b4_npsc_orchestration_backend_negative_boundary::test_b4_orc... | DENY |
| G05 | EBH7-XDM-021 | PASS | tests.qualification.tenant_x.test_tenant_x_r2_fast::test_tx_r2_q07_adversarial_vector_scope_cross... | DENY |
| G05 | EBH7-XDM-022 | PASS | tests.qualification.ebh_7.cross_domain_e2e.test_ebh7_cross_domain_platform_proofs::test_ebh7_xdm_... | DENY |
| G05 | EBH7-XDM-023 | PASS | tests.qualification.ebh_7.cross_domain_e2e.test_ebh7_cross_domain_platform_proofs::test_ebh7_xdm_... | DENY |
| G05 | EBH7-XDM-024 | PASS | tests.qualification.ebh_7.cross_domain_e2e.test_ebh7_cross_domain_platform_proofs::test_ebh7_xdm_... | DENY |
| G05 | EBH7-XDM-025 | PASS | tests.qualification.ebh_7.cross_domain_e2e.test_ebh7_cross_domain_platform_proofs::test_ebh7_xdm_... | DENY |
| G05 | EBH7-XDM-026 | PASS | tests.qualification.ebh_7.cross_domain_e2e.test_ebh7_cross_domain_platform_proofs::test_ebh7_xdm_... | FAIL-CLOSED |
| G05 | EBH7-XDM-027 | PASS | tests.qualification.ebh_7.cross_domain_e2e.test_ebh7_cross_domain_platform_proofs::test_ebh7_xdm_... | DENY |
| G05 | EBH7-XDM-028 | PASS | tests.qualification.ebh_7.cross_domain_e2e.test_ebh7_cross_domain_platform_proofs::test_ebh7_xdm_... | DENY |
| G05 | EBH7-XDM-029 | PASS | tests.qualification.ebh_7.cross_domain_e2e.test_ebh7_cross_domain_platform_proofs::test_ebh7_xdm_... | DENY |
| G05 | EBH7-XDM-030 | PASS | tests.qualification.ebh_7.cross_domain_e2e.test_ebh7_cross_domain_platform_proofs::test_ebh7_xdm_... | ISOLATE |
| G05 | EBH7-XDM-031 | PASS | tests.qualification.ebh_7.cross_domain_e2e.test_ebh7_cross_domain_platform_proofs::test_ebh7_xdm_... | BOUNDED |
| G05 | EBH7-XDM-032 | PASS | tests.qualification.tenant_x.test_tenant_x_r1_fast::test_tx_r1_q12_adversarial_tool_scope_same_te... | DENY |
| G05 | EBH7-XDM-033 | PASS | tests.qualification.dup_x.test_dup_x_mechanical_exit_gates::test_dup_x_section_13_measure_open_by... | REJECT |
| G05 | EBH7-XDM-034 | PASS | tests.qualification.dup_x.test_dup_x_b4_npsc_orchestration_backend_negative_boundary::test_b4_mat... | REJECT |
| G06 | INT2E bundle regression | N/A_WITH_EVIDENCE | import_smoke_modules=33; tests/qualification/ebh_7/test_ebh_7_int2e_bundle_regression.py | Not full platform E2E — changed-bundle import + relational/message_bus resolv... |
| G07 | ECAP C1–C5 + PROHIBITED | PASS | assert_ebh7_rpl_certification_ssot + assert_ebh7_ten_xdm_evidence_ssot; count_prohibited_unresolv... | ECAP C1–C5 derived @ HEAD — agent readiness only (not independent B6 acceptance) |
| G08 | Protocol 03 replay + digests | PASS | assert_ebh7_p03_* integrity bundle (excludes closure audit — replay vs acceptance separation) | Independent auditor must checkout TEST_EXECUTION_SHA and replay MANIFEST pyte... |
| G09 | W6-ID-REGISTRY-ONLY | PASS | tests/qualification/ebh_7/test_ebh_7_semantic_proof_binding_gates.py::test_ebh7_semantic_exact_bl... | CORRECTIVE_PYTEST: TEN/XDM EXACT counts + disposition registry — not scenario... |
| G09 | W6-TENANT-X-FINAL-READY | PASS | docs/project/maintainers/qualification/EBH_7_INDEPENDENT_AUDIT_CORRECTIVE_FINDINGS.md@TENANT-X | HISTORICAL_SUPERSEDED: Removed as platform E2E substitute; mapped proofs only... |
| G09 | W7-HASATTR-DUP-X | PASS | tests/qualification/ebh_7/test_ebh_7_semantic_proof_binding_gates.py::test_ebh7_b6_readiness_mand... | CORRECTIVE_PYTEST: Executable proof bindings — not hasattr(dup_exit) |
| G09 | W5-RPL-ROW-COUNT | PASS | tests.qualification.ebh_7._rpl_h_semantic_evidence:rpl_semantic_verdict_counts | CORRECTIVE_ASSERT: Agent vs normative semantic counts separated from row inve... |
| G09 | W5-VBN-DOC-PRESENCE | PASS | tests.qualification.ebh_7._traceability_matrix:assert_ebh7_traceability_integrity | CORRECTIVE_ASSERT: Traceability matrix — materialization vs qualification col... |
| G09 | TRACE-466-COUNT-ONLY | PASS | tests.qualification.ebh_7._traceability_matrix:assert_ebh7_traceability_behavioral_integrity | CORRECTIVE_ASSERT: Extended rows with owner, evidence, acceptance fields |
| G09 | READINESS-TABLE-PASS | PASS | docs/project/maintainers/qualification/EBH_7_INDEPENDENT_AUDIT_CORRECTIVE_FINDINGS.md@readiness-r... | HISTORICAL_SUPERSEDED: Readiness reconciled to materialization vs qualificati... |
| G09 | GAP-TEN-UNK-05-CLOSED | PASS | tests/qualification/ebh_7/test_ebh_7_semantic_proof_binding_gates.py::test_ebh7_semantic_exact_bl... | CORRECTIVE_PYTEST: Structural N/A evidence only — not platform E2E proof |
| G09 | AUD-01-RPL-AUTO-QUALIFIED | PASS | tests.qualification.ebh_7._rpl_h_normative_evidence:assert_ebh7_rpl_h_normative_evidence_integrity | CORRECTIVE_ASSERT: Normative RPL layer + appendix — not registry membership a... |
| G09 | B6-NORMATIVE-OPEN-ROWS | PASS | tests.qualification.ebh_7._rpl_h_normative_evidence:assert_ebh7_rpl_h_normative_for_b6_readiness | CORRECTIVE_ASSERT: Normative RPL readiness gate — L3 green ≠ B6 PASS |
| G09 | B6-F01-EXTERNAL-WORK-DUAL-INSTANCE | PASS | tests.qualification.ebh_5._external_work_dual_provider_proof:prove_external_work_dual_profile_bin... | CORRECTIVE_ASSERT: Qual alpha/beta + IntegrationProfile selection + contract ops |
| G09 | B6-F02-PRODQ-TEXT-PREFIX-NEGATIVE | PASS | tests/qualification/ebh_7/test_ebh_7_terminal_rpl_gates.py::test_ebh7_internal_rpl_negative | CORRECTIVE_PYTEST: prove_prodq_colocated_adapter_internal_negative @ terminal... |
| G09 | B6-F03-TEN-XDM-DOC-CONFLICT | PASS | tests.qualification.ebh_7._ebh7_certification_ssot:assert_ebh7_ten_xdm_evidence_ssot | CORRECTIVE_ASSERT: SSOT 26/0 TEN and 34/0 XDM @ registry + consistency matrix |
| G09 | B6-F04-P03-TEST-CLOSURE-SHA | PASS | tests.qualification.ebh_7._ebh7_p03_evidence_integrity:assert_ebh7_p03_manifest_role_integrity | CORRECTIVE_ASSERT: TEST_EXECUTION_SHA descends from B6 corrective — not closu... |
| G09 | B6-F07-RPL-EF-PROOF-STRENGTH | PASS | tests.qualification.ebh_5._dual_slug_rpl_ef_contract_evidence:dual_slug_rpl_ef_audit_errors | CORRECTIVE_ASSERT: 28 dual-slug RPL-E contract obligations + Package C v2 CDE... |
| G09 | B6-F08-ECAP-P03-DERIVED-GATES | PASS | tests.qualification.ebh_7._ebh7_closure_evidence_audit:assert_ebh7_ecap_p03_derived_closure_gates | CORRECTIVE_ASSERT: G07/G08 derive from certification + P03 integrity — not st... |
| G09 | B6-F09-FINDING-RESOLUTION-EVIDENCE | PASS | tests.qualification.ebh_7._gate_finding_resolution:assert_ebh7_gate_findings_resolution_complete | CORRECTIVE_ASSERT |

## G10–G12 (outside agent closure)

| ID | Status |
| --- | --- |
| G10 Protocol 03 B6 | OPEN — external auditor PASS/FAIL record |
| G11 OD-EBH7-08 | OPEN — operator signature |
| G12 Roadmap EBH-7 | IN PROGRESS until OD-08 |

**Non-claims:** B6 PASS, OD-EBH7-08 APPROVED, parent CLOSED, global FRZ PASS.
