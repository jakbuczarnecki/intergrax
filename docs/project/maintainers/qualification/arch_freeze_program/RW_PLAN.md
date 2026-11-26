# RW-PLAN — T-AF2-RW-PLAN (canonical)

**Generated:** 2026-11-25T18:09:14.017906+00:00
**Execution SHA:** `35967aaa676686fcf7fd2335f2e4f2bde0f6e53b`
**Node:** T-AF2-RW-PLAN (W2, DOC)

## 0. Reconciliation (35 vs 45 CODE_FIX)

- MEP historical unplanned pack: **35**
- Matrix Appendix A CODE_FIX rows: **39**
- T-IGX-050-D reconciled 225 FRZ rows; CODE_FIX inventory is parsed from Appendix A required_fix column only (bounded by appendix-A end marker). MEP historical 35-pack = WP12-Q12 scope; matrix row count is authoritative (not hardcoded).
- FIND-AF2-23 (18 CODE_FIX without explicit negative tests) is historical WP-12 phase-check-3; current matrix CODE_FIX rows with NT-* or NEG-P/NEG-0: 39 / 39.

## 1. CODE_FIX inventory

| FRZ | Owner ticket | Negative test | Red-first | Regression |
|-----|--------------|---------------|-----------|------------|
| `FRZ-BND-04` | T-IGX-043-G | NT-BND-04 | RED_AT_PREFIX @ 0cee41a88274a53d14bb3558752d55bdde0d0b01 | ENT-A-BND-01..03 |
| `FRZ-BND-05` | T-IGX-043-H | NT-BND-05 | RED_AT_PREFIX @ 0cee41a88274a53d14bb3558752d55bdde0d0b01 | ENT-A-BND-01..03 |
| `FRZ-BND-06` | T-IGX-043-J | NT-BND-06 | RED_AT_PREFIX @ 307a871c059f150a6b744e39135a737399467e88 | tests/unit/runtime/architecture/test_harness_w5_typed_observability_transport_gate.py; ENT-A-BND-01..03 |
| `FRZ-CFG-02` | T-IGX-043-C | NT-CFG-02 | RED_FIRST_UNREPRODUCIBLE @ 754ee000675d71c1734656702beb0aef8ba58fd8 | ENT-A-VBN-01..03 |
| `FRZ-CFG-06` | T-IGX-043-C | NT-CFG-06 | RED_AT_PREFIX @ 0cee41a88274a53d14bb3558752d55bdde0d0b01 | ENT-A-VBN-01..03 |
| `FRZ-CFG-08` | T-IGX-043-F | NT-CFG-08 | RED_FIRST_UNREPRODUCIBLE @ 754ee000675d71c1734656702beb0aef8ba58fd8 | ENT-A-VBN-01..03 |
| `FRZ-CTR-01` | T-IGX-051-D | NT-CTR-01 | RED_AT_PREFIX @ 0cee41a88274a53d14bb3558752d55bdde0d0b01 | ENT-A-CTR-01..03 |
| `FRZ-CTR-02` | T-IGX-051-D | NT-CTR-02 | RED_AT_PREFIX @ 0cee41a88274a53d14bb3558752d55bdde0d0b01 | ENT-A-CTR-01..03 |
| `FRZ-CTR-06` | T-IGX-051-D | NT-CTR-06 | RED_AT_PREFIX @ 0cee41a88274a53d14bb3558752d55bdde0d0b01 | ENT-A-CTR-01..03 |
| `FRZ-DEBT-02` | T-IGX-051-H | NT-DEBT-02 | RED_AT_PREFIX @ 6c9c8155da6227337e17c9f1f230cf78d7949f4b | ENT-A / stage pytest |
| `FRZ-DEBT-03` | T-IGX-051-H | NT-DEBT-03 | RED_AT_PREFIX @ 6c9c8155da6227337e17c9f1f230cf78d7949f4b | ENT-A / stage pytest |
| `FRZ-DEBT-04` | T-IGX-051-F | NT-DEBT-04 | RED_AT_PREFIX @ 8bdb0038d7a8fe506b5490a0ec2acad227e6b938 | ENT-A / stage pytest |
| `FRZ-EXE-02` | T-IGX-043-A | NT-EXE-02 | RED_AT_PREFIX @ b7855b83e98e3cbf2589775dabd64d990385dc24 | ENT-A / stage pytest |
| `FRZ-FRZ-10` | T-B7-PFP-ACTIVATE | NT-FRZ-10 | RED_FIRST_UNREPRODUCIBLE @ 332c5217c390202a911ad5592e88a8f366845587 | ENT-A / stage pytest |
| `FRZ-OWN-02` | T-IGX-043-J | NT-OWN-02 | RED_FIRST_UNREPRODUCIBLE @ 754ee000675d71c1734656702beb0aef8ba58fd8 | ENT-A-OWN-01..03 |
| `FRZ-OWN-03` | T-IGX-043-H | NT-OWN-03 | RED_FIRST_UNREPRODUCIBLE @ 754ee000675d71c1734656702beb0aef8ba58fd8 | ENT-A-OWN-01..03 |
| `FRZ-OWN-05` | T-IGX-043-H | NT-OWN-05 | RED_AT_PREFIX @ 0cee41a88274a53d14bb3558752d55bdde0d0b01 | ENT-A-OWN-01..03 |
| `FRZ-PLG-01` | T-IGX-043-J | NT-PLG-01 | RED_FIRST_UNREPRODUCIBLE @ 22993254b13af1ebae3dfd304c902523b4236129 | ENT-A / stage pytest |
| `FRZ-PLG-02` | T-IGX-043-J | NT-PLG-02 | RED_FIRST_UNREPRODUCIBLE @ 22993254b13af1ebae3dfd304c902523b4236129 | ENT-A / stage pytest |
| `FRZ-PLG-05` | T-IGX-043-A | NT-PLG-05 | RED_AT_PREFIX @ 459dfebc712f54eef6d9e025557930f1171c40d8 | ENT-A / stage pytest |
| `FRZ-PLG-06` | T-IGX-043-J | NT-PLG-06 | RED_FIRST_UNREPRODUCIBLE @ 2b12af1dc5029a223a0d32fffb6d4fc685ef4cd4 | tests/qualification/governance/gr11/catalog.py |
| `FRZ-PLG-07` | T-IGX-043-J | NT-PLG-07 | RED_FIRST_UNREPRODUCIBLE @ 22993254b13af1ebae3dfd304c902523b4236129 | ENT-A / stage pytest |
| `FRZ-PRD-03` | T-IGX-043-B | NT-PRD-03 | RED_FIRST_UNREPRODUCIBLE @ 754ee000675d71c1734656702beb0aef8ba58fd8 | ENT-A-PRD-01..03 |
| `FRZ-PRD-05` | T-IGX-043-H | NT-PRD-05 | RED_AT_PREFIX @ 754ee000675d71c1734656702beb0aef8ba58fd8 | ENT-A-PRD-01..03 |
| `FRZ-PRD-08` | T-IGX-043-J | NT-PRD-08 | RED_FIRST_UNREPRODUCIBLE @ 754ee000675d71c1734656702beb0aef8ba58fd8 | ENT-A-PRD-01..03 |
| `FRZ-REG-09` | T-IGX-051-F | NT-REG-09 | RED_AT_PREFIX @ 754ee000675d71c1734656702beb0aef8ba58fd8 | ENT-A-REG-01; ENT-A-REG-02 |
| `FRZ-REL-07` | T-IGX-043-B | NT-REL-07 | RED_FIRST_UNREPRODUCIBLE @ 66664e2bffa1de7ee5405b4d16d771302d676080 | ENT-A / stage pytest |
| `FRZ-REL-09` | T-IGX-043-B | NT-REL-09 | RED_FIRST_UNREPRODUCIBLE @ ec7f1e27c4a0a81b60ffad3feb2ff7ab696152a2 | ENT-A / stage pytest |
| `FRZ-REL-10` | T-IGX-043-B | NT-REL-10 | RED_FIRST_UNREPRODUCIBLE @ a01b7e415797a0ff1ecfb961ce7d5c5857415106 | ENT-A / stage pytest |
| `FRZ-RPL-04` | T-IGX-051-G | NEG-P | RED_FIRST_UNREPRODUCIBLE @ 754ee000675d71c1734656702beb0aef8ba58fd8 | ENT-A-VBN-01..03 |
| `FRZ-SEC-04` | T-IGX-043-C | NT-SEC-04 | RED_AT_PREFIX @ ccc7287c40670b421cc36cf3fc2eb331cdba310c | ENT-A / stage pytest |
| `FRZ-SEC-06` | T-IGX-051-B | NEG-0 | RED_AT_PREFIX @ 459dfebc712f54eef6d9e025557930f1171c40d8 | ENT-A / stage pytest |
| `FRZ-TEN-01` | T-IGX-043-J | NT-TEN-01 | RED_AT_PREFIX @ 0cee41a88274a53d14bb3558752d55bdde0d0b01 | ENT-A-TEN-01; ENT-A-TEN-02 |
| `FRZ-TEN-02` | PT-T01 | NEG-0 | RED_AT_PREFIX @ 0cee41a88274a53d14bb3558752d55bdde0d0b01 | ENT-A-TEN-01; ENT-A-TEN-02 |
| `FRZ-TEN-07` | PT-T01 | NEG-P | RED_FIRST_UNREPRODUCIBLE @ 0cee41a88274a53d14bb3558752d55bdde0d0b01 | ENT-A-TEN-01; ENT-A-TEN-02 |
| `FRZ-TEN-10` | T-IGX-051-A | NT-TEN-10 | RED_AT_PREFIX @ 0cee41a88274a53d14bb3558752d55bdde0d0b01 | ENT-A-TEN-01; ENT-A-TEN-02 |
| `FRZ-TYP-01` | T-IGX-051-G | NEG-0 | RED_AT_PREFIX @ 0cee41a88274a53d14bb3558752d55bdde0d0b01 | ENT-A-TYP-01..03 |
| `FRZ-TYP-03` | T-IGX-051-G | NEG-P | RED_AT_PREFIX @ 0cee41a88274a53d14bb3558752d55bdde0d0b01 | ENT-A-TYP-01..03 |
| `FRZ-TYP-05` | T-IGX-051-G | NT-TYP-05 | RED_AT_PREFIX @ 0cee41a88274a53d14bb3558752d55bdde0d0b01 | ENT-A-TYP-01..03 |

## 2. Audit-flag conflicts (6)

| Finding | Owner ticket | Disposition |
|---------|--------------|-------------|
| A4-005 | T-IGX-051-H | T-AF2-DEPS-BATCH; PT-WF overlap |
| A4-018 | T-IGX-051-D | T-AF2-RW-RUNTIME; shares PT-WF / FRZ-CTR rows |
| A6-005 | T-IGX-043-I | PT-WD; ENT manifest engine scope |
| A4-006 | None | PT-T01 gate emitter / tenant propagation |
| A3-018 | T-IGX-046-G | PT-G02 STATE-X citation repair |
| A4-016 | T-IGX-043-H | PT-WB dual host LLM resolver |

## 3. Lane order and serialization

See `rw_plan.v1.json` → `lane_order`, `serialization_pairs`, `overlap_hotspots`.

## 4. Path B and tenant isolation

- ReplayX path_b=B on FRZ matrix L66 rows is non-authoritative (DP-34, FRZ_RR Path B).
- Tenant isolation: cross-tenant NEG tests on T-IGX-051-A scope.

## 5. G2 entry conditions

- SCOPE hard predecessors satisfied
- RW train head CI green except known-red register
- PT-Q02 wave audit after G2 qualification

## 6. Dependencies (SCOPE / W3)

W2 preds: ['T-AF2-FRZ-MATRIX', 'T-AF2-DS-EXC-PACKAGE', 'T-AF2-BND-CLOSURE']
SCOPE preds: ['M1', 'DS-1', 'DS-2', 'DS-3', 'DS-4', 'DS-EXC', 'PT-V04', 'PT-B01', 'PT-B02', 'REV-MASTER', 'T-AF2-RW-PLAN', 'T-AF2-RW-PLAN-AUDIT']
W3: ['SCOPE', 'PT-WA']

## 7. Write sets (from approved plans; mechanical scan)

### T-IGX-043-A
- `agents/*/notebooks/01_*_experiment.ipynb`
- `agents/defaults.py`
- `agents/dispute_analyst/notebooks/01_dispute_analyst_experiment.ipynb`
- `agents/dispute_intake/notebooks/01_dispute_intake_experiment.ipynb`
- `agents/dispute_scenario/notebooks/01_dispute_scenario_experiment.ipynb`
- `agents/dispute_strategist/notebooks/01_dispute_strategist_experiment.ipynb`
- `agents/external_contractor_adapter/external_contractor_adapter_agent.py`
- `agents/intergrax_assistant/notebooks/01_intergrax_assistant_experiment.ipynb`
- `agents/legal/notebooks/01_legal_experiment.ipynb`
- `agents/local_indexer/notebooks/01_local_indexer_experiment.ipynb`
- `agents/local_search/notebooks/01_local_search_experiment.ipynb`
- `agents/local_synthesizer/notebooks/01_local_synthesizer_experiment.ipynb`
- `agents/organization_worker/organization_worker_agent.py`
- `agents/signoff_probe/notebooks/01_signoff_probe_experiment.ipynb`
- `applications/*/host/factory.py`
- `applications/*/tests`
- `applications/*/tests/`
- `applications/USAGE.md`
- `applications/_shared/`
- `applications/_shared/runtime_defaults.py`
- `applications/attestation_demo/host/factory.py`
- `applications/dispute_sim_application/host/factory.py`
- `applications/governed_contractor_application/host/factory.py`
- `applications/governed_contractor_application/tests/host/test_governed_contractor_canonical_execution.py`
- `applications/intergrax_assistant_application/host/factory.py`
- … +55 more (see JSON)

### T-IGX-043-B
- `applications/*/tests`
- `applications/*/tests/`
- `applications/dispute_sim_application/host/environment_profile.py`
- `applications/dispute_sim_application/host/factory.py`
- `applications/dispute_sim_application/tests/host/test_dispute_sim_queue_worker_wiring.py`
- `applications/governed_contractor_application/host/environment_profile.py`
- `applications/lab_application`
- `applications/lab_application/host/factory.py`
- `applications/legal_application/host/factory.py`
- `applications/legal_application/host/wiring.py`
- `applications/legal_application/tests/host/test_legal_queue_worker_wiring.py`
- `applications/legal_application/tests/host/test_legal_tool_wiring.py`
- `applications/local_workspace_application/docs/ARCHITECTURE.md`
- `applications/local_workspace_application/host/environment_profile.py`
- `applications/research_application/host/environment_profile.py`
- `applications/research_application/tests/host/test_research_llm_composition.py`
- `intergrax/applications/`
- `intergrax/applications/_shared/environment_wiring.py`
- `intergrax/applications/_shared/memory_provider_admission.py`
- `intergrax/applications/_shared/memory_wiring.py`
- `intergrax/applications/_shared/package_wiring.py`
- `intergrax/applications/_shared/production_capacity_wiring.py`
- `intergrax/applications/_shared/production_queue_resolver.py`
- `intergrax/applications/_shared/queue_worker_wiring.py`
- `intergrax/applications/_shared/runtime_event_delivery_wiring.py`
- … +55 more (see JSON)

### T-IGX-043-C
- `applications/*/host/factory.py`
- `applications/*/tests`
- `applications/*/tests/`
- `applications/_shared`
- `applications/_shared/`
- `applications/_shared/environment_wiring.py`
- `applications/_shared/guardrail_runtime_bridge.py`
- `applications/_shared/harness_host_auxiliary_wiring.py`
- `applications/_shared/interaction_wiring.py`
- `applications/_shared/notification_wiring.py`
- `applications/attestation_demo/host/factory.py`
- `applications/lab_application/host/factory.py`
- `applications/lab_application/host/tool_wiring.py`
- `applications/legal_application/host/`
- `applications/local_workspace_application/host/host_runtime_composition.py`
- `applications/local_workspace_application/model_runtime_proof/runtime.py`
- `applications/local_workspace_application/scripts/`
- `applications/local_workspace_application/scripts/lkw_ollama_embedding_bootstrap.py`
- `applications/local_workspace_application/workspaces/indexed_vector_verifier.py`
- `intergrax/applications/_shared/critical_action_signing_wiring.py`
- `intergrax/applications/_shared/environment_wiring.py`
- `intergrax/applications/_shared/guardrail_runtime_bridge.py`
- `intergrax/applications/_shared/harness_host_auxiliary_wiring.py`
- `intergrax/applications/_shared/harness_host_runtime.py`
- `intergrax/applications/_shared/interaction_wiring.py`
- … +55 more (see JSON)

### T-IGX-043-F
- `intergrax/model_inference/registry/vision_provider.py`
- `intergrax/rag/embedding/registry/profile.py`
- `scripts/governance/iiac_common.py`
- `scripts/maintenance/check_*`
- `scripts/qualification/verify_ebh7_p03_immutable_evidence.py`
- `tests/qualification/config_x`
- `tests/qualification/config_x/`
- `tests/qualification/config_x/_config_x_semantic_production_scan.py`
- `tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py`
- `tests/qualification/config_x/test_config_x_r1_r1_r1_semantic_constant_resolution.py`
- `tests/unit/architecture`

### T-IGX-043-G
- `agents/*`
- `agents/*/tests/`
- `agents/external_contractor_adapter/`
- `agents/local_search/steps/search_job.py`
- `agents/local_search/tests/`
- `agents/tool_selection_qualifier`
- `agents/tool_selection_qualifier/steps/tool_selection_job.py`
- `agents/tool_selection_qualifier/tool_selection.py`
- `agents/web_search_qualifier`
- `agents/web_search_qualifier/search_provider_resolver.py`
- `agents/web_search_qualifier/source_selection/llm_selector.py`
- `agents/web_search_qualifier/steps/web_search_job.py`
- `agents/web_search_qualifier/web_search_qualifier_agent.py`
- `applications/*/tests/`
- `applications/_shared/host_orchestration_backend_spec_builder.py`
- `applications/attestation_demo/host/`
- `applications/contracts/execution_mode.py`
- `applications/contracts/graph_spec.py`
- `applications/local_workspace_application/`
- `applications/local_workspace_application/docs/ARCHITECTURE.md`
- `applications/local_workspace_application/host/agent_factories.py`
- `applications/local_workspace_application/host/environment_profile.py`
- `applications/local_workspace_application/host/factory.py`
- `applications/local_workspace_application/host/wiring.py`
- `applications/local_workspace_application/manifest.py`
- … +55 more (see JSON)

### T-IGX-043-H
- `agents/authoring/acp_run.py`
- `agents/readiness/scoreboard.py`
- `applications/*/host/factory.py`
- `applications/_shared/*_wiring.py`
- `applications/_shared/llm_resolver.py`
- `applications/_shared/wiring.py`
- `applications/governed_contractor_application/`
- `applications/governed_contractor_application/deploy/docker-compose-local-qualification/RUNBOOK.md`
- `applications/governed_contractor_application/deploy/docker-compose-local-qualification/docker-compose.yml`
- `applications/governed_contractor_application/deploy/linux/single-node-production-profile-v1/intergrax-gca-strict-production.env.example`
- `applications/governed_contractor_application/host/package_b_strict_production_serving_command.py`
- `applications/governed_contractor_application/hosting/`
- `applications/governed_contractor_application/hosting/__init__.py`
- `applications/governed_contractor_application/hosting/boundary.py`
- `applications/governed_contractor_application/hosting/profile.py`
- `applications/governed_contractor_application/hosting/runtime.py`
- `applications/local_workspace_application/host/agent_factories.py`
- `applications/local_workspace_application/host/main.py`
- `applications/local_workspace_application/slack_companion/companion.py`
- `applications/local_workspace_application/tests/host/test_lkw_production_host_bootstrap.py`
- `applications/local_workspace_application/tests/host/test_local_workspace_environment_profile.py`
- `applications/local_workspace_application/tests/hosting/test_hosted_foreground.py`
- `applications/local_workspace_application/tests/hosting/test_hosted_profile.py`
- `applications/research_application/.env.example`
- `applications/research_application/host/main.py`
- … +55 more (see JSON)

### T-IGX-043-J
- `applications/_shared/harness_host_runtime.py`
- `tests/unit/applications/test_harness_evaluation_wiring.py`
- `tests/unit/applications/test_wf_fail_closed.py`
- `tests/unit/compat/langchain/`
- `tests/unit/compat/langchain/test_documents.py`
- `tests/unit/experiments/`
- `tests/unit/experiments/test_experiment_workflow.py`
- `tests/unit/runtime/execution/`
- `tests/unit/runtime/execution/test_execution_runtime.py`

### T-IGX-051-A
- `applications/*/mcp/server.py`
- `intergrax/agent_distribution/admin_models.py`
- `intergrax/agent_distribution/admin_service.py`
- `intergrax/agent_distribution/in_memory_stores.py`
- `intergrax/agent_distribution/sqlite_stores.py`
- `intergrax/agent_distribution/stores.py`
- `intergrax/applications/_shared/environment_wiring.py`
- `intergrax/applications/_shared/harness_host_runtime.py`
- `intergrax/applications/_shared/mcp_application_server.py`
- `intergrax/applications/_shared/mcp_catalog_tools.py`
- `intergrax/applications/_shared/mcp_harness_principal.py`
- `intergrax/applications/_shared/mcp_nexus_server.py`
- `intergrax/applications/_shared/memory_provider_admission.py`
- `intergrax/applications/_shared/memory_wiring.py`
- `intergrax/applications/_shared/policy_wiring.py`
- `intergrax/applications/_shared/production_lifecycle.py`
- `intergrax/applications/_shared/wiring_posture.py`
- `intergrax/contracts/`
- `intergrax/contracts/artifact_ref.py`
- `intergrax/integrations/contracts/sandbox_host.py`
- `intergrax/platform/enterprise_audit/impact_map.py`
- `intergrax/rag/governance/embedding_version_policy.py`
- `intergrax/rag/governance/source_freshness.py`
- `intergrax/rag/ingest/ingest_pipeline.py`
- `intergrax/runtime/architecture/online_evaluation.py`
- … +55 more (see JSON)

### T-IGX-051-B
- `applications/_shared`
- `applications/governed_contractor_application/host/`
- `applications/governed_contractor_application/host/package_b_strict_production_serving_command.py`
- `applications/governed_contractor_application/host/single_node_profile_v1_strict_startup.py`
- `intergrax/agent_distribution/admin_service.py`
- `intergrax/agent_distribution/control_plane_governance.py`
- `intergrax/agent_distribution/package_attestation.py`
- `intergrax/agent_distribution/package_trust.py`
- `intergrax/agent_distribution/trust.py`
- `intergrax/applications/_shared/application_security_wiring.py`
- `intergrax/applications/_shared/deploy_activate_pipeline.py`
- `intergrax/applications/_shared/environment_wiring.py`
- `intergrax/applications/_shared/production_agent_capability_runtime.py`
- `intergrax/applications/_shared/production_lifecycle.py`
- `intergrax/applications/_shared/security_assembly_resolver.py`
- `intergrax/applications/_shared/security_runtime_bridge.py`
- `intergrax/contracts/agent_run.py`
- `intergrax/contracts/execution_evidence/persistence_port.py`
- `intergrax/core/plugins/`
- `intergrax/core/plugins/*`
- `intergrax/core/security/raw_secret_gate.py`
- `intergrax/core/security/secret_safety.py`
- `intergrax/core/security_bootstrap.py`
- `intergrax/llm_adapters/governance/quota.py`
- `intergrax/runtime/architecture/online_evaluation_registry.py`
- … +53 more (see JSON)

### T-IGX-051-D
- `agents/model_routing_qualifier/qualification_types.py`
- `agents/model_routing_qualifier/steps/model_routing_job.py`
- `applications/local_workspace_application/benchmarks/local_model_qualification/environment.py`
- `intergrax/contracts/`
- `intergrax/llm_adapters/_shared/openai_completion_mapping.py`
- `intergrax/llm_adapters/_shared/provider_stream_admission.py`
- `intergrax/llm_adapters/_shared/resilience.py`
- `intergrax/llm_adapters/base/base_llm_adapter.py`
- `intergrax/llm_adapters/contracts/*`
- `intergrax/llm_adapters/contracts/provider_error.py`
- `intergrax/llm_adapters/governance/quota.py`
- `intergrax/llm_adapters/providers/*`
- `intergrax/llm_adapters/providers/*.py`
- `intergrax/llm_adapters/providers/ollama_vision.py`
- `intergrax/multimedia/image_smart_loader.py`
- `intergrax/multimedia/images_loader.py`
- `intergrax/runtime/llm/*`
- `scripts/maintenance/check_agents_no_vendor_sdk_imports.py`
- `tests/qualification`
- `tests/qualification/ebh_5`
- `tests/qualification/ebh_6`
- `tests/qualification/ebh_6/_findings_register.py`
- `tests/qualification/harness_02/test_harness_02_gates.py`
- `tests/qualification/pdc_x/d3/test_d3_f01_stream_guard.py`
- `tests/qualification/pdc_x/d3/test_d3_f02_refusal_paths.py`
- … +15 more (see JSON)

### T-IGX-051-F
- `agents/applications`
- `intergrax/applications/_shared/deploy_activate_pipeline.py`
- `intergrax/contracts/evaluation_contracts.py`
- `intergrax/contracts/runtime_event.py`
- `intergrax/runtime/events/payloads/canonical.py`
- `intergrax/runtime/observability/qualification_runtime_trace.py`
- `scripts/gates`
- `scripts/gates/check_agent_promotion_eval_gate.py`
- `scripts/gates/check_eval_digest_enforcement.py`
- `scripts/gates/check_mvp_promotion_gates.py`
- `scripts/gates/check_not_composed_reachability.py`
- `tests/unit/architecture/test_rw_gates_eval_digest_denied.py`
- `tests/unit/architecture/test_rw_gates_not_composed_raises.py`
- `tests/unit/architecture/test_rw_gates_promotion_unrelated_ticket_denied.py`
- `tests/unit/runtime/observability/test_qualification_runtime_observability_o*.py`

### T-IGX-051-G
- `intergrax/contracts/`
- `intergrax/contracts/**/*.py`
- `scripts/maintenance/check_frozen_contract_typ_ratchet.py`
- `scripts/maintenance/measure_frozen_contract_typ.py`
- `scripts/maintenance/measure_frozen_contract_typ05.py`
- `tests/unit/architecture/test_rw_typ_ratchet_*.py`
- `tests/unit/architecture/test_rw_typ_ratchet_baseline_unchanged_passes.py`
- `tests/unit/architecture/test_rw_typ_ratchet_new_any_fails.py`
- `tests/unit/architecture/test_rw_typ_ratchet_new_object_fails.py`

### T-IGX-051-H
- `intergrax/contracts/`
- `intergrax/platform/enterprise_audit/validators.py`
- `intergrax/runtime/long_running/recurrence.py`
- `scripts/governance/iiac_common.py`
- `scripts/maintenance/check_dependency_ownership.py`
- `scripts/maintenance/check_environment_conformance.py`
- `tests/unit`
- `tests/unit/architecture/`
- `tests/unit/architecture/test_ebh_3_dependency_ownership_gate.py`
- `tests/unit/architecture/test_rw_deps_core_ownership_declared.py`
- `tests/unit/test_dependency_governance.py`

## 8. Machine contract

Strict phase check 3: `scripts/governance/verify_rw_plan_phase_check3.py`

JSON SSOT: `rw_plan.v1.json` (schema `rw_plan.v1`)
