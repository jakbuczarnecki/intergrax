# IIAC operator decision — MCP-RC ACT-N11 product re-bind (integrated baseline of the PDC-P7-00 window)

Decision ID: IIAC-OD-MCP-RC-ACT-N11-2026-11-13
Policy version: IIAC-2026-11-05.1
Decision scope: IIAC policy activation (integrated baseline: MCP-RC and PDC-X product delta, 158 ACT-N11 paths vs `2d3eceb6cf7e93dccd4d96919319a6c969b4444b`, audited tree `324d9dba37cb54e1b74a79a05148d2dcc05e6d09`)
Decision status: AUTHORIZED
Audit report digest: c405c65faf53243a0511b0b9584a9f3abda4db08c09ee9b9266c86fe9f6b7a4e

Independent audit dossier digest (MCP-RC first audit, dossier verdict PARTIAL): 30e002c28af0c51070b5236dd674fb0465d439f976bb550c39927e23703eccf1

Independent audit dossier digest (MCP-RC delta re-audit, dossier verdict PASS_SCOPED): fb4ba5a70ad20578513fec793ef4e1421b2ed7ba06d599ce3d89ad3e23aa4f34

Independent audit dossier digest (PDC-P7-00 baseline audit, dossier verdict PASS_SCOPED): 08b1bc2d5b98920fe09008cae5272e7bf63becdad1a125019cf3c26b200a861d

Independent audit dossier digest (PDC-P7-00 addendum 1, dossier verdict PASS_SCOPED): 722cf0654530883c9466e4269eb43c8ae239190145c2ca6ec478b1ed61c3e1c2

Independent audit dossier digest (PDC-P7-00 addendum 2, dossier verdict PASS_SCOPED): fe7e03c7c036ae15de182d4d5dd51f91fe4f6d2df556ffbbafe612e1d5019301

Approved IIAC policy activation IIAC-2026-11-05.1: operator @ 2026-11-14T08:21:13+01:00

Provenance note: the date in the Decision ID is the drafting date of the first draft (2026-11-13), not a signing timestamp; this text was rewritten on 2026-11-14 for the integrated baseline. The line above was written by an operator-designated signing session (IIAC Amendment 01, R-1..R-6) under the delegation record `docs/project/maintainers/qualification/mcp_server_x/operator_decisions/DELEGATION-MCP-RC-2.md` (commit 2e4ef79d5, renewed by `OD-MCP-RC-04` section 4), on the operator's chat commission of 2026-11-13 and the operator's direct confirmation to that session in chat on 2026-11-14. The line is in the plain form because the IIAC activation verifiers accept only that form; it is not claimed to be operator-written. The timestamp is the designated session's system clock at the moment of signing.

## What a signature on this record means (bound verdict)

The activation gate accepts only the verdict `PASS` in the binding manifest (ACT-N05). All independent audits behind this decision carry the dossier verdict `PASS_SCOPED` (the first MCP-RC audit `PARTIAL`, superseded by its re-audit) and their manifests carry the field value `CONDITIONAL`; none can be bound as authored. By the operator's recorded answer (`OD-MCP-RC-04` section 3, Polish label `Tak, wzór P6 (Recommended)`, English rendering by the orchestrator "Yes, the P6 pattern", not an operator-written line), the activation certifier authors a **new binding manifest with `verdict` = `PASS`** in the P6 pattern (stage `IIAC-GOV-1`), and states the scope limits, the open findings, the red-test caveat and the accepted conditions below in its `limitations`. That `PASS` is the activation certifier's own, authored from `PASS_SCOPED` audits; it is not an auditor verdict and the auditors' scoped verdicts are not rewritten. Signing this record accepts that conversion together with every limit listed here.

## Scope

158 product paths under `intergrax/`, `agents/` and `applications/` differ from the current activation pin `audit_execution_sha` `2d3eceb6cf7e93dccd4d96919319a6c969b4444b` (`IIAC_POLICY_ACTIVATION_RECORD.json`, P6 activation). They are the ACT-N11 trigger, derived with the gate's own rule (`git diff --name-only 2d3eceb6c..324d9dba3`, prefixes `intergrax/`, `agents/`, `applications/`): **158 = 23 MCP-RC + 135 non-MCP**. Build-up by audit: first PDC-P7-00 audit 153 paths (at `fa6ff1c96`), addendum 1 adds 3 and counts 156 (at `19d552bee`), addendum 2 adds 2 and counts 158 (at `324d9dba3`). The 23 paths equal `product_paths_delta_vs_base` of `iiac_evidence/PDC-X/MCP-RC/audit_scope.json`; the 158 equal the git-derived set at the audited tree and at the integrated head (certifier preflight addendum 1, section 1; recomputed by the drafting session).

Integrated head at drafting: the product tree of the merged branch equals the audited tree (`git diff --name-only 324d9dba3 HEAD -- intergrax agents applications` is empty), and the 23 MCP-RC paths are byte-identical to `b0e19bf61b346e6941b2e190b703c6922e708b34` (the last MCP-RC product change). `origin/development` has no product-path change after `a156cbbd5` (the second re-pinned baseline, PR #525, contained in `324d9dba3`): `git diff --name-only a156cbbd5 origin/development -- intergrax agents applications` is empty. The operator declared PDC-X frozen on product paths for the rest of the window (`OD-MCP-RC-04` section 1); this is a declaration, not an enforced control.

### 23 MCP-RC paths

| # | Path |
| --- | --- |
| 1 | `applications/dispute_sim_application/mcp/server.py` |
| 2 | `applications/governed_contractor_application/mcp/server.py` |
| 3 | `applications/intergrax_assistant_application/mcp/server.py` |
| 4 | `applications/lab_application/mcp/server.py` |
| 5 | `applications/legal_application/mcp/server.py` |
| 6 | `applications/local_workspace_application/mcp/server.py` |
| 7 | `applications/poc_template_application/mcp/server.py` |
| 8 | `applications/research_application/mcp/server.py` |
| 9 | `intergrax/applications/_shared/fastapi_mcp.py` |
| 10 | `intergrax/applications/_shared/mcp_application_server.py` |
| 11 | `intergrax/applications/_shared/mcp_catalog_invocation_bridge.py` |
| 12 | `intergrax/applications/_shared/mcp_catalog_tools.py` |
| 13 | `intergrax/applications/_shared/mcp_developer_api.py` |
| 14 | `intergrax/applications/_shared/mcp_engine_concurrency.py` |
| 15 | `intergrax/applications/_shared/mcp_harness_principal.py` |
| 16 | `intergrax/applications/_shared/mcp_host_execution_transport_cancel.py` |
| 17 | `intergrax/applications/_shared/mcp_identity_mode.py` |
| 18 | `intergrax/applications/_shared/mcp_invocation_id.py` |
| 19 | `intergrax/applications/_shared/mcp_nexus_server.py` |
| 20 | `intergrax/applications/_shared/mcp_server_engine.py` |
| 21 | `intergrax/applications/_shared/mcp_wire_errors.py` |
| 22 | `intergrax/contracts/mcp_server_definition.py` |
| 23 | `intergrax/tools/mcp_result_bounds.py` |

### 135 non-MCP paths (PDC-X batches on the integrated baseline)

By directory (all 158): `intergrax/runtime` 49, `intergrax/applications` 23, `intergrax/llm_adapters` 20, `intergrax/agent_distribution` 11, `intergrax/hosting` 9, `intergrax/contracts` 6, `intergrax/core` 5, `intergrax/rag` 5, `intergrax/memory` 4, `intergrax/queueing` 4, `intergrax/context` 3, `intergrax/eval` 3, `intergrax/tools` 3, `intergrax/platform` 2, the eight `applications/<app>/mcp/server.py` 1 each, and one each in `intergrax/collaborative_work`, `intergrax/integrations`, `intergrax/skills`. The 135 non-MCP paths (generated from git at the audited tree; the 23 above are excluded):

```text
intergrax/agent_distribution/__init__.py
intergrax/agent_distribution/admin_models.py
intergrax/agent_distribution/admin_service.py
intergrax/agent_distribution/agent_manager_command_facade.py
intergrax/agent_distribution/control_plane_governance.py
intergrax/agent_distribution/events.py
intergrax/agent_distribution/installation.py
intergrax/agent_distribution/installation_service.py
intergrax/agent_distribution/runtime_revision.py
intergrax/agent_distribution/runtime_revision_service.py
intergrax/agent_distribution/upgrade_admission.py
intergrax/applications/_shared/application_security_wiring.py
intergrax/applications/_shared/attestation_runtime_bridge.py
intergrax/applications/_shared/critical_action_signing_wiring.py
intergrax/applications/_shared/evaluation_wiring.py
intergrax/applications/_shared/lkw_hybrid_daemon_wiring.py
intergrax/applications/_shared/llm_resolver.py
intergrax/applications/_shared/security_assembly_resolver.py
intergrax/applications/_shared/security_audit_trail_wiring.py
intergrax/applications/contracts/application_artifacts.py
intergrax/applications/contracts/environment_profile/sub_profiles.py
intergrax/collaborative_work/content_storage.py
intergrax/context/budget/mandatory_base_messages.py
intergrax/context/policy/egress.py
intergrax/context/provider_lifecycle.py
intergrax/contracts/agent_contract_meta.py
intergrax/contracts/agent_lifecycle_state.py
intergrax/contracts/artifact_ref.py
intergrax/contracts/evaluation_contracts.py
intergrax/contracts/task_artifacts.py
intergrax/core/catalog_bootstrap.py
intergrax/core/plugins/admission.py
intergrax/core/plugins/discovery.py
intergrax/core/security/raw_secret_gate.py
intergrax/core/security/security_configuration.py
intergrax/eval/eval_comparison.py
intergrax/eval/evaluators.py
intergrax/eval/typed_run.py
intergrax/hosting/__init__.py
intergrax/hosting/contracts/events.py
intergrax/hosting/engine/components.py
intergrax/hosting/engine/diagnostics.py
intergrax/hosting/engine/engine.py
intergrax/hosting/eventing.py
intergrax/hosting/metrics.py
intergrax/hosting/public_api.py
intergrax/hosting/supervisor/supervisor.py
intergrax/integrations/providers/embedding_provider/ollama/register.py
intergrax/llm_adapters/_shared/adapter_response_builders.py
intergrax/llm_adapters/_shared/conformance.py
intergrax/llm_adapters/_shared/openai_completion_mapping.py
intergrax/llm_adapters/base/base_llm_adapter.py
intergrax/llm_adapters/contracts/capabilities.py
intergrax/llm_adapters/contracts/finish_reason.py
intergrax/llm_adapters/llm_provider_registry.py
intergrax/llm_adapters/providers/aws_bedrock_adapter.py
intergrax/llm_adapters/providers/claude_adapter.py
intergrax/llm_adapters/providers/cohere_native_adapter.py
intergrax/llm_adapters/providers/gemini_adapter.py
intergrax/llm_adapters/providers/mistral_adapter.py
intergrax/llm_adapters/providers/ollama_adapter.py
intergrax/llm_adapters/providers/openai_chat_completions_adapter.py
intergrax/llm_adapters/providers/openai_compat_providers.py
intergrax/llm_adapters/providers/openai_responses_adapter.py
intergrax/llm_adapters/registry/catalog_capabilities.py
intergrax/llm_adapters/registry/failover_adapter.py
intergrax/llm_adapters/registry/model_catalog.yaml
intergrax/llm_adapters/routing/evaluating_adapter.py
intergrax/memory/contracts/memory_control.py
intergrax/memory/default_memory_control_plane.py
intergrax/memory/memory_temporal.py
intergrax/memory/retention_sweep.py
intergrax/platform/enterprise_audit/data/changed_since_impact_map.v1.json
intergrax/platform/enterprise_audit/impact_map.py
intergrax/queueing/contracts/task_queue.py
intergrax/queueing/providers/document_store/colocated_worker.py
intergrax/queueing/providers/document_store/document_store_task_queue.py
intergrax/queueing/task_index.py
intergrax/rag/governance/__init__.py
intergrax/rag/governance/source_freshness.py
intergrax/rag/ingest/ingest_pipeline.py
intergrax/rag/retrieval/retrieval_result.py
intergrax/rag/retrieval/retrieval_service.py
intergrax/runtime/agent_governance/pipeline.py
intergrax/runtime/architecture/agent_promotion.py
intergrax/runtime/architecture/evaluation_assets.py
intergrax/runtime/architecture/evaluation_automation.py
intergrax/runtime/architecture/evaluation_registry_trends.py
intergrax/runtime/architecture/evaluation_scenario_loader.py
intergrax/runtime/architecture/online_evaluation_registry.py
intergrax/runtime/attestation/host_attestation.py
intergrax/runtime/evidence/eval_evidence_contracts.py
intergrax/runtime/execution/agent_runtime_context.py
intergrax/runtime/execution/trace_persistence_models_bridge.py
intergrax/runtime/execution_evidence/attestor.py
intergrax/runtime/integrations/observability.py
intergrax/runtime/llm/model_call_runtime_evidence_adapter.py
intergrax/runtime/long_running/recurrence.py
intergrax/runtime/long_running/scheduler.py
intergrax/runtime/nexus/artifacts/context_reference.py
intergrax/runtime/nexus/artifacts/governed_access.py
intergrax/runtime/nexus/artifacts/in_memory_artifact_store.py
intergrax/runtime/nexus/artifacts/models.py
intergrax/runtime/nexus/artifacts/retention.py
intergrax/runtime/nexus/artifacts/sources.py
intergrax/runtime/nexus/artifacts/spill.py
intergrax/runtime/nexus/artifacts/sqlite_artifact_store.py
intergrax/runtime/nexus/artifacts/store_base.py
intergrax/runtime/nexus/artifacts/taxonomy.py
intergrax/runtime/nexus/context/context_compiler.py
intergrax/runtime/nexus/context/context_engine.py
intergrax/runtime/nexus/context/context_manager.py
intergrax/runtime/nexus/context/degradation_ladder.py
intergrax/runtime/nexus/context/shared_task_context.py
intergrax/runtime/nexus/engine/contracts/runtime_state_contract.py
intergrax/runtime/nexus/engine/runtime_state.py
intergrax/runtime/nexus/orchestration/run_artifact_bundle_builder.py
intergrax/runtime/nexus/tools/plan_context_invocation.py
intergrax/runtime/nexus/tracing/persistence_models.py
intergrax/runtime/nexus/tracing/trace_models.py
intergrax/runtime/observability/export_boundary.py
intergrax/runtime/persistence/sqlite_composition.py
intergrax/runtime/persistence/sqlite_opens.py
intergrax/runtime/registry/agent_registry.py
intergrax/runtime/registry/agent_routing_policy.py
intergrax/runtime/replay/models.py
intergrax/runtime/replay/replay_engine.py
intergrax/runtime/security/defense_plugin_loader.py
intergrax/runtime/security/multi_region_audit_trail.py
intergrax/runtime/security/security_audit_trail.py
intergrax/runtime/task/worker_bootstrap.py
intergrax/runtime/vendor_knowledge/contribution_catalog.py
intergrax/skills/host_lifecycle.py
intergrax/tools/providers/rag/service.py
intergrax/tools/providers/storage/service.py
```

None of the 135 was audited by the MCP-RC audits. They were audited, at the diff-scan and sampling depth stated in the PDC-P7-00 dossiers, by the PDC-P7-00 baseline audit and its addenda; the eight per-domain audits IA-D1..IA-D8 are not done (see the open conditions).

## Independent audits and preflights

| Evidence | Verdict | Audited / execution tree | Sessions | Evidence publication (pin-only edit commit) |
| --- | --- | --- | --- | --- |
| MCP-RC first audit | dossier `PARTIAL` (manifest `CONDITIONAL`) | `93a089d3cca941592f1ad4f65435333c86da82e5` | auditor `MCP-RC-G6-AUDITOR-90803ebc-262a-457b-9315-14a5be0713d2` | `78d4c496071141ca4092b926b838fb80016f1b91` (`c3f962636b104ceccce07d4252f7e60b16bf53f4`) |
| MCP-RC delta re-audit | dossier `PASS_SCOPED` (manifest `CONDITIONAL`) | `b0e19bf61b346e6941b2e190b703c6922e708b34` | auditor `MCP-RC-G6-REAUDIT-AUDITOR-9579becc-3291-43df-b623-f406ec80ac48` | same |
| MCP-RC certifier preflight | `CERTIFIER_PREFLIGHT_PASS_SCOPED` | record head `1b5f26f249beaafe007ffff2f1e396ae393a07dc` | certifier designation `MCP-RC-CERTIFIER-PREFLIGHT-2026-11-13` | same |
| PDC-P7-00 baseline audit | dossier `PASS_SCOPED` (manifest `CONDITIONAL`), 153 paths | `fa6ff1c9616a6df28c11b087eb3f1b0dc4097b5f` | auditor `PDC-P7-00-ACT-N11-G6-BASELINE-AUDITOR-85fb061a-028f-4648-a3d6-284a392f4e27` | `6b195980d20067449ade0d2c2fd2280781c5a25d` (`ae5623efbdfd69ddf4ba67f82861ccad130c9b33`) |
| PDC-P7-00 addendum 1 | dossier `PASS_SCOPED` (manifest `CONDITIONAL`), 156 paths | `19d552beeb6f0677a8bdc3b4ce65da15dd637c7b` | the same auditor, id suffix `-ADDENDUM-1` | `6b195980d20067449ade0d2c2fd2280781c5a25d` (`ae5623efbdfd69ddf4ba67f82861ccad130c9b33`) |
| PDC-P7-00 addendum 2 | dossier `PASS_SCOPED` (manifest `CONDITIONAL`), 158 paths | `324d9dba37cb54e1b74a79a05148d2dcc05e6d09` | the same auditor, id suffix `-ADDENDUM-2` | `29d66ae3fcc7a06a919d7e07541740841e8abfe8` (`fde55618379ff99779172e40214b5879d9d0810d`) |
| PDC-P7-00 certifier preflight | `CERTIFIER_PREFLIGHT_PASS_SCOPED` | head `ae5623efbdfd69ddf4ba67f82861ccad130c9b33` (preflight commit `5c25d4d8f`) | certifier designation `PDC-P7-00-CERTIFIER-PREFLIGHT-2026-11-13` | not applicable |
| PDC-P7-00 certifier preflight addendum 1 | `CERTIFIER_PREFLIGHT_PASS_SCOPED` | head `fde55618379ff99779172e40214b5879d9d0810d` (preflight commit `7801dcb12`) | the same certifier designation | not applicable |

Implementer designations as recorded in the manifests: `T-IGX-038-A-F-MCP-RC-IMPLEMENTER` (MCP-RC); `T-IGX-036-J-T-PDC-X-IMPLEMENTERS+T-IGX-038-A-F-MCP-RC-IMPLEMENTER` (first PDC-P7-00 audit and addendum 1); `T-IGX-036-J-V-PDC-X-IMPLEMENTERS+T-IGX-038-A-F-MCP-RC-IMPLEMENTER` (addendum 2). They are labels; no UUID is recorded and none is invented. The certifier designations are labels, not verified application session ids. Independence of implementer, auditors and certifier is a `process_enforced` attestation (one git identity signs all commits).

Evidence was published in three rounds, each with pin-only edits by an evidence-integration session that is none of implementer, auditor or certifier: round 1 MCP-RC (`iiac_evidence/PDC-X/MCP-RC/EVIDENCE_INTEGRATION_RECORD.md`), round 2 PDC-P7-00 first audit and addendum 1, round 3 addendum 2 (both in `iiac_evidence/PDC-X/P7-00/EVIDENCE_INTEGRATION_RECORD.md`). If an addendum 3 is made (open condition 1), a further round follows and the SHAs in this record move.

Current gate state (certifier preflight check 13 and addendum 1 check 9): `verify_iiac_policy_activation --require-in-force` exits 1 with only `ACT-N11 product path changed after the audited SHA` errors, 156 at the first preflight head and 158 at the integrated head. That is the intended trigger until the two-phase rebind; the drafting session did not re-run it.

## Digests (SHA-256 of committed file bytes, `git -c core.autocrlf=false show <commit>:<path>`)

Recomputed by the drafting session on 2026-11-14. Each audit's four digests equal the digests recorded in its manifest (checked). Paths are relative to `docs/project/maintainers/qualification/`.

| Artifact | SHA-256 |
| --- | --- |
| `iiac_evidence/IIAC-GOV-1/certification/IIAC_ACT01_INDEPENDENT_REAUDIT_DOSSIER.md` (the `Audit report digest` line above) | `c405c65faf53243a0511b0b9584a9f3abda4db08c09ee9b9266c86fe9f6b7a4e` |
| `iiac_evidence/ENT-AUDIT-X/act_n11_recovery/audit_scope.json` | `e06ce433ac1c260390885e19cc60b0687be9d8908d78bd38ae4fb5f1102fe56f` |
| `iiac_evidence/ENT-AUDIT-X/act_n11_recovery/test_matrix.txt` | `d5028f194bb7917d07e8df5264075952dcf8757503975042988caba6e89a240b` |
| `iiac_evidence/HIST-P5-2026-11-09/HIST_P5_REACTIVATION_FINDINGS.json` | `df3648f926dad3199ff40c7c83c7fae4ed7a26bd1c67701a355bc7130073f831` |
| `mcp_server_x/MCP_SERVER_X_RC_INDEPENDENT_AUDIT_DOSSIER.md` | `30e002c28af0c51070b5236dd674fb0465d439f976bb550c39927e23703eccf1` |
| `mcp_server_x/MCP_SERVER_X_RC_INDEPENDENT_REAUDIT_DOSSIER.md` | `fb4ba5a70ad20578513fec793ef4e1421b2ed7ba06d599ce3d89ad3e23aa4f34` |
| `iiac_evidence/PDC-X/MCP-RC/iiac_audit_session_manifest.PDC-X.mcp_rc.json` | `a9ae928e810e5935cbe1ae32cc1a9126a980204739f6c24f51e8d7dcd278aa0b` |
| `iiac_evidence/PDC-X/MCP-RC/reaudit/iiac_audit_session_manifest.PDC-X.mcp_rc_reaudit.json` | `5095559de8d6b2d1c10c476f3ba6c68e0f98a105630d0951a0ad2e268a071c8f` |
| `platform_domain_completeness/PDC_P7_00_ACT_N11_BASELINE_AUDIT_DOSSIER.md` | `08b1bc2d5b98920fe09008cae5272e7bf63becdad1a125019cf3c26b200a861d` |
| `iiac_evidence/PDC-X/P7-00/iiac_audit_p7_00_findings.json` | `720b9c2abf098b8263a62433538731143b4d61cda425254929952ffdd0a0133c` |
| `iiac_evidence/PDC-X/P7-00/audit_scope.json` | `2f6fbdc3100a7bbee9d40039a77e5d2deeff52608f383e686417f6577002b109` |
| `iiac_evidence/PDC-X/P7-00/test_matrix.txt` | `381b79a47b86a0399c18c18fae114d5dd971561d18aeac4e1685f69c3af75d91` |
| `iiac_evidence/PDC-X/P7-00/iiac_audit_session_manifest.PDC-X.p7_00.json` | `5d6acfe1adf4dc529a0afc08893a186319ac25a6d846359c976273537e061a27` |
| `platform_domain_completeness/PDC_P7_00_ACT_N11_BASELINE_AUDIT_ADDENDUM_1.md` | `722cf0654530883c9466e4269eb43c8ae239190145c2ca6ec478b1ed61c3e1c2` |
| `iiac_evidence/PDC-X/P7-00/addendum_1/iiac_audit_p7_00_addendum_1_findings.json` | `224e86a27b60a8213628943098ac988d41d7bd3568d1bb89b4a6a548911c61c9` |
| `iiac_evidence/PDC-X/P7-00/addendum_1/audit_scope.json` | `08e6bf5041348659ffb5e450bae080863b2abe1778416e436b729da488708792` |
| `iiac_evidence/PDC-X/P7-00/addendum_1/test_matrix.txt` | `7c8e9f862828188d09af37fd7d372e085ec9cf0d109bc80c0d67df40efd352aa` |
| `iiac_evidence/PDC-X/P7-00/addendum_1/iiac_audit_session_manifest.PDC-X.p7_00.addendum_1.json` | `faeff3b80372b3c14fa733ba4a38155b0dcafc07738222db83c5e629a172478a` |
| `platform_domain_completeness/PDC_P7_00_ACT_N11_BASELINE_AUDIT_ADDENDUM_2.md` | `fe7e03c7c036ae15de182d4d5dd51f91fe4f6d2df556ffbbafe612e1d5019301` |
| `iiac_evidence/PDC-X/P7-00/addendum_2/iiac_audit_p7_00_addendum_2_findings.json` | `f6f641f2a4bfe774ebc05fff98c878dd6c3df67665d6d4cc7fa4ec8820936f32` |
| `iiac_evidence/PDC-X/P7-00/addendum_2/audit_scope.json` | `9a5ca90b3691056464f0a5b26962b905e02dc901b619a8f1b05430568884b9e6` |
| `iiac_evidence/PDC-X/P7-00/addendum_2/test_matrix.txt` | `0904860856f54ed513955f365d8d872d772132fca89a30b16e6ae4977f7f0b9d` |
| `iiac_evidence/PDC-X/P7-00/addendum_2/iiac_audit_session_manifest.PDC-X.p7_00.addendum_2.json` | `8caf425e1aba482a7144704fc29a5a99d561b541db25deacd24dcf8770508a0b` |
| `platform_domain_completeness/PDC_P7_00_CERTIFIER_PREFLIGHT.md` | `8caf768d1cbdf048e85aa78840fea4b0ece1fb359cfa767ba45f882756f32e04` |
| `platform_domain_completeness/PDC_P7_00_CERTIFIER_PREFLIGHT_ADDENDUM_1.md` | `d33cd18ff77f93b3887279060b13e2fc85d12cd759060084ed617822294075b1` |

The first four rows are the artifacts the activation gate compares if the record keeps the `IIAC-GOV-1` report set (the P6 pattern; all five bound artifacts must sit under `iiac_evidence/`, ACT-N14, so the dossiers under `mcp_server_x/` and `platform_domain_completeness/` cannot be bound as they are). The `Audit report digest` line above must equal the record's `session_b_governance_audit_report_digest`; if the activation certifier binds other artifacts, this line and this table are revised before signing. The digests of the activation certifier's binding manifest and handoff record do not exist yet and are not part of this draft; the certifier writes both and the record cites them.

## Session provenance (operator confirms by signing)

The MCP-RC implementer, the PDC-X implementers, the first MCP-RC auditor, the MCP-RC re-audit auditor, the PDC-P7-00 baseline auditor (three rounds), the MCP-RC and PDC-P7-00 preflight certifier, the evidence-integration sessions, the signing delegate and the activation certifier (a new dedicated session, id recorded in its binding manifest) are separate sessions; none of the auditors, certifiers or the signing delegate is an implementer. This is a `process_enforced` attestation, not cryptographic proof.

## Open conditions and operator-owned items (not resolved by this record)

Signing the activation line accepts the scope limits below as limits; it does not resolve or accept any finding.

1. **F-41 (MEDIUM, pending change).** The ENT-AUDIT impact map (`intergrax/platform/enterprise_audit/impact_map.py`, `intergrax/platform/enterprise_audit/data/changed_since_impact_map.v1.json`) has no rule for the eight `applications/*/mcp/server.py` modules, so `test_xd12_every_pdc_touched_product_module_maps_to_a_domain_check` and `tests/qualification/pdc_x/p6_impact/test_p6_impact_map_coverage.py` are red on the integrated tree and green on `development` alone. `OD-MCP-RC-04` section 5 records the operator's answer that PDC-X adds the rule on `development` as a one-time exception to the freeze. **That rule is not on `development` yet** (no product-path change after `a156cbbd5`). When it lands, the bytes of two paths already inside this audit change: audit addendum 2 is void for those two paths, an auditor addendum 3 and a short certifier preflight addendum are required, the evidence SHAs and every digest in this record that depends on them move, and this record is revised and re-digested before signing. The union stays 158 paths.
2. **F-42 (MEDIUM).** Two `embedding_provider` tests fail (`UnknownIntegrationError`: the integration slug `ollama` is not registered in the catalog) inside the check `VBN-REACHABILITY` of `ent-audit-fast`; probably the known B-REG-P / ollama item, not confirmed as such. No owner is named; `OD-MCP-RC-04` section 2 does not list it.
3. **F-35 (MEDIUM), F-09 (LOW), F-07 (LOW).** Ten COMPAT-X event-evolution gates are red since PDC-X (seven more are red at the pin too); the PDC-X owner repairs them on `development` (`OD-MCP-RC-04` section 2) and they are still red there at the preflight addendum. F-09: pytest does not exit after `test_xd_03_agent_governance.py`; the owner recommendation is open. F-07: passes at the integrated head per the certifier (resolved at test level, not confirmed by an auditor).
4. **F-43, F-44 (LOW) and F-38, F-04/F-37, F-05, F-10.** Routed to the ENT-AUDIT owner, the COMPAT-X owner and the per-domain audits as the dossiers state; F-10 is a recommended product fix that the auditor does not require for activation. Whether host composition wiring is a PDC-X claim is an operator or orchestrator decision.
5. **CI contexts not verified.** The certifier read what `ci-smoke`, `governance-fast`, `evol-x-b4-preflight`, `ent-audit-fast` and `ent-audit-pr-gate` run but obtained no green result for them on a workstation: two `governance-fast` ACT-N11 tests fail until the rebind; fifteen fresh-clone fixtures fail on the Windows workstation (path length); `ent-audit-fast` and `ent-audit-pr-gate` did not finish. No claim is made that the five contexts are green.
6. **MCP-RC carry-overs.** The owner of the strict xfail `tests/unit/applications/test_mcp_surface_opt_in.py::test_scaffold_defaults_mcp_to_opt_in[product]` (AF-05, AF-16) is not named (`governance/tickets/T-IGX-039-A.md`, draft); operator acceptance of RA-02, RA-03, RA-04 and AF-05 is not recorded (the implementer's "accepted as declared" is not an operator acceptance).
7. **Audit coverage limits** (inherited from the dossiers, not re-verified here): the eight per-domain audits IA-D1..IA-D8 are not done and 27 findings are routed to them; the 130 first-audit PDC-X paths were reviewed by diff scan, selective reading, review helpers, 22 probes and 12 mutants, without a mutation sample on the later delta paths; F-18 and F-24..F-28 are unreproduced leads; 40 regression failures are attributed to the pin but not diagnosed; CI contexts, G-DOC-01..12, P8, full `tests/unit`, network and live-provider paths are not covered; the shared environment lacks `tzdata`.
8. **Drift.** Any product commit on `development` after `a156cbbd5` needs another audit addendum and preflight; the freeze is declared, not enforced.

## Non-claims

This decision authorizes only the IIAC activation named in its scope. It does not bypass governance checks, does not close the MCP-SERVER-X stage, does not promote any FRZ status (counter stays 225), does not claim production readiness, does not accept any open finding, does not authorize a merge, a pull request, a ticket approval or P8 changes, and does not edit `IIAC_POLICY_ACTIVATION_RECORD.json`. The binding manifest, the handoff record and the two-phase rebind of that record are separate steps by a separate activation-certifier session. Operator-selected options quoted above are recorded chat answers, not signed lines.

## Provenance

| Item | Value |
| --- | --- |
| Ticket | `governance/tickets/T-IGX-038-F.md` |
| Prior `audit_execution_sha` (current pin) | `2d3eceb6cf7e93dccd4d96919319a6c969b4444b` |
| Campaign base | `6a6279a49a9aa5a33936a1c6a577c7194b4bcebb` |
| MCP-RC last product change | `b0e19bf61b346e6941b2e190b703c6922e708b34` |
| Integrated baselines | `3dd991336233cea1662518df8b35e5e1ea901e00` (PR #524), `a156cbbd5ff521dc7e99dd02ab6bebd4ca234a73` (PR #525, PDC-X `INTEGRATED_BASELINE_SHA`) |
| Audited tree (product) | `324d9dba37cb54e1b74a79a05148d2dcc05e6d09` |
| `origin/mcp-rc` at drafting | `7801dcb1293e31ee0c483184efd59c01a2dc606b` |
| Operator records | `mcp_server_x/operator_decisions/OD-MCP-RC-01.md`, `OD-MCP-RC-02.md`, `OD-MCP-RC-03.md`, `OD-MCP-RC-04.md`, `DELEGATION-MCP-RC-1.md`, `DELEGATION-MCP-RC-2.md` |
| Runbook | `mcp_server_x/MCP_SERVER_X_RC_ACTIVATION_RUNBOOK.md` |
| Precedent | `IIAC_OD_ACTIVATION_MCP_ACT_N11_P6_2026-11-11.md` |
