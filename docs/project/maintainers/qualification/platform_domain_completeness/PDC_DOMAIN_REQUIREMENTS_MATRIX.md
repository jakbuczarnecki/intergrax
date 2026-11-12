# PDC-X — domain requirements matrix (PDC-P2)

**Generated** by `scripts/qualification/pdc_x_requirements.py` from its `ROWS` source; do not edit by hand. Machine-readable twin: [`PDC_DOMAIN_REQUIREMENTS_MATRIX.json`](./PDC_DOMAIN_REQUIREMENTS_MATRIX.json) (all 24 fields). Audit head `1ebd2ab4c2e6b7b03c858f0ede172466cac1f089`. **No row is certified:** `CERTIFICATION_STATUS` = `NOT_CERTIFIED` until PDC-P7.

**Rows:** 172.

## Totals by domain and class

| Domain | ADVERSARIAL_COVERAGE_REQUIRED | ALREADY_CERTIFIED_REUSABLE | BLOCKED_EXACT_REASON | CURRENT_HEAD_REQUALIFICATION | DEFERRED_OUTSIDE_FREEZE_SCOPE | DOCUMENTATION_RECONCILIATION | IMPLEMENTATION_COMPLETION | INTEGRATION_CORRECTION | N/A_WITH_EVIDENCE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PDC-D1 | 6 | 0 | 2 | 8 | 0 | 2 | 2 | 2 | 0 |
| PDC-D2 | 5 | 1 | 1 | 7 | 0 | 0 | 1 | 2 | 1 |
| PDC-D3 | 4 | 2 | 1 | 12 | 0 | 0 | 1 | 3 | 1 |
| PDC-D4 | 4 | 1 | 3 | 4 | 0 | 2 | 4 | 2 | 0 |
| PDC-D5 | 0 | 0 | 0 | 1 | 0 | 2 | 9 | 6 | 0 |
| PDC-D6 | 7 | 1 | 2 | 7 | 0 | 3 | 3 | 4 | 1 |
| PDC-D7 | 3 | 7 | 3 | 3 | 0 | 4 | 1 | 1 | 1 |
| PDC-D8 | 3 | 0 | 1 | 1 | 0 | 0 | 8 | 5 | 1 |

## Requirements

| ID | Requirement | Status | Class | Certification @ delta | Corrective task | FRZ | Freeze blocker |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PDC-D1-R001 | One AgentContract identity and version per agent | PRESENT | CURRENT_HEAD_REQUALIFICATION | AC1-AC6-FINAL-AUDIT (34 files (agent_distribution, runtime/registry) @ 1ebd2ab4c2e6) | PDC-P7-D1 replay | FRZ-OWN-01,FRZ-CTR-01,FRZ-CMP-02 | NO |
| PDC-D1-R002 | Typed agent authoring contract | PRESENT | CURRENT_HEAD_REQUALIFICATION | AC1-AC6-FINAL-AUDIT (34 files (agent_distribution, runtime/registry) @ 1ebd2ab4c2e6) | PDC-P7-D1 replay | FRZ-TYP-01,FRZ-CTR-01 | NO |
| PDC-D1-R003 | Package identity, provenance and attestation verified before install | PRESENT | ADVERSARIAL_COVERAGE_REQUIRED | AC1-AC6-FINAL-AUDIT (34 files (agent_distribution, runtime/registry) @ 1ebd2ab4c2e6) | PDC-P5-D1 | FRZ-SEC-06,FRZ-PLG-05 | YES |
| PDC-D1-R004 | Single agent registry authority (runtime projection, not an install API) | PRESENT | ADVERSARIAL_COVERAGE_REQUIRED | AC1-AC6-FINAL-AUDIT (34 files (agent_distribution, runtime/registry) @ 1ebd2ab4c2e6) | PDC-P5-D1 | FRZ-OWN-04,FRZ-OWN-05,FRZ-REG-03 | YES |
| PDC-D1-R005 | Capability declarations and dependency resolution | PRESENT | CURRENT_HEAD_REQUALIFICATION | AC1-AC6-FINAL-AUDIT (34 files (agent_distribution, runtime/registry) @ 1ebd2ab4c2e6) | PDC-P7-D1 replay | FRZ-CTR-05,FRZ-PLG-01 | NO |
| PDC-D1-R006 | Agent assembly validation (contract + lifecycle metadata) | PRESENT | CURRENT_HEAD_REQUALIFICATION | AC1-AC6-FINAL-AUDIT (34 files (agent_distribution, runtime/registry) @ 1ebd2ab4c2e6) | PDC-P7-D1 replay | FRZ-CTR-01 | NO |
| PDC-D1-R007 | Tenant-scoped install, bind and activate | PRESENT | ADVERSARIAL_COVERAGE_REQUIRED | TENANT-X (1 file in security scope (policy/side_effect_authorization_errors.py) @ 1ebd2ab4c2e6) | PDC-P5-D1 | FRZ-TEN-04,FRZ-TEN-11 | YES |
| PDC-D1-R008 | Installed != configured != effective != authorized distinction | PRESENT | CURRENT_HEAD_REQUALIFICATION | AC1-AC6-FINAL-AUDIT (34 files (agent_distribution, runtime/registry) @ 1ebd2ab4c2e6) | PDC-P7-D1 replay | FRZ-CFG-03 | NO |
| PDC-D1-R009 | Dynamic discovery and acquisition (AC-4); architecture doc still says READY_FOR_INDEPENDENT_AUDIT | PRESENT | DOCUMENTATION_RECONCILIATION | AC1-AC6-FINAL-AUDIT (34 files (agent_distribution, runtime/registry) @ 1ebd2ab4c2e6) | PDC-P5-DOC | FRZ-DOC-01,FRZ-PLG-08 | YES |
| PDC-D1-R010 | Version-pinned runtime binding | PRESENT | CURRENT_HEAD_REQUALIFICATION | AC1-AC6-FINAL-AUDIT (34 files (agent_distribution, runtime/registry) @ 1ebd2ab4c2e6) | PDC-P7-D1 replay | FRZ-EVOL-04,FRZ-CMP-05 | NO |
| PDC-D1-R011 | UPGRADE is an explicit governed operation with a compatibility check | ABSENT | IMPLEMENTATION_COMPLETION | — | PDC-P5-D1 (ADR-PDC-01) | FRZ-CMP-05,FRZ-EVOL-07,FRZ-PLG-08 | YES |
| PDC-D1-R012 | Rollback safety | PRESENT | ADVERSARIAL_COVERAGE_REQUIRED | AC1-AC6-FINAL-AUDIT (34 files (agent_distribution, runtime/registry) @ 1ebd2ab4c2e6) | PDC-P5-D1 | FRZ-REC-02,FRZ-EVOL-07 | YES |
| PDC-D1-R013 | Revocation and emergency revocation reach active executions | PRESENT | ADVERSARIAL_COVERAGE_REQUIRED | AC1-AC6-FINAL-AUDIT (34 files (agent_distribution, runtime/registry) @ 1ebd2ab4c2e6) | PDC-P5-D1 | FRZ-SEC-06,FRZ-GOV-05 | YES |
| PDC-D1-R014 | RETIRE is an explicit operation (state exists, no operation) | PARTIAL | IMPLEMENTATION_COMPLETION | — | PDC-P5-D1 (ADR-PDC-01) | FRZ-PLG-08,FRZ-OBS-05 | YES |
| PDC-D1-R015 | Cross-host agent reuse | PRESENT | CURRENT_HEAD_REQUALIFICATION | AC1-AC6-FINAL-AUDIT (34 files (agent_distribution, runtime/registry) @ 1ebd2ab4c2e6) | PDC-P7-D1 replay | FRZ-RPL-04 | NO |
| PDC-D1-R016 | Activation continuity across restart | PRESENT | CURRENT_HEAD_REQUALIFICATION | STATE-X (3 files (long_running, persistence) @ 1ebd2ab4c2e6) | PDC-P7-D1 replay | FRZ-REC-02,FRZ-STA-06 | NO |
| PDC-D1-R017 | Every lifecycle transition emits canonical evidence | PARTIAL | ADVERSARIAL_COVERAGE_REQUIRED | — | PDC-P5-D1 | FRZ-OBS-01,FRZ-TRC-08 | YES |
| PDC-D1-R018 | Skill disable/remove/upgrade (SKILL-X F-04) kept separate from agent lifecycle | PARTIAL | BLOCKED_EXACT_REASON | SKILL-X (1 file (skills/skill_execution_pinning_wiring.py) @ 1ebd2ab4c2e6) | OD-PDC-SCOPE-01 (implement vs authorized exclusion) | FRZ-PLG-08,FRZ-DEBT-05 | YES |
| PDC-D1-R019 | Runtime factory beyond VENV_BUNDLE (OCI/sidecar deferred) | PARTIAL | BLOCKED_EXACT_REASON | — | OD-PDC-SCOPE-03 (authorized exclusion proposed) | FRZ-DEBT-05,FRZ-DEBT-06 | YES |
| PDC-D1-R020 | One owner for plugin discovery (core/plugins/discovery.py vs vendor_knowledge and rag entry-point loaders) | PARTIAL | INTEGRATION_CORRECTION | — | PDC-P3-01 ownership decision -> PDC-P5-D1 | FRZ-PLG-03,FRZ-OWN-05 | YES |
| PDC-D1-R021 | plug_03/plug_04/plug_final suites have no owning roadmap stage | PRESENT | DOCUMENTATION_RECONCILIATION | — | PDC-P3-01 ownership record | FRZ-REG-05,FRZ-DOC-03 | YES |
| PDC-D2-R001 | Source identity and context provenance | PRESENT | CURRENT_HEAD_REQUALIFICATION | HARNESS-RESIDUAL/CE-01 (2 files (context tracking, nexus context_manager) @ 1ebd2ab4c2e6) | PDC-P7-D2 replay | FRZ-TRC-05 | NO |
| PDC-D2-R002 | Tenant/workspace/session isolation of memory, RAG and context | PRESENT | ALREADY_CERTIFIED_REUSABLE | TENANT-X (1 file in security scope (policy/side_effect_authorization_errors.py) @ 1ebd2ab4c2e6) | — | FRZ-TEN-04,FRZ-SEC-01 | NO |
| PDC-D2-R003 | Memory lifecycle has one owner | PRESENT | CURRENT_HEAD_REQUALIFICATION | EBH-2H (13 files (intergrax/memory) @ 1ebd2ab4c2e6) | PDC-P7-D2 replay | FRZ-OWN-01,FRZ-STA-01 | NO |
| PDC-D2-R004 | RAG retrieval correctness | PRESENT | CURRENT_HEAD_REQUALIFICATION | EBH-2G (53 files (intergrax/rag) @ 1ebd2ab4c2e6) | PDC-P7-D2 replay | FRZ-CTR-01,FRZ-TYP-01 | NO |
| PDC-D2-R005 | Index consistency and source freshness (only an embedding-version warning + optional reindex hook) | PARTIAL | BLOCKED_EXACT_REASON | EBH-2G (53 files (intergrax/rag) @ 1ebd2ab4c2e6) | OD-PDC-SCOPE-06 (freshness policy contract vs authorized exclusion) | FRZ-STA-05,FRZ-DEBT-04 | YES |
| PDC-D2-R006 | Context assembly ownership | PRESENT | CURRENT_HEAD_REQUALIFICATION | HARNESS-RESIDUAL/CE-01 (2 files (context tracking, nexus context_manager) @ 1ebd2ab4c2e6) | PDC-P7-D2 replay | FRZ-OWN-01 | NO |
| PDC-D2-R007 | Mandatory instructions survive compaction and degradation | PRESENT | ADVERSARIAL_COVERAGE_REQUIRED | — | PDC-P5-D2 | FRZ-CTR-01,FRZ-REG-03 | YES |
| PDC-D2-R008 | Retrieval-poisoning filter consumed on both paths; RAG service fallback is fail-open (False when profile absent) | PARTIAL | INTEGRATION_CORRECTION | — | PDC-P5-D6 (shared with PDC-D6-R014) | FRZ-SEC-08,FRZ-CFG-02 | YES |
| PDC-D2-R009 | Tool-output ingestion boundary | PRESENT | ADVERSARIAL_COVERAGE_REQUIRED | — | PDC-P5-D2 | FRZ-SEC-08 | YES |
| PDC-D2-R010 | Token budget and hard model-window enforcement (CE-02 has no closure record) | PRESENT | CURRENT_HEAD_REQUALIFICATION | — | PDC-P7-D2 replay of ce_02 (CE-02 closure via PDC) | FRZ-CTR-01,FRZ-REL-10 | NO |
| PDC-D2-R011 | Degradation step TOKENIZER_HARD_TRIM drops entry_id/tool_calls/tool_call_id | PARTIAL | INTEGRATION_CORRECTION | — | PDC-P5-D2 (ADR-PDC-02) | FRZ-TRC-05,FRZ-REC-04 | YES |
| PDC-D2-R012 | Durable compaction semantics (CE-02 PARTIAL in gap audit) | PARTIAL | ADVERSARIAL_COVERAGE_REQUIRED | — | PDC-P5-D2 (ADR-PDC-02) | FRZ-REC-04,FRZ-STA-06 | YES |
| PDC-D2-R013 | Reconstruction and session continuity after restart | PRESENT | CURRENT_HEAD_REQUALIFICATION | STATE-X (3 files (long_running, persistence) @ 1ebd2ab4c2e6) | PDC-P7-D2 replay | FRZ-REC-01,FRZ-TRC-09 | NO |
| PDC-D2-R014 | Sensitive data in context is classified and redacted | PARTIAL | ADVERSARIAL_COVERAGE_REQUIRED | — | PDC-P5-D2 | FRZ-SEC-08 | YES |
| PDC-D2-R015 | Memory retention enforcement (retention is decision-only, no TTL; MEMORY-DURABILITY-LIFECYCLE planned) | PARTIAL | IMPLEMENTATION_COMPLETION | EBH-2H (13 files (intergrax/memory) @ 1ebd2ab4c2e6) | PDC-P5-D2 (ADR-PDC-02) | FRZ-SEC-09 | YES |
| PDC-D2-R016 | Citation and evidence fidelity | PRESENT | CURRENT_HEAD_REQUALIFICATION | — | PDC-P7-D2 replay | FRZ-TRC-05 | NO |
| PDC-D2-R017 | ContextFragment name used by two different concepts (context contracts vs token-optimization pack) | PRESENT | N/A_WITH_EVIDENCE | — | — | FRZ-OWN-02 | NO |
| PDC-D2-R018 | Compaction never deletes audit evidence | PRESENT | ADVERSARIAL_COVERAGE_REQUIRED | — | PDC-P6 (XD-09) | FRZ-AUD-05,FRZ-REC-10 | YES |
| PDC-D3-R001 | Typed provider-neutral adapter ABI | PRESENT | CURRENT_HEAD_REQUALIFICATION | EBH-2E (9 files (llm_adapters, runtime/llm) @ 1ebd2ab4c2e6) | PDC-P7-D3 replay | FRZ-CTR-01,FRZ-CTR-06 | NO |
| PDC-D3-R002 | ModelCatalog capability metadata | PRESENT | CURRENT_HEAD_REQUALIFICATION | EBH-5 (0 files (llm_adapters); replay RED (ebh_5 239p/4f/0s) @ 1ebd2ab4c2e6) | PDC-P7-D3 replay after PDC-D3-R024 | FRZ-PLG-01 | NO |
| PDC-D3-R003 | Configuration-driven provider selection | PRESENT | CURRENT_HEAD_REQUALIFICATION | EBH-5 (0 files (llm_adapters); replay RED (ebh_5 239p/4f/0s) @ 1ebd2ab4c2e6) | PDC-P7-D3 replay after PDC-D3-R024 | FRZ-CFG-01,FRZ-PLG-04 | NO |
| PDC-D3-R004 | Routing and failover bounded | PRESENT | ADVERSARIAL_COVERAGE_REQUIRED | EBH-2E (9 files (llm_adapters, runtime/llm) @ 1ebd2ab4c2e6) | PDC-P5-D3 | FRZ-REL-07,FRZ-REL-11 | YES |
| PDC-D3-R005 | Model version pinning and runtime attribution | PRESENT | CURRENT_HEAD_REQUALIFICATION | EBH-2E (9 files (llm_adapters, runtime/llm) @ 1ebd2ab4c2e6) | PDC-P7-D3 replay | FRZ-TRC-04,FRZ-GOV-08 | NO |
| PDC-D3-R006 | Streaming / tool-call streaming parity or explicit capability fail-closed (Ollama adapters lack stream_with_tools) | PARTIAL | IMPLEMENTATION_COMPLETION | — | PDC-P5-D3 (ADR-PDC-03) | FRZ-CTR-01,FRZ-REL-06 | YES |
| PDC-D3-R007 | Structured outputs | PRESENT | ALREADY_CERTIFIED_REUSABLE | EBH-2E (9 files (llm_adapters, runtime/llm) @ 1ebd2ab4c2e6) | — | FRZ-TYP-01 | NO |
| PDC-D3-R008 | Tool-call normalization | PRESENT | CURRENT_HEAD_REQUALIFICATION | EBH-2E (9 files (llm_adapters, runtime/llm) @ 1ebd2ab4c2e6) | PDC-P7-D3 replay | FRZ-TYP-01 | NO |
| PDC-D3-R009 | Refusal semantics: typed finish reason exists, refusal payload is a string | PRESENT | ADVERSARIAL_COVERAGE_REQUIRED | — | PDC-P5-D3 | FRZ-REL-06,FRZ-TYP-05 | YES |
| PDC-D3-R010 | Usage, token and cost accounting | PRESENT | CURRENT_HEAD_REQUALIFICATION | EBH-2E (9 files (llm_adapters, runtime/llm) @ 1ebd2ab4c2e6) | PDC-P7-D3 replay | FRZ-OBS-06 | NO |
| PDC-D3-R011 | Cancellation explicit per provider (seam beside the ABI; Ollama has no native remote cancel) | PRESENT | ADVERSARIAL_COVERAGE_REQUIRED | EBH-2E (9 files (llm_adapters, runtime/llm) @ 1ebd2ab4c2e6) | PDC-P5-D3 | FRZ-REL-04,FRZ-REL-05 | YES |
| PDC-D3-R012 | Timeouts | PRESENT | CURRENT_HEAD_REQUALIFICATION | EBH-2E (9 files (llm_adapters, runtime/llm) @ 1ebd2ab4c2e6) | PDC-P7-D3 replay | FRZ-REL-01 | NO |
| PDC-D3-R013 | Provider error taxonomy | PRESENT | CURRENT_HEAD_REQUALIFICATION | EBH-2E (9 files (llm_adapters, runtime/llm) @ 1ebd2ab4c2e6) | PDC-P7-D3 replay | FRZ-REL-06 | NO |
| PDC-D3-R014 | Context-window enforcement | PRESENT | CURRENT_HEAD_REQUALIFICATION | — | PDC-P7-D3 replay (ce_02) | FRZ-REL-10 | NO |
| PDC-D3-R015 | Capability mismatch fails closed | PARTIAL | ADVERSARIAL_COVERAGE_REQUIRED | — | PDC-P5-D3 | FRZ-CFG-02 | YES |
| PDC-D3-R016 | Provider replaceability | PRESENT | CURRENT_HEAD_REQUALIFICATION | EBH-5 (0 files (llm_adapters); replay RED (ebh_5 239p/4f/0s) @ 1ebd2ab4c2e6) | PDC-P7-D3 replay after PDC-D3-R024 | FRZ-RPL-01,FRZ-RPL-04 | NO |
| PDC-D3-R017 | Local/offline models and native Ollama | PRESENT | CURRENT_HEAD_REQUALIFICATION | EBH-2E (9 files (llm_adapters, runtime/llm) @ 1ebd2ab4c2e6) | PDC-P7-D3 replay; live parity = Path B | FRZ-RPL-02 | NO |
| PDC-D3-R018 | Tenant-aware model configuration | PRESENT | ALREADY_CERTIFIED_REUSABLE | TENANT-X (1 file in security scope (policy/side_effect_authorization_errors.py) @ 1ebd2ab4c2e6) | — | FRZ-TEN-05 | NO |
| PDC-D3-R019 | No vendor selection hidden in core (provider DEFAULT_MODEL constants, LLMProfile.lab, rag default embedding provider) | PRESENT | CURRENT_HEAD_REQUALIFICATION | EBH-5 (0 files (llm_adapters) @ 1ebd2ab4c2e6) | PDC-P7-D3 against VBN CORE_HARDCODING_FINDINGS_REGISTER | FRZ-CFG-08,FRZ-CTR-06 | NO |
| PDC-D3-R020 | Provider plugin registration layer (LLM-PROVIDER-PLUGIN-1 Planned) | ABSENT | BLOCKED_EXACT_REASON | — | OD-PDC-SCOPE-03 (authorized exclusion proposed) | FRZ-PLG-02,FRZ-DEBT-05 | YES |
| PDC-D3-R021 | Live provider qualification | PRESENT | N/A_WITH_EVIDENCE | — | — | FRZ-PRD-01 | NO |
| PDC-D3-R022 | Streaming capability flags misreport: six adapters stream but supports_streaming() returns False | PARTIAL | INTEGRATION_CORRECTION | — | PDC-P5-D3 (ADR-PDC-03) | FRZ-CTR-01,FRZ-REG-09 | YES |
| PDC-D3-R023 | Context-window tables duplicated inside providers and ModelCatalog | PARTIAL | INTEGRATION_CORRECTION | — | PDC-P3-01 ownership decision -> PDC-P5-D3 | FRZ-OWN-05 | YES |
| PDC-D4-R001 | Canonical ApplicationEnvironmentProfile and HostedApplicationProfile | PRESENT | CURRENT_HEAD_REQUALIFICATION | HARNESS-W7/HOST-01 (1 file (applications/_shared/environment_wiring.py) @ 1ebd2ab4c2e6) | PDC-P7-D4 replay | FRZ-CTR-01,FRZ-OWN-02 | NO |
| PDC-D4-R002 | Single effective composition root | PRESENT | CURRENT_HEAD_REQUALIFICATION | HARNESS-W7/HOST-01 (1 file (applications/_shared/environment_wiring.py) @ 1ebd2ab4c2e6) | PDC-P7-D4 replay | FRZ-OWN-03,FRZ-BND-05 | NO |
| PDC-D4-R003 | Startup dependency ordering and rollback | PRESENT | ADVERSARIAL_COVERAGE_REQUIRED | — | PDC-P5-D4 | FRZ-REL-10,FRZ-OBS-05 | YES |
| PDC-D4-R004 | Readiness and degraded states | PRESENT | ADVERSARIAL_COVERAGE_REQUIRED | — | PDC-P5-D4 | FRZ-OBS-05 | YES |
| PDC-D4-R005 | Graceful shutdown with in-flight drain | PRESENT | ADVERSARIAL_COVERAGE_REQUIRED | — | PDC-P5-D4 | FRZ-REL-04 | YES |
| PDC-D4-R006 | Instance ownership guard | PRESENT | ADVERSARIAL_COVERAGE_REQUIRED | — | PDC-P5-D4 | FRZ-STA-01 | YES |
| PDC-D4-R007 | Restart supervision with an OS-process crash proof (APP-HOST-5E Planned) | PARTIAL | IMPLEMENTATION_COMPLETION | — | PDC-P5-D4 | FRZ-REC-01,FRZ-REC-02 | YES |
| PDC-D4-R008 | Failure classification | PRESENT | CURRENT_HEAD_REQUALIFICATION | — | PDC-P7-D4 replay | FRZ-REL-06 | NO |
| PDC-D4-R009 | Hosting lifecycle metrics through canonical observability (APP-HOST-3D; no metrics today) | ABSENT | IMPLEMENTATION_COMPLETION | — | PDC-P5-D4 (ADR-PDC-04) | FRZ-OBS-01,FRZ-OBS-05 | YES |
| PDC-D4-R010 | Restart preserves configuration/profile digest generically (APP-HOST-5D) | PARTIAL | IMPLEMENTATION_COMPLETION | — | PDC-P5-D4 (ADR-PDC-04) | FRZ-REC-02,FRZ-TRC-11 | YES |
| PDC-D4-R011 | InteractionProfile facade (APP-HOST-6; no class exists) | ABSENT | BLOCKED_EXACT_REASON | — | OD-PDC-SCOPE-02 (authorized exclusion proposed; ApplicationEnvironmentProfile authority PRESERVE) | FRZ-DEBT-05,FRZ-DEBT-06 | YES |
| PDC-D4-R012 | OS service-manager adapters (APP-HOST-7) | ABSENT | BLOCKED_EXACT_REASON | — | OD-PDC-SCOPE-02 (authorized exclusion proposed) | FRZ-DEBT-05 | YES |
| PDC-D4-R013 | Host plugins in HostedApplicationProfile (APP-HOST-9E) | PARTIAL | BLOCKED_EXACT_REASON | — | OD-PDC-SCOPE-02 (authorized exclusion proposed) | FRZ-PLG-01,FRZ-DEBT-05 | YES |
| PDC-D4-R014 | Stable public hosting API declaration (APP-HOST-9F) | ABSENT | IMPLEMENTATION_COMPLETION | — | PDC-P5-D4 (ADR-PDC-04) | FRZ-CTR-04,FRZ-DOC-05,FRZ-CMP-01 | YES |
| PDC-D4-R015 | Generic HostDeploymentProfile carries LKW-specific fields (TL-FIX-C) | PARTIAL | INTEGRATION_CORRECTION | — | PDC-P5-D4 | FRZ-BND-04,FRZ-CTR-06 | YES |
| PDC-D4-R016 | TL-FIX-D implemented in code (bind_execution_adapter) but plan row still ACCEPTED/PLANNED | PRESENT | DOCUMENTATION_RECONCILIATION | — | PDC-P5-DOC | FRZ-DOC-01 | YES |
| PDC-D4-R017 | No host-owned execution authority | PRESENT | CURRENT_HEAD_REQUALIFICATION | HARNESS-W7/HOST-01 (1 file (applications/_shared/environment_wiring.py) @ 1ebd2ab4c2e6) | PDC-P7-D4 replay | FRZ-EXE-01,FRZ-EXE-02 | NO |
| PDC-D4-R018 | Tenant-specific host configuration | PRESENT | ALREADY_CERTIFIED_REUSABLE | TENANT-X (1 file in security scope (policy/side_effect_authorization_errors.py) @ 1ebd2ab4c2e6) | — | FRZ-TEN-05 | NO |
| PDC-D4-R019 | Sandbox/shadow app isolation is not a universal production guarantee; plan marks APP-CON-8 Done | PARTIAL | DOCUMENTATION_RECONCILIATION | — | PDC-P5-DOC | FRZ-SEC-07,FRZ-DOC-02 | YES |
| PDC-D5-R001 | Typed EvalDataset / EvalScenario / EvalRun contracts | ABSENT | IMPLEMENTATION_COMPLETION | — | PDC-P5-D5 (ADR-PDC-05) | FRZ-CTR-01,FRZ-TYP-01 | YES |
| PDC-D5-R002 | Dataset versioning with content digest (GoldenDatasetAsset has free-string version) | PARTIAL | IMPLEMENTATION_COMPLETION | CTRL-X (0 files (eval, experiments, evaluation scope) @ 1ebd2ab4c2e6) | PDC-P5-D5 (ADR-PDC-05) | FRZ-CMP-03,FRZ-REG-06 | YES |
| PDC-D5-R003 | Evaluation scenario identity | PARTIAL | IMPLEMENTATION_COMPLETION | — | PDC-P5-D5 (ADR-PDC-05) | FRZ-TYP-01 | YES |
| PDC-D5-R004 | Agent/model/prompt/profile/code-SHA references on every run (EvalResult lacks them) | PARTIAL | IMPLEMENTATION_COMPLETION | — | PDC-P5-D5 (ADR-PDC-05) | FRZ-TRC-04,FRZ-TRC-05 | YES |
| PDC-D5-R005 | Task-success, tool-use, groundedness and citation evaluators behind one contract | PARTIAL | INTEGRATION_CORRECTION | — | PDC-P5-D5 (ADR-PDC-05) | FRZ-OWN-05,FRZ-PLG-01 | YES |
| PDC-D5-R006 | Safety evaluation hook (no safety/harm evaluator exists) | ABSENT | IMPLEMENTATION_COMPLETION | — | PDC-P5-D5 (ADR-PDC-05) | FRZ-SEC-07 | YES |
| PDC-D5-R007 | Deterministic vs model-assisted evaluator separation | PARTIAL | INTEGRATION_CORRECTION | — | PDC-P5-D5 (ADR-PDC-05) | FRZ-TYP-05 | YES |
| PDC-D5-R008 | Human review queue | PRESENT | CURRENT_HEAD_REQUALIFICATION | CTRL-X (0 files (eval, experiments, evaluation scope) @ 1ebd2ab4c2e6) | PDC-P7-D5 replay | FRZ-GOV-06 | NO |
| PDC-D5-R009 | Reproducible evaluation runs | PARTIAL | IMPLEMENTATION_COMPLETION | — | PDC-P5-D5 (ADR-PDC-05) | FRZ-REG-06 | YES |
| PDC-D5-R010 | Baseline vs candidate comparison (only a pass-rate delta exists) | PARTIAL | IMPLEMENTATION_COMPLETION | — | PDC-P5-D5 (ADR-PDC-05) | FRZ-EVOL-06 | YES |
| PDC-D5-R011 | Quality regression detection | PARTIAL | IMPLEMENTATION_COMPLETION | — | PDC-P5-D5 (ADR-PDC-05) | FRZ-EVOL-06 | YES |
| PDC-D5-R012 | Cost-to-success and latency/quality comparison | PARTIAL | IMPLEMENTATION_COMPLETION | — | PDC-P5-D5 (ADR-PDC-05) | FRZ-OBS-06 | YES |
| PDC-D5-R013 | Evaluation provenance in the canonical evidence store | PARTIAL | INTEGRATION_CORRECTION | — | PDC-P5-D5 (ADR-PDC-05) | FRZ-OBS-01,FRZ-AUD-05 | YES |
| PDC-D5-R014 | Promotion gates G0-G2 are file-existence checks; evaluation must stay advisory and never authorize | PARTIAL | INTEGRATION_CORRECTION | — | PDC-P5-D5 (ADR-PDC-05) | FRZ-GOV-01,FRZ-GOV-02,FRZ-REG-09 | YES |
| PDC-D5-R015 | tenant_id on OnlineEvaluationObservation (TENANT-X debt recorded by CTRL-X) | PARTIAL | INTEGRATION_CORRECTION | CTRL-X (0 files (eval, experiments, evaluation scope) @ 1ebd2ab4c2e6) | PDC-P5-D5 | FRZ-TEN-01,FRZ-TEN-07 | YES |
| PDC-D5-R016 | CRITIC_VERIFICATION findings are legacy lineage; canonical authority is DECISION_SYSTEM | PRESENT | DOCUMENTATION_RECONCILIATION | — | PDC-P5-DOC | FRZ-DOC-02 | YES |
| PDC-D5-R017 | Production-feedback boundary of evaluation | PARTIAL | DOCUMENTATION_RECONCILIATION | — | PDC-P5-DOC (ADR-PDC-05) | FRZ-DOC-01 | YES |
| PDC-D5-R018 | File-based evaluation registries have no concurrency semantics (DX-07) | PARTIAL | INTEGRATION_CORRECTION | — | PDC-P5-D5 (ADR-PDC-05) | FRZ-STA-03 | YES |
| PDC-D6-R001 | Principal and workload identity | PRESENT | CURRENT_HEAD_REQUALIFICATION | TENANT-X (1 file in security scope (policy/side_effect_authorization_errors.py) @ 1ebd2ab4c2e6) | PDC-P7-D6 replay | FRZ-TEN-01 | NO |
| PDC-D6-R002 | Authentication (architecture scope) | PRESENT | CURRENT_HEAD_REQUALIFICATION | — | PDC-P7-D6 replay; live = Path B | FRZ-SEC-04 | NO |
| PDC-D6-R003 | Single authorization / policy authority | PRESENT | CURRENT_HEAD_REQUALIFICATION | GOV-X2 (16 files (runtime/governance, runtime/policy, delegation) @ 1ebd2ab4c2e6) | PDC-P7-D6 replay | FRZ-GOV-01,FRZ-OWN-04 | NO |
| PDC-D6-R004 | Delegation and authority narrowing | PRESENT | ADVERSARIAL_COVERAGE_REQUIRED | GOV-X2 (16 files (runtime/governance, runtime/policy, delegation) @ 1ebd2ab4c2e6) | PDC-P5-D6 | FRZ-GOV-03,FRZ-GOV-04 | YES |
| PDC-D6-R005 | Tenant isolation evidence for FRZ-SEC-01 | PRESENT | ALREADY_CERTIFIED_REUSABLE | TENANT-X (1 file in security scope (policy/side_effect_authorization_errors.py) @ 1ebd2ab4c2e6) | — | FRZ-SEC-01,FRZ-TEN-12 | NO |
| PDC-D6-R006 | Secret references with late resolution | PRESENT | ADVERSARIAL_COVERAGE_REQUIRED | — | PDC-P5-D6 | FRZ-SEC-02,FRZ-TEN-06 | YES |
| PDC-D6-R007 | Raw secrets never cross layers, logs or traces (no static gate exists) | PARTIAL | IMPLEMENTATION_COMPLETION | — | PDC-P5-D6 (ADR-PDC-06) | FRZ-SEC-03,FRZ-REG-09 | YES |
| PDC-D6-R008 | Critical-action signing falls back to literal 'harness-dev-signing-key' (SEC-BND-05) | PARTIAL | INTEGRATION_CORRECTION | — | PDC-P5-D6 (ADR-PDC-06) | FRZ-CFG-02,FRZ-SEC-02 | YES |
| PDC-D6-R009 | Credential scope and revocation | PRESENT | ADVERSARIAL_COVERAGE_REQUIRED | TENANT-X (1 file in security scope (policy/side_effect_authorization_errors.py) @ 1ebd2ab4c2e6) | PDC-P5-D6 | FRZ-SEC-02,FRZ-TEN-06 | YES |
| PDC-D6-R010 | Sandbox enforcement | PRESENT | CURRENT_HEAD_REQUALIFICATION | — | PDC-P7-D6 replay (AW-7C PHYSQ scoped); provider live = Path B | FRZ-SEC-05 | NO |
| PDC-D6-R011 | Filesystem and process isolation | PRESENT | CURRENT_HEAD_REQUALIFICATION | — | PDC-P7-D6 replay | FRZ-SEC-05 | NO |
| PDC-D6-R012 | Network egress controls | PRESENT | CURRENT_HEAD_REQUALIFICATION | — | PDC-P7-D6 replay | FRZ-SEC-05 | NO |
| PDC-D6-R013 | Tool-input injection guard assert_tool_input_safe has zero consumers | PARTIAL | INTEGRATION_CORRECTION | — | PDC-P5-D6 (ADR-PDC-06) | FRZ-OWN-05,FRZ-SEC-08 | YES |
| PDC-D6-R014 | Retrieval-poisoning defense must fail closed when the profile is absent (SEC-BND-04) | PARTIAL | INTEGRATION_CORRECTION | — | PDC-P5-D6 (ADR-PDC-06) | FRZ-SEC-08,FRZ-CFG-02 | YES |
| PDC-D6-R015 | Data exfiltration / propagation across provider boundaries | PARTIAL | ADVERSARIAL_COVERAGE_REQUIRED | — | PDC-P5-D6 | FRZ-SEC-08 | YES |
| PDC-D6-R016 | Plugin entry-point admission is structural only (no signature/provenance); agent packages are attested | PARTIAL | BLOCKED_EXACT_REASON | — | OD-PDC-SCOPE-07 (plugin provenance enforcement vs authorized scope) | FRZ-SEC-06,FRZ-PLG-05 | YES |
| PDC-D6-R017 | MCP exposure security | PRESENT | BLOCKED_EXACT_REASON | — | await MCP-SERVER-X P8 certification | FRZ-SEC-04,FRZ-TEN-11 | YES |
| PDC-D6-R018 | Side-effect governance | PRESENT | CURRENT_HEAD_REQUALIFICATION | GOV-X2 (16 files (runtime/governance, runtime/policy, delegation) @ 1ebd2ab4c2e6) | PDC-P7-D6 replay | FRZ-GOV-05,FRZ-TRC-06 | NO |
| PDC-D6-R019 | Cross-tenant negative tests across D1-D8 | ABSENT | ADVERSARIAL_COVERAGE_REQUIRED | TENANT-X (1 file in security scope (policy/side_effect_authorization_errors.py) @ 1ebd2ab4c2e6) | PDC-P6 (XD-02/06/10) | FRZ-TEN-12,FRZ-SEC-01 | YES |
| PDC-D6-R020 | Security audit trail is in-memory only (SEC-BND-06) | PARTIAL | IMPLEMENTATION_COMPLETION | — | PDC-P5-D6 (ADR-PDC-06) | FRZ-AUD-05,FRZ-OBS-01 | YES |
| PDC-D6-R021 | Consolidated security architecture and threat model (no doc pair; broken IDENTITY_TRUST.md link) | PARTIAL | DOCUMENTATION_RECONCILIATION | — | PDC-P5-DOC (ADR-PDC-06) | FRZ-SEC-07,FRZ-DOC-02 | YES |
| PDC-D6-R022 | Retention/deletion/redaction responsibility explicit | PARTIAL | IMPLEMENTATION_COMPLETION | — | PDC-P5-D8 + PDC-P5-D2 (ADR-PDC-08, ADR-PDC-02) | FRZ-SEC-09 | YES |
| PDC-D6-R023 | Encryption / residency responsibility matrix | ABSENT | DOCUMENTATION_RECONCILIATION | — | PDC-P5-DOC (ADR-PDC-06) | FRZ-SEC-10 | YES |
| PDC-D6-R024 | Fail-closed security defaults | PARTIAL | ADVERSARIAL_COVERAGE_REQUIRED | — | PDC-P5-D6 | FRZ-CFG-02 | YES |
| PDC-D6-R025 | HARNESS-FINAL 'ENTERPRISE QUALIFIED' for plugin trust/secrets/security contradicts FRZ-SEC OPEN | PRESENT | DOCUMENTATION_RECONCILIATION | — | PDC-P5-DOC | FRZ-DOC-02 | YES |
| PDC-D6-R026 | Production auth-bypass live proof | PRESENT | N/A_WITH_EVIDENCE | — | — | FRZ-SEC-04 | NO |
| PDC-D6-R027 | Demo attestation signing seed reachable only in lab/proof paths | PRESENT | ADVERSARIAL_COVERAGE_REQUIRED | — | PDC-P5-D6 | FRZ-CFG-05,FRZ-SEC-02 | YES |
| PDC-D7-R001 | Task definition and queue contract | PRESENT | ALREADY_CERTIFIED_REUSABLE | HARNESS-W8/BG-01 (0 files (queueing, background_tasks, background_execution) @ 1ebd2ab4c2e6) | — | FRZ-CTR-01 | NO |
| PDC-D7-R002 | Worker admission through canonical execution | PRESENT | ALREADY_CERTIFIED_REUSABLE | HARNESS-W8/BG-01 (0 files (queueing, background_tasks, background_execution) @ 1ebd2ab4c2e6) | — | FRZ-EXE-02,FRZ-EXE-07 | NO |
| PDC-D7-R003 | Durable dispatch | PRESENT | ALREADY_CERTIFIED_REUSABLE | HARNESS-W8/BG-01 (0 files (queueing, background_tasks, background_execution) @ 1ebd2ab4c2e6) | — | FRZ-REC-01 | NO |
| PDC-D7-R004 | One-shot delayed execution | PRESENT | CURRENT_HEAD_REQUALIFICATION | HARNESS-RESIDUAL/SCHED-01 (9 files (runtime/long_running) @ 1ebd2ab4c2e6) | PDC-P7-D7 replay | FRZ-EXE-07 | NO |
| PDC-D7-R005 | Queue-level delay/ETA (TaskRequest has none; delayed work owned by SchedulerLedger) | ABSENT | N/A_WITH_EVIDENCE | HARNESS-RESIDUAL/SCHED-01 (9 files (runtime/long_running) @ 1ebd2ab4c2e6) | — | FRZ-OWN-01 | NO |
| PDC-D7-R006 | Recurrence (interval/cron) as occurrences on SchedulerLedger; no second engine | ABSENT | BLOCKED_EXACT_REASON | — | OD-PDC-SCOPE-05 (implement via ADR-PDC-07 vs authorized exclusion) | FRZ-EXE-07,FRZ-DEBT-05 | YES |
| PDC-D7-R007 | Time-zone and DST semantics for recurrence | ABSENT | BLOCKED_EXACT_REASON | — | OD-PDC-SCOPE-05 | FRZ-DEBT-05 | YES |
| PDC-D7-R008 | Recurrence misfire / catch-up policy | PARTIAL | BLOCKED_EXACT_REASON | — | OD-PDC-SCOPE-05 | FRZ-DEBT-05 | YES |
| PDC-D7-R009 | Aggregate retry budget | PRESENT | ADVERSARIAL_COVERAGE_REQUIRED | TOOL-LIFE-X (1 file (long_running/postgresql_task_checkpoint_store.py) @ 1ebd2ab4c2e6) | PDC-P5-D7 | FRZ-REL-02,FRZ-REL-07 | YES |
| PDC-D7-R010 | Idempotency and duplicate handling | PRESENT | ALREADY_CERTIFIED_REUSABLE | HARNESS-W8/BG-01 (0 files (queueing, background_tasks, background_execution) @ 1ebd2ab4c2e6) | — | FRZ-REL-03 | NO |
| PDC-D7-R011 | At-least-once delivery consequences documented | PARTIAL | DOCUMENTATION_RECONCILIATION | — | PDC-P5-DOC | FRZ-REL-03,FRZ-DOC-01 | YES |
| PDC-D7-R012 | Lease/fencing proven across processes (today: two schedulers in one process) | PRESENT | ADVERSARIAL_COVERAGE_REQUIRED | HARNESS-RESIDUAL/SCHED-01 (9 files (runtime/long_running) @ 1ebd2ab4c2e6) | PDC-P5-D7 | FRZ-REL-09,FRZ-STA-03 | YES |
| PDC-D7-R013 | Dead-letter / terminal-failed semantics (no DLQ anywhere) | ABSENT | IMPLEMENTATION_COMPLETION | — | PDC-P5-D7 (ADR-PDC-07) | FRZ-REL-07,FRZ-REL-09 | YES |
| PDC-D7-R014 | Graceful worker shutdown with host drain | PRESENT | ADVERSARIAL_COVERAGE_REQUIRED | — | PDC-P6 (XD-05) | FRZ-REL-04 | YES |
| PDC-D7-R015 | Crash recovery and resume | PRESENT | CURRENT_HEAD_REQUALIFICATION | STATE-X (3 files (long_running, persistence) @ 1ebd2ab4c2e6) | PDC-P7-D7 replay | FRZ-REC-01,FRZ-REC-03 | NO |
| PDC-D7-R016 | Full execution identity carried on enqueue (adoption gap) | PARTIAL | INTEGRATION_CORRECTION | — | PDC-P5-D7 | FRZ-EXE-06,FRZ-TRC-12 | YES |
| PDC-D7-R017 | Tenant scope of background work | PRESENT | ALREADY_CERTIFIED_REUSABLE | HARNESS-W8/BG-01 (0 files (queueing, background_tasks, background_execution) @ 1ebd2ab4c2e6) | — | FRZ-TEN-08 | NO |
| PDC-D7-R018 | Governance of background work | PRESENT | ALREADY_CERTIFIED_REUSABLE | HARNESS-W8/BG-01 (0 files (queueing, background_tasks, background_execution) @ 1ebd2ab4c2e6) | — | FRZ-GOV-05 | NO |
| PDC-D7-R019 | Observability of background work | PRESENT | CURRENT_HEAD_REQUALIFICATION | HARNESS-W8/BG-01 (0 files (queueing, background_tasks, background_execution) @ 1ebd2ab4c2e6) | PDC-P7-D7 replay | FRZ-OBS-05 | NO |
| PDC-D7-R020 | Operational control (list/cancel/purge) | PRESENT | ALREADY_CERTIFIED_REUSABLE | HARNESS-W8/BG-01 (0 files (queueing, background_tasks, background_execution) @ 1ebd2ab4c2e6) | — | FRZ-REL-04 | NO |
| PDC-D7-R021 | Horizontal scaling boundary documented (DIST-SCALE-X scoped) | PARTIAL | DOCUMENTATION_RECONCILIATION | — | PDC-P5-DOC | FRZ-REL-08 | YES |
| PDC-D7-R022 | TaskRegistry/WorkerRuntime wraps TaskExecutionRegistry (layered, not duplicate); ownership unrecorded | PRESENT | DOCUMENTATION_RECONCILIATION | — | PDC-P3-01 ownership record | FRZ-OWN-01,FRZ-DOC-03 | YES |
| PDC-D7-R023 | BACKGROUND_TASKS plan statuses stale; doc not registered in runtime hub (PDC-P1-F01) | PRESENT | DOCUMENTATION_RECONCILIATION | — | PDC-P5-DOC | FRZ-DOC-01,FRZ-DOC-03 | YES |
| PDC-D8-R001 | Artifact taxonomy (business / execution evidence / context / temporary tool output) with one owner per class; no global storage engine | ABSENT | IMPLEMENTATION_COMPLETION | — | PDC-P3 ADR-PDC-08 -> PDC-P5-D8 | FRZ-OWN-01,FRZ-DOC-03 | YES |
| PDC-D8-R002 | Mandatory content digest on durable artifact refs (optional outside Collaborative Work) | PARTIAL | IMPLEMENTATION_COMPLETION | — | PDC-P5-D8 (ADR-PDC-08) | FRZ-REC-10,FRZ-AUD-05 | YES |
| PDC-D8-R003 | Versioning (business artifacts certified; run artifacts unversioned) | PARTIAL | IMPLEMENTATION_COMPLETION | — | PDC-P5-D8 (ADR-PDC-08) | FRZ-CMP-03 | YES |
| PDC-D8-R004 | Producer identity and execution lineage | PARTIAL | INTEGRATION_CORRECTION | — | PDC-P5-D8 | FRZ-TRC-01,FRZ-TRC-03 | YES |
| PDC-D8-R005 | Tenant/workspace scope of artifacts | PARTIAL | ADVERSARIAL_COVERAGE_REQUIRED | — | PDC-P5-D8 | FRZ-TEN-04 | YES |
| PDC-D8-R006 | Governed artifact read/access authorization port (store explicitly has no security) | ABSENT | IMPLEMENTATION_COMPLETION | — | PDC-P5-D8 (ADR-PDC-08) | FRZ-GOV-05,FRZ-SEC-08 | YES |
| PDC-D8-R007 | Durable execution artifact store (only InMemoryArtifactStore) | PARTIAL | IMPLEMENTATION_COMPLETION | — | PDC-P5-D8 (ADR-PDC-08) | FRZ-STA-07,FRZ-REC-01 | YES |
| PDC-D8-R008 | Retention/TTL enforcement (ArtifactRetentionPolicy has zero consumers) | PARTIAL | IMPLEMENTATION_COMPLETION | — | PDC-P5-D8 (ADR-PDC-08) | FRZ-SEC-09 | YES |
| PDC-D8-R009 | Deletion and legal hold; audit evidence non-deletable (Collaborative Work defers hard delete) | ABSENT | IMPLEMENTATION_COMPLETION | — | PDC-P5-D8 (ADR-PDC-08) | FRZ-SEC-09,FRZ-AUD-05 | YES |
| PDC-D8-R010 | Governed secure download (only provider presigned_url) | PARTIAL | BLOCKED_EXACT_REASON | — | OD-PDC-SCOPE-04 (implement vs authorized exclusion) | FRZ-SEC-08,FRZ-DEBT-05 | YES |
| PDC-D8-R011 | Partial outputs are owned by PartialResultContract (task metadata), not by artifacts | PRESENT | N/A_WITH_EVIDENCE | — | — | FRZ-OWN-01 | NO |
| PDC-D8-R012 | Large output / tool-result spill to artifact refs (none exists) | ABSENT | IMPLEMENTATION_COMPLETION | — | PDC-P5-D8 (ADR-PDC-08) | FRZ-REL-10 | YES |
| PDC-D8-R013 | Durable context references to artifacts | PARTIAL | INTEGRATION_CORRECTION | — | PDC-P5-D8 | FRZ-REC-04 | YES |
| PDC-D8-R014 | Artifact recovery after restart | ABSENT | ADVERSARIAL_COVERAGE_REQUIRED | — | PDC-P5-D8 | FRZ-REC-01 | YES |
| PDC-D8-R015 | Cross-provider portability of artifact content storage | PRESENT | CURRENT_HEAD_REQUALIFICATION | EBH-5 (0 files (llm_adapters) @ 1ebd2ab4c2e6) | PDC-P7-D8 replay | FRZ-RPL-04 | NO |
| PDC-D8-R016 | Auditability of artifact operations | PARTIAL | INTEGRATION_CORRECTION | — | PDC-P5-D8 | FRZ-OBS-06,FRZ-AUD-05 | YES |
| PDC-D8-R017 | Identical contracts in contracts/task_artifacts.py and applications/contracts/application_artifacts.py (copy has 0 production importers) | PRESENT | INTEGRATION_CORRECTION | — | PDC-P5-D8 (COMPAT-X R5 shim policy) | FRZ-OWN-02,FRZ-OWN-05,FRZ-CMP-08 | YES |
| PDC-D8-R018 | Three types named ArtifactRef (pydantic ArtifactRef vs TraceArtifactRef aliases) | PRESENT | INTEGRATION_CORRECTION | — | PDC-P5-D8 | FRZ-TYP-01,FRZ-OWN-02 | YES |
| PDC-D8-R019 | Compaction never deletes evidence artifacts (same invariant as PDC-D2-R018) | PRESENT | ADVERSARIAL_COVERAGE_REQUIRED | — | PDC-P6 (XD-09) | FRZ-AUD-05 | YES |
| PDC-D1-R022 | plug_03 external chunker entry-point flow fails at HEAD (test_plug_03_full_qualification_evidence_batch) | PARTIAL | INTEGRATION_CORRECTION | — | PDC-P5-REG (root cause: stale gate vs regression) | FRZ-PLG-02,FRZ-REG-02 | YES |
| PDC-D3-R024 | EBH-5 replaceability gates red at HEAD: proof matrix/registry drift (48 vs 56 qualified concerns; RPL-F N/A vs PASS) | PARTIAL | INTEGRATION_CORRECTION | EBH-5 (0 files (llm_adapters) @ 1ebd2ab4c2e6) | PDC-P5-REG (root cause: stale matrix vs regression) | FRZ-RPL-01,FRZ-REG-05 | YES |
| PDC-D4-R020 | GR-12 product host composition fails under STRICT agent contract authority (revision-bound authority required) | PARTIAL | INTEGRATION_CORRECTION | — | PDC-P5-REG (root cause: stale gate vs regression) | FRZ-OWN-03,FRZ-REG-02 | YES |
| PDC-D6-R028 | GR-10 inference governance gates red at HEAD (pre-model enforcement AST, evidence-persistence wiring, checkpoint authority) | PARTIAL | INTEGRATION_CORRECTION | GOV-X2 (16 files (runtime/governance, runtime/policy, delegation) @ 1ebd2ab4c2e6) | PDC-P5-REG (root cause: stale gate vs regression) | FRZ-GOV-05,FRZ-REG-02 | YES |
