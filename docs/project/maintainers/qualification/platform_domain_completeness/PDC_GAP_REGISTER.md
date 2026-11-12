# PDC-X — gap register (PDC-P2-11)

**Generated** by `scripts/qualification/pdc_x_requirements.py`. One entry per requirement that needs work beyond replay. Rows classed `CURRENT_HEAD_REQUALIFICATION`, `ALREADY_CERTIFIED_REUSABLE` or `N/A_WITH_EVIDENCE` are closed by PDC-P7 replay and appear only in the requirements matrix. `BLOCKED_EXACT_REASON` rows wait on the named Operator Decision.

| Lane | Count |
| --- | ---: |
| MCP-SERVER-X | 1 |
| OD-PDC-SCOPE-01 | 1 |
| OD-PDC-SCOPE-02 | 3 |
| OD-PDC-SCOPE-03 | 2 |
| OD-PDC-SCOPE-04 | 1 |
| OD-PDC-SCOPE-05 | 3 |
| OD-PDC-SCOPE-06 | 1 |
| OD-PDC-SCOPE-07 | 1 |
| PDC-P3 ADR-PDC-08 | 1 |
| PDC-P3-01 | 4 |
| PDC-P5-D1 | 8 |
| PDC-P5-D2 | 6 |
| PDC-P5-D3 | 6 |
| PDC-P5-D4 | 9 |
| PDC-P5-D5 | 15 |
| PDC-P5-D6 | 12 |
| PDC-P5-D7 | 4 |
| PDC-P5-D8 | 15 |
| PDC-P5-DOC | 11 |
| PDC-P5-REG | 4 |
| PDC-P6 | 4 |

## MCP-SERVER-X

| ID | Gap | Class | Test / evidence | Depends on |
| --- | --- | --- | --- | --- |
| PDC-D6-R017 | MCP exposure security | BLOCKED_EXACT_REASON | MCP-SERVER-X qualification package | MCP-SERVER-X |

## OD-PDC-SCOPE-01

| ID | Gap | Class | Test / evidence | Depends on |
| --- | --- | --- | --- | --- |
| PDC-D1-R018 | Skill disable/remove/upgrade (SKILL-X F-04) kept separate from agent lifecycle | BLOCKED_EXACT_REASON | tests/qualification/pdc_x/d1 skill disable/remove (if implemented) | — |

## OD-PDC-SCOPE-02

| ID | Gap | Class | Test / evidence | Depends on |
| --- | --- | --- | --- | --- |
| PDC-D4-R011 | InteractionProfile facade (APP-HOST-6; no class exists) | BLOCKED_EXACT_REASON | n/a if excluded | — |
| PDC-D4-R012 | OS service-manager adapters (APP-HOST-7) | BLOCKED_EXACT_REASON | n/a if excluded | — |
| PDC-D4-R013 | Host plugins in HostedApplicationProfile (APP-HOST-9E) | BLOCKED_EXACT_REASON | n/a if excluded | — |

## OD-PDC-SCOPE-03

| ID | Gap | Class | Test / evidence | Depends on |
| --- | --- | --- | --- | --- |
| PDC-D1-R019 | Runtime factory beyond VENV_BUNDLE (OCI/sidecar deferred) | BLOCKED_EXACT_REASON | n/a if excluded | — |
| PDC-D3-R020 | Provider plugin registration layer (LLM-PROVIDER-PLUGIN-1 Planned) | BLOCKED_EXACT_REASON | n/a if excluded | — |

## OD-PDC-SCOPE-04

| ID | Gap | Class | Test / evidence | Depends on |
| --- | --- | --- | --- | --- |
| PDC-D8-R010 | Governed secure download (only provider presigned_url) | BLOCKED_EXACT_REASON | tests/qualification/pdc_x/d8 governed download (if in scope) | — |

## OD-PDC-SCOPE-05

| ID | Gap | Class | Test / evidence | Depends on |
| --- | --- | --- | --- | --- |
| PDC-D7-R006 | Recurrence (interval/cron) as occurrences on SchedulerLedger; no second engine | BLOCKED_EXACT_REASON | tests/qualification/pdc_x/d7 recurrence occurrences + DST | ADR-PDC-07 |
| PDC-D7-R007 | Time-zone and DST semantics for recurrence | BLOCKED_EXACT_REASON | tests/qualification/pdc_x/d7 DST transition | PDC-D7-R006 |
| PDC-D7-R008 | Recurrence misfire / catch-up policy | BLOCKED_EXACT_REASON | tests/qualification/pdc_x/d7 misfire policy | PDC-D7-R006 |

## OD-PDC-SCOPE-06

| ID | Gap | Class | Test / evidence | Depends on |
| --- | --- | --- | --- | --- |
| PDC-D2-R005 | Index consistency and source freshness (only an embedding-version warning + optional reindex hook) | BLOCKED_EXACT_REASON | tests/qualification/pdc_x/d2 stale source detected | — |

## OD-PDC-SCOPE-07

| ID | Gap | Class | Test / evidence | Depends on |
| --- | --- | --- | --- | --- |
| PDC-D6-R016 | Plugin entry-point admission is structural only (no signature/provenance); agent packages are attested | BLOCKED_EXACT_REASON | tests/qualification/pdc_x/d6 unsigned plugin rejected (if in scope) | — |

## PDC-P3 ADR-PDC-08

| ID | Gap | Class | Test / evidence | Depends on |
| --- | --- | --- | --- | --- |
| PDC-D8-R001 | Artifact taxonomy (business / execution evidence / context / temporary tool output) with one owner per class; no global storage engine | IMPLEMENTATION_COMPLETION | taxonomy + ownership gate | ADR-PDC-08 |

## PDC-P3-01

| ID | Gap | Class | Test / evidence | Depends on |
| --- | --- | --- | --- | --- |
| PDC-D1-R020 | One owner for plugin discovery (core/plugins/discovery.py vs vendor_knowledge and rag entry-point loaders) | INTEGRATION_CORRECTION | tests/qualification/pdc_x/d1 single discovery owner gate | — |
| PDC-D1-R021 | plug_03/plug_04/plug_final suites have no owning roadmap stage | DOCUMENTATION_RECONCILIATION | inventory ownership record | — |
| PDC-D3-R023 | Context-window tables duplicated inside providers and ModelCatalog | INTEGRATION_CORRECTION | tests/qualification/pdc_x/d3 single context-window source | — |
| PDC-D7-R022 | TaskRegistry/WorkerRuntime wraps TaskExecutionRegistry (layered, not duplicate); ownership unrecorded | DOCUMENTATION_RECONCILIATION | ownership matrix entry | — |

## PDC-P5-D1

| ID | Gap | Class | Test / evidence | Depends on |
| --- | --- | --- | --- | --- |
| PDC-D1-R003 | Package identity, provenance and attestation verified before install | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/d1 tampered/unsigned package rejected | — |
| PDC-D1-R004 | Single agent registry authority (runtime projection, not an install API) | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/d1 no-shadow-registry gate | — |
| PDC-D1-R007 | Tenant-scoped install, bind and activate | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/d1 cross-tenant install/bind/activate denied | — |
| PDC-D1-R011 | UPGRADE is an explicit governed operation with a compatibility check | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/d1 upgrade compat accept/reject + rollback | ADR-PDC-01 |
| PDC-D1-R012 | Rollback safety | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/d1 rollback after failed activation | — |
| PDC-D1-R013 | Revocation and emergency revocation reach active executions | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/d1 revoke during execution -> fail closed | — |
| PDC-D1-R014 | RETIRE is an explicit operation (state exists, no operation) | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/d1 retire removes routing + keeps evidence | ADR-PDC-01 |
| PDC-D1-R017 | Every lifecycle transition emits canonical evidence | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/d1 transition evidence completeness | — |

## PDC-P5-D2

| ID | Gap | Class | Test / evidence | Depends on |
| --- | --- | --- | --- | --- |
| PDC-D2-R007 | Mandatory instructions survive compaction and degradation | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/d2 mandatory retention under pressure | — |
| PDC-D2-R009 | Tool-output ingestion boundary | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/d2 injected tool output not promoted to instruction | — |
| PDC-D2-R011 | Degradation step TOKENIZER_HARD_TRIM drops entry_id/tool_calls/tool_call_id | INTEGRATION_CORRECTION | tests/qualification/pdc_x/d2 hard trim preserves tool-call linkage | ADR-PDC-02 |
| PDC-D2-R012 | Durable compaction semantics (CE-02 PARTIAL in gap audit) | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/d2 compaction round-trip + reconstruction | ADR-PDC-02 |
| PDC-D2-R014 | Sensitive data in context is classified and redacted | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/d2 sensitive fragment not leaked to provider | — |
| PDC-D2-R015 | Memory retention enforcement (retention is decision-only, no TTL; MEMORY-DURABILITY-LIFECYCLE planned) | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/d2 retention expiry + vector tombstone | ADR-PDC-02 |

## PDC-P5-D3

| ID | Gap | Class | Test / evidence | Depends on |
| --- | --- | --- | --- | --- |
| PDC-D3-R004 | Routing and failover bounded | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/d3 failover exhaustion bounded | — |
| PDC-D3-R006 | Streaming / tool-call streaming parity or explicit capability fail-closed (Ollama adapters lack stream_with_tools) | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/d3 unsupported tool streaming fails closed | ADR-PDC-03 |
| PDC-D3-R009 | Refusal semantics: typed finish reason exists, refusal payload is a string | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/d3 refusal normalized across providers | — |
| PDC-D3-R011 | Cancellation explicit per provider (seam beside the ABI; Ollama has no native remote cancel) | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/d3 cancel outcome typed (terminated / transport_closed / not_supported) | — |
| PDC-D3-R015 | Capability mismatch fails closed | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/d3 mismatch denied before call | — |
| PDC-D3-R022 | Streaming capability flags misreport: six adapters stream but supports_streaming() returns False | INTEGRATION_CORRECTION | tests/qualification/pdc_x/d3 capability flags match implemented methods | ADR-PDC-03 |

## PDC-P5-D4

| ID | Gap | Class | Test / evidence | Depends on |
| --- | --- | --- | --- | --- |
| PDC-D4-R003 | Startup dependency ordering and rollback | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/d4 startup failure rolls back in reverse order | — |
| PDC-D4-R004 | Readiness and degraded states | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/d4 degraded component reported | — |
| PDC-D4-R005 | Graceful shutdown with in-flight drain | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/xd_05 drain with background work | — |
| PDC-D4-R006 | Instance ownership guard | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/d4 second instance rejected | — |
| PDC-D4-R007 | Restart supervision with an OS-process crash proof (APP-HOST-5E Planned) | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/d4 subprocess kill + supervised restart | — |
| PDC-D4-R009 | Hosting lifecycle metrics through canonical observability (APP-HOST-3D; no metrics today) | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/d4 lifecycle metrics emitted via observability port | ADR-PDC-04 |
| PDC-D4-R010 | Restart preserves configuration/profile digest generically (APP-HOST-5D) | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/d4 digest mismatch on restart rejected | ADR-PDC-04 |
| PDC-D4-R014 | Stable public hosting API declaration (APP-HOST-9F) | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/d4 public API manifest + COMPAT gate | ADR-PDC-04 |
| PDC-D4-R015 | Generic HostDeploymentProfile carries LKW-specific fields (TL-FIX-C) | INTEGRATION_CORRECTION | tests/qualification/pdc_x/d4 no product vocabulary in generic profile | — |

## PDC-P5-D5

| ID | Gap | Class | Test / evidence | Depends on |
| --- | --- | --- | --- | --- |
| PDC-D5-R001 | Typed EvalDataset / EvalScenario / EvalRun contracts | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/d5 contract schema + persistence | ADR-PDC-05 |
| PDC-D5-R002 | Dataset versioning with content digest (GoldenDatasetAsset has free-string version) | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/d5 digest mismatch rejected | ADR-PDC-05 |
| PDC-D5-R003 | Evaluation scenario identity | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/d5 scenario identity stable | ADR-PDC-05 |
| PDC-D5-R004 | Agent/model/prompt/profile/code-SHA references on every run (EvalResult lacks them) | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/d5 run provenance complete | ADR-PDC-05 |
| PDC-D5-R005 | Task-success, tool-use, groundedness and citation evaluators behind one contract | INTEGRATION_CORRECTION | tests/qualification/pdc_x/d5 evaluator registry covers kinds | ADR-PDC-05 |
| PDC-D5-R006 | Safety evaluation hook (no safety/harm evaluator exists) | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/d5 safety evaluator kind via guardrail provider | ADR-PDC-05 |
| PDC-D5-R007 | Deterministic vs model-assisted evaluator separation | INTEGRATION_CORRECTION | tests/qualification/pdc_x/d5 judge results isolated from rule results | ADR-PDC-05 |
| PDC-D5-R009 | Reproducible evaluation runs | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/xd_08 digest-identical rerun | ADR-PDC-05 |
| PDC-D5-R010 | Baseline vs candidate comparison (only a pass-rate delta exists) | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/d5 candidate regression detected | ADR-PDC-05 |
| PDC-D5-R011 | Quality regression detection | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/d5 threshold breach flagged | ADR-PDC-05 |
| PDC-D5-R012 | Cost-to-success and latency/quality comparison | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/d5 comparison carries cost and latency | ADR-PDC-05 |
| PDC-D5-R013 | Evaluation provenance in the canonical evidence store | INTEGRATION_CORRECTION | tests/qualification/pdc_x/d5 run recorded as evidence | ADR-PDC-05 |
| PDC-D5-R014 | Promotion gates G0-G2 are file-existence checks; evaluation must stay advisory and never authorize | INTEGRATION_CORRECTION | tests/qualification/pdc_x/xd_08 eval PASS cannot authorize promotion | ADR-PDC-05 |
| PDC-D5-R015 | tenant_id on OnlineEvaluationObservation (TENANT-X debt recorded by CTRL-X) | INTEGRATION_CORRECTION | tests/qualification/pdc_x/d5 cross-tenant observation denied | — |
| PDC-D5-R018 | File-based evaluation registries have no concurrency semantics (DX-07) | INTEGRATION_CORRECTION | tests/qualification/pdc_x/d5 concurrent writes safe or rejected | ADR-PDC-05 |

## PDC-P5-D6

| ID | Gap | Class | Test / evidence | Depends on |
| --- | --- | --- | --- | --- |
| PDC-D2-R008 | Retrieval-poisoning filter consumed on both paths; RAG service fallback is fail-open (False when profile absent) | INTEGRATION_CORRECTION | tests/qualification/pdc_x/d6 missing profile -> filter enforced | — |
| PDC-D6-R004 | Delegation and authority narrowing | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/d6 child widening denied | — |
| PDC-D6-R006 | Secret references with late resolution | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/d6 secret resolved only at use site | — |
| PDC-D6-R007 | Raw secrets never cross layers, logs or traces (no static gate exists) | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/d6 caplog/trace secret scan + static gate | ADR-PDC-06 |
| PDC-D6-R008 | Critical-action signing falls back to literal 'harness-dev-signing-key' (SEC-BND-05) | INTEGRATION_CORRECTION | tests/qualification/pdc_x/d6 missing signing secret fails closed outside lab | ADR-PDC-06 |
| PDC-D6-R009 | Credential scope and revocation | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/xd_06 revoked credential mid-retry denied | — |
| PDC-D6-R013 | Tool-input injection guard assert_tool_input_safe has zero consumers | INTEGRATION_CORRECTION | tests/qualification/pdc_x/d6 single tool-input defense path | ADR-PDC-06 |
| PDC-D6-R014 | Retrieval-poisoning defense must fail closed when the profile is absent (SEC-BND-04) | INTEGRATION_CORRECTION | tests/qualification/pdc_x/d6 missing profile -> filter enforced | ADR-PDC-06 |
| PDC-D6-R015 | Data exfiltration / propagation across provider boundaries | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/d6 tenant data not sent to unapproved provider | — |
| PDC-D6-R020 | Security audit trail is in-memory only (SEC-BND-06) | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/d6 audit trail survives restart | ADR-PDC-06 |
| PDC-D6-R024 | Fail-closed security defaults | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/d6 missing security config fails closed | — |
| PDC-D6-R027 | Demo attestation signing seed reachable only in lab/proof paths | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/d6 demo seed unreachable in strict profile | — |

## PDC-P5-D7

| ID | Gap | Class | Test / evidence | Depends on |
| --- | --- | --- | --- | --- |
| PDC-D7-R009 | Aggregate retry budget | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/d7 retry budget exhaustion terminal | — |
| PDC-D7-R012 | Lease/fencing proven across processes (today: two schedulers in one process) | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/d7 two-process fencing over SQLite | — |
| PDC-D7-R013 | Dead-letter / terminal-failed semantics (no DLQ anywhere) | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/d7 poison task reaches terminal dead-letter state | ADR-PDC-07 |
| PDC-D7-R016 | Full execution identity carried on enqueue (adoption gap) | INTEGRATION_CORRECTION | tests/qualification/pdc_x/d7 enqueue envelope carries canonical identity | — |

## PDC-P5-D8

| ID | Gap | Class | Test / evidence | Depends on |
| --- | --- | --- | --- | --- |
| PDC-D6-R022 | Retention/deletion/redaction responsibility explicit | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/xd_10 TTL deletion honored | ADR-PDC-08 |
| PDC-D8-R002 | Mandatory content digest on durable artifact refs (optional outside Collaborative Work) | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/d8 digest mismatch rejected | ADR-PDC-08 |
| PDC-D8-R003 | Versioning (business artifacts certified; run artifacts unversioned) | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/d8 run artifact version append-only | ADR-PDC-08 |
| PDC-D8-R004 | Producer identity and execution lineage | INTEGRATION_CORRECTION | tests/qualification/pdc_x/d8 artifact lineage reconstructable | — |
| PDC-D8-R005 | Tenant/workspace scope of artifacts | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/xd_10 cross-tenant read denied | — |
| PDC-D8-R006 | Governed artifact read/access authorization port (store explicitly has no security) | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/d8 unauthorized read denied | ADR-PDC-08 |
| PDC-D8-R007 | Durable execution artifact store (only InMemoryArtifactStore) | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/d8 artifacts survive restart | ADR-PDC-08 |
| PDC-D8-R008 | Retention/TTL enforcement (ArtifactRetentionPolicy has zero consumers) | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/d8 TTL expiry enforced | ADR-PDC-08 |
| PDC-D8-R009 | Deletion and legal hold; audit evidence non-deletable (Collaborative Work defers hard delete) | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/d8 evidence delete refused; legal hold respected | ADR-PDC-08 |
| PDC-D8-R012 | Large output / tool-result spill to artifact refs (none exists) | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/xd_09 spill round-trip | ADR-PDC-08 |
| PDC-D8-R013 | Durable context references to artifacts | INTEGRATION_CORRECTION | tests/qualification/pdc_x/xd_09 context ref resolves after compaction | — |
| PDC-D8-R014 | Artifact recovery after restart | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/d8 restart recovery | PDC-D8-R007 |
| PDC-D8-R016 | Auditability of artifact operations | INTEGRATION_CORRECTION | tests/qualification/pdc_x/d8 artifact ops in evidence | — |
| PDC-D8-R017 | Identical contracts in contracts/task_artifacts.py and applications/contracts/application_artifacts.py (copy has 0 production importers) | INTEGRATION_CORRECTION | tests/qualification/pdc_x/d8 single definition gate | — |
| PDC-D8-R018 | Three types named ArtifactRef (pydantic ArtifactRef vs TraceArtifactRef aliases) | INTEGRATION_CORRECTION | tests/qualification/pdc_x/d8 no ambiguous ArtifactRef alias | — |

## PDC-P5-DOC

| ID | Gap | Class | Test / evidence | Depends on |
| --- | --- | --- | --- | --- |
| PDC-D1-R009 | Dynamic discovery and acquisition (AC-4); architecture doc still says READY_FOR_INDEPENDENT_AUDIT | DOCUMENTATION_RECONCILIATION | doc status matches AGENT_PLATFORM_AC1_AC6_FINAL_ARCHITECTURE_AUDIT | — |
| PDC-D4-R016 | TL-FIX-D implemented in code (bind_execution_adapter) but plan row still ACCEPTED/PLANNED | DOCUMENTATION_RECONCILIATION | plan status matches code | — |
| PDC-D4-R019 | Sandbox/shadow app isolation is not a universal production guarantee; plan marks APP-CON-8 Done | DOCUMENTATION_RECONCILIATION | bounded claim stated consistently in arch + plan | — |
| PDC-D5-R016 | CRITIC_VERIFICATION findings are legacy lineage; canonical authority is DECISION_SYSTEM | DOCUMENTATION_RECONCILIATION | audit findings re-dispositioned as legacy | — |
| PDC-D5-R017 | Production-feedback boundary of evaluation | DOCUMENTATION_RECONCILIATION | boundary documented | — |
| PDC-D6-R021 | Consolidated security architecture and threat model (no doc pair; broken IDENTITY_TRUST.md link) | DOCUMENTATION_RECONCILIATION | threat model covers D1-D8 surfaces | — |
| PDC-D6-R023 | Encryption / residency responsibility matrix | DOCUMENTATION_RECONCILIATION | responsibility matrix published | — |
| PDC-D6-R025 | HARNESS-FINAL 'ENTERPRISE QUALIFIED' for plugin trust/secrets/security contradicts FRZ-SEC OPEN | DOCUMENTATION_RECONCILIATION | scoped wording reconciled | — |
| PDC-D7-R011 | At-least-once delivery consequences documented | DOCUMENTATION_RECONCILIATION | delivery semantics section | — |
| PDC-D7-R021 | Horizontal scaling boundary documented (DIST-SCALE-X scoped) | DOCUMENTATION_RECONCILIATION | scaling boundary section | — |
| PDC-D7-R023 | BACKGROUND_TASKS plan statuses stale; doc not registered in runtime hub (PDC-P1-F01) | DOCUMENTATION_RECONCILIATION | plan + hub registration | — |

## PDC-P5-REG

| ID | Gap | Class | Test / evidence | Depends on |
| --- | --- | --- | --- | --- |
| PDC-D1-R022 | plug_03 external chunker entry-point flow fails at HEAD (test_plug_03_full_qualification_evidence_batch) | INTEGRATION_CORRECTION | tests/qualification/plug_03 green | PDC-D1-R020 |
| PDC-D3-R024 | EBH-5 replaceability gates red at HEAD: proof matrix/registry drift (48 vs 56 qualified concerns; RPL-F N/A vs PASS) | INTEGRATION_CORRECTION | tests/qualification/ebh_5 green | — |
| PDC-D4-R020 | GR-12 product host composition fails under STRICT agent contract authority (revision-bound authority required) | INTEGRATION_CORRECTION | tests/qualification/governance gr12 green | — |
| PDC-D6-R028 | GR-10 inference governance gates red at HEAD (pre-model enforcement AST, evidence-persistence wiring, checkpoint authority) | INTEGRATION_CORRECTION | tests/qualification/governance gr10 green | — |

## PDC-P6

| ID | Gap | Class | Test / evidence | Depends on |
| --- | --- | --- | --- | --- |
| PDC-D2-R018 | Compaction never deletes audit evidence | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/xd_09 evidence survives compaction | — |
| PDC-D6-R019 | Cross-tenant negative tests across D1-D8 | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/xd_02, xd_06, xd_10 | — |
| PDC-D7-R014 | Graceful worker shutdown with host drain | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/xd_05 drain + lease release | — |
| PDC-D8-R019 | Compaction never deletes evidence artifacts (same invariant as PDC-D2-R018) | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/xd_09 | PDC-D2-R018 |
