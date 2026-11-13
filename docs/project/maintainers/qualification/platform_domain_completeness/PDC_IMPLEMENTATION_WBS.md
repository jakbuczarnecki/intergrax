# PDC-X — implementation WBS (PDC-P4)

**Generated** by `scripts/qualification/pdc_x_wbs.py` from `PDC_DOMAIN_REQUIREMENTS_MATRIX.json` and the checked-in batch table; do not edit by hand. Ticket T-IGX-036-G.

**Coverage:** 107 work-requiring requirement rows, each assigned to exactly one batch (checked).
Rows classed `CURRENT_HEAD_REQUALIFICATION`, `ALREADY_CERTIFIED_REUSABLE`, `N/A_WITH_EVIDENCE` or `DEFERRED_OUTSIDE_FREEZE_SCOPE` close at PDC-P7 by replay or carry their authorizing OD, so they are not P5 work.

**Integration owner:** PDC orchestrator session (single integration owner for shared hotspots). Shared hotspots: `intergrax/applications/_shared/`, `intergrax/contracts/`. At most one batch per wave may write under each hotspot (checked).

**Ticket letters** for P5 batches are reserved here. Each ticket file is scaffolded with its own plan at batch start, so draft tickets do not go stale while product merges wait on MCP-SERVER-X P8.

## Batches

| Batch | Lane | Wave | Ticket (reserved) | Rows | ADR | Depends on | Product code | Merge gate | Auditor |
| --- | --- | --- | --- | ---: | --- | --- | --- | --- | --- |
| B-P3 | GOV | W0-done | T-IGX-036-D | 2 | — | — | no | Closed in PDC-P3 (ownership record) | IA of the owning domain (P7 replay) |
| B-MCP | EXT | W0-external | MCP-SERVER-X | 1 | — | — | no | External: owned by MCP-SERVER-X P8 | MCP-SERVER-X P8 auditor; PDC IA-D6 consumes the result |
| B-DOC | DOC | WA | T-IGX-036-H | 8 | — | — | no | No product path: may merge before MCP-SERVER-X P8 | IA of the affected domain (doc delta at P7) |
| B-REG-T | REG | WB | T-IGX-036-I | 3 | — | — | no | No product path: may merge before MCP-SERVER-X P8 | IA-D3 (ebh_5), IA-D4 (GR-12), IA-D6 (GR-10) |
| B-REG-P | REG | W1 | T-IGX-036-M | 1 | — | — | YES | OD-PDC-SEQ-01: implement in an isolated worktree; merge only after MCP-SERVER-X P8 publication; ACT-N11 re-activation at PDC-P7-00 | IA-D1 (plug_03 replay) |
| B-D1 | D1 | W1 | T-IGX-036-J | 10 | ADR-PDC-01 | — | YES | OD-PDC-SEQ-01: implement in an isolated worktree; merge only after MCP-SERVER-X P8 publication; ACT-N11 re-activation at PDC-P7-00 | IA-D1 |
| B-D3 | D3 | W1 | T-IGX-036-K | 7 | ADR-PDC-03 | — | YES | OD-PDC-SEQ-01: implement in an isolated worktree; merge only after MCP-SERVER-X P8 publication; ACT-N11 re-activation at PDC-P7-00 | IA-D3 |
| B-D7 | D7 | W1 | T-IGX-036-L | 7 | ADR-PDC-07 | — | YES | OD-PDC-SEQ-01: implement in an isolated worktree; merge only after MCP-SERVER-X P8 publication; ACT-N11 re-activation at PDC-P7-00 | IA-D7 |
| B-D2 | D2 | W2 | T-IGX-036-N | 7 | ADR-PDC-02 | B-D1 | YES | OD-PDC-SEQ-01: implement in an isolated worktree; merge only after MCP-SERVER-X P8 publication; ACT-N11 re-activation at PDC-P7-00 | IA-D2 |
| B-D5 | D5 | W2 | T-IGX-036-O | 16 | ADR-PDC-05 | B-D3 | YES | OD-PDC-SEQ-01: implement in an isolated worktree; merge only after MCP-SERVER-X P8 publication; ACT-N11 re-activation at PDC-P7-00 | IA-D5 |
| B-D6 | D6 | W3a | T-IGX-036-P | 15 | ADR-PDC-06 | B-D2, B-D5 | YES | OD-PDC-SEQ-01: implement in an isolated worktree; merge only after MCP-SERVER-X P8 publication; ACT-N11 re-activation at PDC-P7-00 | IA-D6 |
| B-D8 | D8 | W3b | T-IGX-036-Q | 17 | ADR-PDC-08 | B-D6 | YES | OD-PDC-SEQ-01: implement in an isolated worktree; merge only after MCP-SERVER-X P8 publication; ACT-N11 re-activation at PDC-P7-00 | IA-D8 |
| B-D4 | D4 | W3c | T-IGX-036-R | 9 | ADR-PDC-04 | B-D7, B-D8 | YES | OD-PDC-SEQ-01: implement in an isolated worktree; merge only after MCP-SERVER-X P8 publication; ACT-N11 re-activation at PDC-P7-00 | IA-D4 |
| B-P6 | ALL | W4 | T-IGX-036-S | 4 | — | B-D1, B-D2, B-D3, B-D4, B-D5, B-D6, B-D7, B-D8, B-REG-P, B-REG-T | no | OD-PDC-SEQ-01: implement in an isolated worktree; merge only after MCP-SERVER-X P8 publication; ACT-N11 re-activation at PDC-P7-00 | IA-GLOBAL (P8) and each IA-Dn for its XD rows |

## B-P3 — Ownership records closed in P3

- **Ticket:** T-IGX-036-D · **Wave:** W0-done · **ADR:** — · **Auditor:** IA of the owning domain (P7 replay)
- **Write set:** — (no writes)
- **Tests:** PDC_OWNERSHIP_MATRIX.md
- **Evidence (exact SHA):** implementation SHA and publication SHA recorded in `PDC_SHA_LADDER.md`; test log under `.tmp/proof/pdc-x-p5/B-P3/`; audited by IA of the owning domain (P7 replay) at the PDC-P7 audited SHA.
- **Merge gate:** Closed in PDC-P3 (ownership record)

| Requirement | Class | Test case |
| --- | --- | --- |
| PDC-D1-R021 | DOCUMENTATION_RECONCILIATION | inventory ownership record |
| PDC-D7-R022 | DOCUMENTATION_RECONCILIATION | ownership matrix entry |

## B-MCP — MCP exposure security (MCP-SERVER-X P8)

- **Ticket:** MCP-SERVER-X · **Wave:** W0-external · **ADR:** — · **Auditor:** MCP-SERVER-X P8 auditor; PDC IA-D6 consumes the result
- **Write set:** — (no writes)
- **Tests:** MCP-SERVER-X qualification package
- **Evidence (exact SHA):** implementation SHA and publication SHA recorded in `PDC_SHA_LADDER.md`; test log under `.tmp/proof/pdc-x-p5/B-MCP/`; audited by MCP-SERVER-X P8 auditor; PDC IA-D6 consumes the result at the PDC-P7 audited SHA.
- **Merge gate:** External: owned by MCP-SERVER-X P8

| Requirement | Class | Test case |
| --- | --- | --- |
| PDC-D6-R017 | BLOCKED_EXACT_REASON | MCP-SERVER-X qualification package |

## B-DOC — Documentation reconciliations (wave A)

- **Ticket:** T-IGX-036-H · **Wave:** WA · **ADR:** — · **Auditor:** IA of the affected domain (doc delta at P7)
- **Write set:** `docs/project/architecture/AGENT_DISTRIBUTION.md`, `docs/project/architecture/TIER3_APPLICATION_ENVIRONMENT.md`, `docs/project/architecture/BACKGROUND_TASKS.md`, `docs/project/architecture/intergrax_runtime_architecture.md`, `docs/project/maintainers/plans/TIER3_APPLICATION_ENVIRONMENT.md`, `docs/project/maintainers/plans/CRITIC_VERIFICATION.md`, `docs/project/maintainers/plans/BACKGROUND_TASKS.md`, `docs/project/maintainers/qualification/HARNESS_FINAL_CURRENT_HEAD_CERTIFICATION.md`
- **Tests:** check_docs_domain_pairs.py; pdc_x_inventory.py --check
- **Evidence (exact SHA):** implementation SHA and publication SHA recorded in `PDC_SHA_LADDER.md`; test log under `.tmp/proof/pdc-x-p5/B-DOC/`; audited by IA of the affected domain (doc delta at P7) at the PDC-P7 audited SHA.
- **Merge gate:** No product path: may merge before MCP-SERVER-X P8

| Requirement | Class | Test case |
| --- | --- | --- |
| PDC-D1-R009 | DOCUMENTATION_RECONCILIATION | doc status matches AGENT_PLATFORM_AC1_AC6_FINAL_ARCHITECTURE_AUDIT |
| PDC-D4-R016 | DOCUMENTATION_RECONCILIATION | plan status matches code |
| PDC-D4-R019 | DOCUMENTATION_RECONCILIATION | bounded claim stated consistently in arch + plan |
| PDC-D5-R016 | DOCUMENTATION_RECONCILIATION | audit findings re-dispositioned as legacy |
| PDC-D6-R025 | DOCUMENTATION_RECONCILIATION | scoped wording reconciled |
| PDC-D7-R011 | DOCUMENTATION_RECONCILIATION | delivery semantics section |
| PDC-D7-R021 | DOCUMENTATION_RECONCILIATION | scaling boundary section |
| PDC-D7-R023 | DOCUMENTATION_RECONCILIATION | plan + hub registration |

## B-REG-T — Stale-gate repair: ebh_5 (EBH-7 reclassification), GR-12 fixture authority, GR-10 AST gates

- **Ticket:** T-IGX-036-I · **Wave:** WB · **ADR:** — · **Auditor:** IA-D3 (ebh_5), IA-D4 (GR-12), IA-D6 (GR-10)
- **Write set:** `tests/qualification/ebh_5/`, `tests/qualification/governance/`, `tests/unit/applications/task_control_product_host_test_support.py`, `docs/project/maintainers/qualification/EBH_5_CROSS_LAYER_REPLACEABILITY_MATRIX.md`
- **Tests:** tests/qualification/ebh_5 + tests/qualification/governance green
- **Evidence (exact SHA):** implementation SHA and publication SHA recorded in `PDC_SHA_LADDER.md`; test log under `.tmp/proof/pdc-x-p5/B-REG-T/`; audited by IA-D3 (ebh_5), IA-D4 (GR-12), IA-D6 (GR-10) at the PDC-P7 audited SHA.
- **Merge gate:** No product path: may merge before MCP-SERVER-X P8

| Requirement | Class | Test case |
| --- | --- | --- |
| PDC-D3-R024 | INTEGRATION_CORRECTION | tests/qualification/ebh_5 green |
| PDC-D4-R020 | INTEGRATION_CORRECTION | tests/qualification/governance gr12 green |
| PDC-D6-R028 | INTEGRATION_CORRECTION | tests/qualification/governance gr10 green |

## B-REG-P — Regression: duplicate `ollama` catalog slug breaks full integration bootstrap (7aa9cfa18)

- **Ticket:** T-IGX-036-M · **Wave:** W1 · **ADR:** — · **Auditor:** IA-D1 (plug_03 replay)
- **Write set:** `intergrax/integrations/providers/embedding_provider/`
- **Tests:** tests/unit/rag/test_rag_plugin_discovery.py + tests/qualification/plug_03 green
- **Evidence (exact SHA):** implementation SHA and publication SHA recorded in `PDC_SHA_LADDER.md`; test log under `.tmp/proof/pdc-x-p5/B-REG-P/`; audited by IA-D1 (plug_03 replay) at the PDC-P7 audited SHA.
- **Merge gate:** OD-PDC-SEQ-01: implement in an isolated worktree; merge only after MCP-SERVER-X P8 publication; ACT-N11 re-activation at PDC-P7-00

| Requirement | Class | Test case |
| --- | --- | --- |
| PDC-D1-R022 | INTEGRATION_CORRECTION | tests/qualification/plug_03 green |

## B-D1 — Agent lifecycle: upgrade/retire, skill lifecycle, discovery owner, adversarial lifecycle

- **Ticket:** T-IGX-036-J · **Wave:** W1 · **ADR:** ADR-PDC-01 · **Auditor:** IA-D1
- **Write set:** `intergrax/agent_distribution/`, `intergrax/skills/`, `intergrax/runtime/registry/`, `intergrax/core/plugins/discovery.py`, `intergrax/rag/bootstrap/entry_point_load.py`, `intergrax/runtime/vendor_knowledge/contribution_catalog.py`, `intergrax/contracts/agent_lifecycle_state.py`, `tests/qualification/pdc_x/d1/`
- **Tests:** tests/qualification/pdc_x/d1/
- **Evidence (exact SHA):** implementation SHA and publication SHA recorded in `PDC_SHA_LADDER.md`; test log under `.tmp/proof/pdc-x-p5/B-D1/`; audited by IA-D1 at the PDC-P7 audited SHA.
- **Merge gate:** OD-PDC-SEQ-01: implement in an isolated worktree; merge only after MCP-SERVER-X P8 publication; ACT-N11 re-activation at PDC-P7-00

| Requirement | Class | Test case |
| --- | --- | --- |
| PDC-D1-R003 | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/d1 tampered/unsigned package rejected |
| PDC-D1-R004 | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/d1 no-shadow-registry gate |
| PDC-D1-R007 | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/d1 cross-tenant install/bind/activate denied |
| PDC-D1-R011 | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/d1 upgrade compat accept/reject + rollback |
| PDC-D1-R012 | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/d1 rollback after failed activation |
| PDC-D1-R013 | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/d1 revoke during execution -> fail closed |
| PDC-D1-R014 | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/d1 retire removes routing + keeps evidence |
| PDC-D1-R017 | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/d1 transition evidence completeness |
| PDC-D1-R018 | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/d1 skill disable/remove (if implemented) |
| PDC-D1-R020 | INTEGRATION_CORRECTION | tests/qualification/pdc_x/d1 single discovery owner gate |

## B-D3 — Model runtime: declared streaming/tool-streaming/cancel capabilities, conformance gate, catalog-owned context windows

- **Ticket:** T-IGX-036-K · **Wave:** W1 · **ADR:** ADR-PDC-03 · **Auditor:** IA-D3
- **Write set:** `intergrax/llm_adapters/`, `tests/qualification/pdc_x/d3/`
- **Tests:** tests/qualification/pdc_x/d3/
- **Evidence (exact SHA):** implementation SHA and publication SHA recorded in `PDC_SHA_LADDER.md`; test log under `.tmp/proof/pdc-x-p5/B-D3/`; audited by IA-D3 at the PDC-P7 audited SHA.
- **Merge gate:** OD-PDC-SEQ-01: implement in an isolated worktree; merge only after MCP-SERVER-X P8 publication; ACT-N11 re-activation at PDC-P7-00

| Requirement | Class | Test case |
| --- | --- | --- |
| PDC-D3-R004 | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/d3 failover exhaustion bounded |
| PDC-D3-R006 | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/d3 unsupported tool streaming fails closed |
| PDC-D3-R009 | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/d3 refusal normalized across providers |
| PDC-D3-R011 | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/d3 cancel outcome typed (terminated / transport_closed / not_supported) |
| PDC-D3-R015 | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/d3 mismatch denied before call |
| PDC-D3-R022 | INTEGRATION_CORRECTION | tests/qualification/pdc_x/d3 capability flags match implemented methods |
| PDC-D3-R023 | INTEGRATION_CORRECTION | tests/qualification/pdc_x/d3 single context-window source |

## B-D7 — Background: recurrence on SchedulerLedger, dead-letter state, identity on enqueue, two-process fencing

- **Ticket:** T-IGX-036-L · **Wave:** W1 · **ADR:** ADR-PDC-07 · **Auditor:** IA-D7
- **Write set:** `intergrax/queueing/`, `intergrax/background_tasks/`, `intergrax/runtime/long_running/`, `intergrax/runtime/background_execution/`, `tests/qualification/pdc_x/d7/`
- **Tests:** tests/qualification/pdc_x/d7/
- **Evidence (exact SHA):** implementation SHA and publication SHA recorded in `PDC_SHA_LADDER.md`; test log under `.tmp/proof/pdc-x-p5/B-D7/`; audited by IA-D7 at the PDC-P7 audited SHA.
- **Merge gate:** OD-PDC-SEQ-01: implement in an isolated worktree; merge only after MCP-SERVER-X P8 publication; ACT-N11 re-activation at PDC-P7-00

| Requirement | Class | Test case |
| --- | --- | --- |
| PDC-D7-R006 | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/d7 recurrence occurrences + DST |
| PDC-D7-R007 | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/d7 DST transition |
| PDC-D7-R008 | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/d7 misfire policy |
| PDC-D7-R009 | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/d7 retry budget exhaustion terminal |
| PDC-D7-R012 | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/d7 two-process fencing over SQLite |
| PDC-D7-R013 | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/d7 poison task reaches terminal dead-letter state |
| PDC-D7-R016 | INTEGRATION_CORRECTION | tests/qualification/pdc_x/d7 enqueue envelope carries canonical identity |

## B-D2 — Context: trim fidelity, durable compaction (CE-02 closure), memory retention, RAG freshness, adversarial context

- **Ticket:** T-IGX-036-N · **Wave:** W2 · **ADR:** ADR-PDC-02 · **Auditor:** IA-D2
- **Write set:** `intergrax/context/`, `intergrax/runtime/nexus/context/`, `intergrax/runtime/token_optimization/`, `intergrax/runtime/context_lifecycle/`, `intergrax/memory/`, `intergrax/rag/governance/`, `intergrax/rag/retrieval/retrieval_result.py`, `intergrax/rag/retrieval/retrieval_service.py`, `intergrax/rag/ingest/ingest_pipeline.py`, `tests/qualification/pdc_x/d2/`
- **Tests:** tests/qualification/pdc_x/d2/ + ce_02 replay
- **Evidence (exact SHA):** implementation SHA and publication SHA recorded in `PDC_SHA_LADDER.md`; test log under `.tmp/proof/pdc-x-p5/B-D2/`; audited by IA-D2 at the PDC-P7 audited SHA.
- **Merge gate:** OD-PDC-SEQ-01: implement in an isolated worktree; merge only after MCP-SERVER-X P8 publication; ACT-N11 re-activation at PDC-P7-00

| Requirement | Class | Test case |
| --- | --- | --- |
| PDC-D2-R005 | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/d2 stale source detected |
| PDC-D2-R007 | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/d2 mandatory retention under pressure |
| PDC-D2-R009 | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/d2 injected tool output not promoted to instruction |
| PDC-D2-R011 | INTEGRATION_CORRECTION | tests/qualification/pdc_x/d2 hard trim preserves tool-call linkage |
| PDC-D2-R012 | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/d2 compaction round-trip + reconstruction |
| PDC-D2-R014 | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/d2 sensitive fragment not leaked to provider |
| PDC-D2-R015 | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/d2 retention expiry + vector tombstone |

## B-D5 — Evaluation: typed EvalDataset/EvalRun/EvalComparison, evaluator kinds, evidence persistence, advisory promotion

- **Ticket:** T-IGX-036-O · **Wave:** W2 · **ADR:** ADR-PDC-05 · **Auditor:** IA-D5
- **Write set:** `intergrax/eval/`, `intergrax/experiments/`, `intergrax/runtime/evaluation/`, `intergrax/runtime/architecture/evaluation_assets.py`, `intergrax/runtime/architecture/evaluation_automation.py`, `intergrax/runtime/architecture/evaluation_registry_trends.py`, `intergrax/runtime/architecture/evaluation_scenario_loader.py`, `intergrax/runtime/architecture/online_evaluation_registry.py`, `intergrax/runtime/architecture/agent_promotion.py`, `intergrax/runtime/evidence/eval_evidence_contracts.py`, `intergrax/contracts/evaluation_contracts.py`, `intergrax/applications/_shared/evaluation_wiring.py`, `scripts/gates/check_mvp_promotion_gates.py`, `tests/qualification/pdc_x/d5/`
- **Tests:** tests/qualification/pdc_x/d5/
- **Evidence (exact SHA):** implementation SHA and publication SHA recorded in `PDC_SHA_LADDER.md`; test log under `.tmp/proof/pdc-x-p5/B-D5/`; audited by IA-D5 at the PDC-P7 audited SHA.
- **Merge gate:** OD-PDC-SEQ-01: implement in an isolated worktree; merge only after MCP-SERVER-X P8 publication; ACT-N11 re-activation at PDC-P7-00

| Requirement | Class | Test case |
| --- | --- | --- |
| PDC-D5-R001 | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/d5 contract schema + persistence |
| PDC-D5-R002 | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/d5 digest mismatch rejected |
| PDC-D5-R003 | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/d5 scenario identity stable |
| PDC-D5-R004 | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/d5 run provenance complete |
| PDC-D5-R005 | INTEGRATION_CORRECTION | tests/qualification/pdc_x/d5 evaluator registry covers kinds |
| PDC-D5-R006 | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/d5 safety evaluator kind via guardrail provider |
| PDC-D5-R007 | INTEGRATION_CORRECTION | tests/qualification/pdc_x/d5 judge results isolated from rule results |
| PDC-D5-R009 | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/xd_08 digest-identical rerun |
| PDC-D5-R010 | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/d5 candidate regression detected |
| PDC-D5-R011 | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/d5 threshold breach flagged |
| PDC-D5-R012 | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/d5 comparison carries cost and latency |
| PDC-D5-R013 | INTEGRATION_CORRECTION | tests/qualification/pdc_x/d5 run recorded as evidence |
| PDC-D5-R014 | INTEGRATION_CORRECTION | tests/qualification/pdc_x/xd_08 eval PASS cannot authorize promotion |
| PDC-D5-R015 | INTEGRATION_CORRECTION | tests/qualification/pdc_x/d5 cross-tenant observation denied |
| PDC-D5-R017 | DOCUMENTATION_RECONCILIATION | boundary documented |
| PDC-D5-R018 | INTEGRATION_CORRECTION | tests/qualification/pdc_x/d5 concurrent writes safe or rejected |

## B-D6 — Security: fail-closed signing/poisoning defaults, single tool-input defense, durable audit trail, secret gate, plugin provenance

- **Ticket:** T-IGX-036-P · **Wave:** W3a · **ADR:** ADR-PDC-06 · **Auditor:** IA-D6
- **Write set:** `intergrax/runtime/security/`, `intergrax/core/security/`, `intergrax/core/catalog_bootstrap.py`, `intergrax/core/plugins/admission.py`, `intergrax/runtime/execution_evidence/attestor.py`, `intergrax/tools/providers/rag/service.py`, `intergrax/applications/_shared/critical_action_signing_wiring.py`, `intergrax/applications/_shared/security_assembly_resolver.py`, `intergrax/applications/_shared/security_audit_trail_wiring.py`, `intergrax/applications/_shared/application_security_wiring.py`, `docs/project/maintainers/architecture/satellites/`, `intergrax/runtime/attestation/host_attestation.py`, `intergrax/runtime/attestation/__init__.py`, `intergrax/applications/_shared/attestation_runtime_bridge.py`, `intergrax/core/plugins/discovery.py`, `intergrax/runtime/nexus/tools/plan_context_invocation.py`, `scripts/maintenance/check_critical_action_signing.py`, `scripts/maintenance/check_immutable_security_audit_trail.py`, `tests/unit/runtime/architecture/test_audit_ideal_depth_gate.py`, `tests/qualification/pdc_x/d6/`
- **Tests:** tests/qualification/pdc_x/d6/
- **Evidence (exact SHA):** implementation SHA and publication SHA recorded in `PDC_SHA_LADDER.md`; test log under `.tmp/proof/pdc-x-p5/B-D6/`; audited by IA-D6 at the PDC-P7 audited SHA.
- **Merge gate:** OD-PDC-SEQ-01: implement in an isolated worktree; merge only after MCP-SERVER-X P8 publication; ACT-N11 re-activation at PDC-P7-00

| Requirement | Class | Test case |
| --- | --- | --- |
| PDC-D2-R008 | INTEGRATION_CORRECTION | tests/qualification/pdc_x/d6 missing profile -> filter enforced |
| PDC-D6-R004 | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/d6 child widening denied |
| PDC-D6-R006 | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/d6 secret resolved only at use site |
| PDC-D6-R007 | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/d6 caplog/trace secret scan + static gate |
| PDC-D6-R008 | INTEGRATION_CORRECTION | tests/qualification/pdc_x/d6 missing signing secret fails closed outside lab |
| PDC-D6-R009 | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/xd_06 revoked credential mid-retry denied |
| PDC-D6-R013 | INTEGRATION_CORRECTION | tests/qualification/pdc_x/d6 single tool-input defense path |
| PDC-D6-R014 | INTEGRATION_CORRECTION | tests/qualification/pdc_x/d6 missing profile -> filter enforced |
| PDC-D6-R015 | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/d6 tenant data not sent to unapproved provider |
| PDC-D6-R016 | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/d6 unsigned plugin rejected (if in scope) |
| PDC-D6-R020 | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/d6 audit trail survives restart |
| PDC-D6-R021 | DOCUMENTATION_RECONCILIATION | threat model covers D1-D8 surfaces |
| PDC-D6-R023 | DOCUMENTATION_RECONCILIATION | responsibility matrix published |
| PDC-D6-R024 | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/d6 missing security config fails closed |
| PDC-D6-R027 | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/d6 demo seed unreachable in strict profile |

## B-D8 — Artifacts: taxonomy, mandatory digest, governed read/download port, durable store, retention, spill, dedupe

- **Ticket:** T-IGX-036-Q · **Wave:** W3b · **ADR:** ADR-PDC-08 · **Auditor:** IA-D8
- **Write set:** `intergrax/contracts/artifact_ref.py`, `intergrax/contracts/task_artifacts.py`, `intergrax/applications/contracts/application_artifacts.py`, `intergrax/runtime/nexus/artifacts/`, `intergrax/runtime/nexus/tracing/trace_models.py`, `intergrax/collaborative_work/`, `intergrax/runtime/execution/run_artifact_bundle_composition.py`, `intergrax/runtime/nexus/orchestration/run_artifact_bundle_builder.py`, `intergrax/applications/_shared/run_artifact_bundle_builder.py`, `docs/project/capabilities/architecture/ARTIFACT_LIFECYCLE.md`, `docs/project/capabilities/plan/ARTIFACT_LIFECYCLE.md`, `tests/qualification/state_x/_state_x_explicit_mechanism_classifications.py`, `intergrax/tools/providers/storage/service.py`, `intergrax/runtime/replay/models.py`, `intergrax/runtime/replay/replay_engine.py`, `tests/qualification/pdc_x/d8/`
- **Tests:** tests/qualification/pdc_x/d8/
- **Evidence (exact SHA):** implementation SHA and publication SHA recorded in `PDC_SHA_LADDER.md`; test log under `.tmp/proof/pdc-x-p5/B-D8/`; audited by IA-D8 at the PDC-P7 audited SHA.
- **Merge gate:** OD-PDC-SEQ-01: implement in an isolated worktree; merge only after MCP-SERVER-X P8 publication; ACT-N11 re-activation at PDC-P7-00

| Requirement | Class | Test case |
| --- | --- | --- |
| PDC-D6-R022 | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/xd_10 TTL deletion honored |
| PDC-D8-R001 | IMPLEMENTATION_COMPLETION | taxonomy + ownership gate |
| PDC-D8-R002 | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/d8 digest mismatch rejected |
| PDC-D8-R003 | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/d8 run artifact version append-only |
| PDC-D8-R004 | INTEGRATION_CORRECTION | tests/qualification/pdc_x/d8 artifact lineage reconstructable |
| PDC-D8-R005 | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/xd_10 cross-tenant read denied |
| PDC-D8-R006 | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/d8 unauthorized read denied |
| PDC-D8-R007 | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/d8 artifacts survive restart |
| PDC-D8-R008 | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/d8 TTL expiry enforced |
| PDC-D8-R009 | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/d8 evidence delete refused; legal hold respected |
| PDC-D8-R010 | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/d8 governed download (if in scope) |
| PDC-D8-R012 | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/xd_09 spill round-trip |
| PDC-D8-R013 | INTEGRATION_CORRECTION | tests/qualification/pdc_x/xd_09 context ref resolves after compaction |
| PDC-D8-R014 | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/d8 restart recovery |
| PDC-D8-R016 | INTEGRATION_CORRECTION | tests/qualification/pdc_x/d8 artifact ops in evidence |
| PDC-D8-R017 | INTEGRATION_CORRECTION | tests/qualification/pdc_x/d8 single definition gate |
| PDC-D8-R018 | INTEGRATION_CORRECTION | tests/qualification/pdc_x/d8 no ambiguous ArtifactRef alias |

## B-D4 — Hosting: lifecycle suite + crash/restart proof, metrics, digest-preserving restart, stable API, TL-FIX-C

- **Ticket:** T-IGX-036-R · **Wave:** W3c · **ADR:** ADR-PDC-04 · **Auditor:** IA-D4
- **Write set:** `intergrax/hosting/`, `intergrax/applications/_shared/environment_wiring.py`, `intergrax/applications/contracts/environment_profile/sub_profiles.py`, `intergrax/applications/_shared/lkw_hybrid_daemon_wiring.py`, `tests/qualification/pdc_x/d4/`
- **Tests:** tests/qualification/pdc_x/d4/
- **Evidence (exact SHA):** implementation SHA and publication SHA recorded in `PDC_SHA_LADDER.md`; test log under `.tmp/proof/pdc-x-p5/B-D4/`; audited by IA-D4 at the PDC-P7 audited SHA.
- **Merge gate:** OD-PDC-SEQ-01: implement in an isolated worktree; merge only after MCP-SERVER-X P8 publication; ACT-N11 re-activation at PDC-P7-00

| Requirement | Class | Test case |
| --- | --- | --- |
| PDC-D4-R003 | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/d4 startup failure rolls back in reverse order |
| PDC-D4-R004 | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/d4 degraded component reported |
| PDC-D4-R005 | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/xd_05 drain with background work |
| PDC-D4-R006 | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/d4 second instance rejected |
| PDC-D4-R007 | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/d4 subprocess kill + supervised restart |
| PDC-D4-R009 | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/d4 lifecycle metrics emitted via observability port |
| PDC-D4-R010 | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/d4 digest mismatch on restart rejected |
| PDC-D4-R014 | IMPLEMENTATION_COMPLETION | tests/qualification/pdc_x/d4 public API manifest + COMPAT gate |
| PDC-D4-R015 | INTEGRATION_CORRECTION | tests/qualification/pdc_x/d4 no product vocabulary in generic profile |

## B-P6 — Cross-domain integration XD-01..12, DUP-X/EBH-4 replay, full ECAP change-delta

- **Ticket:** T-IGX-036-S · **Wave:** W4 · **ADR:** — · **Auditor:** IA-GLOBAL (P8) and each IA-Dn for its XD rows
- **Write set:** `tests/qualification/pdc_x/xd/`
- **Tests:** tests/qualification/pdc_x/xd/
- **Evidence (exact SHA):** implementation SHA and publication SHA recorded in `PDC_SHA_LADDER.md`; test log under `.tmp/proof/pdc-x-p5/B-P6/`; audited by IA-GLOBAL (P8) and each IA-Dn for its XD rows at the PDC-P7 audited SHA.
- **Merge gate:** OD-PDC-SEQ-01: implement in an isolated worktree; merge only after MCP-SERVER-X P8 publication; ACT-N11 re-activation at PDC-P7-00

| Requirement | Class | Test case |
| --- | --- | --- |
| PDC-D2-R018 | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/xd_09 evidence survives compaction |
| PDC-D6-R019 | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/xd_02, xd_06, xd_10 |
| PDC-D7-R014 | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/xd_05 drain + lease release |
| PDC-D8-R019 | ADVERSARIAL_COVERAGE_REQUIRED | tests/qualification/pdc_x/xd_09 |
