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
| B-P6-R | ALL | W5 | T-IGX-036-T | 0 | — | B-D1, B-D2, B-D3, B-D4, B-D5, B-D6, B-D7, B-D8, B-REG-P, B-REG-T, B-P6 | YES | OD-PDC-SEQ-01: implement in an isolated worktree; merge only after MCP-SERVER-X P8 publication; ACT-N11 re-activation at PDC-P7-00 | IA-GLOBAL (P8) |
| B-P6-M | ALL | W6 | T-IGX-036-U | 0 | — | B-P6-R | YES | OD-PDC-SEQ-01: implement in an isolated worktree; merge only after MCP-SERVER-X P8 publication; ACT-N11 re-activation at PDC-P7-00 | IA-GLOBAL (P8) |
| B-P6-G | ALL | W7 | T-IGX-036-V | 0 | — | B-P6-R, B-P6-M | YES | OD-PDC-SEQ-01: implement in an isolated worktree; merge only after MCP-SERVER-X P8 publication; ACT-N11 re-activation at PDC-P7-00 | IA-GLOBAL (P8) |
| B-P6-F | ALL | W8 | T-IGX-036-X | 0 | — | B-P6-G | YES | OD-PDC-SEQ-01: implement in an isolated worktree; merge only after MCP-SERVER-X P8 publication; ACT-N11 re-activation at PDC-P7-00 | IA-GLOBAL (P8) |
| B-P6-E | ALL | W9 | T-IGX-036-W | 0 | — | B-P6-G, B-P6-F | no | No product path: may merge before MCP-SERVER-X P8 | IA-GLOBAL (P8) |
| B-P6-H | ALL | W10 | T-IGX-036-Y | 0 | — | B-P6-E | no | No product path: may merge before MCP-SERVER-X P8 | IA-GLOBAL (P8) |
| B-P6-I | ALL | W11 | T-IGX-036-Z | 0 | — | B-P6-F | YES | OD-PDC-SEQ-01: implement in an isolated worktree; merge only after MCP-SERVER-X P8 publication; ACT-N11 re-activation at PDC-P7-00 | IA-GLOBAL (P8) |
| B-P6-J | ALL | W12 | T-IGX-040-A | 0 | — | B-P6-I | YES | OD-PDC-SEQ-01: implement in an isolated worktree; merge only after MCP-SERVER-X P8 publication; ACT-N11 re-activation at PDC-P7-00 | IA-GLOBAL (P8) |
| B-P7-D2 | D2 | W13 | T-IGX-041-B | 0 | ADR-PDC-02 | B-P6-J | YES | OD-PDC-SEQ-01: implement in an isolated worktree; merge only after MCP-SERVER-X P8 publication; ACT-N11 re-activation at PDC-P7-00 | IA-D2 (re-audit, loop 1 of 2) |
| B-P7-D5 | D5 | W13 | T-IGX-041-E | 0 | — | B-P6-J | YES | OD-PDC-SEQ-01: implement in an isolated worktree; merge only after MCP-SERVER-X P8 publication; ACT-N11 re-activation at PDC-P7-00 | IA-D5 (P7 re-audit) |
| B-P7-D1 | D1 | W14 | T-IGX-041-A | 0 | ADR-PDC-01 | B-P6-J | YES | OD-PDC-SEQ-01: implement in an isolated worktree; merge only after MCP-SERVER-X P8 publication; ACT-N11 re-activation at PDC-P7-00 | IA-D1 (re-audit, loop 1 of 2) |
| B-P7-D7 | D7 | W13 | T-IGX-041-G | 0 | — | B-P6-J | YES | OD-PDC-SEQ-01: implement in an isolated worktree; merge only after MCP-SERVER-X P8 publication; ACT-N11 re-activation at PDC-P7-00 | IA-D7 (P7 re-audit) |
| B-P7-D4 | D4 | W13 | T-IGX-041-D | 0 | ADR-PDC-04 | B-P6-J | YES | OD-PDC-SEQ-01: implement in an isolated worktree; merge only after MCP-SERVER-X P8 publication; ACT-N11 re-activation at PDC-P7-00 | IA-D4 (re-audit, loop 1 of 2) |
| B-P7-D3 | D3 | W13 | T-IGX-041-C | 0 | ADR-PDC-03 | B-P6-J | YES | OD-PDC-SEQ-01: implement in an isolated worktree; merge only after MCP-SERVER-X P8 publication; ACT-N11 re-activation at PDC-P7-00 | IA-D3 (re-audit, loop 1 of 2) |
| B-P7-D8 | D8 | W13 | T-IGX-041-H | 0 | ADR-PDC-08 | B-P6-J | YES | OD-PDC-SEQ-01: implement in an isolated worktree; merge only after MCP-SERVER-X P8 publication; ACT-N11 re-activation at PDC-P7-00 | IA-D8 (P7 re-audit) |
| B-P7-D6 | D6 | W13 | T-IGX-041-F | 0 | ADR-PDC-06 | B-P6-J | YES | OD-PDC-SEQ-01: implement in an isolated worktree; merge only after MCP-SERVER-X P8 publication; ACT-N11 re-activation at PDC-P7-00 | IA-D6 (re-audit, loop 1 of 2) |
| B-P7-TX | TX | W13 | T-IGX-041-I | 0 | — | B-P6-J | no | No product path: may merge before MCP-SERVER-X P8 | IA-GLOBAL (P8) |
| B-P7-CX | ALL | W14b | T-IGX-041-L | 0 | — | B-P7-D1, B-P7-D2, B-P7-D3, B-P7-D4, B-P7-D5, B-P7-D6, B-P7-D7, B-P7-D8, B-P7-TX | no | No product path: may merge before MCP-SERVER-X P8 | IA-GLOBAL (P8) |

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
- **Write set:** `intergrax/integrations/providers/embedding_provider/`, `tests/qualification/pdc_x/reg_p/`
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

## B-P6-R — Integration repairs found by the P5 lanes and the XD runs (tzdata, capability flags, freshness, launcher path, circular import, trace alias callers, feature-doc registration)

- **Ticket:** T-IGX-036-T · **Wave:** W5 · **ADR:** — · **Auditor:** IA-GLOBAL (P8)
- **Write set:** `pyproject.toml`, `uv.lock`, `intergrax/runtime/llm/model_call_runtime_evidence_adapter.py`, `intergrax/rag/retrieval/agentic_loop.py`, `intergrax/applications/_shared/lkw_hybrid_daemon_wiring.py`, `intergrax/runtime/integrations/observability.py`, `intergrax/runtime/execution/agent_runtime_context.py`, `intergrax/runtime/execution/trace_persistence_models_bridge.py`, `intergrax/runtime/nexus/context/context_manager.py`, `intergrax/runtime/nexus/context/shared_task_context.py`, `intergrax/runtime/nexus/engine/runtime_state.py`, `intergrax/runtime/nexus/engine/contracts/runtime_state_contract.py`, `intergrax/runtime/nexus/tracing/persistence_models.py`, `docs/project/capabilities/README.md`, `docs/project/architecture/intergrax_runtime_architecture.md`, `intergrax/runtime/nexus/tools/tool_runtime.py`, `intergrax/runtime/nexus/tools/plan_context_invocation.py`, `intergrax/runtime/task/worker_bootstrap.py`, `intergrax/runtime/persistence/sqlite_opens.py`, `intergrax/runtime/persistence/sqlite_composition.py`, `intergrax/runtime/nexus/artifacts/context_reference.py`, `intergrax/runtime/nexus/artifacts/spill.py`, `intergrax/memory/default_memory_control_plane.py`, `intergrax/runtime/agent_governance/pipeline.py`, `intergrax/llm_adapters/contracts/capabilities.py`, `tests/qualification/ebh_5/_sqlite_touchpoint_registry.py`, `intergrax/runtime/observability/export_boundary.py`, `tests/qualification/pdc_x/d4/test_d4_no_product_vocabulary.py`, `tests/qualification/pdc_x/xd/test_xd_02_agent_context.py`, `tests/qualification/pdc_x/xd/test_xd_03_agent_governance.py`, `tests/qualification/pdc_x/xd/test_xd_09_artifacts_context.py`, `tests/qualification/pdc_x/xd/test_xd_10_artifacts_security.py`, `tests/unit/memory/test_mem_ent10_memory_security_governance.py`, `tests/qualification/ebh_5/test_ebh_5_proof_integrity_gates.py`, `tests/qualification/pdc_x/p6_repair/`
- **Tests:** tests/qualification/pdc_x/p6_repair/
- **Evidence (exact SHA):** implementation SHA and publication SHA recorded in `PDC_SHA_LADDER.md`; test log under `.tmp/proof/pdc-x-p5/B-P6-R/`; audited by IA-GLOBAL (P8) at the PDC-P7 audited SHA.
- **Merge gate:** OD-PDC-SEQ-01: implement in an isolated worktree; merge only after MCP-SERVER-X P8 publication; ACT-N11 re-activation at PDC-P7-00

| Requirement | Class | Test case |
| --- | --- | --- |
| — | — | — |

## B-P6-M — Model binding found by XD-01: activated revision pins the LLM profile digest; required model features of an agent are checked when the adapter is bound

- **Ticket:** T-IGX-036-U · **Wave:** W6 · **ADR:** — · **Auditor:** IA-GLOBAL (P8)
- **Write set:** `intergrax/agent_distribution/runtime_revision.py`, `intergrax/agent_distribution/admin_service.py`, `intergrax/contracts/agent_contract_meta.py`, `intergrax/applications/_shared/llm_resolver.py`, `tests/qualification/pdc_x/xd/test_xd_01_agent_model.py`, `tests/qualification/pdc_x/p6_model/`
- **Tests:** tests/qualification/pdc_x/p6_model/
- **Evidence (exact SHA):** implementation SHA and publication SHA recorded in `PDC_SHA_LADDER.md`; test log under `.tmp/proof/pdc-x-p5/B-P6-M/`; audited by IA-GLOBAL (P8) at the PDC-P7 audited SHA.
- **Merge gate:** OD-PDC-SEQ-01: implement in an isolated worktree; merge only after MCP-SERVER-X P8 publication; ACT-N11 re-activation at PDC-P7-00

| Requirement | Class | Test case |
| --- | --- | --- |
| — | — | — |

## B-P6-G — ENT-AUDIT impact-map coverage of the PDC-touched product modules (XD finding 8)

- **Ticket:** T-IGX-036-V · **Wave:** W7 · **ADR:** — · **Auditor:** IA-GLOBAL (P8)
- **Write set:** `intergrax/platform/enterprise_audit/impact_map.py`, `intergrax/platform/enterprise_audit/data/changed_since_impact_map.v1.json`, `tests/qualification/ent_audit_x/unit/test_changed_since_impact_map.py`, `tests/qualification/pdc_x/xd/test_xd_12_evol_ent_audit.py`, `tests/qualification/pdc_x/p6_impact/`
- **Tests:** tests/qualification/pdc_x/p6_impact/
- **Evidence (exact SHA):** implementation SHA and publication SHA recorded in `PDC_SHA_LADDER.md`; test log under `.tmp/proof/pdc-x-p5/B-P6-G/`; audited by IA-GLOBAL (P8) at the PDC-P7 audited SHA.
- **Merge gate:** OD-PDC-SEQ-01: implement in an isolated worktree; merge only after MCP-SERVER-X P8 publication; ACT-N11 re-activation at PDC-P7-00

| Requirement | Class | Test case |
| --- | --- | --- |
| — | — | — |

## B-P6-F — ECAP VBN-REACHABILITY: the single ollama catalog row resolves the embedding category; ebh_5 harness registers the owning row first

- **Ticket:** T-IGX-036-X · **Wave:** W8 · **ADR:** — · **Auditor:** IA-GLOBAL (P8)
- **Write set:** `intergrax/integrations/providers/model_serving_runtime/ollama/bundle.py`, `intergrax/integrations/providers/model_serving_runtime/ollama/register.py`, `intergrax/integrations/providers/embedding_provider/ollama/bundle.py`, `intergrax/integrations/providers/embedding_provider/ollama/register.py`, `tests/qualification/ebh_5/_package_c_v2_primitives.py`, `tests/qualification/pdc_x/p6_ecapfix/`
- **Tests:** tests/qualification/pdc_x/p6_ecapfix/; test_ebh_5_behavioral_package_c_v2.py green
- **Evidence (exact SHA):** implementation SHA and publication SHA recorded in `PDC_SHA_LADDER.md`; test log under `.tmp/proof/pdc-x-p5/B-P6-F/`; audited by IA-GLOBAL (P8) at the PDC-P7 audited SHA.
- **Merge gate:** OD-PDC-SEQ-01: implement in an isolated worktree; merge only after MCP-SERVER-X P8 publication; ACT-N11 re-activation at PDC-P7-00

| Requirement | Class | Test case |
| --- | --- | --- |
| — | — | — |

## B-P6-E — P6 acceptance record and EBH-7 re-pin (fresh L3 replay at the integrated baseline)

- **Ticket:** T-IGX-036-W · **Wave:** W9 · **ADR:** — · **Auditor:** IA-GLOBAL (P8)
- **Write set:** `docs/project/maintainers/qualification/ebh_7_p03_independent_replay/`, `docs/project/maintainers/qualification/EBH_7_SHA_LADDER.md`, `docs/project/maintainers/qualification/EBH_7_B6_F10_ECAP_NORMATIVE_MATRIX.md`, `docs/project/maintainers/qualification/platform_domain_completeness/PDC_P6_ACCEPTANCE_RECORD.md`, `docs/project/maintainers/qualification/platform_domain_completeness/PDC_SHA_LADDER.md`, `docs/project/maintainers/qualification/platform_domain_completeness/PDC_REG_DIAGNOSIS.md`, `.github/workflows/unit-tests.yml`, `tests/qualification/ebh_7/_constants.py`
- **Tests:** PDC_P6_ACCEPTANCE_RECORD.md; tests/qualification/ebh_7 green at the new pin
- **Evidence (exact SHA):** implementation SHA and publication SHA recorded in `PDC_SHA_LADDER.md`; test log under `.tmp/proof/pdc-x-p5/B-P6-E/`; audited by IA-GLOBAL (P8) at the PDC-P7 audited SHA.
- **Merge gate:** No product path: may merge before MCP-SERVER-X P8

| Requirement | Class | Test case |
| --- | --- | --- |
| — | — | — |

## B-P6-H — EBH-7 B6-F10 computed certification at the P6 pin: closed-stage family evidence (FRZ-AUD, FRZ-EVOL), C4 path classification, replay-profile coverage

- **Ticket:** T-IGX-036-Y · **Wave:** W10 · **ADR:** — · **Auditor:** IA-GLOBAL (P8)
- **Write set:** `tests/qualification/ebh_7/_ebh7_f10_canonical_inputs.py`, `tests/qualification/ebh_7/_ebh7_f10_ecap_normative_matrix.py`, `tests/qualification/ebh_7/test_ebh_7_b6_f10_c1_c4_closure_gates.py`, `docs/project/maintainers/qualification/ebh_7_p03_independent_replay/MANIFEST.json`, `docs/project/maintainers/qualification/ebh_7_p03_independent_replay/ARTIFACT_DIGESTS.json`
- **Tests:** tests/qualification/ebh_7/test_ebh_7_b6_f10_c1_c4_closure_gates.py; the real F10 gate and five cascade tests stay red until the final re-pin
- **Evidence (exact SHA):** implementation SHA and publication SHA recorded in `PDC_SHA_LADDER.md`; test log under `.tmp/proof/pdc-x-p5/B-P6-H/`; audited by IA-GLOBAL (P8) at the PDC-P7 audited SHA.
- **Merge gate:** No product path: may merge before MCP-SERVER-X P8

| Requirement | Class | Test case |
| --- | --- | --- |
| — | — | — |

## B-P6-I — Omnibus repair of the cheap P6 debt: pdc_x process exit (R-05), stale baseline-red test fixtures, unused embedding_provider/ollama manifest

- **Ticket:** T-IGX-036-Z · **Wave:** W11 · **ADR:** — · **Auditor:** IA-GLOBAL (P8)
- **Write set:** `intergrax/integrations/providers/embedding_provider/ollama/manifest.py`, `applications/local_workspace_application/tests/test_run_artifact_metadata.py`, `tests/integration/runtime/nexus/tracing/test_sqlite_run_trace_store.py`, `tests/unit/integrations/providers/test_p5_m6_p4_providers.py`, `tests/unit/rag/embedding/test_hf_embedding_provider_execution.py`, `tests/qualification/pdc_x/conftest.py`, `tests/qualification/pdc_x/xd/test_xd_03_agent_governance.py`, `tests/qualification/pdc_x/p6_debt/`
- **Tests:** tests/qualification/pdc_x/p6_debt/; the pdc_x pytest process exits after its summary line
- **Evidence (exact SHA):** implementation SHA and publication SHA recorded in `PDC_SHA_LADDER.md`; test log under `.tmp/proof/pdc-x-p5/B-P6-I/`; audited by IA-GLOBAL (P8) at the PDC-P7 audited SHA.
- **Merge gate:** OD-PDC-SEQ-01: implement in an isolated worktree; merge only after MCP-SERVER-X P8 publication; ACT-N11 re-activation at PDC-P7-00

| Requirement | Class | Test case |
| --- | --- | --- |
| — | — | — |

## B-P6-J — MCP-RC handoff repairs: the single openai catalog row dispatches on integration_category (F-49, F-50 pins); ENT-AUDIT impact rule for the Tier-3 applications/ root (F-41)

- **Ticket:** T-IGX-040-A · **Wave:** W12 · **ADR:** — · **Auditor:** IA-GLOBAL (P8)
- **Write set:** `intergrax/integrations/providers/embedding_provider/openai/register.py`, `intergrax/platform/enterprise_audit/impact_map.py`, `intergrax/platform/enterprise_audit/data/changed_since_impact_map.v1.json`, `tests/qualification/pdc_x/p6_ecapfix/`, `tests/qualification/pdc_x/p6_impact/`
- **Tests:** tests/qualification/pdc_x/p6_ecapfix/ and p6_impact/; xd_12 and p6_impact coverage tests green on development and on a tree that contains the MCP-RC change
- **Evidence (exact SHA):** implementation SHA and publication SHA recorded in `PDC_SHA_LADDER.md`; test log under `.tmp/proof/pdc-x-p5/B-P6-J/`; audited by IA-GLOBAL (P8) at the PDC-P7 audited SHA.
- **Merge gate:** OD-PDC-SEQ-01: implement in an isolated worktree; merge only after MCP-SERVER-X P8 publication; ACT-N11 re-activation at PDC-P7-00

| Requirement | Class | Test case |
| --- | --- | --- |
| — | — | — |

## B-P7-D2 — D2 repair after the P7 independent audit (IA-D2): tool-call payloads counted by the hard context budget and oldest-first last-resort drop (F-D2-01 compiler half, F-D2-17), tenant-qualified and governed TASK memory scope (F-D2-02, F-D2-14), isolated reindex hooks (F-D2-11), bounded retention-sweep clock and deadline (F-D2-13), 35 red D2-cited RAG and context test files root-caused (F-D2-08, F-D2-09, F-D2-10, F-D2-18); F-D2-03..07, 12, 15, 16, 19 are declared exceptions

- **Ticket:** T-IGX-041-B · **Wave:** W13 · **ADR:** ADR-PDC-02 · **Auditor:** IA-D2 (re-audit, loop 1 of 2)
- **Write set:** `intergrax/context/budget/message_token_cost.py`, `intergrax/context/budget/mandatory_base_messages.py`, `intergrax/context/budget/token_counter.py`, `intergrax/runtime/nexus/context/context_compiler.py`, `intergrax/runtime/nexus/context/degradation_ladder.py`, `intergrax/runtime/nexus/context/context_preflight.py`, `intergrax/memory/default_memory_control_plane.py`, `intergrax/memory/contracts/memory_control.py`, `intergrax/memory/retention_sweep.py`, `intergrax/rag/governance/embedding_version_policy.py`, `intergrax/runtime/policy/pre_context_policy_audit.py`, `scripts/maintenance/check_pre_context_policy_wiring.py`, `tests/qualification/pdc_x/d2_repair/`, `tests/unit/rag/document_loaders/parsers/test_parser_pipeline.py`, `tests/unit/rag/document_loaders/registry/test_document_handler_registry.py`, `tests/unit/rag/document_loaders/test_document_loader.py`, `tests/unit/rag/document_splitters/test_native_strategies.py`, `tests/unit/rag/indexing/test_indexing_manager_indexes_documents.py`, `tests/unit/rag/indexing/test_single_index_strategy.py`, `tests/unit/rag/ingest/test_hierarchical_dual_index_wiring.py`, `tests/unit/rag/ingest/test_ingest_lci2b_loader_boundary.py`, `tests/unit/rag/ingest/test_ingest_pipeline_graph_wiring.py`, `tests/unit/rag/retrieval/test_ebh_2g_r1_retrieval_contract_typing_gate.py`, `tests/unit/rag/retrievers/test_hybrid_retriever.py`, `tests/unit/rag/vectorstore/test_hybrid_capability_routing.py`, `tests/unit/rag/vectorstore/test_source_ownership_contract.py`, `tests/unit/rag/vectorstore/test_vectorstore_cross_tenant_isolation.py`, `tests/integration/rag/document_loaders/parsers/test_doc_smart_document_handler.py`, `tests/integration/rag/document_loaders/parsers/test_document_handlers_framework.py`, `tests/integration/rag/document_loaders/parsers/test_excel_smart_document_handler.py`, `tests/integration/rag/document_loaders/parsers/test_html_smart_document_handler.py`, `tests/integration/rag/document_loaders/parsers/test_image_smart_document_handler.py`, `tests/integration/rag/document_loaders/parsers/test_pdf_smart_document_handler.py`, `tests/integration/rag/document_loaders/parsers/test_video_smart_document_handler.py`, `tests/integration/rag/document_loaders/test_fixture_dataset_ingestion.py`, `tests/integration/rag/embedding/test_hf_embedding_pipeline_integration.py`, `tests/integration/rag/embedding/test_ollama_embedding_pipeline_integration.py`, `tests/integration/rag/embedding/test_vllm_embedding_pipeline_integration.py`, `tests/integration/rag/rerankers/test_cross_encoder_reranker.py`, `tests/integration/rag/rerankers/test_embedding_cosine_reranker.py`, `tests/integration/rag/rerankers/test_semantic_reranker.py`, `tests/integration/rag/retrievers/test_retrieval_integration.py`, `tests/integration/rag/retrievers/test_retriever_pipeline_basic.py`, `tests/integration/rag/test_graph_reingest_qualification.py`, `tests/integration/rag/test_neo4j_live_qualification.py`, `tests/integration/rag/test_same_source_reingest_serialization.py`, `tests/integration/context/test_mem_xint6_cross_layer_e2e_certification.py`, `tests/unit/runtime/architecture/test_ideal_harness_l3_w2_depth_gate.py`, `tests/qualification/pdc_x/d2/test_d2_compaction_roundtrip.py`, `tests/qualification/pdc_x/d2/test_d2_context_fidelity.py`, `tests/qualification/memory_behavior/fixtures.py`, `tests/qualification/memory_behavior/scenarios/task.py`, `tests/unit/runtime/nexus/context/test_context_compiler.py`, `tests/unit/context/test_ce_02_r1d_safe_degradation.py`, `tests/unit/context/budget/test_ce02_r1i_mandatory_provider_hard_budget.py`, `tests/qualification/ce_02/test_ce_02_qualification_batch.py`, `tests/qualification/pdc_x/d2/test_d2_memory_retention.py`
- **Tests:** tests/qualification/pdc_x/d2_repair/ plus the D2-cited suites of the ticket test contract; the 35 red files of IA-D2 run 5 green or classified; D3 estimator change (T-IGX-041-C, PR 569) merged first
- **Evidence (exact SHA):** implementation SHA and publication SHA recorded in `PDC_SHA_LADDER.md`; test log under `.tmp/proof/pdc-x-p5/B-P7-D2/`; audited by IA-D2 (re-audit, loop 1 of 2) at the PDC-P7 audited SHA.
- **Merge gate:** OD-PDC-SEQ-01: implement in an isolated worktree; merge only after MCP-SERVER-X P8 publication; ACT-N11 re-activation at PDC-P7-00

| Requirement | Class | Test case |
| --- | --- | --- |
| — | — | — |

## B-P7-D5 — P7 repair of the independent D5 audit (loop 1): tenant isolation of online evaluation observations and the three hosts that hold a tenant (F-D5-01), ExperimentSession.run NameError and governance identity admission (F-D5-07), evaluation evidence status and comparison coverage (F-D5-04), promotion certification subject binding (F-D5-03 strengthening part), MVP promotion gate G0/G2 symbols and anchored authorization (F-D5-02 strengthening part), contract regex hygiene, R017 boundary documentation (F-D5-08); F-D5-05 and F-D5-06 are declared exceptions

- **Ticket:** T-IGX-041-E · **Wave:** W13 · **ADR:** — · **Auditor:** IA-D5 (P7 re-audit)
- **Write set:** `intergrax/runtime/architecture/online_evaluation_registry.py`, `intergrax/runtime/architecture/user_satisfaction.py`, `intergrax/runtime/architecture/online_evaluation.py`, `intergrax/runtime/architecture/agent_promotion.py`, `intergrax/runtime/evidence/eval_evidence_contracts.py`, `intergrax/eval/eval_comparison.py`, `intergrax/experiments/workflow.py`, `scripts/gates/check_mvp_promotion_gates.py`, `intergrax/contracts/evaluation_contracts.py`, `intergrax/applications/_shared/harness_host_runtime.py`, `intergrax/applications/_shared/runtime_config_bridge.py`, `intergrax/applications/_shared/scenario_runtime_baseline.py`, `tests/qualification/pdc_x/d5_repair/`, `docs/project/architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`, `docs/project/maintainers/plans/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`
- **Tests:** tests/qualification/pdc_x/d5_repair/ plus the cited d5, xd_08, experiments, online evaluation and host suites green; every in-scope finding pinned red-then-green with negative tests
- **Evidence (exact SHA):** implementation SHA and publication SHA recorded in `PDC_SHA_LADDER.md`; test log under `.tmp/proof/pdc-x-p5/B-P7-D5/`; audited by IA-D5 (P7 re-audit) at the PDC-P7 audited SHA.
- **Merge gate:** OD-PDC-SEQ-01: implement in an isolated worktree; merge only after MCP-SERVER-X P8 publication; ACT-N11 re-activation at PDC-P7-00

| Requirement | Class | Test case |
| --- | --- | --- |
| — | — | — |

## B-P7-D1 — D1 repair after the P7 independent audit (IA-D1): activation, rollback and the deploy/activate pipeline re-check revocation, authorized emergency revocation entry point, application-ownership guard on install/upgrade, no leftover VERIFIED upgrade candidate, 18 red D1 suites root-caused

- **Ticket:** T-IGX-041-A · **Wave:** W14 · **ADR:** ADR-PDC-01 · **Auditor:** IA-D1 (re-audit, loop 1 of 2)
- **Write set:** `intergrax/agent_distribution/admin_service.py`, `intergrax/applications/_shared/deploy_activate_pipeline.py`, `intergrax/applications/_shared/durable_registry_projection_input_store.py`, `tests/qualification/pdc_x/d1_repair/`, `tests/qualification/pdc_x/d1/test_d1_adversarial_lifecycle.py`, `tests/unit/agents/authoring/conftest.py`, `tests/unit/agents/authoring/patterns/test_pattern_probe_runs.py`, `tests/unit/agents/authoring/test_acp_execution_budget_ledger_factory.py`, `tests/unit/agents/authoring/test_acp_routing_trace_integration.py`, `tests/unit/agents/authoring/test_acp_wave3_trace.py`, `tests/unit/agents/authoring/test_rb2b2_uer_consumer_requalification.py`, `tests/unit/agent_distribution/test_ac6_architecture_gates.py`, `tests/unit/agent_distribution/test_agent_distribution_activation_remediation.py`, `tests/unit/agent_distribution/test_agent_distribution_build_remediation.py`, `tests/integration/agent_distribution/test_canonical_agent_lifecycle_e2e.py`, `tests/integration/agent_distribution/test_enterprise_agent_lifecycle_durable_e2e.py`, `tests/integration/agent_distribution/test_enterprise_projection_rehydration_e2e.py`, `tests/integration/agent_distribution/test_ac6_trust_lifecycle_e2e.py`
- **Tests:** tests/qualification/pdc_x/d1_repair/ plus the D1-cited suites of the ticket test contract; the 18 ids of the IA-D1 report (F-D1-11) green
- **Evidence (exact SHA):** implementation SHA and publication SHA recorded in `PDC_SHA_LADDER.md`; test log under `.tmp/proof/pdc-x-p5/B-P7-D1/`; audited by IA-D1 (re-audit, loop 1 of 2) at the PDC-P7 audited SHA.
- **Merge gate:** OD-PDC-SEQ-01: implement in an isolated worktree; merge only after MCP-SERVER-X P8 publication; ACT-N11 re-activation at PDC-P7-00

| Requirement | Class | Test case |
| --- | --- | --- |
| — | — | — |

## B-P7-D7 — P7 repair of D7 Background and Scheduling (IA-D7 findings F-D7-01, 02, 04, 06, 07, 08, 10, 11, 12, 13): per-claim isolation and a crash-safe recurrence chain, tenant-qualified occurrence identity, UTC-normalized run_at_utc, bounded recurrence, sticky terminal queue states, conditional-write claims, head-of-line fix, reconciled documents; F-D7-03, 05, 09 are declared exceptions

- **Ticket:** T-IGX-041-G · **Wave:** W13 · **ADR:** — · **Auditor:** IA-D7 (P7 re-audit)
- **Write set:** `intergrax/runtime/long_running/scheduler.py`, `intergrax/runtime/long_running/recurrence.py`, `intergrax/runtime/long_running/scheduled_resume.py`, `intergrax/runtime/long_running/scheduled_resume_metadata.py`, `intergrax/runtime/long_running/scheduler_claim.py`, `intergrax/runtime/long_running/store.py`, `intergrax/runtime/long_running/postgresql_task_checkpoint_store.py`, `intergrax/queueing/providers/document_store/document_store_task_queue.py`, `tests/qualification/pdc_x/d7_repair/`, `tests/qualification/pdc_x/d7/test_d7_recurrence.py`, `tests/qualification/pdc_x/d7/test_d7_fencing_two_process.py`, `tests/qualification/pdc_x/d7/_fencing_worker.py`, `docs/project/architecture/BACKGROUND_TASKS.md`, `docs/project/maintainers/plans/BACKGROUND_TASKS.md`
- **Tests:** tests/qualification/pdc_x/d7_repair/ plus the cited d7, bg_01, sched_01 and XD-05/06 suites green; every in-scope finding pinned red-then-green with negative tests
- **Evidence (exact SHA):** implementation SHA and publication SHA recorded in `PDC_SHA_LADDER.md`; test log under `.tmp/proof/pdc-x-p5/B-P7-D7/`; audited by IA-D7 (P7 re-audit) at the PDC-P7 audited SHA.
- **Merge gate:** OD-PDC-SEQ-01: implement in an isolated worktree; merge only after MCP-SERVER-X P8 publication; ACT-N11 re-activation at PDC-P7-00

| Requirement | Class | Test case |
| --- | --- | --- |
| — | — | — |

## B-P7-D4 — D4 repair after the P7 independent audit (IA-D4): cancellation rolls back lease, components and runtime; bounded runtime readiness probe and bounded event/metric publishes; public API fingerprint sees model config and constraints; composition-root gate with the declared lab MCP exception; TL-FIX-C and isolation-bound documentation reconciled; symlink tests skip without privilege

- **Ticket:** T-IGX-041-D · **Wave:** W13 · **ADR:** ADR-PDC-04 · **Auditor:** IA-D4 (re-audit, loop 1 of 2)
- **Write set:** `intergrax/hosting/engine/engine.py`, `intergrax/hosting/eventing.py`, `intergrax/hosting/supervisor/supervisor.py`, `intergrax/hosting/public_api.py`, `tests/qualification/pdc_x/d4_repair/`, `tests/qualification/pdc_x/d4/test_d4_crash_restart_subprocess.py`, `tests/unit/hosting/instance/test_file_guard.py`, `tests/unit/hosting/instance/test_w3_file_guard_regression.py`, `docs/project/architecture/TIER3_APPLICATION_ENVIRONMENT.md`, `docs/project/maintainers/plans/TIER3_APPLICATION_ENVIRONMENT.md`, `docs/project/maintainers/plans/satellites/TIER3_APPLICATION_ENVIRONMENT_embedded_detail.md`, `docs/project/architecture/APPLICATION_HOSTING.md`
- **Tests:** tests/qualification/pdc_x/d4_repair/ plus the D4-cited suites of the ticket test contract (pdc_x/d4, host_01, tests/unit/hosting)
- **Evidence (exact SHA):** implementation SHA and publication SHA recorded in `PDC_SHA_LADDER.md`; test log under `.tmp/proof/pdc-x-p5/B-P7-D4/`; audited by IA-D4 (re-audit, loop 1 of 2) at the PDC-P7 audited SHA.
- **Merge gate:** OD-PDC-SEQ-01: implement in an isolated worktree; merge only after MCP-SERVER-X P8 publication; ACT-N11 re-activation at PDC-P7-00

| Requirement | Class | Test case |
| --- | --- | --- |
| — | — | — |

## B-P7-D3 — P7 repair of the independent D3 audit (loop 1): streaming guard, quota and cancel registration on the default OpenAI, Gemini/Vertex and LangChain Ollama stream paths (F-D3-01); refusal and content-filter normalization on every stream and buffered path (F-D3-02); word-boundary retry classification (F-D3-07 defect part); tool_calls in the token estimator (F-D2-01 estimator half); stale llm_adapters unit suites (F-D3-05); timeout_sec doc honesty (F-D3-03 doc part)

- **Ticket:** T-IGX-041-C · **Wave:** W13 · **ADR:** ADR-PDC-03 · **Auditor:** IA-D3 (re-audit, loop 1 of 2)
- **Write set:** `intergrax/llm_adapters/base/base_llm_adapter.py`, `intergrax/llm_adapters/contracts/call_config.py`, `intergrax/llm_adapters/_shared/retry.py`, `intergrax/llm_adapters/_shared/openai_completion_mapping.py`, `intergrax/llm_adapters/_shared/stream_finish_signal.py`, `intergrax/llm_adapters/providers/openai_responses_adapter.py`, `intergrax/llm_adapters/providers/openai_chat_completions_adapter.py`, `intergrax/llm_adapters/providers/gemini_adapter.py`, `intergrax/llm_adapters/providers/ollama_adapter.py`, `intergrax/llm_adapters/providers/native_ollama_adapter.py`, `intergrax/llm_adapters/providers/mistral_adapter.py`, `intergrax/llm_adapters/providers/claude_adapter.py`, `intergrax/llm_adapters/providers/cohere_native_adapter.py`, `intergrax/llm_adapters/providers/aws_bedrock_adapter.py`, `platform_proofs/scenarios/ai_incident_investigation/application/investigator_agent.py`, `tests/qualification/pdc_x/d3/`, `tests/unit/llm_adapters/test_estimate_tokens_tool_calls.py`, `tests/unit/llm_adapters/test_ollama_tool_calling.py`, `tests/unit/llm_adapters/test_registry_model_contract.py`, `tests/unit/llm_adapters/test_llm_provider_registry.py`, `tests/unit/llm_adapters/test_p2_002_c_provider_registration.py`, `tests/unit/llm_adapters/registry_state_test_support.py`, `tests/unit/llm_adapters/test_adapter_conformance.py`, `tests/unit/llm_adapters/test_structured_output_flags.py`
- **Tests:** tests/unit/llm_adapters, ce_02 and pdc_x/d3 green in the combined run with the llm extras; new pinned tests red on the unchanged code; mutants M03 and M12 killed
- **Evidence (exact SHA):** implementation SHA and publication SHA recorded in `PDC_SHA_LADDER.md`; test log under `.tmp/proof/pdc-x-p5/B-P7-D3/`; audited by IA-D3 (re-audit, loop 1 of 2) at the PDC-P7 audited SHA.
- **Merge gate:** OD-PDC-SEQ-01: implement in an isolated worktree; merge only after MCP-SERVER-X P8 publication; ACT-N11 re-activation at PDC-P7-00

| Requirement | Class | Test case |
| --- | --- | --- |
| — | — | — |

## B-P7-D8 — P7 repair of the independent D8 audit (loop 1): RuntimeContext.create_artifact tenant, producer and lineage (F-D8-07), version-safe and audited retention sweep (F-D8-05), expiry enforced at read (F-D8-06), stored workspace required on read and lifecycle operations (F-D8-04 partial), raw storage tools refuse the Collaborative Work content key space (F-D8-03 bounded), governed port honours a returned decision and audits backend failures (F-D8-08, F-D8-12), documentation of library-only parts (F-D8-17); F-D8-01 and F-D8-02 are declared exceptions

- **Ticket:** T-IGX-041-H · **Wave:** W13 · **ADR:** ADR-PDC-08 · **Auditor:** IA-D8 (P7 re-audit)
- **Write set:** `intergrax/runtime/execution/agent_runtime_context.py`, `intergrax/runtime/nexus/artifacts/store_base.py`, `intergrax/runtime/nexus/artifacts/sqlite_artifact_store.py`, `intergrax/runtime/nexus/artifacts/in_memory_artifact_store.py`, `intergrax/runtime/nexus/artifacts/retention.py`, `intergrax/runtime/nexus/artifacts/sources.py`, `intergrax/runtime/nexus/artifacts/governed_access.py`, `intergrax/tools/providers/storage/service.py`, `tests/qualification/pdc_x/d8_repair/`, `tests/qualification/pdc_x/d8/test_d8_governed_port.py`, `tests/qualification/pdc_x/d8/test_xd09_spill_and_context_reference.py`, `tests/qualification/pdc_x/d8/test_xd10_tenant_and_ttl.py`, `tests/qualification/pdc_x/xd/test_xd_02_agent_context.py`, `tests/qualification/pdc_x/xd/test_xd_09_artifacts_context.py`, `tests/qualification/pdc_x/xd/test_xd_10_artifacts_security.py`, `tests/qualification/pdc_x/xd/test_xd_11_observability.py`, `docs/project/capabilities/architecture/ARTIFACT_LIFECYCLE.md`, `docs/project/capabilities/plan/ARTIFACT_LIFECYCLE.md`
- **Tests:** tests/qualification/pdc_x/d8_repair/ plus the cited d8, xd_02, xd_09, xd_10, xd_11 and storage tool suites green; every in-scope finding pinned red-then-green with negative tests
- **Evidence (exact SHA):** implementation SHA and publication SHA recorded in `PDC_SHA_LADDER.md`; test log under `.tmp/proof/pdc-x-p5/B-P7-D8/`; audited by IA-D8 (P7 re-audit) at the PDC-P7 audited SHA.
- **Merge gate:** OD-PDC-SEQ-01: implement in an isolated worktree; merge only after MCP-SERVER-X P8 publication; ACT-N11 re-activation at PDC-P7-00

| Requirement | Class | Test case |
| --- | --- | --- |
| — | — | — |

## B-P7-D6 — D6 repair after the P7 independent audit (IA-D6), loop 1: evidence-plane security audit trail writes only through a validating port (F-D6-01), overstating immutable/replicated wording relabeled (F-D6-02), plugin provenance bound to the attested distribution (F-D6-05), stale sandbox governance-deny test root-caused (F-D6-07), provider text kept out of the credential error chain (F-D6-12)

- **Ticket:** T-IGX-041-F · **Wave:** W13 · **ADR:** ADR-PDC-06 · **Auditor:** IA-D6 (re-audit, loop 1 of 2)
- **Write set:** `intergrax/runtime/security/security_audit_trail.py`, `intergrax/runtime/security/multi_region_audit_trail.py`, `intergrax/core/plugins/admission.py`, `intergrax/integrations/credentials/secrets_store_resolver.py`, `tests/qualification/pdc_x/d6_repair/`, `tests/unit/runtime/sandbox/test_cap_c1c_sandbox_evidence_and_contract_purity.py`
- **Tests:** tests/qualification/pdc_x/d6_repair/ red on unchanged code and green after; d6, xd_02/06/10, compat_x F-35 and event-evolution gates, sandbox unit suite and credential tests green except the known reds named in the ticket
- **Evidence (exact SHA):** implementation SHA and publication SHA recorded in `PDC_SHA_LADDER.md`; test log under `.tmp/proof/pdc-x-p5/B-P7-D6/`; audited by IA-D6 (re-audit, loop 1 of 2) at the PDC-P7 audited SHA.
- **Merge gate:** OD-PDC-SEQ-01: implement in an isolated worktree; merge only after MCP-SERVER-X P8 publication; ACT-N11 re-activation at PDC-P7-00

| Requirement | Class | Test case |
| --- | --- | --- |
| — | — | — |

## B-P7-TX — P7 re-pin of the TENANT-X class-field carrier-count gate (F-D2-03, F-D3-06, F-D6-06): classify the carriers added and removed since the pin, re-pin the total once after the last domain repair, strengthen the classification check; test-only

- **Ticket:** T-IGX-041-I · **Wave:** W13 · **ADR:** — · **Auditor:** IA-GLOBAL (P8)
- **Write set:** `tests/qualification/tenant_x/_replay_x_carrier_reconciliation.py`, `tests/qualification/tenant_x/test_tenant_x_replay_carrier_reclassification.py`
- **Tests:** tests/qualification/tenant_x green at the final head; negative tests prove an unclassified, swapped or unreasoned carrier still fails; PHASE 2 runs only after the D1, D2 and D6 repair PRs are merged
- **Evidence (exact SHA):** implementation SHA and publication SHA recorded in `PDC_SHA_LADDER.md`; test log under `.tmp/proof/pdc-x-p5/B-P7-TX/`; audited by IA-GLOBAL (P8) at the PDC-P7 audited SHA.
- **Merge gate:** No product path: may merge before MCP-SERVER-X P8

| Requirement | Class | Test case |
| --- | --- | --- |
| — | — | — |

## B-P7-CX — Re-pin of the COMPAT-X counters (debt item Z-D1 of T-IGX-036-Z) at the last product change: pinned counters, id-exact evidence fixture and four accepted shim-delta paths classified as heuristic hits; tests and fixtures only

- **Ticket:** T-IGX-041-L · **Wave:** W14b · **ADR:** — · **Auditor:** IA-GLOBAL (P8)
- **Write set:** `tests/qualification/compat_x/_compat_x_semantic_identity_reconciliation.py`, `tests/qualification/compat_x/test_compat_x_pdc_x_p7_repin_delta.py`, `tests/fixtures/compat_x/accepted_shim_delta_replay_x_05549f9.json`, `tests/fixtures/compat_x/repin_pdc_x_p7_delta.json`
- **Tests:** tests/qualification/compat_x green at the re-pinned counters; the ten red tests of the plan pass; new negative tests (hidden removal, extra candidate, balanced substitution, unexpected shim path) fail without their guard
- **Evidence (exact SHA):** implementation SHA and publication SHA recorded in `PDC_SHA_LADDER.md`; test log under `.tmp/proof/pdc-x-p5/B-P7-CX/`; audited by IA-GLOBAL (P8) at the PDC-P7 audited SHA.
- **Merge gate:** No product path: may merge before MCP-SERVER-X P8

| Requirement | Class | Test case |
| --- | --- | --- |
| — | — | — |
