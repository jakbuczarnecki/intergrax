# Enterprise Architecture Audit Report — EBH-7 B6-F10 certification correction

**Platform scope banner:** This report covers the **Intergrax Agent OS / Harness AI** four-tier platform (`intergrax/` · `intergrax/runtime/` · `agents/` · `applications/`) at EBH-7 closure scope. It is **pre-audit agent output** for independent B6 acceptance — **not** independent B6 PASS, **not** OD-EBH7-08, **not** global FRZ PASS.

Gate verdicts in this report are **computed** by `tests/qualification/ebh_7/_ebh7_f10_ecap_normative_matrix.py` from closed normative sets and inserted by `scripts/qualification/render_ebh7_f10_normative_matrix.py`. Obligation-level detail: [`EBH_7_B6_F10_ECAP_NORMATIVE_MATRIX.md`](EBH_7_B6_F10_ECAP_NORMATIVE_MATRIX.md).

## A. HEAD and baseline

- **TEST_EXECUTION_SHA (certified executable tree):** MANIFEST `TEST_EXECUTION_SHA` — see [`EBH_7_SHA_LADDER.md`](EBH_7_SHA_LADDER.md)
- **Last independently accepted baseline (C4):** `5a3a05347aaea93a8e4086fc90bc08e6d5eeaf92` (ROADMAP-REPLAY-X B6 PASS)
- **EBH-7 authorized entry baseline:** `81679e67225f42dfbc1c508516e19488b6fbf453`
- **DUP-X historical acceptance (C2):** `a7d83925b375b5f08eb0f175ced2c17fa026bba9` + current delta census
- **Audit mode:** FULL-HEAD closed-world obligations + DELTA vs accepted baseline

## B. Roadmap progress

- **Current parent stage:** EBH-7 — B6-F10 certification correction; independent B6 acceptance pending
- **Downstream (not entered):** EVOL-X · ENT-AUDIT-X · ARCH-FREEZE · SCENARIO-READINESS-X

## C. Enterprise compliance coverage

Canonical closed sets: 26 FRZ families (freeze checklist IDs), Tier-0..3 (`AGENTS.md`), DUP-01..14 + 15 DUP-X responsibility families + 12 normative C2 surfaces, roadmap §3.0.8 domains 1–31 + every inventory concern, every path changed since the accepted baseline.

ECAP-C5 (this report) is evaluated against the block below and the F10 matrix by the read-only gate; its computed verdict is published only in the F10 matrix.

<!-- F10-COMPUTED:BEGIN (generated — scripts/qualification/render_ebh7_f10_normative_matrix.py) -->
**Computed from closed normative sets @ TEST_EXECUTION_SHA `dc18e36d971c93c8f631fbc86f73149d1b81b4d7`** (C5 verified by read-only gate; no hand-assigned verdicts).

| ECAP gate | Computed verdict | Obligations | PASS | N/A WITH EVIDENCE | FAIL |
| --- | --- | --- | --- | --- | --- |
| ECAP-C1 | BLOCKED | 32 | 0 | 4 | 28 |
| ECAP-C2 | BLOCKED | 43 | 0 | 0 | 43 |
| ECAP-C3 | BLOCKED | 105 | 0 | 6 | 99 |
| ECAP-C4 | BLOCKED | 9 | 0 | 0 | 9 |

**C1–C4 computed disposition:** EBH-7 — B6-F10 BLOCKED / EXACT MISSING OBLIGATIONS
- C1 missing obligations: C1-FAMILY-ID-IDENTITY, C1-FAM-BND, C1-FAM-CFG, C1-FAM-CMP, C1-FAM-CTL, C1-FAM-CTR, C1-FAM-DOC, C1-FAM-EXE, C1-FAM-GOV, C1-FAM-HRN, C1-FAM-OBS, C1-FAM-OWN, C1-FAM-PLG, C1-FAM-PRD, C1-FAM-REC, C1-FAM-REG, C1-FAM-REL, C1-FAM-RPL, C1-FAM-SEC, C1-FAM-STA, C1-FAM-TEN, C1-FAM-TRC, C1-FAM-TYP, C1-ROLLUP-VBN, C1-TIER-0, C1-TIER-1, C1-TIER-2, C1-TIER-3
- C2 missing obligations: C2-DUP-CLASS-IDENTITY, C2-DUP-01, C2-DUP-02, C2-DUP-03, C2-DUP-04, C2-DUP-05, C2-DUP-06, C2-DUP-07, C2-DUP-08, C2-DUP-09, C2-DUP-10, C2-DUP-11, C2-DUP-12, C2-DUP-13, C2-DUP-14, C2-RF-EXE-01, C2-RF-BGW-01, C2-RF-GOV-01, C2-RF-CAP-01, C2-RF-PRV-01, C2-RF-REG-01, C2-RF-RCV-01, C2-RF-STA-01, C2-RF-PRV-02, C2-RF-OBS-01, C2-RF-PLG-01, C2-RF-TEN-01, C2-RF-LEG-01, C2-RF-CMP-01, C2-RF-T3-01, C2-SURFACE-execution-dispatch, C2-SURFACE-governance, C2-SURFACE-state-persistence-recovery, C2-SURFACE-provider-backend-selection, C2-SURFACE-configuration-profile, C2-SURFACE-registries, C2-SURFACE-tenant-context, C2-SURFACE-observability, C2-SURFACE-trace-evidence, C2-SURFACE-scheduler-background, C2-SURFACE-compatibility, C2-SURFACE-composition, C2-COUNTERS
- C3 missing obligations: C3-TAXONOMY-IDENTITY, C3-VBN-1-relational-persistence, C3-VBN-2-nosql-document, C3-VBN-3-key-value, C3-VBN-5-vector-database, C3-VBN-6-graph-database, C3-VBN-7-object-blob, C3-VBN-8-event-streaming, C3-VBN-9-queues-workers, C3-VBN-11-transaction-idempotency, C3-VBN-12-checkpoints-execution-state, C3-VBN-13-agent-memory, C3-VBN-14-rag-retrieval, C3-VBN-15-llm-embedding, C3-VBN-16-agent-runtimes, C3-VBN-17-hosting-substrate, C3-VBN-18-external-integrations-mcp, C3-VBN-21-tracing, C3-VBN-22-diagnostics, C3-VBN-23-alerting, C3-VBN-24-audit-evidence, C3-VBN-25-secrets, C3-VBN-26-auth-identity, C3-VBN-27-config-profile, C3-VBN-29-telemetry-otlp, C3-VBN-31-discovered, C3-CONCERN-integration.relational_store, C3-CONCERN-integration.document_store, C3-CONCERN-integration.key_value_cache, C3-CONCERN-integration.message_bus, C3-CONCERN-integration.object_storage, C3-CONCERN-integration.vector_store, C3-CONCERN-integration.search_provider, C3-CONCERN-integration.notification_channel, C3-CONCERN-integration.conversation_channel, C3-CONCERN-integration.collaboration_suite, C3-CONCERN-integration.issue_tracker, C3-CONCERN-integration.wiki_knowledge, C3-CONCERN-integration.observability_backend, C3-CONCERN-integration.browser_automation, C3-CONCERN-integration.cloud_platform, C3-CONCERN-integration.secrets_store, C3-CONCERN-integration.graph_store, C3-CONCERN-integration.document_parser, C3-CONCERN-integration.rerank_provider, C3-CONCERN-integration.feature_flag, C3-CONCERN-integration.ci_cd, C3-CONCERN-integration.security_scanner, C3-CONCERN-integration.sandbox_host, C3-CONCERN-integration.identity_provider, C3-CONCERN-integration.speech_provider, C3-CONCERN-integration.workflow_orchestrator, C3-CONCERN-integration.vision_serving, C3-CONCERN-integration.ml_inference_host, C3-CONCERN-integration.model_serving_runtime, C3-CONCERN-integration.billing_meter, C3-CONCERN-integration.crm, C3-CONCERN-integration.llm_guardrail, C3-CONCERN-integration.managed_retrieval, C3-CONCERN-integration.embedding_provider, C3-CONCERN-integration.external_work, C3-CONCERN-llm.chat_completion, C3-CONCERN-llm.embedding, C3-CONCERN-llm.rerank, C3-CONCERN-llm.fallback_policy, C3-CONCERN-llm.model_routing_qualifier, C3-CONCERN-persistence.relational_sqlite_lab, C3-CONCERN-int_config.configure_existing, C3-CONCERN-execution.integration_bound_resolution, C3-CONCERN-observability.otlp_transport, C3-CONCERN-plugin.integration_catalog, C3-CONCERN-plugin.llm_adapter_registry, C3-CONCERN-tool.marketplace_activation, C3-CONCERN-agent.effective_profile, C3-CONCERN-speech.provider, C3-CONCERN-env.integration_slug, C3-CONCERN-application.composition_root, C3-CONCERN-lab.integration_preset, C3-CONCERN-tokenizer.selection, C3-CONCERN-tool.observability_backend, C3-CONCERN-runtime.checkpoint.task, C3-CONCERN-runtime.human.decision, C3-CONCERN-runtime.execution.idempotency, C3-CONCERN-runtime.execution.decision_finalization, C3-CONCERN-runtime.events.runtime_store, C3-CONCERN-runtime.trace.run, C3-CONCERN-runtime.agent.checkpoint, C3-CONCERN-runtime.agent.compensation_queue, C3-CONCERN-runtime.tool_life.durable, C3-CONCERN-runtime.enterprise_reliability.provider_invocation, C3-CONCERN-platform.single_node_profile_v1.sqlite_triad, C3-CONCERN-distributed.sqlite_kv, C3-CONCERN-memory.sqlite.conversational, C3-CONCERN-lab.adaptive.signal_proposal_stores, C3-CONCERN-prodq.colocated_runtime_adapter_isinstance, C3-INVENTORY-CLOSED-WORLD, C3-DISCOVERED-MECHANISMS, C3-PROHIBITED-ZERO, C3-CONFIG-X-54
- C4 missing obligations: C4-BASELINE, C4-UNCLASSIFIED, C4-CLASS-composition-configuration, C4-CLASS-documentation-only, C4-CLASS-qualification-infrastructure, C4-CLASS-runtime-behavior, C4-CLASS-security-governance-tenant, C4-CLASS-state-recovery, C4-CLASS-vendor-backend
<!-- F10-COMPUTED:END -->

## D. Architecture findings

| Finding ID | Class | Scope | Disposition |
| --- | --- | --- | --- |
| B6-F10 | Evidence gap — declarative matrix | ECAP C1–C5 | Rebuilt as computed certification; verdict per block in §C |
| F10-C3-REG-01 | Qualification regression (EBH-5 Q07) | `integration.relational_store`, `tool.marketplace_activation`, `agent.effective_profile` | EBH-7 registered composite / parametrized proof refs rejected by EBH-5 validator; validator now resolves every node and binds parameter to `concern_id` |
| F10-C2-OWNER-DELTA | Stale DUP-X owner pointers | RF-EXE/BGW/GOV/RCV/STA/TEN/CMP/T3 | Current-HEAD owner delta bindings validated by path resolution / discovery census (DUP-X SSOT unchanged) |
| F10-C1-TIER-EXC-01 | Ratified tier exception | `intergrax/cli/external_work.py` → Tier-3 demo fixture | Lazy CLI demo import ratified in AC1–AC6 architecture audit; scan fails on any other violation or stale exception |
| F10-C1-STALE-GATE-REG | Stale family gate node | FRZ-REG (EBH-6 `_family_l9_evidence.py`) | Gate pointed to removed `test_aj_evidence_required_honest_not_zero`; rebound to its renamed successor `test_aj_evidence_required_matches_t2_bundle_gaps`; EBH-6 gate-node test now requires the function to exist (previously file-only) |
| F10-C4-STATE-REG-01 | Production regression (state-recovery) | `intergrax/memory/stores/sqlite_conversational_memory_store.py` | `load_memory` called `typing.Literal` `MessageRole(...)` → `TypeError` on every reload (introduced by `8afff322`); fixed with closed-role validation @ `a803191ad`; C4 regression `test_ebh7_fixed_substrate_rpl_positive[memory.sqlite.conversational]` |
| F10-P03-PROFILE-SHADOW | Replay coverage gap | MANIFEST `pytest_profile` | Four `ebh_7` files listed before `tests/qualification/ebh_7/` made pytest 8 collect only those four (195-node B5 runs); shadowing members removed; coverage helper no longer credits a shadowed directory (adversarial gate) |
| F10-P03-DIRTY-TREE | Non-reproducible census | EBH-7 W1 mechanism registry; EBH-6 wiring discovery | Clean-checkout GHA run `41227765200` failed: 447 of the historical 685 MECH rows came from the gitignored `docker/runtime-context/` copy; discovery now excludes it (238 on local and CI). Local-only `.tmp/proof` staging check skips when absent |
| F10-CFG-ISO-01 | Test isolation | CONFIG-X FRZ-CFG-07 gates | `clear_catalog()` without bootstrap reset broke later W5-B2 composition tests; fixture restores default integration catalog state |

Historical corrective findings F01–F09 and F11: unchanged (see `_gate_findings_register.py`).

## E. Vendor/backend dependency changes

- C3 domain coverage is computed per roadmap domain (inventory concern or N/A WITH EVIDENCE); CONFIG-X 54/54 and PROHIBITED = 0 are separate obligations and do not substitute domain coverage.
- Production vendor-backend paths changed since the accepted baseline are classified in C4 (`vendor-backend`) with bound regressions.

## F. Cross-layer ownership and authority

- Tier import boundaries: AST scan of Tier-0/Tier-2 sources executed inside the F10 gate + hardening-3 boundary gate.
- DUP-X: class/family/surface identity, counters and closed-world census re-executed @ executed HEAD.

## G. Runtime reliability and failure semantics

- Runtime-behavior / security-tenant / state-recovery changes since the accepted baseline are bound to invoker, tenant-scope, tenant E2E and fixed-substrate RPL regressions (C4).

## H. Impact on historical certification

- B6-F10 code (qualification infrastructure + the F10-C4-STATE-REG-01 production fix) → B6-F10 TEST_EXECUTION_SHA `496016c0cf6cc5cf1f9a29d885677605dd0415d3` (historical, ladder; GHA run `41231430749`); final evidence reconciliation (strict JUnit metadata + per-node skip dispositions; qualification tooling only, no production change) → current TEST_EXECUTION_SHA `dc18e36d971c93c8f631fbc86f73149d1b81b4d7` and a new GitHub Actions replay; prior B5 run `41214574614` retained as historical B5 evidence only — it never executed the full `ebh_7` directory (F10-P03-PROFILE-SHADOW).

## I. Scenario readiness / architecture-change risk

- SCENARIO-READINESS-X: downstream, not in scope.

## J. Blockers vs missing evidence

- Exact missing obligations, if any, are listed in the computed block (§C) and in the F10 matrix — no hand-assigned READY.
- Independent B6 acceptance remains with the external auditor.

## K. Closure verdict and next action

- **Agent disposition:** the computed disposition in §C (C1–C4) and the F10 matrix (C1–C5) — never overridden by text.
- **Non-claims:** B6 PASS · OD-EBH7-08 · EBH-7 CLOSED · EVOL-X · global FRZ PASS
