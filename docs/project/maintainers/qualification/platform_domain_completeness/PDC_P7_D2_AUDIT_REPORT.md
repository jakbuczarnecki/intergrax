# PDC-P7 independent audit of domain D2 (CONTEXT-KNOWLEDGE) - IA-D2

| Field | Value |
| --- | --- |
| **Audit** | IA-D2, independent auditor of PDC-X domain D2, PDC-P7 wave 2 |
| **Auditor session** | `PDC-P7-IA-D2-pdc-x-ia-d2-e1f727` (clean session; never implemented, signed or certified anything of D2) |
| **Policy** | `IIAC-2026-11-05.1`, `OD-PDC-IIAC-SCOPE` Option A; verdict vocabulary PASS / PASS_SCOPED / PARTIAL / FAIL |
| **Audited SHA** | `c08efe46a74a20595468d8d8bf06eab2dca5f573` (named by the orchestrator as `origin/development` at start) |
| **Audit execution SHA** | `b7889220607b462195308fa186f4ba66fb78f3a8` (descends from the audited SHA; the only differences are governance tickets and one documentation file, no product, script or test path) |
| **Date** | 2026-11-15, Windows 11, Python 3.12.11, default `uv run` environment (no optional extras) |
| **Ticket anchor** | `T-IGX-036-N` (batch B-D2, ADR-PDC-02); `Ref:` trailer on the commit |
| **Overall verdict** | **PARTIAL** (IIAC manifest value `CONDITIONAL`, see limitations) |
| **Rows** | 18 requirement rows: 10 PASS, 1 PASS_SCOPED, 7 PARTIAL, 0 FAIL |
| **Findings** | 0 HIGH, 8 MEDIUM, 10 LOW, 2 INFO |

No product, test, script, governance or ticket file was edited. This report certifies nothing: no FRZ promotion, closure or production-readiness claim.

## 1. Summary

The B-D2 repairs hold where they were aimed. Hard trim keeps message identity and drops tool-call pairs atomically, mandatory instructions survive pressure and are never truncated,
unlinked tool output is no longer promoted to a system block, provider-declared CONFIDENTIAL/RESTRICTED fragments are withheld from the model, the retention sweep expires and tombstones
through `MemoryControlPlane`, and the freshness contract returns typed verdicts. All nine runtime mutants of these guards were killed by the D2 tests, so those tests can fail.

The domain is PARTIAL because of seven rows whose claim is wider than what the code and tests establish: the token budget ignores tool-call payloads (R010), TASK-scope memory is tenant-blind
and the TENANT-X current-head replay is red (R002), the sensitive-data control depends on labels nobody sets and redacts nothing (R014), the retention sweep and freshness assessment are reachable from no shipped
composition (R015, R005), citation fidelity has no enforcing code or discriminating test (R016), and the RAG suites cited for R004 are not green. Eight MEDIUM findings carry these; none is HIGH
because every one is either latent (no shipped composition reaches the faulty branch) or bounded (fail-closed second gate, stale test double). Untrusted RAG, web and memory fragments still reach the
model as `role=system` (F-D2-04), the one design-level weakness adjacent to the R009 repair.

## 2. Method

1. **Scope read.** `OD-PDC-IIAC-SCOPE`, the 18 `PDC-D2-R*` rows of `PDC_DOMAIN_REQUIREMENTS_MATRIX.json`, the D2 rows of `PDC_OWNERSHIP_MATRIX.md`, ADR-PDC-02, the B-D2 batch in `PDC_IMPLEMENTATION_WBS.md`, `T-IGX-036-N` and its plan (including Amendment 1), the P7-00 ACT-N11 dossier and addenda 1-4 (leads only).
2. **Code reading at the owner paths.** `degradation_ladder.py`, `context_compiler.py`, `context_engine.py`, `formatter.py`, `provider_lifecycle.py`, `policy/egress.py`, `mandatory_base_messages.py`, `default_memory_control_plane.py`, `memory_control.py`, `retention_sweep.py`, `source_freshness.py`, `embedding_version_policy.py`, `retrieval_service.py`, `ingest_pipeline.py`, `plan_context_invocation.py`, `tools/providers/rag/service.py`, `citation_chain.py`, `context_lifecycle/*`, plus production-caller greps for every new capability.
3. **Cited tests, one pytest command at a time** (7 runs, narrow paths, output under `.tmp/session/ia-d2/`, copied to `docs/project/maintainers/qualification/iiac_evidence/PDC-X/D2/logs/`): see `docs/project/maintainers/qualification/iiac_evidence/PDC-X/D2/test_matrix.txt`.
4. **Adversarial probes** (`adversarial_scripts/probe_d2.py`, `probe_d2_b.py`, `probe_tenant_carriers.py`): required source vs egress, source roles of untrusted fragments, two-tenant TASK scope, raising reindex hook, tool-call payload budgeting, hard-budget drop order, forged `tool_call_id`.
5. **Mutation checks** (`adversarial_scripts/run_mutants.py`, runtime monkeypatch, no file edit): nine guards disabled one at a time (M1-M9), the claiming D2 tests must fail. All nine killed (M4 was first written with a wrong patch signature, rerun correctly; logs in `logs/mutants/`).
6. **Routed P7-00 leads** re-verified (section 6).

Rules applied to verdicts: a row is PASS only if the claim as stated by matrix, ADR or plan holds at the SHA, a discriminating test or probe exists, and no bypass was found; PARTIAL when part of the claim is unmet, unreachable in a composed host or only unverifiable; PASS_SCOPED only for an Operator-authorized exclusion. Pre-existing reds are reported when the row's cited evidence depends on them.

## 3. Per-row verdicts

| Requirement | Verdict | Basis (evidence) | Findings |
| --- | --- | --- | --- |
| PDC-D2-R001 | **PASS** | ce_01 15 passed (run1); canonicalize_fragment rejects forged provider provenance (provider_lifecycle.py:278-292); no bypass found. | - |
| PDC-D2-R002 | **PARTIAL** | USER scope holds: xd_02 passes, tenant_x all but one test passes (1220 passed in run3), mutant M9 killed; TASK scope is tenant-blind (probe P3); tenant_x current-head replay red. | F-D2-02, F-D2-03, F-D2-14 |
| PDC-D2-R003 | **PASS** | No production caller mutates user memory outside DefaultMemoryControlPlane (grep); memory_behavior and tests/unit/memory pass (1220 passed in run3); org memory is outside the plane scopes. | F-D2-02, F-D2-19 |
| PDC-D2-R004 | **PARTIAL** | Cited suites not green: 48 failed / 13 collection errors, ~25 stale doubles, ~14+12 missing extras, one unparseable test file. | F-D2-08, F-D2-09 |
| PDC-D2-R005 | **PARTIAL** | Stamp + typed verdict + hook call work (d2 freshness and wiring tests pass; M6 killed) but nothing composes them; hook failure breaks retrieval; fingerprint is mtime+size. | F-D2-06, F-D2-11, F-D2-12 |
| PDC-D2-R006 | **PASS** | ce_01 passes; single assembly path through ContextPlanner/DefaultNexusContextEngine/ContextCompiler found; static pre-context gate script is stale. | F-D2-10 |
| PDC-D2-R007 | **PASS** | d2 mandatory-retention tests pass; M7 (mandatory protection removed) killed; mandatory fragments skipped by compaction (compaction_apply.py:37-41); over-budget mandatory fails closed. | F-D2-15, F-D2-18 |
| PDC-D2-R008 | **PASS_SCOPED** | Tool path default-ON for absent profile (d6 poisoning 14 passed); nexus path fail-closed in production_mode only - Operator Amendment 1(c). | F-D2-04, F-D2-20 |
| PDC-D2-R009 | **PASS** | Unlinked TOOL_OUTPUT is rejected at canonicalization, linked output is role=tool; M4 (guard removed) killed. Row text is satisfied; adjacent untrusted sources remain role=system and ids are not matched. | F-D2-04, F-D2-16, F-D2-19 |
| PDC-D2-R010 | **PARTIAL** | ce_02 21 passed, deterministic and window math tested; but tool_calls payload is outside every budget count (probe P5). | F-D2-01 |
| PDC-D2-R011 | **PASS** | Hard trim keeps entry_id/name/tool_calls/tool_call_id and drops pairs atomically; M1 and M2 killed; end-to-end linkage tests pass. | F-D2-01, F-D2-17 |
| PDC-D2-R012 | **PASS** | Round trip, restart, protected-region, drift and corruption tests pass (4 tests); reuses the CE-02/UCL path; negative cases use broad except escapes. | F-D2-18 |
| PDC-D2-R013 | **PASS** | xd_07 restart reconstruction passes; tests/unit/ucl and unit/memory pass; no adversarial restart probe beyond the suite. | - |
| PDC-D2-R014 | **PARTIAL** | Egress filter works for labelled fragments (d2 R014 2 tests, M3 killed) but classification is provider-declared (default INTERNAL), nothing redacts, ceiling fixed. | F-D2-05, F-D2-15 |
| PDC-D2-R015 | **PARTIAL** | Sweep expires, forgets and tombstones the vector entry through the plane (7 tests, M5/M8 killed); not composed in any host, USER-profile only, producer-controlled as_of. | F-D2-06, F-D2-13 |
| PDC-D2-R016 | **PARTIAL** | CitationChainLink has no consumer; cited golden/citation tests assert non-emptiness only; RAG citation gate passes. | F-D2-07 |
| PDC-D2-R017 | **PASS** | N/A_WITH_EVIDENCE confirmed: UNIFIED_CONTEXT_LIFECYCLE.md:390 classifies the pack separately; the token_optimization ContextFragment is imported only inside token_optimization. | F-D2-19 |
| PDC-D2-R018 | **PASS** | xd_09 passes (evidence artifact survives compaction/TTL deletion refused); context_lifecycle has no delete/prune path (grep). | F-D2-19 |

Note on R005, R015: the component behaves as designed and the Operator accepted the additive, optional shape (Amendment 1). They are PARTIAL because ADR-PDC-02 promises enforcement (retention) and background reindex admission (freshness) and no shipped composition performs either (F-D2-06).

## 4. Test and probe results

| Run | Scope | Result |
| --- | --- | --- |
| 1 | `tests/qualification/pdc_x/d2`, `tests/qualification/ce_02`, `tests/qualification/ce_01` | 77 passed, exit 0 (162 s) |
| 2 | `pdc_x/xd` xd_02, xd_07, xd_09 + `tests/qualification/memory_behavior` | 66 passed, exit 0 |
| 3 | `tests/unit/context`, `tests/unit/memory`, `tests/unit/ucl`, `tests/qualification/tenant_x` | 1220 passed, 1 skipped, **1 failed** (`test_tx_p0_r3_head_q02_class_field_carrier_count_stable`, F-D2-03) |
| 4-5 | `tests/unit/rag`, `tests/integration/rag`, `tests/integration/context`, ideal depth gate | run 4 aborted on 13 collection errors; run 5 with `--continue-on-collection-errors`: **48 failed, 746 passed, 25 skipped, 13 errors** (F-D2-08..10) |
| 6 | `pdc_x/xd/test_xd_03_agent_governance.py` (R-05 hang check) | 9 passed, process exited after 10 s: **R-05 not reproduced** |
| 7 | `pdc_x/d6 -k poison` (R008 evidence) | 14 passed |
| M1-M9 | runtime mutants of D2 guards | 9/9 killed |
| probes | P1-P8 | see `logs/probe_d2.out.jsonl`, `logs/probe_d2_b.out.jsonl` |

Known classified reds were not re-reported: EBH-7 identity gate, the five `roadmap_replay_x` tests, `dist_scale_x`/Docker scope (C10/C15 freeze) were not part of the D2 runs.

## 5. Findings

### F-D2-01 - MEDIUM - Hard model-window budget counts only message.content; assistant tool_calls payload is never budgeted

- **Rows:** PDC-D2-R010, PDC-D2-R011  ·  **Verification:** PROBE_CONFIRMED
- **Evidence:** context_compiler.py:70 (classify_candidates) and :283 (_enforce_hard_budget.total) and degradation_ladder.py:86/_hard_trim_messages all size a message as count_tokens(message.content). base_llm_adapter.py:632 estimate_tokens_for_messages also joins only m.content. Probe P5 (adversarial_scripts/probe_d2.py): an assistant message with content '' and a 200,000-character tool_calls[].function.arguments, compiled with input_budget_tokens=200, returns total_tokens=4, trimmed=False, all four messages kept (200,078 characters of unbudgeted tool_calls).
- **Reproduction:** `uv run python docs/project/maintainers/qualification/iiac_evidence/PDC-X/D2/adversarial_scripts/probe_d2.py  (probe P5_tool_call_args)`
- **Impact:** The 'hard model-window enforcement' claim of R010 does not cover tool-call arguments, function names or role overhead; only the 256-token default margin protects the window. A long agent loop (or a model that writes large arguments) can exceed the real window while every ledger says the context fits.
- **Suggested repair scope:** Count the serialized tool_calls (id, name, arguments) and tool name/role overhead in classify_candidates, _hard_trim_messages, _enforce_hard_budget and the adapter estimator; trim or drop whole tool-call groups when the payload alone exceeds the budget. Add a ce_02 case with a large arguments payload.
- **Owner path:** `intergrax/runtime/nexus/context/context_compiler.py; intergrax/runtime/nexus/context/degradation_ladder.py; intergrax/llm_adapters/base/base_llm_adapter.py (D3-owned estimator, coordinate)`

### F-D2-02 - MEDIUM - TASK scope of the memory control plane drops the tenant before the capability call and skips governance

- **Rows:** PDC-D2-R002, PDC-D2-R003  ·  **Verification:** PROBE_CONFIRMED
- **Evidence:** default_memory_control_plane.py:846-887 (_remember_task/_forget_task): scope.tenant_id is validated against the identity by assert_memory_scope_authorized, then task_memory.write/delete(namespace, key) is called with no tenant, identity or governance evaluation; TaskMemoryCapability (memory_control.py) has no tenant parameter; _assert_identity_tenant_is_bound_tenant guards USER scope only. Probe P3: on one plane/capability, tenant-a writes ns/k, tenant-b overwrites ns/k, tenant-c forgets ns/k -> the key is deleted and all three calls succeed (capability saw only ('ns','k')).
- **Reproduction:** `uv run python .../D2/adversarial_scripts/probe_d2.py  (probe P3_task_scope_tenant)`
- **Impact:** Tenant isolation of task memory is the capability implementer's problem by contract, but the contract gives it no tenant. Any shared task-memory capability collides across tenants (overwrite, read, delete). Latent today: memory_wiring.build_default_memory_control_plane wires only user_profile_manager, so TASK/SESSION scopes are unsupported in the shipped composition.
- **Suggested repair scope:** Extend TaskMemoryCapability with tenant (or identity) and key the namespace by tenant in the plane; apply the same bound-tenant guard and a governance evaluation to TASK scope; add a two-tenant negative test next to test_xd02_*.
- **Owner path:** `intergrax/memory/default_memory_control_plane.py:846-887; intergrax/memory/contracts/memory_control.py (TaskMemoryCapability)`

### F-D2-03 - MEDIUM - TENANT-X current-head replay (cited evidence of R002) is red: carrier-count pin 831 != 819, new D2 tenant carriers unclassified

- **Rows:** PDC-D2-R002  ·  **Verification:** TEST_RUN
- **Evidence:** tests/qualification/tenant_x/test_tenant_x_p0_current_head.py::test_tx_p0_r3_head_q02_class_field_carrier_count_stable fails: 'assert 831 == 819' (TENANT_X_REPLAY_HEAD_CARRIER_TOTAL in tests/qualification/tenant_x/_replay_x_carrier_reconciliation.py:12; logs/run3_*). 1220 other tests of that run pass (1 skipped). The 12 unreconciled carriers include at least three added by B-D2 in a file created after the pin (retention_sweep.py, 4b0f927a2, 2026-11-12): MemoryRetentionSweepPayload/Output/Report.tenant_id (adversarial_scripts/probe_tenant_carriers.py lists head carriers in files changed since 5ccb72fcc that are absent from the classified delta set; the remaining ones belong to eval, artifacts, queueing, observability etc. and the exact 12 cannot be isolated without the pinned inventory).
- **Reproduction:** `uv run pytest tests/qualification/tenant_x/test_tenant_x_p0_current_head.py -q`
- **Impact:** The 'tenant_x replay' named as the R002 test case is not green at the audited SHA; the delta since the TENANT-X-FINAL pin has not been reconciled by the TENANT-X owner, so there is no current-head evidence that the new D2 carriers are tenant-checked. Not one of the known-red classes (EBH-7 identity gate, roadmap_replay_x, dist_scale_x).
- **Suggested repair scope:** TENANT-X owner: classify every new class-field carrier (authority/propagation role) in _replay_x_carrier_reconciliation.py and re-pin the count; the D2 carriers need an explicit role (payload carries no authority, the handler rechecks the admitted tenant).
- **Owner path:** `tests/qualification/tenant_x/_replay_x_carrier_reconciliation.py (test-only); intergrax/memory/retention_sweep.py as classification subject`

### F-D2-04 - MEDIUM - Only TOOL_OUTPUT was hardened against instruction promotion; RAG, web, memory and attachment fragments still reach the model as role=system

- **Rows:** PDC-D2-R009, PDC-D2-R008  ·  **Verification:** PROBE_CONFIRMED
- **Evidence:** formatter.py:47-52 renders every fragment that is neither SESSION_HISTORY nor a linked TOOL_OUTPUT as ChatMessage(role='system', content='[context:<source>:<id>] ...'). Probe P2: a RAG fragment 'IGNORE PREVIOUS INSTRUCTIONS' is delivered with role 'system'. The R009 fix (provider_lifecycle.py:293-302 rejects TOOL_OUTPUT without tool_call_id; formatter renders linked output as role 'tool') is correctly guarded by tests and by mutant M4 (killed), but retrieved or fetched third-party text keeps system-level authority.
- **Reproduction:** `uv run python .../D2/adversarial_scripts/probe_d2.py  (probe P2_source_roles)`
- **Impact:** A poisoned document, web page or memory entry that passes (or bypasses, see R008 non-production path) the retrieval-poisoning filter is an instruction to the model, the same class of defect R009 closed for tool output. Mandatory-system protection in the compiler (mandatory_base_message_indices) additionally treats non-fragment system messages as untouchable.
- **Suggested repair scope:** Render untrusted-source fragments (RAG, web search, attachments, long-term memory, knowledge) as delimited data under a non-instruction role or an explicit untrusted envelope; keep role=system for trusted instruction sources; extend the d2 R009 test to every untrusted ContextFragmentSource.
- **Owner path:** `intergrax/context/formatter.py:47-52; intergrax/context/contracts.py (ContextFragmentSource trust classes)`

### F-D2-05 - MEDIUM - Sensitive-data control is provider-declared only: no classifier, no redaction, unclassified fragments default to INTERNAL and egress

- **Rows:** PDC-D2-R014  ·  **Verification:** CODE_READ_BY_AUDITOR+GREP
- **Evidence:** contracts.py:288 ContextFragment.sensitivity defaults to DataClassification.INTERNAL; egress.py filters only CONFIDENTIAL/RESTRICTED against a ceiling hard-coded to INTERNAL (context_engine.py:331 passes no ceiling, no profile option). grep for sensitivity= in intergrax/ finds no producer that classifies memory, RAG, tool or history content (matches are the context contracts/invariants, tests and an unrelated calendar field). Base messages (history, user turn) are not fragments and are not covered. R014 asks that sensitive data is 'classified and redacted'; no redaction path exists in D2 paths. Tests d2 R014 and mutant M3 (killed) prove the filter works for fragments a provider has labelled.
- **Reproduction:** `grep -rn "sensitivity=" intergrax --include=*.py ; uv run pytest tests/qualification/pdc_x/d2/test_d2_context_fidelity.py -k sensitive`
- **Impact:** Confidential content flows to an external model unless a provider author labels it; the 'classified' half of the claim rests on every provider doing the right thing, and 'redacted' is not implemented.
- **Suggested repair scope:** Derive sensitivity from the source record (memory governance classification, RAG document classification, tool result classification) at fragment construction; make the ceiling a typed, provider-locality-aware policy; add redaction (or explicit refusal) for classified spans; decide the default for unclassified fragments (fail-closed in strict profiles).
- **Owner path:** `intergrax/context/policy/egress.py; intergrax/context/contracts.py; providers under intergrax/context/providers and intergrax/runtime/nexus/context/provider_source_adapters.py`

### F-D2-06 - MEDIUM - Memory retention sweep and RAG freshness are not reachable from any shipped composition; sweep covers USER-profile entries only

- **Rows:** PDC-D2-R015, PDC-D2-R005  ·  **Verification:** GREP+CODE_READ_BY_AUDITOR
- **Evidence:** grep over intergrax/, agents/, applications/ (and scripts/ for the sweep names): register_memory_retention_sweep_task, MemoryRetentionSweep, memory_retention_sweep_task_request, RetrievalService(source_observer=...), RetrievalResult.freshness and register_reindex_queue_hook have no caller outside their own modules and tests. memory_wiring.py:619-622 builds the plane with user_profile_manager only. No recurrence/schedule enqueues the sweep. retention_sweep.py:161 lists candidates through MemoryRetentionCandidateReader (user-profile entries); entity-temporal, procedural, long-horizon, session-turn-index and organization-profile memory have no retention enforcement. ADR-PDC-02 says reindexing is 'admitted as background work'; the only mechanism is a process-global synchronous callback list (embedding_version_policy.py:23,157) that nothing registers.
- **Reproduction:** `grep -rn "register_memory_retention_sweep_task\|source_observer\|register_reindex_queue_hook" intergrax agents applications scripts --include=*.py`
- **Impact:** Component behaviour is proven (mutants M5, M6, M8 killed), but in a composed host expired memory is never swept and stale sources are never detected or reindexed: the HIGH-impact FRZ-SEC-09 retention claim is enforced nowhere by default. Same class as P7-00 finding F-04 (opt-in, no production caller).
- **Suggested repair scope:** Compose the sweep in the host/worker root (task registration + a recurrence rule) and the freshness observer + reindex admission in the RAG composition root; extend retention to the other memory stores or document them as out of scope with an Operator decision; add a composition test that fails when the sweep task is not registered.
- **Owner path:** `intergrax/applications/_shared/memory_wiring.py; worker/host composition roots; intergrax/memory/retention_sweep.py; intergrax/rag/retrieval/resolve.py`

### F-D2-07 - MEDIUM - Citation and evidence fidelity has no enforcing code or discriminating test: CitationChainLink is an unconsumed 28-line model; the cited golden harness only asserts non-empty data

- **Rows:** PDC-D2-R016  ·  **Verification:** CODE_READ_BY_AUDITOR+TEST_RUN
- **Evidence:** intergrax/runtime/context/citation_chain.py is a pydantic data holder (no check that fragment_id exists, that excerpt belongs to the fragment or that output_ref is cited); grep finds no consumer in intergrax/. The cited test case 'context golden harness' resolves to tests/unit/runtime/architecture/test_ideal_harness_l3_w2_depth_gate.py: test_ideal_w2_citation_chain asserts len(chain.links)==1 and test_ideal_w2_scenario_and_context_libraries asserts 'cases' is non-empty; tests/fixtures/context_golden/cases.json holds two literal cases and scripts/maintenance/check_context_golden.py prints OK for them. The only fidelity logic is RAG citation_from_chunk (excerpt from chunk text, metadata copied) covered by tests/unit/rag/retrieval/test_rag_citation_engine_gate.py (passes).
- **Reproduction:** `uv run pytest tests/unit/runtime/architecture/test_ideal_harness_l3_w2_depth_gate.py -k citation ; .venv/Scripts/python.exe scripts/maintenance/check_context_golden.py`
- **Impact:** Nothing can fail if citation links are fabricated or point to excluded fragments; the matrix row is 'PRESENT' but unverifiable.
- **Suggested repair scope:** Either bind CitationChain to the assembled-context ledger (validate fragment ids against fragments_included, excerpt containment, tenant) or reclassify R016 as RAG-citation-only and test fabricated/foreign citations.
- **Owner path:** `intergrax/runtime/context/citation_chain.py; intergrax/rag/retrieval/citation.py; tests/qualification/pdc_x/d2 (new case)`

### F-D2-08 - MEDIUM - The cited RAG unit/integration suites are not green at the audited SHA (stale test doubles, collection errors, missing extras)

- **Rows:** PDC-D2-R004  ·  **Verification:** TEST_RUN
- **Evidence:** uv run pytest tests/unit/rag tests/integration/rag tests/integration/context tests/unit/runtime/architecture/test_ideal_harness_l3_w2_depth_gate.py --continue-on-collection-errors: 48 failed, 746 passed, 25 skipped, 13 collection errors (logs/run5_*). Attribution by error text (auditor classification, approximate): ~14 failures + 12 collection errors need optional extras (langchain_core, qdrant_client, docx, pandas, fitz, sentence_transformers; the default `uv run` environment has none); ~25 are stale test doubles/contract drift - FakeVectorstore/_NoopVectorstore lack list_source_record_ids(root_document_id=...) and abstract methods (contract change a61f091e3 2026-08-09 / ae14377c9 2026-08-13), _BridgeHandler/_DummyHandler lack abstract load(), tests import the removed PipelineDocumentHandler, tests pass RagProfile(graph_store_backend=...); further failures: FakeVectorStoreManager.supports_native_hybrid_search, InmemoryVectorStoreIntegration.list_collections, same-source reingest generation tests, test_ideal_w2_w2_script_gates (F-D2-10). Nine failures (test_ingest_lci2b_loader_boundary, test_ingest_pipeline_graph_wiring) end in ingest_pipeline.py:596 _list_current_source_ids, a B-D2 write-set file; PLAN_T-IGX-036-N acceptance required 'existing suites for the touched packages stay green'.
- **Reproduction:** `uv run pytest tests/unit/rag tests/integration/rag tests/integration/context -q --continue-on-collection-errors`
- **Impact:** R004 (RAG retrieval correctness) names these suites as its evidence; ~25 stale tests cannot detect regressions and the vector-store cross-tenant isolation unit file is not even collectable in the default environment. The reds pre-date B-D2 (not introduced by it) but were never repaired or classified.
- **Suggested repair scope:** Update the doubles to the vector-store/document-handler contracts, delete or port tests of removed APIs, add the RAG extras to the qualification profile (or mark env-dependent modules explicitly), re-run and record a green baseline.
- **Owner path:** `tests/unit/rag/**, tests/integration/rag/** (test-only); extras selection in the qualification runner`

### F-D2-09 - LOW - tests/integration/rag/test_neo4j_live_qualification.py has an IndentationError (line 613) and cannot be collected

- **Rows:** PDC-D2-R004  ·  **Verification:** TEST_RUN
- **Evidence:** python -c ast.parse(...) -> IndentationError: unexpected indent at line 613 ('with pytest.raises(RuntimeError, match="cypher error"):'); last modified by ed54bf41f (2026-08-14); collection error in logs/run4_*.
- **Reproduction:** `python -c "import ast;ast.parse(open('tests/integration/rag/test_neo4j_live_qualification.py',encoding='utf-8').read())"`
- **Impact:** The live Neo4j GraphRAG qualification is silently lost from every run (it is also skipped without INTERGRAX_RUN_NEO4J_LIVE, so the break is invisible).
- **Suggested repair scope:** Fix the indentation; add a compileall/py_compile gate over tests/ so a syntax error cannot hide behind a skip marker.
- **Owner path:** `tests/integration/rag/test_neo4j_live_qualification.py`

### F-D2-10 - LOW - Stale pre-context policy wiring gate (scan root moved) and a docstring-coupled context test are red

- **Rows:** PDC-D2-R006, PDC-D2-R016  ·  **Verification:** TEST_RUN
- **Evidence:** scripts/maintenance/check_pre_context_policy_wiring.py -> 'missing scan root: ...\intergrax\agents\uaep.py' (UAEP was relocated by 42cbe7e1a on 2026-09-20; runtime/policy/pre_context_policy_audit.py still scans the old path); the loop in test_ideal_w2_w2_script_gates stops at the first failing script, so the later scripts are not exercised by that test (run individually: check_tool_injection_defense and check_context_golden pass). tests/integration/context/test_mem_xint6_cross_layer_e2e_certification.py::test_typed_sources_documented_on_provider_context asserts a docstring phrase in context/contracts.py that no longer exists.
- **Reproduction:** `.venv/Scripts/python.exe scripts/maintenance/check_pre_context_policy_wiring.py`
- **Impact:** The static guard that the pre-context policy hook stays wired in assembly has been red since 2026-09-20 (runtime wiring itself is exercised: run_pre_context_policy_gate in context_engine.py and ce_01/ce_02 pass).
- **Suggested repair scope:** Point the audit at the relocated UAEP path (or derive the roots from the registry), make the script loop report every failing script, replace the docstring assertion with a typed check.
- **Owner path:** `intergrax/runtime/policy/pre_context_policy_audit.py; scripts/maintenance/check_pre_context_policy_wiring.py; tests/unit/runtime/architecture/test_ideal_harness_l3_w2_depth_gate.py`

### F-D2-11 - LOW - A raising reindex hook turns a successful retrieval into an exception; hooks are process-global and tenant-blind (P7-00 F-17 confirmed)

- **Rows:** PDC-D2-R005  ·  **Verification:** PROBE_CONFIRMED
- **Evidence:** embedding_version_policy.py:157-158 calls every registered hook without try/except; retrieval_service.py:226 evaluates freshness inside the result construction and only the observer call is guarded (:236-253). Probe P4: a hook raising RuntimeError makes RetrievalService._assess_freshness raise RuntimeError. ReindexQueueRequest carries source_path only (no tenant/workspace); the hook list is a module global.
- **Reproduction:** `uv run python .../D2/adversarial_scripts/probe_d2.py  (probe P4_raising_hook)`
- **Impact:** An operator hook failure breaks retrieval for every request that touches a stale source; a multi-tenant host cannot attribute a reindex request to a tenant.
- **Suggested repair scope:** Isolate each hook (log + count), make request_reindex fail-soft in the retrieval path, add tenant/scope to ReindexQueueRequest, scope hooks per service instance.
- **Owner path:** `intergrax/rag/governance/embedding_version_policy.py:142-159; intergrax/rag/governance/source_freshness.py:239-262; intergrax/rag/retrieval/retrieval_service.py:236-253`

### F-D2-12 - LOW - Freshness identity and fingerprint: mutable metadata key, absolute server path stamped into chunk metadata and citations, size+mtime revision (P7-00 F-18 partly confirmed)

- **Rows:** PDC-D2-R005  ·  **Verification:** CODE_READ_BY_AUDITOR (not exploited: no observer is wired)
- **Evidence:** source_freshness.py:149-155 chunk_source_id keys on metadata source_path/source/source_id (callbacks win over the stamp: ingest_pipeline.py `{**freshness_stamp, **callback}`); ingest_pipeline.py:84 stamps source_path=str(path) (absolute) which citation_from_chunk copies into Citation.metadata (citation.py:34-47); observe_file_source (source_freshness.py:130-141) stats the id it is given and uses '<mtime_ns>-<size>' as revision (content edits that keep size and mtime are invisible); a loader-supplied provenance.source_revision in another format than the observer's revision is a permanent false 'stale'. Cross-tenant verdict collapse (the dossier's claim) is not reachable because retrieval is scoped per tenant before the assessment.
- **Reproduction:** `read source_freshness.py:130-155 and ingest_pipeline.py:73-120`
- **Impact:** Path disclosure to the model/user; a file-stat observer built on chunk metadata is a stat oracle; weak change detection.
- **Suggested repair scope:** Stamp a relative/opaque source id and a content digest, keep the observer key outside caller-writable metadata, document the revision format contract.
- **Owner path:** `intergrax/rag/ingest/ingest_pipeline.py; intergrax/rag/governance/source_freshness.py`

### F-D2-13 - LOW - Retention sweep task: producer-controlled as_of, payload-selected user, unbounded thread join (P7-00 F-15; tenant-fallback claim not reproduced)

- **Rows:** PDC-D2-R015  ·  **Verification:** CODE_READ_BY_AUDITOR
- **Evidence:** retention_sweep.py:337 as_of_iso from the payload sets the evaluation instant (far future expires every entry that has valid_until or a policy window; forget is irreversible); :331 identity_resolver(tenant, user) is invoked for the payload user_id; :288-291 thread.join() has no timeout. The dossier's 'falls back to the payload tenant' is imprecise: :325 falls back to the queue-admitted `tenant_id` argument and :326 requires body.tenant_id == tenant_id (test_background_retention_sweep_rejects_a_payload_for_another_tenant passes).
- **Reproduction:** `read retention_sweep.py:309-346`
- **Impact:** Anyone who can enqueue the task inside a tenant can trigger premature irreversible expiry for any user of that tenant; cancellation/deadline cannot interrupt the blocking join.
- **Suggested repair scope:** Drop payload as_of (use the injected clock) or bound it to now+skew, restrict the payload user to the admitted principal or a service identity, add a join timeout/cancellation.
- **Owner path:** `intergrax/memory/retention_sweep.py:296-346`

### F-D2-14 - LOW - Tenant-bound plane guard depends on isinstance(UserProfileManagerMemoryCapability) and a private manager attribute (P7-00 F-36 confirmed)

- **Rows:** PDC-D2-R002  ·  **Verification:** CODE_READ_BY_AUDITOR
- **Evidence:** default_memory_control_plane.py:213-216 reads self._manager._tenant_id; :379-386 applies the guard only when the capability is that concrete class and only in USER scope. A duck-typed capability, or TASK/SESSION scope (F-D2-02), is unguarded. Mutant M9 (guard no-op) is killed by test_xd02_a_foreign_tenant_identity_on_a_tenant_bound_plane_recalls_nothing.
- **Reproduction:** `read default_memory_control_plane.py:211-216,379-386`
- **Impact:** The guard protects the shipped wrapper only; a replacement capability must re-implement the check.
- **Suggested repair scope:** Add bound_tenant_id to the capability protocol and make the guard protocol-based (fail closed when absent in strict profiles).
- **Owner path:** `intergrax/memory/default_memory_control_plane.py; intergrax/memory/contracts/memory_control.py`

### F-D2-15 - LOW - Required source whose fragments are all egress-blocked fails closed with an untyped ValueError that names the wrong cause (P7-00 F-19 partly refuted)

- **Rows:** PDC-D2-R014, PDC-D2-R007  ·  **Verification:** PROBE_CONFIRMED
- **Evidence:** Probe P1: request.required_sources={RAG}, only fragment CONFIDENTIAL -> assemble raises ValueError('required source missing after collect: rag') from runtime/policy/context_assembly_policy.py:40 (post-collect policy gate), not RequiredContextSourceUnavailableError. validate_required_sources_fulfilled still runs before filter_fragments_by_provider_egress (context_engine.py:321-334), so the typed validation is satisfied by fragments that are then withheld; the second gate catches it. No silent withholding occurred.
- **Reproduction:** `uv run python .../D2/adversarial_scripts/probe_d2.py  (probe P1_required_vs_egress)`
- **Impact:** Fail-closed holds; the error class and reason code lose the egress cause (diagnosis, retry classification).
- **Suggested repair scope:** Run the egress filter before required-source validation or map the post gate error to RequiredContextSourceUnavailableError with reason_code sensitivity.provider_egress_blocked.
- **Owner path:** `intergrax/runtime/nexus/context/context_engine.py:321-334; intergrax/runtime/policy/context_assembly_policy.py:40`

### F-D2-16 - LOW - A provider-declared tool_call_id is only checked for presence: an orphan tool result reaches model-facing context

- **Rows:** PDC-D2-R009, PDC-D2-R011  ·  **Verification:** PROBE_CONFIRMED
- **Evidence:** provider_lifecycle.py:293-302 accepts any non-blank tool_call_id; formatter.py:33-45 emits role='tool'; merge_iterative_tool_feedback_messages (formatter.py:96) appends unmatched tool messages at the end. Probe P8: TOOL_OUTPUT fragment with tool_call_id 'call-does-not-exist' -> messages [tool(call-does-not-exist), user], linkage violation 'result without call'. The accepted test test_linked_tool_output_stays_a_tool_message_never_a_system_instruction itself uses an unmatched id (call-9).
- **Reproduction:** `.venv/Scripts/python.exe .../D2/adversarial_scripts/probe_d2_b.py`
- **Impact:** Many provider APIs reject a tool message without a preceding assistant tool call: a provider can make a run fail (availability). Not an instruction-promotion path.
- **Suggested repair scope:** Validate the id against the assistant tool_calls of the assembled window (drop or fail closed); fix the test to use a matched id.
- **Owner path:** `intergrax/context/provider_lifecycle.py; intergrax/context/formatter.py`

### F-D2-17 - LOW - Last-resort hard budget drops the newest non-mandatory messages first and keeps the oldest

- **Rows:** PDC-D2-R011, PDC-D2-R010  ·  **Verification:** PROBE_CONFIRMED
- **Evidence:** context_compiler.py:345-357 iterates range(len-1, -1, -1). Probe P6: 6 history messages, budget 130 -> kept ['s','h0','h1','u'], dropped h2..h5 (the most recent history).
- **Reproduction:** `uv run python .../D2/adversarial_scripts/probe_d2.py  (probe P6_drop_order)`
- **Impact:** When the ladder is exhausted the model loses the freshest context and keeps the stalest; contradicts the oldest-first ladder step.
- **Suggested repair scope:** Iterate oldest-first among non-mandatory units (still dropping tool-call groups atomically) and add an ordering assertion to the R011 compiler test.
- **Owner path:** `intergrax/runtime/nexus/context/context_compiler.py:337-370`

### F-D2-18 - LOW - Weak negative tests in the compaction round trip: broad except-and-return escapes and a reconstruction check against in-test originals

- **Rows:** PDC-D2-R012  ·  **Verification:** CODE_READ_BY_AUDITOR
- **Evidence:** test_d2_compaction_roundtrip.py:153-156 `except ValueError: return`; :209-212 `except Exception: return`; test_d2_context_fidelity.py:274-280 accepts either an error or an intact instruction; the reconstruction assertion (:125-127) maps source_refs to the in-test `originals`, not to a durable message store. Any unrelated ValueError/Exception makes these tests pass.
- **Reproduction:** `read the cited lines`
- **Impact:** Tests can pass for the wrong reason (reduced evidence for R012 and R007 negative cases). Positive round-trip assertions and the other D2 tests were mutation-checked (M1..M9 killed).
- **Suggested repair scope:** Assert the specific exception type/message (or the typed refusal) and reconstruct from the durable store.
- **Owner path:** `tests/qualification/pdc_x/d2/test_d2_compaction_roundtrip.py; test_d2_context_fidelity.py`

### F-D2-19 - INFO - Ownership citations in the matrix do not name the code that carries the claim

- **Rows:** PDC-D2-R003, PDC-D2-R009, PDC-D2-R017, PDC-D2-R018  ·  **Verification:** CODE_READ_BY_AUDITOR
- **Evidence:** R018: UclArtifactOwnership (context_lifecycle/contracts.py:690) is workspace ownership of optimization artifacts, not evidence protection; the claim holds structurally (no DELETE/prune path in intergrax/runtime/context_lifecycle/*.py) and by xd_09 (D8 store). R009: ToolInjectionDefenseMiddleware.after is a no-op and its before() guards tool INPUT; the output boundary lives in provider_lifecycle/formatter. R003: OrganizationProfileManager.add_memory_entry mutates organization memory outside MemoryControlPlane (plane scopes are TASK/USER/SESSION). R017: two classes named ContextFragment still exist (classified separately in UNIFIED_CONTEXT_LIFECYCLE.md:390, no import collision found).
- **Reproduction:** `read the cited locations`
- **Impact:** Documentation/traceability only.
- **Suggested repair scope:** Correct the PUBLIC_CONTRACT/owner columns of R009, R018 and the scope statement of R003 at the next matrix revision.
- **Owner path:** `docs/project/maintainers/qualification/platform_domain_completeness/PDC_DOMAIN_REQUIREMENTS_MATRIX.json`

### F-D2-20 - INFO - Scoped result and environment notes

- **Rows:** PDC-D2-R008  ·  **Verification:** CODE_READ_BY_AUDITOR+TEST_RUN
- **Evidence:** R008: tools/providers/rag/service.py:212-223 now defaults the defense ON for an absent profile in every mode (retrieval_poisoning_defense_enabled_or_default); runtime/nexus/tools/plan_context_invocation.py:152-165 is fail-closed only when ctx.config.production_mode is true (Operator Amendment 1(c) of PLAN_T-IGX-036-P: 'fail-closed in strict profiles'); in non-production mode with no security_profile that path still runs without the filter. d6 poisoning tests: 14 passed (logs/run7_*). R-05 debt: pytest exited by itself after every run of this audit, including tests/qualification/pdc_x/xd/test_xd_03_agent_governance.py (9 passed, exit in 10 s; logs/run6_*). The default environment lacks optional extras (see F-D2-08); no package was installed.
- **Reproduction:** `uv run pytest tests/qualification/pdc_x/d6 -k poison -q`
- **Impact:** PASS_SCOPED for R008 is an Operator-authorized scope limit, not a defect.
- **Suggested repair scope:** none
- **Owner path:** `intergrax/runtime/nexus/tools/plan_context_invocation.py (D6 batch)`


## 6. P7-00 leads routed to IA-D2

| P7-00 lead | Disposition at this SHA |
| --- | --- |
| F-05 | FINDING-6 tenant-bound memory plane re-verified at this SHA (xd_02 + mutant M9); residual tenant gaps are F-D2-02/03/14; FINDING-8/9/10 belong to other domains |
| F-15 | F-D2-13 (as_of and join confirmed; tenant-fallback wording not reproduced) |
| F-17 | F-D2-11 (confirmed by probe P4) |
| F-18 | F-D2-12 (path stamp, weak fingerprint, mutable key confirmed by code; cross-tenant collapse not reachable) |
| F-19 | F-D2-15 (ordering confirmed; outcome is fail-closed through a second gate) |
| F-28 | routed to IA-D8; not audited here |
| F-36 | F-D2-14 (confirmed) |
| F-09 | not reproduced (pytest exits after test_xd_03_agent_governance.py) |

## 7. Overall verdict

**PARTIAL.** No row fails and no HIGH finding exists; the eleven PASS/PASS_SCOPED rows are backed by discriminating tests (mutation-checked) and code reading. Seven rows are PARTIAL (R002, R004, R005, R010, R014, R015, R016).
Per `OD-PDC-IIAC-SCOPE` the repair is a ticket of the owning domain by a different session, followed by a fresh re-audit (max two loops). Suggested grouping: (a) context core, F-D2-01/04/15/16/17 in `intergrax/context` and `intergrax/runtime/nexus/context`;
(b) memory, F-D2-02/13/14 plus F-D2-06 retention composition; (c) RAG, F-D2-06 freshness composition, F-D2-11, F-D2-12; (d) test-only repair, F-D2-03, F-D2-08, F-D2-09, F-D2-10, F-D2-18, F-D2-07 (decide first whether R016 binds code or is reclassified);
(e) F-D2-05 needs an Operator-level design decision (classification source and default for unclassified fragments).

## 8. Limitations (what this audit did not cover)

- Other domains' audit reports were not read. D6/D8 behaviour was touched only where a D2 row cites it (R008 d6 poisoning tests, R018 xd_09).
- The default `uv run` environment has no optional extras; 12 collection errors and ~14 failures are environment dependent (qdrant, langchain_core, docx, pandas, fitz, sentence_transformers). The vector-store cross-tenant isolation unit file (`test_vectorstore_cross_tenant_isolation.py`) and the Qdrant/Chroma/pgvector/Neo4j live qualifications were not executed; nothing was installed and the C10/C15 freeze (no Docker, no new vendor integration) was respected.
- Failure attribution of the RAG suite (stale doubles vs extras) is by error text, not by bisecting each test against the pin.
- Not run: the full `tests/unit` tree, EBH-7 and `roadmap_replay_x`, CI contexts, `igx_gov ticket`, FRZ counters, P8 cross-domain checks, performance and load behaviour, non-Windows platforms.
- R013 and R018 were verified through the existing restart/evidence suites and structural greps, without a new restart-corruption probe. R006 "single assembly owner" was checked by search over `intergrax/`, not by a proof of exhaustiveness.
- Token estimators were exercised with the fake adapter and character counters; real tokenizer behaviour (tiktoken, provider-specific) was not measured.
- F-D2-12 and F-D2-13 are code-read findings without an exploit run (no observer or sweep is composed). F-D2-03 attributes at least three of the 12 unreconciled tenant carriers to D2; the full set was not isolated.
- IIAC manifest verdict enum has no PARTIAL; `CONDITIONAL` is used as the closest value (as in the P7-00 manifest). `certifier_session_id` and `evidence_publication_sha`/`certification_publication_sha` are placeholders until a certifier publishes. The `Ref:` trailer regex for `T-IGX-036-N` applies (ticket exists); this report adds no product path.
