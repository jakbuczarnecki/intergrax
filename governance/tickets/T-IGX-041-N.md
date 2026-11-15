# T-IGX-041-N — PDC-X batch B-P8-R2: P8 repair loop 1, test hygiene of the closure suites and the incident scenario

## Scope

Test, data and classification repair after the independent global audit IA-GLOBAL (`PDC_INDEPENDENT_AUDIT.md`, pull request `jakbuczarnecki/intergrax#579`, verdict PARTIAL, audited SHA `cba32d2479cd134911d2fdf4fb1db4587c8a94d8`), under decision 2 of `DELEGATION-PDC-CLOSE.md` (one P8 repair loop) and the limits of `DELEGATION-PDC-P7-REPAIR.md` (option B; no hotspot path outside exactly named files; gates only strengthened, classified or re-pinned). The batch writes no product path and no hotspot path. In scope: the test-side residue of G-05 (stale keyword, missing occurrence persistence, an invalid proof descriptor value and a new descriptor gate), G-04 (the `FakeVectorstore` double, one reingest test that queries the coordinator with the wrong key, the wrong CI claim of `RAG_RED_CLASSIFICATION.md`) and G-10 (three stale D5-contract tests and the two `test_audit_ideal_depth_gate.py` reds). Excluded and routed to T-IGX-041-M or an Operator decision: the authority, manifest and registry layers behind the 82 `AgentBinding` failures of the incident scenario (hotspot file `scenario_runtime_baseline.py` and the `ModelCallRuntimeEvidenceAdapter` token count), the two graph reingest tests and `test_graphrag_source_reingest_replaces_vectors_and_graph_safely` (likely product defect in the graph visibility keys), the rebuild-path and trace-schema D5 tests, and the four declared `fixture_strict_adapter_parity` tests. The plan is not a certification verdict.

## Plan digest

Plan digest: c11c4b605c4b6aa815141afe1da536b6ed088b4c1028bc5f107849c9ea0d2d65

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-041-N.md`](plans/PLAN_T-IGX-041-N.md).

## Approved

Approved: operator-delegate @ 2026-11-15T15:20:00+01:00
Delegation: docs/project/maintainers/qualification/platform_domain_completeness/operator_decisions/DELEGATION-PDC-CLOSE.md@7857632dd064e9fd6583d975d781cca516c5bca5

Delegate checks (scope of `DELEGATION-PDC-CLOSE.md` decision 2, which applies the limits of `DELEGATION-PDC-P7-REPAIR.md`; the record is on `origin/development`, last changed by `7857632dd`; plan read at commit `c8386450c` of the batch branch). A first signing attempt at `bb566b278` was refused because Deliverable 3 justified `LOCAL_READ_ONLY` for `proof.json` with an "in-process order provider" that does not exist (the provider is a Docker HTTP Order Service on `127.0.0.1:18091`); the planner amended the plan to pin `LOCAL_MUTATING` and the checks were redone from scratch:

- The plan digest recomputed with `scripts/governance/plan_digest.py` equals the digest above (`c11c4b605c4b6aa815141afe1da536b6ed088b4c1028bc5f107849c9ea0d2d65`); the amendment changed two plan lines and this digest line.
- The write set of the plan (10 entries) equals the write set of batch row `B-P8-R2` in `pdc_x_wbs.py` and the regenerated `PDC_IMPLEMENTATION_WBS.md` (compared as lists, equal; `pdc_x_wbs.py --check` OK). It contains no path under `intergrax/`, `agents/`, `applications/` or `testing_support/`, no hotspot path (`intergrax/applications/_shared/`, `intergrax/contracts/`), nothing under `scripts/governance/` or `.github/`. The branch commits touch only the ticket, the plan, the WBS, the DAG and `pdc_x_wbs.py`.
- Reproduced by the delegate at the base: `tests/unit/platform_proofs/scenarios/ai_incident_investigation` gives 99 failed, 186 passed, 9 skipped, split 82 (`AgentBinding requires agent_type or import_path`) + 7 (`occurrence_persistence`) + 2 (`decision_accepted`) + 6 (`ManifestLoadError` of the indirect-prompt-injection `proof.json`) + 2 (`FAIL_MANIFEST` assertions at `test_canonical_reproduction.py:81` and `:145`); the three D5 and depth files give 7 failed, 91 passed; the RAG test files give 5 failed (`FakeVectorstore`) and 3 failed with 3 passed (reingest); a probe with `publication_scope_id` in the coordinator key makes `test_stale_inflight_publication_cannot_become_active_after_takeover` pass (probe file removed). Spot-checked file:line claims (about 35): `agent_resolution.py:35`, `runtime_composition.py:137`, `scenario_runtime_baseline.py:347-370, 373-389, 408-415`, `completion_reconciliation.py:153-160` and the three stale keywords, the constructors of `ProblemLifecycleEngine` and `DiagnosticReadService` and the helper `document_store_occurrence_persistence_for_tests`, `vectorstore_manager.py:401`, `rag-guard.yml` selection (`-m gate`, no `tests/integration/rag` in any workflow), `ingest_pipeline.py:345-351`, the graph keys without `publication_scope_id`, `host_execution_capacity_policy.py:44`, `production_capacity_wiring.py:47-55` with the PRODUCT plus `production_adapters_enabled` branch, `qualification_runtime_trace.py:101`, `validating_evidence_persistence_port.py:57`, `VisionProfile.create_adapter().slug == "onnxruntime"`, `product_defaults()` capacity 8/8; the audit (`PDC_INDEPENDENT_AUDIT.md` at `54f1d21d5`) states 103 = 4 + ~99 and the five D5 reds at `f87df3b2a`.
- Amended Deliverable 3: `order_provider_client.py:1,19` and `:28` (`reset`, `POST /debug/reset` at `:32`), the Docker run section of the scenario README, the README statement "actual provider write requests = 0", `intergrax_proof_contracts.py:46-50` (four members), `intergrax_platform_proof_descriptor_loader.py:89` and the absence of in-process provider implementations are true. The only execution check on `safety_class` is `intergrax_proof_runner.py:220` (`EXTERNAL_MUTATING`), so `LOCAL_MUTATING` triggers none; no file pins the content or digest of `proof.json` (IIAC matrix entries list the path only). The pinned class is the higher of the two plausible members and the plan forbids a lower one.
- Findings: in-scope table (G-05 residuals a, b, c; G-04 doubles, reingest test a., text; G-10 test side and depth) and excluded table (authority and rebuild layers, token count, graph reingest, rewire test, trace-schema test, the four declared parity tests, further D2 reds, other findings) each carry reason and reopen condition.
- No gate is weakened or deleted: the keyword, constructor, fixture and patch-target edits keep their assertions; `29_1` keeps its assertion on the adapter slug; `30_4` replaces an expectation the product closed deliberately (commit `aa5bf08ac`) by a pin of the fail-closed `ControlPlaneCompositionError`, a strengthening; the new `n_repair` descriptor gate is a new strengthening gate; no skip, xfail or `importorskip` is added. No registry, scheduler, engine, trust, policy or audit authority is added; C10/C15 respected; the plan states it is not a certification verdict.
- The implementer session and this delegate session are different sessions; the delegate edited nothing except this approval block and the lifecycle state. Not checked by the delegate: the 82-test layers behind the authority and rebuild paths beyond the implementer's experiment logs, and that each remaining red in PHASE 2 carries an excluded cause (the plan makes that an acceptance item).
- Merge gate (as the record states): the pull request merges under authorizations 1 and 3 of `DELEGATION-PDC-STANDING.md` after green CI with known red classes only and a head match; WBS gate "no product path: may merge before MCP-SERVER-X P8". A third loop, a new kind of finding, a hotspot path outside named files, a change of the meaning of a gate or a new authority returns to the Operator. Write-set amendments are not delegated.

## Test contract

```text
uv run pytest tests/qualification/pdc_x/n_repair tests/unit/platform_proofs/scenarios/ai_incident_investigation tests/unit/rag/indexing/test_indexing_manager_indexes_documents.py tests/integration/rag/test_same_source_reingest_serialization.py tests/integration/rag/test_graph_reingest_qualification.py tests/unit/applications/test_harness_host_runtime_llm.py tests/unit/applications/test_scenario_runtime_baseline.py tests/unit/runtime/architecture/test_audit_ideal_depth_gate.py -q -p no:randomly
uv run ruff check <touched files>
uv run python scripts/qualification/pdc_x_wbs.py --check
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run python -m scripts.governance.igx_gov ticket
```

Known reds that remain by design and are not repaired here: the excluded rows of the plan (82 incident scenario tests behind the authority and rebuild layers, the three graph reingest tests, the rebuild-path and trace-schema tests of `test_scenario_runtime_baseline.py`), the four `fixture_strict_adapter_parity` tests, IIAC activation, EBH-7 identity.

## Out of scope

- The excluded findings (plan table) and every product or hotspot path (`intergrax/`, `agents/`, `applications/`, `testing_support/`).
- G-01, G-02, G-03, G-06 and G-07..G-12 of the audit (ticket T-IGX-041-M, T-IGX-041-K or the P9 registers).
- `scripts/governance/`, `.github/`, FRZ files and P8 originals.
- IIAC activation and rebind, the EBH-7 re-pin, OD signatures, certification and the re-audit.

## Implementation checkpoint

Logs under `.tmp/proof/pdc-x-041n/` (not committed). Environment: `uv sync --frozen` with `dev-ci`, `llm-all`, `dev-ci-rag`.

- New gate `tests/qualification/pdc_x/n_repair/test_n_proof_descriptor_gate.py`: red on the unchanged `proof.json` (2 failed, 5 passed), green after the descriptor repair (`safety_class` `LOCAL_MUTATING`, as pinned by the plan; no gate pins the file content, `IIAC_GLOBAL_AUTHORITY_IMPACT_MATRIX.json` lists its path only).
- Repaired and green: `test_completion_alignment.py` (21 passed), `test_indexing_manager_indexes_documents.py` (5 passed with `langchain-core`), `test_stale_inflight_publication_cannot_become_active_after_takeover`, `test_harness_host_runtime_llm.py` (3 passed), `test_build_scenario_runtime_strict_with_explicit_manifest`, `test_audit_ideal_depth_gate.py::29_1` and `::30_4`, and 5 of the 8 tests of `test_diagnostic_platform_integration.py`.
- Signed test contract, one pytest run after all repairs (`contract_final.log`): 95 failed, 306 passed, 9 skipped in 72 s. An earlier full run before the last edit of `test_diagnostic_platform_integration.py` gave 96 failed, 305 passed, 9 skipped; the pytest process of such runs does not exit after its summary line (observation R-05) and had to be stopped.
- The 95 failures are the excluded rows of the plan, all with a named cause:
  - 85 in the incident scenario with `AgentBinding requires agent_type or import_path` (the 82 of the plan plus 3 that were hidden behind the residual causes now repaired);
  - 6 in the incident scenario that run the real skeleton proof as a subprocess and now fail inside it (`test_parent_runner_integration_*` in `test_dual_canonical_evidence.py` and `test_skeleton_integration.py`, `test_canonical_cli_reproduces_scenario_proof`): the descriptor now loads, so the proof runs; the cause behind its failure was not traced (consistent with the `AgentBinding` cause, not proven);
  - 4 outside the scenario directory: `test_stale_graph_topology_is_inactive_after_generation_takeover`, `test_graphrag_source_reingest_replaces_vectors_and_graph_safely`, `test_rewire_scenario_decision_wiring_reapplies_validation_engine`, `test_execute_scenario_task_persists_finalized_trace_for_orchestration` (owner T-IGX-041-M).
- Newly seen layers behind repaired tests: the proof descriptor repair exposes the 6 skeleton-proof failures above; the `test_harness_host_runtime_llm` repair exposed that the autouse fixture of `tests/unit/applications/conftest.py` stubs the LLM resolver, so the provider-missing test calls `monkeypatch.undo()` first (inside the named file); the strict fixture of `test_scenario_runtime_baseline.py` needed explicit capacity caps and an explicit dependency concurrency admission (`build_dependency_concurrency_admission_for_tool_ids`), not the reference-host builder named in the plan, because that builder returns nothing when the lab tool profile has no available tools; `test_limitation_kind_rendered_in_platform_context_boundary` copied `DiagnosticProblemDetail` by hand and lacked its new field `occurrence_aggregate_health` (passed through, same file).
- `RAG_RED_CLASSIFICATION.md`: a section "Corrections after IA-GLOBAL (G-04, T-IGX-041-N)" was appended; the original rows are unchanged.
- Pyright on the touched files: 78 errors before, 64 after (the new gate file has 0); ruff check and format clean on all touched Python files. `ruff format` re-wrapped some lines of the legacy test files (formatting only).
- `pdc_x_wbs.py --check`, `pdc_x_requirements.py --check`, `test_pdc_x_wbs.py`: OK after the merge of `origin/development`. In the merge, `B-P8-R2` stays in wave W15 (already used by `B-P7-RP`, T-IGX-041-J); the duplicate wave-text entry of this ticket was dropped and the entry of T-IGX-041-J kept, so the generated text of W15 describes T-IGX-041-J only.

### Deviations from the signed plan

- Item 7 (second test of `test_harness_host_runtime_llm.py`): the plan lists deriving the orchestration profile from the environment's own; the test additionally calls `monkeypatch.undo()` to drop the package's autouse LLM stub (same file, same intent).
- Item 8 (`test_build_scenario_runtime_strict_with_explicit_manifest`): the fixture sets the capacity caps and an explicit admission instead of calling the reference-host builder (the plan allowed "the product's reference-host default builder or an explicit `max_inflight_nodes`"); the builder alone was tried and did not give the admission.
- Item 2 (`test_diagnostic_platform_integration.py`): one more stale constructor argument (`occurrence_aggregate_health`) was passed through in the same file.
- No file outside the signed write set was written; no product path and no hotspot path.

## Lifecycle state

plan_approved
