# T-IGX-041-M — PDC-X P8 repair loop 1: four findings of the independent global audit

## Scope

Repair ticket for the product defects found by the independent global auditor (IA-GLOBAL, session `ia-global`; interim findings at `development` `cba32d247`, full table in sections 7 and 9 of `PDC_INDEPENDENT_AUDIT.md`, PR #579), the one P8 repair loop of decision 2 of `DELEGATION-PDC-CLOSE.md`, under the limits of `DELEGATION-PDC-P7-REPAIR.md` (option B: real defects; hotspot paths only as exactly named files; gates only strengthened or classified; no new authority). Product defects only; test hygiene, register and debt wording belong to other tickets. Every item was re-verified in code by the planner. In scope: (G-01, HIGH) three defects of the RAG loader family: `doc_smart_document_handler.py:11-13` imports `PipelineDocumentHandler` from a module that does not define it, so `create_default_documents_loader()` raises `ImportError`; `_KnowledgeDocumentWithParserRuntime` is defined twice (`contracts/parser_native_handle.py`, `compat/legacy_runtime_document.py`), so a native parser handle attached by a handler is invisible to `get_parser_native_handle` (red `test_parser_pipeline`); `InmemoryVectorStoreIntegration` has no `list_collections` (red `test_hierarchical_dual_index_wiring`). (G-02, MEDIUM) regression of F-D7-02: an exception in the eligibility reads of `LongRunningScheduler._process_claimed_schedule` (`scheduler.py:207-251`), or an `ExecutionTerminalError` that both resume gates turn into "not resumable", leaves a recurring claim `RUNNING` with no successor or ends the chain as `SKIPPED`: release the claim and retry. (G-03, MEDIUM) `workspace.export_artifact` and `workspace.import_artifact` reach object storage with arbitrary keys, and the guard of T-IGX-041-H is bypassable by case and trailing-dot variants and by absolute keys on the filesystem store (reproduced). (G-06, MEDIUM) two PDC-touched product paths map to no ENT-AUDIT check (three `pdc_x` tests red): exact-file entries in the CHANGED-SINCE impact map, with a negative test that unmapped siblings still fail. (LOW) F-D6-12: `from None` leaves provider text in `__context__`. (MEDIUM, test-only) F-D1-11: pin that a changed integration binding slug is still refused by the durable projection input store; no hotspot file is edited. Excluded with reason and reopen condition (plan section "Excluded findings"): tenant or prefix binding of all other object keys, non-CAS `recover_interrupted_running` (documentation limit only), other vector store wrappers, the extras-profile reds and reingest/takeover reds of G-04, F-D7-20, F-D3-02 pins, G-11 stream resilience, G-05, G-09, G-10, G-12 and the LOW debt. Product paths: yes (eleven files, no hotspot). Batch: `B-P8-R1` (wave W14a). Plan: `PLAN_T-IGX-041-M.md`. Order: must merge before the EBH-7 pin P, together with T-IGX-041-K and T-IGX-041-L; T-IGX-041-L re-pins the COMPAT-X counters at the last product change and must be re-verified after this ticket. This ticket is not a certification verdict and not a closure of any domain; the product-path changes keep the IIAC activation stale (ACT-N11) until the P8-04 rebind and the EBH-7 executable-identity gate red until the final re-pin.

## Plan digest

Plan digest: d70a6df7e6521717aa34058327368401660ff2747639c31853c04ac059c0e9af

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-041-M.md`](plans/PLAN_T-IGX-041-M.md).

## Approved

(pending; the plan is signed by a delegate session under decision 2 of `DELEGATION-PDC-CLOSE.md` within the limits of `DELEGATION-PDC-P7-REPAIR.md`; the implementer never writes this section)

## Test contract

```text
uv run pytest tests/qualification/pdc_x/p8_repair -q
uv run pytest tests/qualification/pdc_x/d7_repair tests/qualification/pdc_x/d7 tests/qualification/sched_01 -q
uv run pytest tests/qualification/pdc_x/d8_repair tests/qualification/pdc_x/d8 tests/qualification/pdc_x/d6_repair tests/qualification/pdc_x/d1_repair/test_d1_repair_durable_projection_inputs.py -q
uv run pytest tests/unit/tools/providers/test_catalog_expansion.py tests/unit/tools/providers/test_t10_catalog_tools.py -q
uv run pytest tests/unit/rag/document_loaders tests/unit/rag/ingest/test_hierarchical_dual_index_wiring.py tests/integration/rag/document_loaders/test_fixture_dataset_ingestion.py -q
uv run pytest tests/qualification/pdc_x/p6_impact tests/qualification/pdc_x/xd/test_xd_12_evol_ent_audit.py -q
uv run pytest tests/qualification/compat_x -q
uv run ruff check <touched files>
uv run ruff format --check <touched files>
uv run pyright <touched files>
uv run python scripts/qualification/pdc_x_wbs.py --check
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

- Every item of the plan section "Excluded findings"; the declared exceptions of `DELEGATION-PDC-CLOSE.md`; any hotspot path (`intergrax/applications/_shared/`, `intergrax/contracts/`); `scripts/governance/`, `.github/`; the ENT-AUDIT manifest and checks; the existing storage tools and their gate; every existing test.
- The EBH-7 re-pin, the IIAC activation record, OD signatures, certification.

## Lifecycle state

plan_pending
