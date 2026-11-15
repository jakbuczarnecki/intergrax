# T-IGX-041-M — PDC-X P8 repair loop 1: four findings of the independent global audit

## Scope

Repair ticket for the interim findings of the independent global auditor (IA-GLOBAL, session `ia-global`, `development` at `cba32d247`), the one P8 repair loop of decision 2 of `DELEGATION-PDC-CLOSE.md`, under the limits of `DELEGATION-PDC-P7-REPAIR.md` (option B: real defects and hygiene; hotspot paths only as exactly named files; gates only strengthened or classified; no new authority). All four findings were re-verified in code by the planner. In scope: (1) HIGH, D2 residue F-D2-08: `doc_smart_document_handler.py:11-13` imports `PipelineDocumentHandler` from a module that does not define it, so `create_default_documents_loader()` raises `ImportError`; (2) MEDIUM, regression of T-IGX-041-G (F-D7-02): an exception in the eligibility reads of `LongRunningScheduler._process_claimed_schedule` (`scheduler.py:207-251`) leaves a recurring claim `RUNNING` with no successor, so the chain ends silently; (3) MEDIUM, F-D8-03 sibling of T-IGX-041-H: `workspace.export_artifact` and `workspace.import_artifact` (`tools/providers/workspace/service.py:127-193`) reach object storage with arbitrary keys; bounded repair: they refuse the Collaborative Work content key space like the raw `storage.*` tools; (4) two PDC-touched product paths (`intergrax/experiments/workflow.py`, `intergrax/runtime/policy/pre_context_policy_audit.py`) map to no ENT-AUDIT check, which keeps three `pdc_x` tests red; classification entries in the CHANGED-SINCE impact map (exact files, 85-check manifest unchanged), with a negative test that unmapped siblings still fail. Excluded with reason and reopen condition (plan section "Excluded findings"): tenant or prefix binding of all other object-storage keys and an audit event for the raw storage tools, release failure of the store, occurrence overlap, optional-extras reds, new ENT-AUDIT checks, any other IA-GLOBAL finding. Product paths: yes (six files, no hotspot). Batch: `B-P8-R1` (wave W14a). Plan: `PLAN_T-IGX-041-M.md`. Order: must merge before the EBH-7 pin P, together with T-IGX-041-K and T-IGX-041-L; T-IGX-041-L re-pins the COMPAT-X counters at the last product change and must be re-verified after this ticket. This ticket is not a certification verdict and not a closure of any domain; the product-path changes keep the IIAC activation stale (ACT-N11) until the P8-04 rebind and the EBH-7 executable-identity gate red until the final re-pin.

## Plan digest

Plan digest: 372ddec03e1d1be5d4412680a5ec62148ee1bf05b7114b3bff706df0caa84c31

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-041-M.md`](plans/PLAN_T-IGX-041-M.md).

## Approved

(pending; the plan is signed by a delegate session under decision 2 of `DELEGATION-PDC-CLOSE.md` within the limits of `DELEGATION-PDC-P7-REPAIR.md`; the implementer never writes this section)

## Test contract

```text
uv run pytest tests/qualification/pdc_x/p8_repair -q
uv run pytest tests/qualification/pdc_x/d7_repair tests/qualification/pdc_x/d7 tests/qualification/sched_01 -q
uv run pytest tests/qualification/pdc_x/d8_repair tests/qualification/pdc_x/d8 tests/unit/tools/providers/test_catalog_expansion.py tests/unit/tools/providers/test_t10_catalog_tools.py -q
uv run pytest tests/unit/rag/document_loaders tests/integration/rag/document_loaders/test_fixture_dataset_ingestion.py -q
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
