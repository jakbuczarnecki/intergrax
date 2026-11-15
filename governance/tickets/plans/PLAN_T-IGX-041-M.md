# PLAN T-IGX-041-M — PDC-X P8 repair loop 1: four findings of the independent global audit (IA-GLOBAL, interim)

Parent: PDC-X campaign, P8 repair loop 1. Origin: the interim findings of the independent global auditor (session `ia-global`, `development` at `cba32d247`). Authority: decision 2 of `operator_decisions/DELEGATION-PDC-CLOSE.md` (one P8 repair loop; the plan is signed by a delegate under the limits of `DELEGATION-PDC-P7-REPAIR.md`: option B, hotspot paths only as exactly named files, gates only strengthened or classified, no new authority; a third loop, a finding of a new kind, a hotspot path outside the named files, a change of the meaning of a gate or a new authority returns to the Operator). Batch row: `B-P8-R1` in `PDC_IMPLEMENTATION_WBS.md`. This plan is written by the implementer lane (PHASE 1) and is **unsigned** (`plan_pending`); the implementer never writes `Approved:` or `Delegation:` lines. It is **not a certification verdict** and certifies nothing.

Every statement below was verified by the planner at `cba32d247` (reading the cited lines, or running the cited command in a throwaway script that is not committed). The auditor's line ranges were checked and corrected where they differed.

## Problem

### Finding 1 (HIGH, D2 residue, F-D2-08) — the default documents loader cannot be created

- `intergrax/rag/document_loaders/handlers/doc_smart_document_handler.py:11-13` imports `PipelineDocumentHandler` from `intergrax/rag/document_loaders/contracts/base_document_handler.py`. That module defines only `BaseDocumentHandler` (`:14`). `PipelineDocumentHandler` is defined at `intergrax/rag/document_loaders/pipeline/document_handler_loading.py:81` (`__all__` at `:99`).
- Observed: `import intergrax.rag.document_loaders.handlers.doc_smart_document_handler` raises `ImportError: cannot import name 'PipelineDocumentHandler' from 'intergrax.rag.document_loaders.contracts.base_document_handler'`. `create_default_documents_loader()` (`bootstrap/default_loader.py:35`) imports the handler at `:43` and registers it at `:54`, so it raises the same error when no registry is passed.
- The seven sibling handlers import the class from the pipeline module (`audio_smart_document_handler.py:11`, `excel_smart_document_handler.py:15`, `html_document_handler.py:11`, `image_smart_document_handler.py:14-17`, `pdf_smart_document_handler.py:14`, `text_smart_document_handler.py:10`, `video_smart_document_handler.py:10`).
- Observed with the import redirected in memory (no file edited): after the one-line redirect `create_default_documents_loader()` returns a `DocumentsLoader` in the default environment, so no second defect hides behind the import.
- T-IGX-041-B already classified this as a product defect outside its write set: `tests/qualification/pdc_x/d2_repair/RAG_RED_CLASSIFICATION.md:39` ("handler registry import fails"; `test_fixture_dataset_ingestion.py::test_docx_ingestion` and `::test_txt_ingestion`). The fix is not hotspot code (`intergrax/rag/` is not under `intergrax/applications/_shared/` or `intergrax/contracts/`).

### Finding 2 (MEDIUM, F-D7-02 regression of T-IGX-041-G) — an exception before the successor is persisted ends a recurrence chain silently

- `LongRunningScheduler._process_claimed_schedule` (`intergrax/runtime/long_running/scheduler.py:207-232`) first reads the eligibility of the claimed target (`_eligible_target`, `:234-251`: `get_by_token`, `_can_materialize_checkpoint`, `_can_resume_checkpoint`) and only then persists the successor (`_materialize_successor`, `:214`, `:267-291`). Before T-IGX-041-G, `_process_due_schedules` materialized the next occurrence in a `finally` (commit `52f35a41f`), so an exception in these reads still left a successor.
- Now an exception from any of the three reads propagates out of `_process_claimed_schedule` to the isolation block in `_process_due_schedules` (`:197-202`), which logs "its claim is left to the lease (at-most-once)" and records the failure; neither the successor nor a "chain ended" record is written, and the claim is not released (`_release_claim`, `:293-303`, is called only from the materialization branch).
- Observed (throwaway probe, recurring rule over a SQLite store whose `get_by_token` raises once): the tick re-raises; the ledger then holds one row `running` and no `pending` row; a later tick processes 0 and creates nothing. `claim_due` marks a `RUNNING` row whose wall-clock lease expired `UNCERTAIN` and never re-fires it (`intergrax/runtime/long_running/store.py:504-521`, the `UPDATE` with `lease_expires_at_utc < ?` at `:509`). The chain has ended without any explicit record.
- A one-shot entry (no recurrence rule) is not affected beyond the existing at-most-once contract and stays unchanged.

### Finding 3 (MEDIUM, F-D8-03 sibling of T-IGX-041-H) — the workspace artifact tools reach object storage with arbitrary keys

- `workspace_export_artifact` (`intergrax/tools/providers/workspace/service.py:127-158`) calls `object_storage.put(storage_key, ...)` at `:149` and `workspace_import_artifact` (`:161-193`) calls `object_storage.get(storage_key)` at `:168`, where `storage_key` is `params.storage_key.strip()` taken from the tool input (`_require_object_storage` at `:113-117`). Nothing refuses a key.
- T-IGX-041-H closed this for the four raw `storage.*` tools: `intergrax/tools/providers/storage/service.py` defines `GOVERNED_CONTENT_KEY_PREFIX = "intergrax/collaborative-work/artifact-content/"` (`:33`), `GovernedStorageKeyError` and `_refuse_governed_content_key` (`:40`), called by `storage_get` (`:63`), `storage_put` (`:77`), `storage_delete` (`:105`), `storage_exists` (`:111`), with an AST gate `tests/qualification/pdc_x/d8_repair/test_d8r_storage_tools_gate.py` (`:93`, `:119`). The two workspace tools use the same `ObjectStorage` (`ToolWiringContext.object_storage`) and are agent-callable (`intergrax/skills/providers/storage/manifests.py:12-13`), so an agent can overwrite (`put`) or read (`get`) Collaborative Work artifact content through them.
- No other tool takes a caller-chosen object key: the other occurrences of `object_storage` under `intergrax/tools/` are the wiring-context field (`registry/wiring.py:98`), a category slug (`providers/cloud_platform/contracts.py:22`) and the backend health probe (`providers/health/category_probes.py:69`).

### Finding 4 (impact map classification) — two changed product paths map to no ENT-AUDIT check

- Three tests fail on `cba32d247` (run once, log under `.tmp/proof/pdc-x-041m/impact_red_baseline.log`): `tests/qualification/pdc_x/p6_impact/test_p6_impact_map_coverage.py::test_every_pdc_touched_product_module_maps_to_at_least_one_check`, `.../test_p6_impact_tier3_root_rule.py::test_synthetic_mcp_rc_range_leaves_no_pdc_touched_product_module_unmapped`, `tests/qualification/pdc_x/xd/test_xd_12_evol_ent_audit.py::test_xd12_every_pdc_touched_product_module_maps_to_a_domain_check`. Message: `2 PDC-touched product modules map to no check, e.g. ['intergrax/experiments/workflow.py', 'intergrax/runtime/policy/pre_context_policy_audit.py']`. The 63 other tests of those files pass.
- The two paths were changed by P7 repairs that added no impact rule: `intergrax/experiments/workflow.py` by `c076c7d0b` (T-IGX-041-E) and `intergrax/runtime/policy/pre_context_policy_audit.py` by `6fd2f2198` (T-IGX-041-B). The map is `PDC_DOMAIN_IMPACT_RULES` (`intergrax/platform/enterprise_audit/impact_map.py:24`, longest prefix wins, every id must exist in the 85-check manifest) merged by `build_impact_map_from_manifest`; the shipped SSOT `intergrax/platform/enterprise_audit/data/changed_since_impact_map.v1.json` is its generator output (`scripts/enterprise_audit/build_impact_map.py --emit-ssot`, asserted by `test_shipped_ssot_equals_generator_output`). T-IGX-040-A added a rule the same way (`b12832cc0`).
- Which manifest checks exercise the two paths (verified):
  - `intergrax/experiments/`: `tests/qualification/harness_01/nexus_import_inventory.py:376` classifies the `intergrax/experiments/` prefix as `TOOLING_INTERNAL`, and `tests/qualification/harness_01/test_harness_01_gates.py:37` imports that inventory; `ENT-A-REL-02` and `ENT-A-REL-03` have that file as their entrypoint. `workflow.py` imports `build_debug_minimal_nexus_loop` (`:32`).
  - `intergrax/runtime/policy/pre_context_policy_audit.py`: it is run by `scripts/maintenance/check_pre_context_policy_wiring.py`, which the IDEAL-L3 gate scripts call (`scripts/gates/check_ideal_harness_l3_gates.py:44`, `scripts/gates/check_audit_ideal_gates.py:103`); **no manifest check has it as an entrypoint**, and the manifest must stay at exactly 85 checks (`test_manifest_still_has_exactly_85_checks`). The nearest governance gates are `ENT-A-GOV-01..03` (gov_x2 catalog and proof-node gates; the catalog names `tests/unit/runtime/policy/test_gr6_r1_decision_requirement_enforcement.py` at `tests/qualification/governance/gov_x2/catalog.py:39`). This is a **classification choice**, flagged to the signer (see "Decisions").

## Deliverables (tests first, shown red, then the fixes)

New test package `tests/qualification/pdc_x/p8_repair/` (new files only; every existing test stays byte-identical). Red logs go under `.tmp/proof/pdc-x-041m/`.

1. **Finding 1** — `test_r1_default_loader.py`: (a) `DocSmartDocumentHandler` imports and subclasses the `PipelineDocumentHandler` of `pipeline/document_handler_loading.py`; (b) `create_default_documents_loader()` returns a `DocumentsLoader` and its registry holds all eight default handlers; (c) a drift gate (AST scan of `intergrax/rag/document_loaders/handlers/*.py`): every import of `PipelineDocumentHandler` comes from `intergrax.rag.document_loaders.pipeline.document_handler_loading`, and a synthetic handler module with the old import path makes the gate fail. Red before the fix. Fix: change the import of `doc_smart_document_handler.py` (one statement).
2. **Finding 2** — `test_r2_recurrence_eligibility.py` (SQLite store, `RecordingExecutor` and `RecurrenceMaterializer` as in `tests/qualification/pdc_x/d7_repair/_support.py`, imported read-only): (a) a recurring claim whose `get_by_token` raises once is released back to `PENDING`, nothing is executed, and the next tick fires it once and persists exactly one successor (same for `_can_resume_checkpoint` raising); (b) a missing or non-resumable target still ends the chain with `SKIPPED` and no successor (unchanged, positive control); (c) a one-shot entry whose eligibility read raises keeps today's behaviour (the failure propagates, the claim is left to the lease, no release). Red before the fix. Fix in `scheduler.py` only: for a claim that carries a recurrence rule while a materializer is configured, an exception (`Exception`, not cancellation) from the eligibility reads is logged, the claim is handed back with the existing `_release_claim`, and `_process_claimed_schedule` returns 0 — the same handling as the store-error branch of `_materialize_successor`. No port change.
3. **Finding 3** — `test_r3_workspace_storage_keys.py` (a recording `ObjectStorage` double as in `test_d8r_storage_tools_gate.py`): export and import refuse every key under `GOVERNED_CONTENT_KEY_PREFIX` including the normalized variants (`..`, `.`, repeated slashes, backslashes) with `GovernedStorageKeyError` **before** any workspace or storage access (no `get`, `put` or `read_artifact_bytes` call recorded); ordinary keys still work (positive controls: export then import round trip); an AST gate: in `workspace/service.py` every function that reads `params.storage_key` calls the refusal before any `object_storage` access, and a synthetic copy without the call fails the gate. Red before the fix. Fix: `storage/service.py` gains one public function `refuse_governed_content_key(key)` that delegates to `_refuse_governed_content_key` (the private name and its call sites stay, so the existing gate is untouched); `workspace/service.py` calls it first in `workspace_export_artifact` and `workspace_import_artifact`. Documentation: the sentence at `docs/project/capabilities/architecture/ARTIFACT_LIFECYCLE.md:52` names the two workspace tools and keeps the statement that other keys are not scoped to a tenant.
4. **Finding 4** — `test_r4_impact_map_classification.py`: (a) `check_ids_for_changed_path` maps `intergrax/experiments/workflow.py` to exactly `ENT-A-REL-02`, `ENT-A-REL-03` and `intergrax/runtime/policy/pre_context_policy_audit.py` to exactly `ENT-A-GOV-01`, `ENT-A-GOV-02`, `ENT-A-GOV-03`, all inside the manifest; (b) negative: the entries are **exact files**, so synthetic sibling paths (`intergrax/experiments/_p8_unmapped_probe.py`, `intergrax/runtime/policy/_p8_unmapped_probe.py`) still map to no check and are reported by the same expression the coverage test uses; (c) the three red tests listed above pass at the final head without being edited. Red before the fix. Fix: two entries in `PDC_DOMAIN_IMPACT_RULES` (sorted position, with the evidence comment) and the SSOT JSON regenerated with `scripts/enterprise_audit/build_impact_map.py --emit-ssot`. The manifest stays at 85 checks; no check, entrypoint or gate is added or removed.
5. **Docs** — `docs/project/architecture/BACKGROUND_TASKS.md:126` gets one sentence for the eligibility-failure handling (the paragraph that already states the successor-before-execution and store-error rules).

## Write set (exact; no hotspot path)

Product: `intergrax/rag/document_loaders/handlers/doc_smart_document_handler.py`, `intergrax/runtime/long_running/scheduler.py`, `intergrax/tools/providers/workspace/service.py`, `intergrax/tools/providers/storage/service.py`, `intergrax/platform/enterprise_audit/impact_map.py`, `intergrax/platform/enterprise_audit/data/changed_since_impact_map.v1.json`.
Tests (new): `tests/qualification/pdc_x/p8_repair/` (`__init__.py`, the four test files, one support module if needed).
Docs: `docs/project/architecture/BACKGROUND_TASKS.md`, `docs/project/capabilities/architecture/ARTIFACT_LIFECYCLE.md`, plus the governance documents of this batch (ticket, this plan, `PDC_IMPLEMENTATION_WBS.md`, `PDC_DEPENDENCY_DAG.md`, the `B-P8-R1` row of `scripts/qualification/pdc_x_wbs.py`).

Not edited: `intergrax/applications/_shared/`, `intergrax/contracts/` (hotspots), `scripts/governance/`, `.github/`, the ENT-AUDIT manifest and checks, the existing storage tools and their gate, every existing test, the EBH-7 pins, the IIAC activation record, `tests/qualification/tenant_x/`, `tests/qualification/compat_x/`, `tests/qualification/ebh_7/`.

## Excluded findings (not repaired in this ticket)

| # | Excluded | Reason | Reopen condition |
| --- | --- | --- | --- |
| E1 | Tenant or prefix binding of every other object-storage key, and a `GovernanceAuditEvent` for the raw `storage.*` and `workspace.*` tools (the rest of F-D8-03) | Needs routing the tools through the governed artifact port or a new tenant key-space design: a new authority and a feature, not a defect fix; T-IGX-041-H already declared it bounded | A governed per-tenant object-key namespace for agent tools is commissioned |
| E2 | A recurring claim whose `release_claim` itself fails or is unsupported by the store | Left to its lease (at-most-once), as for the existing materialization branch; the port default raises `NotImplementedError` and `_release_claim` already logs it | A store without `release_claim` is composed in a host |
| E3 | Overlap of occurrence N+1 with a still running occurrence N (documented in `BACKGROUND_TASKS.md`) | Design property of successor-before-execution (T-IGX-041-G deviation 6), not a regression | A host composition enabling recurrence is commissioned (F-D7-05) |
| E4 | Optional-extras reds of the RAG suites (`docx`, `pandas`, `fitz`, `langchain_core` and others) | Environment, not product; classified by T-IGX-041-B | The RAG extras are added to the qualification profile |
| E5 | New ENT-AUDIT checks or entrypoints for `pre_context_policy_audit.py` and `workflow.py` | The manifest is fixed at 85 checks and a new check changes the meaning of a gate; only classification is allowed | The Operator commissions a manifest extension |
| E6 | Any interim finding of IA-GLOBAL other than the four named in the commission | Not given to this planner | Reported by the orchestrator |

## Decisions for the signer

1. **Policy file mapping** (finding 4): the plan maps `pre_context_policy_audit.py` to `ENT-A-GOV-01..03` because no manifest check executes the script. Alternative: none of the 85 checks is closer; mapping it to nothing keeps three tests red.
2. **Exact files, not directories**: the two impact rules name exact files so that unmapped siblings keep failing; a directory rule would silently classify future files.
3. **Eligibility failure handling** (finding 2): release and retry (chain intact, occurrence not lost) instead of materializing the successor in a `finally` and losing the occurrence; the tick does not re-raise for this branch, as for the store-error branch of `_materialize_successor`.

## Ordering and constraints

- Must merge **before the EBH-7 pin P** (it changes product paths), together with T-IGX-041-K and T-IGX-041-L. `T-IGX-041-L` re-pins the COMPAT-X counters at the last product change, so it must be (re-)verified after this ticket merges. the new `refuse_governed_content_key` is a plain function, not an alias or re-export; PHASE 2 runs `uv run pytest tests/qualification/compat_x -q` once and reports any counter movement to the orchestrator instead of editing compat_x.
- The impact-map coverage tests inspect every product module changed since the PDC baseline: any product path added by T-IGX-041-K or -L, or by a later ticket, must map to a check, or those three tests turn red again. PHASE 2 re-runs them at the final head.
- The product-path changes keep the IIAC activation stale (ACT-N11) until the P8-04 rebind and the EBH-7 executable-identity gate red until the final re-pin (known red classes). This plan is **not a certification verdict**, claims no FRZ change and no closure.

## Protocol

1. Token-budget preflight before each phase; PHASE 2 starts only on the orchestrator's message after the delegate's signature; it merges `origin/development` first and re-verifies every statement above.
2. One `uv run pytest` at a time, narrow paths; logs under `.tmp/proof/pdc-x-041m/`; scratch under `.tmp/session/041-M/`. `ruff check`, `ruff format --check` and `pyright` on touched files; no file-level `# ruff: noqa`.
3. Commits carry exactly one `Ref: governance/tickets/T-IGX-041-M.md` trailer and no AI attribution. Upstream `origin/development` before the first push, explicit refspec, one PR `fix(pdc): P8 repair loop 1 (T-IGX-041-M)`, never merge.
4. No gate is weakened, skipped or deleted; a write-set or scope change returns to the signer through the orchestrator.

## Acceptance

- The four red groups are red before and green after; the three impact tests pass without edits; `git diff --stat origin/development...HEAD` lists only the write set.
- Existing consumers stay green: `tests/qualification/pdc_x/d7_repair`, `tests/qualification/pdc_x/d7`, `tests/qualification/sched_01`; `tests/qualification/pdc_x/d8_repair`, `tests/qualification/pdc_x/d8`, `tests/unit/tools/providers/test_catalog_expansion.py`, `tests/unit/tools/providers/test_t10_catalog_tools.py`; `tests/unit/rag/document_loaders`; the impact and ENT-AUDIT suites (`tests/qualification/pdc_x/p6_impact`, `tests/qualification/pdc_x/xd/test_xd_12_evol_ent_audit.py`, `tests/qualification/ent_audit_x`).

## Test contract

```text
uv run pytest tests/qualification/pdc_x/p8_repair -q
uv run pytest tests/qualification/pdc_x/d7_repair tests/qualification/pdc_x/d7 tests/qualification/sched_01 -q
uv run pytest tests/qualification/pdc_x/d8_repair tests/qualification/pdc_x/d8 tests/unit/tools/providers/test_catalog_expansion.py tests/unit/tools/providers/test_t10_catalog_tools.py -q
uv run pytest tests/unit/rag/document_loaders tests/integration/rag/document_loaders/test_fixture_dataset_ingestion.py -q
uv run pytest tests/qualification/pdc_x/p6_impact tests/qualification/pdc_x/xd/test_xd_12_evol_ent_audit.py -q
uv run pytest tests/qualification/compat_x -q
uv run ruff check <touched files>   |   uv run ruff format --check <touched files>   |   uv run pyright <touched files>
uv run python scripts/qualification/pdc_x_wbs.py --check
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

- Everything under Excluded findings; the declared exceptions of `DELEGATION-PDC-CLOSE.md`; the EBH-7 re-pin; the IIAC activation record; OD signatures; certification.
