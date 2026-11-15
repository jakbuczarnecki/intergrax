# T-IGX-041-H — PDC-X batch B-P7-D8: repair of the D8 Artifact Lifecycle domain after the independent audit IA-D8 (loop 1)

## Scope

Repair ticket of domain D8 (Artifact Lifecycle) under Operator option B (`DELEGATION-PDC-P7-REPAIR.md`), after the independent audit `PDC_P7_D8_AUDIT_REPORT.md` (verdict PARTIAL). In scope: `RuntimeContext.create_artifact` (F-D8-07: missing tenant, producer and lineage), the retention sweep (F-D8-05: delete conditional on the observed version, no sweep without an audit recorder unless explicitly allowed, positive limit), expiry enforced at read (F-D8-06), the stored workspace required on read, hold, release and delete (F-D8-04, partial), the four raw `storage.*` tools refusing the Collaborative Work content key space (F-D8-03, bounded), the governed port honouring a returned DENY decision and auditing backend failures (F-D8-08, F-D8-12), a hostile-source test (F-D8-16 part) and the documentation of what is library-only (F-D8-17 documentation part). Excluded with reason and reopen condition (exception or debt register of the P9 closure package): F-D8-01 and F-D8-02 (not wired, no evidence producer), the Operator-decision remainders of F-D8-03 and F-D8-04, and the LOW and INFO findings listed in the plan. Product code: yes, no hotspot file (`intergrax/contracts/` and `intergrax/applications/_shared/` are untouched). Seven existing test files are named in the write set for fixture-level edits only. This ticket is not a certification verdict. Plan: `PLAN_T-IGX-041-H.md`.

## Plan digest

Plan digest: 232913e88f9210eb47b743731601c127e8b396ff1428b984865e9f925c0ed314

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-041-H.md`](plans/PLAN_T-IGX-041-H.md).

## Approved

Approved: operator-delegate @ 2026-11-15T10:22:02+01:00
Delegation: docs/project/maintainers/qualification/platform_domain_completeness/operator_decisions/DELEGATION-PDC-P7-REPAIR.md@8f4b139c1f5352c9861872f0b5077996b8f0f891

Delegate checks (scope of `DELEGATION-PDC-P7-REPAIR.md` read at `origin/development`, last changing commit `8f4b139c1`; signed at the batch branch, implementer commit `199bf3b60`; all passed before this line was written):

- The plan digest recomputed with `scripts/governance/plan_digest.py` equals the digest above (`232913e88f9210eb47b743731601c127e8b396ff1428b984865e9f925c0ed314`).
- The plan write set (8 product files, `tests/qualification/pdc_x/d8_repair/`, 7 existing test files, 2 docs: 18 entries) equals, as sets compared by script, the write set of batch row `B-P7-D8` in `PDC_IMPLEMENTATION_WBS.md` and the entry in `scripts/qualification/pdc_x_wbs.py`; `pdc_x_wbs.py --check` is OK (107 rows, 23 batches, no write conflicts). The implementer commit changes only the ticket, the plan, the WBS, the DAG and the batch register; nothing under `scripts/governance/` or `.github/`.
- No hotspot file is in the write set (`Hotspot files: none`); `intergrax/contracts/task_artifacts.py`, `intergrax/contracts/artifact_ref.py` and `intergrax/applications/_shared/queue_worker_wiring.py` are not touched by the implementer commit and are named as untouched.
- In-scope findings (F-D8-07, -05, -06, -04 partial, -03 bounded, -08, -12, -09 negative limit, -16 hostile-source test, -17 documentation part) and excluded findings (F-D8-01, -02, remainders of F-D8-03/04/09/16/17, F-D8-10, -11, -13, -14, -15, -18, -19, -20) are all listed with class, reason and reopen condition. Exclusions are of the allowed kinds: F-04 family plus missing evidence producer (F-D8-01, -02), Operator decision (F-D8-03/04 remainders), LOW/INFO debt. The severity tally of the plan (1 HIGH, 6 MEDIUM, 9 LOW, 4 INFO) matches the audit report headings.
- Code claims spot-checked against the base (about 25, all true), among them: `agent_runtime_context.py:235-244` builds `Artifact(...)` without `tenant_id`, producer or lineage while `models.py:37` requires `tenant_id` and `to_durable_ref` raises without producer and lineage (`:70-73`); `RuntimeState` exposes `tenant_id`/`run_id`, `plan_context_invocation.py:69-72` uses all three; `retention.py` `_record` returns without an auditor, `sweep` selects with `list_expired` and calls `delete_artifact` without a version, `limit` is unchecked; `in_memory_artifact_store.py:87-96` and `sqlite_artifact_store.py:296-312` delete every version, `list_expired` uses `<=` in both; `sources.py:41-80` never consults `expires_at_utc` and hold/release/delete ignore the stored workspace; `governed_access.py:283` checks the workspace only `if request.workspace_id is not None`, `_guard` (`:345-390`) discards the return value of `authorize_tool` (declared to return `ToolAuthorizationDecision`, `ports.py:26-31`) and has no `backend_error` outcome; `storage/service.py` exposes `storage_get/put/delete/exists` plus an unchanged `storage_presigned_url`, `ToolWiringContext` has no tenant, `content_storage.py:31` `_PHYSICAL_KEY_PREFIX` = `intergrax/collaborative-work/artifact-content/v1/`; `ARTIFACT_LIFECYCLE.md:36` carries the "enforced by `retention.py`" and "run by canonical execution" wording, and the needles of `test_d8_feature_doc_pair.py` do not include it; `compose_worker_artifact_services` and `build_nexus_task_execution_registry` default `audit`/`artifact_audit` to `None`; nothing under `intergrax/`, `applications/`, `agents/` assigns `RuntimeContext.artifact_store` or calls the runtime `create_artifact`, and `ArtifactReferenceResolver`, `reference_fragment` and `CollaborativeWorkContentSource` appear only in their own modules; `_d8_support.py` stores artifacts in `ws-1` and fixes `T0` at 2026-11-13 12:00 UTC.
- No gate weakened or deleted: every change denies more (sweep audit requirement, conditional delete, expiry and workspace at read, honoured DENY decision, storage key refusal). The existing sweeper tests all pass an audit recorder, and `test_p6r_d8_worker_bootstrap.py` only checks registration, so the audit requirement does not break them. Existing-test edits are listed (seven files, fixture level: workspace id on the request, injected clock; no assertion removed, weakened or skipped; any other existing test that fails stops the batch). They are stale-test hygiene in the sense that these tests encode the audited defects (optional workspace, no expiry at read); the 19 failing tests were measured by the planner on a throwaway tree and were not re-measured by the delegate.
- No registry, scheduler, engine, trust, policy or audit authority is added; the plan states it is not a certification verdict. The implementer session and this delegate session are different sessions; the delegate edited nothing except this approval block and the lifecycle state.
- Merge gate (stated as the delegation record requires): product pull request merges under authorizations 1 and 3 of `DELEGATION-PDC-STANDING.md` and decision 8 of `DELEGATION-PDC-WAVE2.md`: green CI, only the known red classes (IIAC activation staleness, EBH-7 identity gate), `headRefOid` equal to the verified SHA; additionally OD-PDC-SEQ-01 (merge only after MCP-SERVER-X P8 publication). Write-set amendments are not delegated. Not decided here: the numbering of the standing authorizations was not re-verified.

## Test contract

```text
uv run pytest tests/qualification/pdc_x/d8_repair tests/qualification/pdc_x/d8 tests/qualification/pdc_x/xd/test_xd_02_agent_context.py tests/qualification/pdc_x/xd/test_xd_09_artifacts_context.py tests/qualification/pdc_x/xd/test_xd_10_artifacts_security.py tests/qualification/pdc_x/xd/test_xd_11_observability.py tests/qualification/pdc_x/p6_repair/test_p6r_d8_worker_bootstrap.py tests/unit/tools/providers/storage tests/unit/tools/providers/test_t10_catalog_tools.py tests/unit/collaborative_work/test_artifact_content_storage.py -q
uv run ruff check <touched product, script and test files>
uv run pyright <touched product and test files>
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run python scripts/docs/check_docs_domain_pairs.py
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

- Every excluded finding of the plan (F-D8-01, F-D8-02, the Operator-decision remainders of F-D8-03 and F-D8-04, F-D8-09 remainder, F-D8-10, F-D8-11, F-D8-13 to F-D8-20); host composition of the artifact store, governance port, audit recorder, sweep trigger and evidence producer; the spill resolver; `queue_worker_wiring.py`, `worker_bootstrap.py`, `intergrax/contracts/`; the requirement matrix; `scripts/governance/`, `.github/`.
- The TENANT-X carrier re-pin (ticket `T-IGX-041-I`), the IIAC activation and its P8-04 rebind (stays stale after these product changes, known red class), the EBH-7 re-pin (stays red until the final re-pin, known red class), OD signatures, FRZ changes, certification, the re-audit.

## Lifecycle state

plan_approved
