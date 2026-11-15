# PLAN T-IGX-041-H — PDC-X batch B-P7-D8: repair of the D8 Artifact Lifecycle domain after the independent audit IA-D8 (loop 1)

Parent: PDC-X implementation WBS (`PDC_IMPLEMENTATION_WBS.md`, batch **B-P7-D8**, wave W13). Origin: `PDC_P7_D8_AUDIT_REPORT.md` (auditor `IA-D8`, audited SHA `c08efe46a74a20595468d8d8bf06eab2dca5f573`, overall verdict **PARTIAL**: 1 HIGH, 6 MEDIUM, 9 LOW, 4 INFO; rows R004 to R010, R012, R013 and R016 PARTIAL, no row FAIL). Authority: Operator option B in `operator_decisions/DELEGATION-PDC-P7-REPAIR.md` (repair real defects and hygiene in existing code; "component exists but has no production composition" and "claimed feature has no implementation" are declared as exceptions, not repaired; hotspot files only when named exactly; gates only strengthened). Requirement rows touched: PDC-D8-R004, R005, R006, R007, R008, R009, R016 and the D6-owned row PDC-D6-R022 (R001 to R003, R010 to R015, R017 to R019 are not changed). This is the single repair ticket of domain D8. The TENANT-X carrier re-pin is not owned here (ticket `T-IGX-041-I`). This plan is not a certification verdict.

Every factual statement below was re-checked in code at `f87df3b2a` (`origin/development`; the audited D8 product code is unchanged since `c08efe46a`) or measured on a throwaway tree. The audit report is treated as a claim, not as proof.

**No path under `intergrax/contracts/` or `intergrax/applications/_shared/` is changed by this ticket** (in particular `intergrax/contracts/task_artifacts.py`, `intergrax/contracts/artifact_ref.py` and `intergrax/applications/_shared/queue_worker_wiring.py` stay untouched).

## Problem

The D8 mechanisms are real at library level; the repairable defects are at their edges.

1. **F-D8-07 (MEDIUM, real bug).** `RuntimeContext.create_artifact` (`intergrax/runtime/execution/agent_runtime_context.py`, the documented gateway for execution artifacts) builds `Artifact(...)` without the required `tenant_id` and raises `TypeError` on every call; it also sets no producer and no lineage, so even a fixed call could not be projected to a durable reference (`Artifact.to_durable_ref` raises when either is missing). `RuntimeState` exposes `tenant_id`, `run_id` and `request.agent_id` (the spill path in `plan_context_invocation.py` already uses all three).
2. **F-D8-05 (MEDIUM).** `ArtifactRetentionSweeper.sweep` selects expired latest versions with `list_expired` and then calls `delete_artifact`, which removes **every** version; a non-expiring version written between the two calls is destroyed with the expired one (probes F02 of the audit, in-memory and SQLite). Independently, a sweeper built without an audit recorder deletes and records nothing (`_record` returns when there is no auditor); `compose_worker_artifact_services` and `build_nexus_task_execution_registry` default `audit`/`artifact_audit` to `None`.
3. **F-D8-06 (MEDIUM).** `ExecutionArtifactSource` and `GovernedArtifactReadPort` never consult `expires_at_utc`, so an expired artifact stays readable until a sweep runs (probe P5).
4. **F-D8-04 (MEDIUM).** `GovernedArtifactReadPort.read` checks the stored workspace only `if request.workspace_id is not None`, so a request without a workspace reads an artifact stored in one (probe P6). `ExecutionArtifactSource.place_legal_hold`, `release_legal_hold` and `delete` do not look at the stored workspace at all.
5. **F-D8-03 (MEDIUM).** `storage.get`, `storage.put`, `storage.delete` and `storage.exists` (`intergrax/tools/providers/storage/service.py`) act on any key of the `ObjectStorage` that also backs Collaborative Work content (physical keys start with `intergrax/collaborative-work/artifact-content/v1/`), with no tenant, authorization or audit (probe P4). `ToolWiringContext` (`intergrax/tools/registry/wiring.py`) carries no tenant, so a tenant binding of arbitrary keys is not possible there.
6. **F-D8-08 and F-D8-12 (LOW, same function).** `GovernedArtifactReadPort._guard` discards the decision returned by `authorize_tool` (it is fail-closed only because the real boundary raises), and an unexpected backend exception produces no outcome event.
7. **Overstating documentation (F-D8-17 family).** `docs/project/capabilities/architecture/ARTIFACT_LIFECYCLE.md` says retention, TTL and legal hold "are enforced by `retention.py`" and that the sweep is "run by canonical execution"; in the shipped composition nothing builds a sweep task, and nothing assigns `RuntimeContext.artifact_store` (F-D8-01, F-D8-02).

## In-scope findings

| Finding | Severity | Decision | Evidence for IN |
| --- | --- | --- | --- |
| F-D8-07 | MEDIUM | **IN** (bug) | `Artifact.tenant_id` is a required field (`models.py`); `create_artifact` omits it; no caller, so no test ever hit it |
| F-D8-05 | MEDIUM | **IN**: delete conditional on the observed version (race), and a sweep without an audit recorder is refused unless explicitly allowed | read in `retention.py` and both stores; the audit-producer part (a durable sink in a host) stays excluded under F-D8-01 |
| F-D8-06 | MEDIUM | **IN**: an expired, unprotected version is "not found" at read | read in `sources.py`; clock injectable |
| F-D8-04 | MEDIUM | **IN, partial**: a stored workspace must be matched by the request on read and on hold/release/delete; the tenant-wide capability model is excluded | prototype below |
| F-D8-03 | MEDIUM | **IN, bounded**: the four raw storage tools refuse the Collaborative Work content key space (normalized key, with a gate test); tenant binding of arbitrary keys is excluded | `service.py` and `wiring.py` read |
| F-D8-08, F-D8-12 | LOW | **IN as by-products** (same function as F-D8-04) | `governed_access.py:_guard` read |
| F-D8-09 (negative `limit` only) | LOW | **IN as by-product** | `sweep(limit=...)` unchecked |
| F-D8-16 (hostile-source test only) | LOW | **IN as by-product**: a pinned test where a source returns another tenant's reference | the check exists in `read`; the audit's grep of the tests found none that pins it |
| F-D8-17 (documentation part) | INFO | **IN**: architecture and plan pair state what is library-only | text above |

**Measured on a throwaway tree** (`origin/development`, not committed): applying the workspace rule and a wall-clock expiry rule to `governed_access.py` and `sources.py` makes 19 existing tests fail, all in seven files: `tests/qualification/pdc_x/d8/test_d8_governed_port.py`, `test_xd09_spill_and_context_reference.py`, `test_xd10_tenant_and_ttl.py`, and `tests/qualification/pdc_x/xd/test_xd_02_agent_context.py`, `test_xd_09_artifacts_context.py`, `test_xd_10_artifacts_security.py`, `test_xd_11_observability.py` (reads of artifacts stored in workspace `ws-1` without a workspace on the request; spilled output stored at the fixed instant `T0` = 2026-11-13 12:00 UTC, which is before the system clock, read with the system clock). These tests exercise the defects (they assume an optional workspace and no expiry at read), so they need fixture-level edits; this is why the seven files are named in the write set below. The other changes of this plan were not prototyped.

## Excluded findings

Declared exceptions (F-04 family and missing-feature-vs-claim go to the exception register of the P9 closure package) and debt. None is silently dropped.

| Finding | Class | Reason | Reopen condition |
| --- | --- | --- | --- |
| F-D8-01 (HIGH) retention not enforced by any shipped composition; nothing produces `AUDIT_EVIDENCE` | F-04 family plus missing producer (exception register, "not wired" and "not implemented") | no host passes an artifact store, governance and audit recorder to the worker bootstrap or `RuntimeContext`; nothing builds or submits a sweep task; no code sets `AUDIT_EVIDENCE`. Fixing that is a composition decision plus a new producer, not a repair; `queue_worker_wiring.py` and `task_artifacts.py` are left untouched | an Operator decision that host composition of D8 is a PDC-X claim, then a composition ticket (store, governance port, audit recorder, sweep trigger, evidence producer) |
| F-D8-02 D8 runtime dormant in shipped hosts; spill without a product resolver | F-04 family (exception register, "not wired") | no product code assigns `RuntimeContext.artifact_store`; `ArtifactReferenceResolver`, `reference_fragment`, `CollaborativeWorkContentSource` have no non-test caller; spill must stay off until a read tool ships with it | same composition ticket, which must ship the `artifact.reference` read tool together with spill |
| F-D8-03 remainder: bind arbitrary storage keys to a tenant or route the tools through the governed port | Operator decision | `ToolWiringContext` has no tenant; a prefix scheme changes the key contract for every user of the storage tools | Operator decision on the storage tool contract (tenant on the wiring context or retirement of the raw tools) |
| F-D8-04 remainder: capability `artifact.read` is tenant-wide; no run or principal binding | Operator decision | the authorization model belongs to the governance owner; the port only delegates | Operator decision on principal or run binding in `ArtifactAccessContext` |
| F-D8-09 remainder: starvation by protected rows, one poisoned row aborts a sweep, naive datetimes in the stores | LOW, debt | availability hardening with no data disclosure; touches query shape in both stores | next D8 loop or a retention-hardening ticket |
| F-D8-10, F-D8-11, F-D8-13 (`artifact_ref.py` retention-class downgrade, trailing-newline digests, cross-tenant ancestors) | LOW, debt | they live in `intergrax/contracts/artifact_ref.py` and `intergrax/contracts/task_artifacts.py` (hotspot files); not needed by any in-scope fix | next D8 loop with the two files named exactly |
| F-D8-14 legal hold overwritable and lossy | LOW, debt | needs a schema change of the holds table | hold-trail ticket |
| F-D8-15 bundle builder follows symlinks, empty sha256 on bundle entry types | LOW, debt | code-read only, not reproduced (host cannot create symlinks); the entry types are in `task_artifacts.py` (hotspot) | a host that can create symlinks, or the next D8 loop |
| F-D8-16 remainder: a composition test that fails when the shipped worker registers no sweep | tied to F-D8-01 | such a test would assert the exception row | the F-D8-01 composition ticket |
| F-D8-17 remainder (ownership matrix wording, emitter annotation, `taxonomy.py` owner note), F-D8-18, F-D8-19, F-D8-20 | INFO | no action required for the verdict (F-D8-19 is an unrelated long_running test) | none |

## Deliverables

Tests first. New tests live in a new directory `tests/qualification/pdc_x/d8_repair/` (with `__init__.py`). Each is shown failing on the unchanged code for the stated reason, then the fix is applied. The seven existing test files named in the write set are edited **only** in request, clock and source construction helpers (the workspace id an artifact was stored under, the injected clock); no assertion is removed, weakened or skipped, and every edit is listed in the checkpoint. Any other existing test that fails after a fix stops the batch (write-set amendments are not delegated).

### 1. Pinned tests (red first, with negative tests)

- `test_d8r_runtime_gateway.py` (F-D8-07): `RuntimeContext.create_artifact` against a `SQLiteArtifactStore` (red today: `TypeError`); the stored artifact carries the state's tenant, a producer naming the request agent and a lineage with the run id, is retrievable with `get` under that tenant only (another tenant gets `KeyError`) and projects to a `DurableArtifactRef`; without a store it still raises `RuntimeError("ArtifactStore not configured.")`.
- `test_d8r_sweep_version_safety.py` (F-D8-05, F-D8-09): on both stores, a version put after expiry selection is not destroyed (`delete_artifact(..., expected_version=<observed>)` raises the new `ArtifactVersionChanged` and leaves every version; red today: no such parameter, all versions deleted); a sweep whose selection went stale reports the artifact as changed and deletes nothing; a stale selection followed by a second sweep of an expired newest version deletes it; the unconditional `delete_artifact(tenant, id)` still deletes all versions and still refuses evidence and legal hold (unchanged behaviour pinned); `sweep(limit=0)` and negative limits raise `ValueError`.
- `test_d8r_sweep_requires_audit.py` (F-D8-05): a sweeper without an audit recorder refuses to sweep (`ArtifactSweepAuditRequired`), deletes nothing and records nothing; with an audit recorder, or with `allow_unaudited_deletion=True`, it sweeps as before; the registered sweep handler returns a failed result with code `audit_recorder_required` instead of deleting.
- `test_d8r_read_expiry.py` (F-D8-06): an expired, unprotected artifact is not found at read (red today: readable) and the denial is audited as `not_found`; an artifact one second before expiry is readable; the boundary instant counts as expired (consistent with `list_expired`, `<=`); a legal-hold artifact and an audit-evidence artifact past their expiry stay readable; a naive `expires_at_utc` is treated as UTC; a specific expired version is not found while a later non-expired version is readable.
- `test_d8r_workspace_scope.py` (F-D8-04): reading an artifact stored in `ws-1` with no workspace, or with another workspace, is denied `workspace_mismatch` and audited (red today for the no-workspace case); the same workspace reads; an artifact stored without a workspace reads with or without one (unchanged); legal hold, release and delete on a workspace-scoped artifact need the matching workspace (red today: no check) and a denied attempt changes nothing in the store.
- `test_d8r_governed_guard.py` (F-D8-08, F-D8-12, F-D8-16): a governance port that returns a DENY decision without raising denies the read (red today: read succeeds), a `REQUIRE_APPROVAL` return denies as `approval_required`, an ALLOW return reads; a source raising `RuntimeError` produces an outcome event with check `backend_error` and the exception propagates (red today: only the authorization event); a hostile source returning another tenant's reference is denied `tenant_mismatch` (pins the stored-tenant re-check).
- `test_d8r_storage_tools_gate.py` (F-D8-03): `storage.get`, `storage.put`, `storage.delete` and `storage.exists` refuse the key `artifact_content_object_key(...)` derives for a tenant and workspace, its `v1` prefix and the namespace `intergrax/collaborative-work/artifact-content/`, also with a leading slash, repeated slashes, `./` segments, backslashes and `..` segments, and nothing reaches the storage object (a recording double sees no call; red today: every call is executed); ordinary keys (`docs/a.txt`) behave as before; the guard's prefix equals `_PHYSICAL_KEY_PREFIX` of `collaborative_work.content_storage` minus its version segment (a drift gate), and an AST gate asserts that each of the four tool functions starts with the guard call.

### 2. Fixes

1. `intergrax/runtime/execution/agent_runtime_context.py`: `create_artifact` passes `tenant_id=state.tenant_id`, a producer (`ArtifactProvenance(created_by_agent_id=state.request.agent_id or "")`) and a lineage (`ArtifactLineage(run_id=state.run_id)`); class and retention class stay the model defaults (no `AUDIT_EVIDENCE` producer is added).
2. `intergrax/runtime/nexus/artifacts/store_base.py`, `sqlite_artifact_store.py`, `in_memory_artifact_store.py`: `ArtifactStore.delete_artifact(tenant_id, artifact_id, *, expected_version: int | None = None)`; when given, deletion happens only if the latest version still equals it, checked inside the same transaction (SQLite) or lock (in-memory) as the protection check; otherwise a new `ArtifactVersionChanged(RuntimeError)` (defined in `store_base.py`) is raised and nothing is deleted. No new public method name is added.
3. `intergrax/runtime/nexus/artifacts/retention.py`: the sweeper passes `expected_version=artifact.version`, reports artifacts whose selection went stale in a new defaulted field `changed_ids` of `RetentionSweepReport`, validates `limit >= 1`, and refuses to sweep without an audit recorder unless constructed with `allow_unaudited_deletion=True` (new typed error `ArtifactSweepAuditRequired`); the handler maps that error to a failed `ToolExecutionResult` with code `audit_recorder_required`.
4. `intergrax/runtime/nexus/artifacts/sources.py`: `ExecutionArtifactSource(store, *, clock=None)` (default `SystemTimeProvider.utc_now`) treats an expired, unprotected version as not found (protection is judged with the existing `protection_reason` over the artifact's history and the store's legal-hold state); its hold, release and delete operations check the stored workspace against the request.
5. `intergrax/runtime/nexus/artifacts/governed_access.py`: `read` requires the request workspace to equal the stored one whenever the stored one is set; `_guard` honours the returned decision (DENY and REQUIRE_APPROVAL deny and are audited with the existing codes) and records an outcome event with check `backend_error` for an unexpected exception before re-raising it. No policy engine, registry or authority is added.
6. `intergrax/tools/providers/storage/service.py`: a private guard function, called first in `storage_get`, `storage_put`, `storage_delete` and `storage_exists`, normalizes the key (strip, split on `/` and `\`, drop empty and `.` segments, refuse any `..`) and raises a typed error for keys inside `intergrax/collaborative-work/artifact-content/`. `storage_presigned_url` is unchanged.
7. Documentation: `docs/project/capabilities/architecture/ARTIFACT_LIFECYCLE.md` states what is enforced in the library and what no shipped composition wires (retention sweep trigger, store handed to the runtime, evidence producer, spill resolver), replacing the unqualified "enforced" and "run by canonical execution" wording, and adds a "Declared exceptions" table with the reopen conditions above; `docs/project/capabilities/plan/ARTIFACT_LIFECYCLE.md` records this repair and keeps the wiring item open. The needles required by `test_d8_feature_doc_pair.py` stay.
8. Seven existing test files, fixture level only (see above).

Design constraints: no new registry, scheduler, engine, trust, policy or audit authority; no gate weakened, skipped or xfailed; `scripts/governance/`, `.github/`, `intergrax/contracts/`, `intergrax/applications/_shared/`, the requirement matrix and `worker_bootstrap.py` stay untouched.

## Write set

Product: `intergrax/runtime/execution/agent_runtime_context.py`, `intergrax/runtime/nexus/artifacts/store_base.py`, `intergrax/runtime/nexus/artifacts/sqlite_artifact_store.py`, `intergrax/runtime/nexus/artifacts/in_memory_artifact_store.py`, `intergrax/runtime/nexus/artifacts/retention.py`, `intergrax/runtime/nexus/artifacts/sources.py`, `intergrax/runtime/nexus/artifacts/governed_access.py`, `intergrax/tools/providers/storage/service.py`

Hotspot files: none.

Tests, new files only: `tests/qualification/pdc_x/d8_repair/`

Tests, existing files, fixture-level edits only: `tests/qualification/pdc_x/d8/test_d8_governed_port.py`, `tests/qualification/pdc_x/d8/test_xd09_spill_and_context_reference.py`, `tests/qualification/pdc_x/d8/test_xd10_tenant_and_ttl.py`, `tests/qualification/pdc_x/xd/test_xd_02_agent_context.py`, `tests/qualification/pdc_x/xd/test_xd_09_artifacts_context.py`, `tests/qualification/pdc_x/xd/test_xd_10_artifacts_security.py`, `tests/qualification/pdc_x/xd/test_xd_11_observability.py`

Docs: `docs/project/capabilities/architecture/ARTIFACT_LIFECYCLE.md`, `docs/project/capabilities/plan/ARTIFACT_LIFECYCLE.md`

Nothing outside this set changes. A need to write elsewhere (including any further existing test) stops the batch and goes to the orchestrator and the Operator.

## Protocol

1. Write the pinned tests first and show each failing for the stated reason (log under `.tmp/proof/pdc-x-p5/B-P7-D8/`).
2. One `uv run pytest` per verification step (several paths in one command); no broad suites, other sessions run pytest in parallel worktrees; ruff and pyright on touched files. The pre-commit gate runs ruff check and format on whole staged files, so a touched file with older lint or formatting findings is cleaned in the same commit; the checkpoint says so.
3. Commit with exactly one `Ref: governance/tickets/T-IGX-041-H.md` trailer and no AI attribution; set the upstream to `origin/development` before the first push, push with an explicit refspec, open one PR against `development`, never merge. The implementer never writes `Approved:` or activation lines and never certifies.
4. Scratch and logs only under `.tmp/`; tests use placeholder tenants and ids, never real credentials.
5. The worker bootstrap test `tests/qualification/pdc_x/p6_repair/test_p6r_d8_worker_bootstrap.py` is skipped when the optional celery extra is missing; the implementer reports whether it ran.

## Acceptance

- Each pinned test was red on the unchanged code for its stated reason and is green after the fix; negative tests exist for every fix; the seven edited existing files differ only in fixtures and every edit is listed.
- `create_artifact` works against a SQLite store; a sweep never deletes a version written after selection, never runs unaudited by default, and rejects a non-positive limit; an expired unprotected artifact is not readable; a workspace-scoped artifact is read, held, released and deleted only with its workspace; a returned DENY decision denies; the four raw storage tools cannot address Collaborative Work content.
- No file outside the write set changed; no hotspot touched; no gate weakened; no new authority; the exclusion table is reproduced in the checkpoint with the reopen conditions.
- **Status statements.** This ticket is not a certification verdict. Its product changes lie under ACT-N11 paths, so the IIAC activation record stays stale until the P8-04 rebind (known red class: `ACT-N11 product path changed after the audited SHA`), and the EBH-7 identity gate stays red until the campaign's final re-pin (known red class; these paths are outside the executable-identity allowlist). The TENANT-X carrier count is re-pinned once in `T-IGX-041-I`, not here. A fresh auditor re-audits D8 at the corrected SHA (loop 1 of 2).

## Test contract

```text
uv run pytest tests/qualification/pdc_x/d8_repair tests/qualification/pdc_x/d8 tests/qualification/pdc_x/xd/test_xd_02_agent_context.py tests/qualification/pdc_x/xd/test_xd_09_artifacts_context.py tests/qualification/pdc_x/xd/test_xd_10_artifacts_security.py tests/qualification/pdc_x/xd/test_xd_11_observability.py tests/qualification/pdc_x/p6_repair/test_p6r_d8_worker_bootstrap.py tests/unit/tools/providers/storage tests/unit/tools/providers/test_t10_catalog_tools.py tests/unit/collaborative_work/test_artifact_content_storage.py -q
uv run ruff check <touched product, script and test files>
uv run pyright <touched product and test files>
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run python scripts/docs/check_docs_domain_pairs.py
uv run python -m scripts.governance.igx_gov ticket
```

If a listed module needs an optional extra the environment lacks, the implementer syncs it once or reports the module as an environment red; it is not dropped silently.

## Out of scope

- Every row of the Excluded findings table; host composition of the artifact store, governance port, audit recorder, sweep trigger and evidence producer; spill resolver and the `artifact.reference` tool; `queue_worker_wiring.py`, `worker_bootstrap.py`, `intergrax/contracts/`; the requirement matrix and its row statuses; `scripts/governance/`, `.github/`.
- The TENANT-X carrier re-pin, the IIAC activation and its P8-04 rebind, the EBH-7 re-pin, OD signatures, FRZ changes, certification, the re-audit itself.
