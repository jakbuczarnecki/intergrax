# PLAN T-IGX-041-I — PDC-X P7: classify the new TENANT-X class-field carriers and re-pin the carrier-count gate once

Parent: PDC-X campaign, repair loop 1 after the P7 independent audits. Origin: findings F-D2-03, F-D3-06 and F-D6-06 of the independent audits (`PDC_P7_D2_AUDIT_REPORT.md`, `PDC_P7_D3_AUDIT_REPORT.md`, `PDC_P7_D6_AUDIT_REPORT.md`): `test_tx_p0_r3_head_q02_class_field_carrier_count_stable` is red because the carrier count at head differs from the pin. Authority: Operator decision 2 of `DELEGATION-PDC-P7-REPAIR.md` (the TENANT-X carrier-count gate may be re-pinned once, after the last domain repair) and its limit "gates are only strengthened, classified or re-pinned, never weakened or deleted". Batch row: `B-P7-TX` in `PDC_IMPLEMENTATION_WBS.md`. This plan is written by the planner lane (PHASE 1) and is **unsigned** (`plan_pending`); the implementer never writes `Approved:` or `Delegation:` lines. It is not a certification verdict and certifies nothing.

Base of the measurements below: `origin/development` at `dd58baa4d`. They are **known so far**: the repairs T-IGX-041-A (D1), T-IGX-041-B (D2) and T-IGX-041-F (D6) were not merged when this plan was written and may add or remove carriers. PHASE 2 recomputes everything at the final head before it edits anything.

## Problem

1. The gate. `tests/qualification/tenant_x/test_tenant_x_p0_current_head.py:90-98` (`test_tx_p0_r3_head_q02_class_field_carrier_count_stable`) imports `TENANT_X_REPLAY_HEAD_CARRIER_TOTAL` and `assert_replay_carrier_delta_classified` from `tests/qualification/tenant_x/_replay_x_carrier_reconciliation.py` and asserts two things: `carrier_semantics_report().class_field_carrier_total == TENANT_X_REPLAY_HEAD_CARRIER_TOTAL` (`:97`; the pin is `819` at `_replay_x_carrier_reconciliation.py:12`) and `assert_replay_carrier_delta_classified()` (`:98`).
2. What a carrier is. `class_field_tenant_surface_ids()` (`tests/qualification/tenant_x/_inventory.py:70`) is the set of `surface_id`s of the discovered candidates whose `discovery_kind == "class.field.tenant"`, produced by the AST scan of `intergrax/`, `agents/` and `applications/` in `tests/qualification/tenant_x/_discovery.py`; a surface id has the form `<path>::<Class>.tenant_id::class.field.tenant`. `class_field_carrier_total` equals the size of that set.
3. What the pin is meant to be. The helper defines `TENANT_X_FINAL_IMMUTABLE_CARRIER_TOTAL = 800` (`:11`), the pin `TENANT_X_REPLAY_HEAD_CARRIER_TOTAL = 819` (`:12`), `TENANT_X_REPLAY_CARRIER_DELTA = 819 - 800 = 19` (`:13`) and a frozen set `REPLAY_DELTA_CLASS_FIELD_SURFACE_IDS` of exactly 19 surface ids classified as "introduced after TENANT-X-FINAL" (`:16-38`). `assert_replay_carrier_delta_classified()` (`:41`) checks `len(set) == delta` and that every id of the set is still a discovered candidate. It was written in commit `91649e5d8` (2026-10-30, ROADMAP-REPLAY-X L2 corrective wave); at that commit the scan finds 819 class-field carriers, and all 19 classified ids are present both there and at `dd58baa4d`.
4. The red state. At `dd58baa4d` the scan finds **831** class-field carriers (the gate fails with `831 == 819`). Net +12 relative to the pin: **15 added and 3 removed** since `91649e5d8` (table below). The 19 already classified ids are unchanged. No other test of `tests/qualification/tenant_x` is named by the three audits as red.
5. A weakness this ticket also closes. The classification check only requires `len(set) == delta` and membership in the discovered set; it does not require the set to equal the actual new carriers, and it cannot see a carrier that is replaced by another (one removed, one added: same count). The count assertion catches an unclassified addition only by arithmetic.

### Known so far: class-field carriers new since the pin (head `dd58baa4d` against `91649e5d8`)

"Origin" is the first commit that touched the module after the pin (`git log 91649e5d8..HEAD`). Family, requirement and role are the values the existing classifier assigns today (`get_tenant_x_inventory()`, read with a throwaway script), not new decisions; `identity_source` is `UNKNOWN` for all of them.

| # | Surface (module :: field) | Origin batch | family | provenance requirement | propagation role |
| --- | --- | --- | --- | --- | --- |
| 1 | `intergrax/applications/_shared/evaluation_wiring.py` :: `ApplicationEvaluationWiring.tenant_id` | B-D5 | RUNTIME_EXECUTION | PROVENANCE_REQUIRED_AT_BOUNDARY | IDENTITY_CARRIER |
| 2 | `intergrax/contracts/evaluation_contracts.py` :: `EvalComparison.tenant_id` | B-D5 | UNKNOWN | UNKNOWN_REQUIREMENT | IDENTITY_CARRIER |
| 3 | `intergrax/contracts/evaluation_contracts.py` :: `EvalRun.tenant_id` | B-D5 | UNKNOWN | UNKNOWN_REQUIREMENT | IDENTITY_CARRIER |
| 4 | `intergrax/eval/typed_run.py` :: `EvalRunProvenance.tenant_id` | B-D5 | RUNTIME_EXECUTION | PROVENANCE_REQUIRED_AT_BOUNDARY | IDENTITY_CARRIER |
| 5 | `intergrax/runtime/architecture/online_evaluation_registry.py` :: `TenantOnlineEvaluationObservation.tenant_id` | B-D5 | RUNTIME_EXECUTION | PROVENANCE_REQUIRED_AT_BOUNDARY | IDENTITY_CARRIER |
| 6 | `intergrax/memory/retention_sweep.py` :: `MemoryRetentionSweepOutput.tenant_id` | B-D2 (`4b0f927a2`) | MEMORY_INDEX | PASSIVE_CARRIER_ONLY | IDENTITY_CARRIER |
| 7 | `intergrax/memory/retention_sweep.py` :: `MemoryRetentionSweepPayload.tenant_id` | B-D2 | MEMORY_INDEX | PASSIVE_CARRIER_ONLY | IDENTITY_CARRIER |
| 8 | `intergrax/memory/retention_sweep.py` :: `MemoryRetentionSweepReport.tenant_id` | B-D2 | MEMORY_INDEX | PASSIVE_CARRIER_ONLY | IDENTITY_CARRIER |
| 9 | `intergrax/contracts/artifact_ref.py` :: `ArtifactScope.tenant_id` | B-D8 (`92ee26bf3`) | UNKNOWN | UNKNOWN_REQUIREMENT | IDENTITY_CARRIER |
| 10 | `intergrax/runtime/nexus/artifacts/context_reference.py` :: `ArtifactReferenceToken.tenant_id` | B-D8 | RUNTIME_EXECUTION | PROVENANCE_REQUIRED_AT_BOUNDARY | IDENTITY_CARRIER |
| 11 | `intergrax/runtime/nexus/artifacts/governed_access.py` :: `ArtifactReadRequest.tenant_id` | B-D8 | RUNTIME_EXECUTION | PROVENANCE_REQUIRED_AT_BOUNDARY | IDENTITY_CARRIER |
| 12 | `intergrax/runtime/nexus/artifacts/governed_access.py` :: `GovernedDownloadGrant.tenant_id` | B-D8 | RUNTIME_EXECUTION | PROVENANCE_REQUIRED_AT_BOUNDARY | IDENTITY_CARRIER |
| 13 | `intergrax/runtime/nexus/artifacts/retention.py` :: `ArtifactRetentionSweepOutput.tenant_id` | B-D8 | RUNTIME_EXECUTION | PROVENANCE_REQUIRED_AT_BOUNDARY | IDENTITY_CARRIER |
| 14 | `intergrax/runtime/nexus/artifacts/retention.py` :: `RetentionSweepReport.tenant_id` | B-D8 | RUNTIME_EXECUTION | PROVENANCE_REQUIRED_AT_BOUNDARY | IDENTITY_CARRIER |
| 15 | `intergrax/runtime/long_running/recurrence.py` :: `RecurrenceRule.tenant_id` | B-D7 (`52f35a41f`) | RUNTIME_EXECUTION | PROVENANCE_REQUIRED_AT_BOUNDARY | IDENTITY_CARRIER |

Removed since the pin (the field no longer exists in the module, which B-D8 `92ee26bf3` last changed; `intergrax/applications/contracts/application_artifacts.py` no longer contains the word `tenant`):

| # | Surface |
| --- | --- |
| R1 | `intergrax/applications/contracts/application_artifacts.py` :: `ApplicationArtifactRef.tenant_id` |
| R2 | `intergrax/applications/contracts/application_artifacts.py` :: `SandboxArtifactRef.tenant_id` |
| R3 | `intergrax/applications/contracts/application_artifacts.py` :: `WorkspaceArtifactRef.tenant_id` |

Arithmetic at `dd58baa4d`: `800 + 19 + 15 - 3 = 831`. The set of 831 ids is identical at `146e7acf1`, at `c08efe46a` (the audited SHA) and at `dd58baa4d`, so none of the merged P7 repairs changed it; all 15 additions and 3 removals come from the P5 batches. The T-IGX-041-G repair (D7) did not change the pinned total (measured 831 before and after on its branch; recurrence's `RecurrenceRule.tenant_id` was already counted).

## Deliverables

PHASE 2 starts only on the orchestrator's message after the D1, D2 and D6 repair PRs are merged and runs on a branch from the then-current `origin/development`. Tests first, shown red, then the re-pin.

1. **Recompute at the final head** (throwaway script under `.tmp/session/041-I/`, not committed): with the scanner of the tree under test, list the class-field surface ids at the pin commit `91649e5d8` (via `git ls-tree` and `git cat-file`, applying `_discovery._is_candidate_file` and `_discovery._discover_from_ast`, plus the canonical anchors of kind `class.field.tenant`) and at the final head. The script reproduces 819 and 831 at the two revisions known above. `ADDED = head - pin_commit`, `REMOVED = pin_commit - head`. The known-so-far sets are the expectation; any other member (for example from D1, D2 or D6) is reported in the PR description.
2. **Pinned tests first** (new file `tests/qualification/tenant_x/test_tenant_x_replay_carrier_reclassification.py`, written against a pure helper that PHASE 2 adds; shown red before the helper exists):
   - an extra carrier that is not in the classified set makes the check fail (count and membership);
   - a classified id that is no longer discovered fails; a removed id that is still discovered fails;
   - a swap (one carrier removed and one other added, same total) fails through the ids digest;
   - a classified entry without a role from the closed vocabulary or without a non-empty reason fails;
   - the arithmetic `800 + 19 + len(ADDED) - len(REMOVED) == pin` holds for the real data and fails for a mutated pin;
   - the existing 19-id set is byte-identical and still discovered.
3. **Classification entries** in `_replay_x_carrier_reconciliation.py`: a mapping `REPLAY_P7_ADDED_CLASSIFICATION: surface_id -> (role, reason)` with `role` in the closed vocabulary `TENANT_SCOPED` (the field carries the owning tenant of the record or request and is an isolation input) or `TENANT_FREE` (explicitly not an isolation input; a reason is mandatory), plus `REPLAY_P7_REMOVED_SURFACE_IDS` (frozenset). The implementer reads each class to write the reason; the inventory values in the table above are recorded facts, **not** changed (the classifier files are not in the write set). A carrier whose inventory requirement is `UNKNOWN_REQUIREMENT` (rows 2, 3, 9 today) is recorded with its role and reason like the others; the unknown is not hidden.
4. **Re-pin once.** `TENANT_X_REPLAY_HEAD_CARRIER_TOTAL` becomes `800 + 19 + len(ADDED) - len(REMOVED)` as computed in step 1 (831 if no in-flight repair changes the set). The existing frozenset of 19 ids stays untouched. `assert_replay_carrier_delta_classified()` is replaced by a strengthened version built on a pure function `replay_carrier_delta_violations(...)` returning a list of violation strings: delta equals `len(19-set) + len(ADDED) - len(REMOVED)`; every classified id is discovered; every removed id is not discovered; the discovered set is consistent with the classified sets through an **ids digest**: `TENANT_X_REPLAY_CLASS_FIELD_IDS_SHA256`, the SHA-256 of the sorted class-field surface ids joined by newlines at the final head. The test file `test_tenant_x_p0_current_head.py` is not edited (it calls the helper without arguments).
5. **Evidence in the PR description:** the final ADDED and REMOVED lists with roles and reasons, the digest, the red logs, the head SHA, and which of D1, D2 and D6 contributed carriers.

## Write set (exact)

- `tests/qualification/tenant_x/_replay_x_carrier_reconciliation.py` (pin, classification entries, strengthened helper)
- `tests/qualification/tenant_x/test_tenant_x_replay_carrier_reclassification.py` (new)
- Governance documents of this batch: `governance/tickets/T-IGX-041-I.md`, this plan, `PDC_IMPLEMENTATION_WBS.md`, `PDC_DEPENDENCY_DAG.md` and the batch table of `scripts/qualification/pdc_x_wbs.py` (row `B-P7-TX`, generated with `--write`).

Not edited: every path under `intergrax/`, `agents/`, `applications/`; `tests/qualification/tenant_x/test_tenant_x_p0_current_head.py`, `_inventory.py`, `_discovery.py`, `_carrier_semantics.py`, `_classifiers.py` and every other existing file of that directory; `intergrax/contracts/`, `intergrax/applications/_shared/`, `scripts/governance/`, `.github/`, the EBH-7 pins, the IIAC activation record.

## Protocol

1. Token-budget preflight before each phase. PHASE 2 starts only on the orchestrator's message; it first merges `origin/development` (normal merge) and reruns the recomputation.
2. One `uv run pytest` at a time over narrow paths; logs under `.tmp/proof/pdc-x-041i/`; scratch under `.tmp/session/041-I/`. `ruff check`, `ruff format --check` and `pyright` on the two touched Python files; no file-level `# ruff: noqa`.
3. Commits carry exactly one `Ref: governance/tickets/T-IGX-041-I.md` trailer and no AI attribution. Set the upstream to `origin/development` before the first push, push with an explicit refspec, open one PR against `development`, never merge.
4. The gate is never weakened, skipped, xfailed or deleted; the pin is changed exactly once in this ticket, at the final head. A write-set or scope change returns to the signer through the orchestrator.

## Acceptance

- `test_tx_p0_r3_head_q02_class_field_carrier_count_stable` passes at the final head; the whole of `tests/qualification/tenant_x` passes except any test that is red on `origin/development` for an unrelated reason (named in the PR).
- Every member of ADDED has a role and a reason; every member of REMOVED is absent at head; the arithmetic and the digest hold; the 19-id set is byte-identical.
- The new negative tests were red before the helper and are green after; a mutated pin, an unclassified extra carrier, a swap, a missing reason and a still-present removed id each fail.
- `git diff --stat origin/development...HEAD` lists only the write set.

## Test contract

```text
uv run pytest tests/qualification/tenant_x -q
uv run ruff check tests/qualification/tenant_x/_replay_x_carrier_reconciliation.py tests/qualification/tenant_x/test_tenant_x_replay_carrier_reclassification.py
uv run ruff format --check tests/qualification/tenant_x/_replay_x_carrier_reconciliation.py tests/qualification/tenant_x/test_tenant_x_replay_carrier_reclassification.py
uv run pyright tests/qualification/tenant_x/_replay_x_carrier_reconciliation.py tests/qualification/tenant_x/test_tenant_x_replay_carrier_reclassification.py
uv run python scripts/qualification/pdc_x_wbs.py --check
uv run python -m scripts.governance.igx_gov ticket
```

## Not a certification verdict; known red classes

This ticket is a test-only re-pin by classification. It is **not a certification verdict** and claims no FRZ change, no closure and no production readiness. It adds no product path; it does not re-pin EBH-7, so the EBH-7 executable-identity gate stays red until the campaign's final re-pin, and the IIAC activation record stays stale until the P8-04 rebind (both known red classes of the delegation's merge gate).

## Out of scope

- Any change to product code, to the classifiers or discovery of TENANT-X, to other TENANT-X pins (propagation edge count, mint/rewrite counts) or to the 19-id set.
- Repairing the carriers' behaviour or tenant checks (owned by the domain repairs); the EBH-7 re-pin; the IIAC activation; OD signatures; certification.
- Re-pinning more than once: a later change of the carrier set after PHASE 2 needs a new decision.
