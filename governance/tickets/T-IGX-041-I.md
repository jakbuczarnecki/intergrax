# T-IGX-041-I — PDC-X P7: classify the new TENANT-X class-field carriers and re-pin the carrier-count gate once

## Scope

Test-only re-pin, by Operator decision 2 of `DELEGATION-PDC-P7-REPAIR.md` (2026-11-15: the TENANT-X carrier-count gate may be re-pinned once, after the last domain repair, by classification and re-pin only). `test_tx_p0_r3_head_q02_class_field_carrier_count_stable` fails on `development` (`831` carriers against the pin `819`; audit findings F-D2-03, F-D3-06, F-D6-06). Known so far at `dd58baa4d`: 15 class-field carriers added and 3 removed since the pin commit `91649e5d8` (all from the P5 batches B-D2, B-D5, B-D7 and B-D8; the merged P7 repairs did not change the set). The plan classifies each carrier with a role and a reason, re-pins the total once at the final head, replaces the weak classification check by a strengthened one (exact arithmetic, removed ids absent, ids digest, closed role vocabulary) and adds negative tests proving that an unclassified, swapped or unreasoned carrier still fails. PHASE 2 runs only after the D1, D2 and D6 repair PRs (T-IGX-041-A, -B, -F) are merged, because they may add or remove carriers; it recomputes the set at the final head. Product code: no. Batch: `B-P7-TX`. Plan: `PLAN_T-IGX-041-I.md`. This ticket is not a certification verdict; the EBH-7 identity gate stays red until the final re-pin of the campaign.

## Plan digest

Plan digest: 11df8fb703e590b4d258a136fb6bd9c08b1c0e1f98ee80d66e326cf2473a3951

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-041-I.md`](plans/PLAN_T-IGX-041-I.md).

## Approved

Approved: operator-delegate @ 2026-11-15T11:51:16+01:00
Delegation: docs/project/maintainers/qualification/platform_domain_completeness/operator_decisions/DELEGATION-PDC-P7-REPAIR.md@8f4b139c1f5352c9861872f0b5077996b8f0f891

Delegate checks (scope of `DELEGATION-PDC-P7-REPAIR.md` read at `origin/development`, last changing commit `8f4b139c1`; signed at the batch branch, implementer commit `cb754447b`, base `dd58baa4d` = `origin/development`; all passed before this line was written):

- The plan digest recomputed with `scripts/governance/plan_digest.py` equals the digest above (`11df8fb703e590b4d258a136fb6bd9c08b1c0e1f98ee80d66e326cf2473a3951`).
- The plan write set (`_replay_x_carrier_reconciliation.py`, new `test_tenant_x_replay_carrier_reclassification.py`) equals the write set of batch row `B-P7-TX` in `PDC_IMPLEMENTATION_WBS.md` and the entry in `scripts/qualification/pdc_x_wbs.py`; `pdc_x_wbs.py --check` is OK (107 rows, 28 batches, no write conflicts). The implementer commit changes only the ticket, the plan, the WBS, the DAG and the batch register; nothing under `intergrax/`, `agents/`, `applications/`, `scripts/governance/` or `.github/`; no hotspot.
- The gate was re-run by the delegate: `uv run pytest tests/qualification/tenant_x/test_tenant_x_p0_current_head.py` gives `assert 831 == 819` at `test_tenant_x_p0_current_head.py:97` (1 failed, 12 passed). The carrier set was recomputed with the head scanner of the tree under test: 819 class-field ids at the pin commit `91649e5d8` (via `git ls-tree` and `git cat-file`), 831 at head; ADDED = 15 and REMOVED = 3, identical to the two tables of the plan (same surfaces, in the same modules); the frozen 19-id set is a subset of both revisions; `800 + 19 + 15 - 3 = 831`. The first commit after the pin for each added module is the stated P5 batch (`4b0f927a2` B-D2, `92ee26bf3` B-D8, `52f35a41f` B-D7, `85a8c441d` B-D5); `application_artifacts.py` contains no `tenant` and was last changed by `92ee26bf3`. The statement that the audited SHA and `146e7acf1` give the same 831 ids was not re-scanned by the delegate; the head set and its P5 origins are verified.
- Line references verified: gate `test_tenant_x_p0_current_head.py:90-98`; `_replay_x_carrier_reconciliation.py:11-13`, the 19-id frozenset and `assert_replay_carrier_delta_classified` at `:41`; `_inventory.py:70` `class_field_tenant_surface_ids`; `_carrier_semantics.py:359` `class_field_carrier_total=len(class_fields)`. The only other importers of the helper are audit evidence scripts, not tests, so replacing `assert_replay_carrier_delta_classified()` by a strengthened version with the same call signature breaks no test.
- Classification and re-pin only, never weakening or deleting the gate: the count assertion, the membership check and the 19-id set stay; the plan adds exact arithmetic, removed ids absent, an ids digest and a closed role vocabulary with a mandatory reason, and negative tests (extra unclassified carrier, classified id no longer discovered, removed id still discovered, swap with equal total, missing role or reason, mutated pin). The inventory classifiers are not edited; unknown requirements are recorded, not hidden. The pin is changed exactly once.
- PHASE 2 is sequenced after the D1, D2 and D6 repair PRs (T-IGX-041-A, -B, -F), the only repairs not yet on `origin/development` (T-IGX-041-C, -D, -E, -G, -H are merged at `dd58baa4d`); it starts only on the orchestrator's message, merges `origin/development`, recomputes the set at the final head before editing, and reports any additional member in the PR description. This signature covers the plan and the classification procedure, not a pin value computed on another head.
- No registry, scheduler, engine, trust, policy or audit authority is added; the plan states it is not a certification verdict and leaves the EBH-7 re-pin and the IIAC activation untouched. The implementer session and this delegate session are different sessions; the delegate edited nothing except this approval block and the lifecycle state.
- Merge gate (stated as the delegation record requires): the test-only pull request merges under authorizations 1 and 3 of `DELEGATION-PDC-STANDING.md` and decision 8 of `DELEGATION-PDC-WAVE2.md`: green CI, only the known red classes (IIAC activation staleness, EBH-7 identity gate), `headRefOid` equal to the verified SHA. Write-set amendments are not delegated. Not decided here: the numbering of the standing authorizations was not re-verified.

## Test contract

```text
uv run pytest tests/qualification/tenant_x -q
uv run ruff check tests/qualification/tenant_x/_replay_x_carrier_reconciliation.py tests/qualification/tenant_x/test_tenant_x_replay_carrier_reclassification.py
uv run ruff format --check tests/qualification/tenant_x/_replay_x_carrier_reconciliation.py tests/qualification/tenant_x/test_tenant_x_replay_carrier_reclassification.py
uv run pyright tests/qualification/tenant_x/_replay_x_carrier_reconciliation.py tests/qualification/tenant_x/test_tenant_x_replay_carrier_reclassification.py
uv run python scripts/qualification/pdc_x_wbs.py --check
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

- Every path under `intergrax/`, `agents/`, `applications/`; the TENANT-X classifiers, discovery and the other pins of that package; the existing 19-id classified set (stays byte-identical).
- `scripts/governance/`, `.github/`, `intergrax/contracts/`, `intergrax/applications/_shared/`, the EBH-7 re-pin, the IIAC activation record, OD signatures, certification.

## Lifecycle state

plan_approved
