# T-IGX-041-I — PDC-X P7: classify the new TENANT-X class-field carriers and re-pin the carrier-count gate once

## Scope

Test-only re-pin, by Operator decision 2 of `DELEGATION-PDC-P7-REPAIR.md` (2026-11-15: the TENANT-X carrier-count gate may be re-pinned once, after the last domain repair, by classification and re-pin only). `test_tx_p0_r3_head_q02_class_field_carrier_count_stable` fails on `development` (`831` carriers against the pin `819`; audit findings F-D2-03, F-D3-06, F-D6-06). Known so far at `dd58baa4d`: 15 class-field carriers added and 3 removed since the pin commit `91649e5d8` (all from the P5 batches B-D2, B-D5, B-D7 and B-D8; the merged P7 repairs did not change the set). The plan classifies each carrier with a role and a reason, re-pins the total once at the final head, replaces the weak classification check by a strengthened one (exact arithmetic, removed ids absent, ids digest, closed role vocabulary) and adds negative tests proving that an unclassified, swapped or unreasoned carrier still fails. PHASE 2 runs only after the D1, D2 and D6 repair PRs (T-IGX-041-A, -B, -F) are merged, because they may add or remove carriers; it recomputes the set at the final head. Product code: no. Batch: `B-P7-TX`. Plan: `PLAN_T-IGX-041-I.md`. This ticket is not a certification verdict; the EBH-7 identity gate stays red until the final re-pin of the campaign.

## Plan digest

Plan digest: 11df8fb703e590b4d258a136fb6bd9c08b1c0e1f98ee80d66e326cf2473a3951

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-041-I.md`](plans/PLAN_T-IGX-041-I.md).

## Approved

(pending; the plan is signed by a delegate session under `DELEGATION-PDC-P7-REPAIR.md`, which covers T-IGX-041-A..I; the implementer never writes this section)

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

plan_pending
