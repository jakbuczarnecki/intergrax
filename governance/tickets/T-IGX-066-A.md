# T-IGX-066-A — Wave 06 OD-REPLAYX-09: ROADMAP-REPLAY-X prod-Δ disposition (wariant A)

## Scope

EQO Wave 06 **AUTH-02**. Reconcile ROADMAP-REPLAY-X mechanical exit gates @ `development` from DUP-X `a7d83925` through evidence pin `da5ca5b53` without re-pinning DUP-X. Add ticket-scoped W06 prod-Δ batches (11 classification bands), refresh ECAP six-counter table @ post-PASS SHA, restore B6 R2 bidirectional reconciliation and vendor-neutrality replay counter to **0**.

Operator decision id: **OD-REPLAYX-09** (variant A). Continuation of PRE-ENT line `T-IGX-034-E`; does not supersede OD-REPLAYX-08 scoped closure.

## Plan digest

Plan digest: 56cff29df9b016111a0696bd8065766a28f3d0ab31af039c2c6f3ea45ff2537a

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-066-A.md`](plans/PLAN_T-IGX-066-A.md).

## Approved

Approved: operator @ 2026-11-23T07:45:00+01:00



## Write set

See plan section 4 (`_w06_eqo_prod_delta.py`, `_prod_delta_disposition.py`, W06 qual docs, ECAP refresh).

## Test contract

```text
uv run pytest tests/qualification/roadmap_replay_x/test_roadmap_replay_x_mechanical_exit_gates.py -q
uv run pytest tests/qualification/roadmap_replay_x/test_roadmap_replay_x_scoped_path_net_zero.py -q
```

Plus batch regression bundle in plan section 7.

## Out of scope

FRZ promotion; DUP-X re-pin; mechanical assertion weakening; `intergrax/` product semantic edits; `T-IGX-053-A` FCS gate; EQO global CLOSED / FCS / M3.

## Lifecycle state

quality_gate

## Checkpoint

@ post-impl: mechanical **16/16 PASS**, scoped_path_net_zero PASS — `.tmp/session/w06/pytest_w06_post_impl.log` (27 nodeids incl. globals). Pre-impl baseline: `.tmp/session/w06/pytest_replay_mechanical_da5ca5b.log` (3 FAIL).