# T-IGX-034-E — PRE-ENT Wave 1: ROADMAP-REPLAY-X mechanical exit gates (scoped prod delta)

## Scope

Wave 1 `replay` worktree batch: reconcile ROADMAP-REPLAY-X mechanical gates on `development` after predecessor product batches landed post DUP-X. Scope PRE-ENT L3 prod-Δ dispositions so `historical_evidence_invalidated_by_later_changes` and ECAP bidirectional counters return **0** without FRZ promotion or parent closure claims.

Deliverables: `_pre_ent_l3_prod_delta.py`, `_prod_delta_disposition.py` merge, `PRE_ENT_REPLAY_L3_PROD_DELTA.md`, drift log + ECAP C2 narrative alignment.

## Plan digest

Plan digest: 8dec4277bac8fd47ff41102445b9b6db7e50a1e997b4dbf966d8c6ca9e1f76f3

## Approved

(pending — PRE-ENT campaign)

## Test contract

```text
uv run pytest tests/qualification/roadmap_replay_x/test_roadmap_replay_x_mechanical_exit_gates.py -q
```

## Out of scope

FRZ PASS / row promotion, roadmap §5 publication, ENT-AUDIT-X, independent G6 audit, product-path fixes outside qualification SSOT, push/merge unless operator directs.

## Lifecycle state

in_progress

## Checkpoint

Execution @ HEAD `4a932d0cf5487b72e433805e372e3855fcd3d987` (2026-11-10): mechanical exit gates RED → L3 prod-Δ disposition reconciliation (this ticket).
