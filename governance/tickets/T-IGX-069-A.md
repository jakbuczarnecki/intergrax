# T-IGX-069-A — EQO W0-01 / W0-05: publish live `EQO_PIN_BIND.json` (isolated)

## Scope

Doc-only publication on `development` from `origin/development` base, **without** local commits T-IGX-065-A or train/W3 deltas.

- Replace `docs/project/maintainers/qualification/enterprise_quality_oversight/eqo_closure/EQO_PIN_BIND.json` from `.tmp/session/enterprise-roadmap-stage02b/W0_01_PROPOSED_PIN_BIND.json` (live `actual` refs after fetch; `publication_sha` set only in post-merge SHA ladder step per plan — not in the publishing commit body).
- W0-05 inventory §2–4 already published @ `8ad8aa733dd6b62094c5705b84a7dccfd605259c`; no inventory republish.

## Plan digest

Plan digest: d3afc459e1898964923f4901bfa4fd780e7568e212567dec375c54910982065d

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-069-A.md`](plans/PLAN_T-IGX-069-A.md)

## Write set

```text
docs/project/maintainers/qualification/enterprise_quality_oversight/eqo_closure/EQO_PIN_BIND.json
```

## Test contract

```text
uv run python -m scripts.governance.igx_gov ticket
```

On isolated checkout at `origin/development` after applying write set only.

## Out of scope

Product paths, tests, FRZ, roadmap, WBS status coercion, PT-T01, train commits, `eqo_closure/` files other than `EQO_PIN_BIND.json`.

## Approved

Approved: Operator @ 2026-11-23T09:25:00+01:00

## Lifecycle state

plan_approved

## Checkpoint

(Isolated worktree `t-igx-069-a-w0pub` — publication in flight.)