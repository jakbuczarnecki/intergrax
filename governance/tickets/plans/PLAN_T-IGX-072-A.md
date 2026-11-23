# PLAN T-IGX-072-A — Cursor integrator correction (INT-F01..F03)

## Classification

| Field | Value |
| --- | --- |
| Ticket | `T-IGX-072-A` |
| IIAC / change impact | Control-plane correction (OD-AF-INT A integrator only) |
| Collision note | Replaces contested `T-IGX-071-A` integrator ID; EQO-WB02 retains its own `T-IGX-071-A` draft |

## Work packages

### INT-F01 — CI required-check semantics

Preserve GitHub `--required` verification when configured. For `kind: docs` integrations targeting `development`, add explicit mandatory check profile (no bypass on empty required list):

- `ci-smoke`
- `ticket-governance`
- `governance-policy-pin`
- `governance-fast`
- `evol-x-b4-preflight`
- `ent-audit-fast`
- `ent-audit-pr-gate`

Fail closed on missing, pending, skipped, or failed mandatory/applicable checks. Product (`kind: product`) semantics unchanged.

### INT-F02 — UTF-8 subprocess decoding

Deterministic UTF-8 decoding for structured GitHub CLI output; controlled `Blocked` on invalid UTF-8 or malformed JSON; safe replacement decoding for diagnostic stderr only.

### INT-F03 — Idempotent publication

Reuse a single matching open PR (same repo, source branch, target, HEAD SHA); skip `gh pr create`; fail closed on duplicates, divergence, or inconsistent state.

## Target files

- `scripts/cursor/igx_integrator.py`
- `tests/unit/governance/test_cursor_integrator.py`
- `governance/tickets/T-IGX-072-A.md`
- `governance/tickets/plans/PLAN_T-IGX-072-A.md`
- `docs/project/maintainers/qualification/cursor_integration/operator_decisions/DELEGATION-T-IGX-072-A-INT.md`

## Test contract

```text
uv run pytest tests/unit/governance/test_cursor_integrator.py tests/unit/governance/test_cursor_integration_guard.py tests/unit/governance/test_claude_surface_unchanged.py -q
uv run python -m scripts.governance.igx_gov ticket
```

## Negative security tests

- Empty required + incomplete doc profile → blocked
- Skipped mandatory doc check → blocked
- Invalid UTF-8 JSON from `gh` → blocked
- Duplicate open PR records → blocked
- Remote branch divergence → blocked

## Independent review

Mandatory `igx-reviewer` session before publication; separate from implementer.

## Authorized publication

After approval and audit: merge to `development` via governed path (`fix/t-igx-072-a-int` PR or fast-forward push per branch protection).

## No-change scope

- Claude Code surface; SFX product implementation; FRZ/IIAC pins; GitHub settings; hook bypass