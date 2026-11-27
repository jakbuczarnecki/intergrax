# T-IGX-058-D — Cursor development push: decouple publication from working-tree cleanliness

## Scope

Remove the Cursor-only `git status --porcelain` clean-tree gate from `scripts/hooks/cursor_development_push.py`. The Operator decision supersedes the clean-tree retention requirement in `T-IGX-058-C` only; that ticket is preserved unchanged.

Cursor development publication is independent of working-tree cleanliness; existing commit-range, branch, ancestry, governance and remote protections remain mandatory.

## Plan digest

Plan digest: n/a (operator corrective instruction EC-W3-CORRECTIVE-IMPLEMENTATION-04 Package A)

## Approved

Approved: operator @ 2026-11-27T10:00:00+01:00

Authorization: EC-W3-CORRECTIVE-IMPLEMENTATION-04 explicit permanent removal of the Cursor pre-push clean-working-tree requirement.

## Write set

```text
governance/tickets/T-IGX-058-D.md
scripts/hooks/cursor_development_push.py
tests/unit/governance/test_cursor_development_push.py
```

## Test contract

```text
uv run --no-sync pytest tests/unit/governance/test_cursor_development_push.py tests/unit/governance/test_cursor_integration_guard.py tests/unit/governance/test_cursor_candidate_publication.py tests/unit/governance/test_claude_surface_unchanged.py -q
uv run --no-sync ruff check scripts/hooks/cursor_development_push.py tests/unit/governance/test_cursor_development_push.py
uv run --no-sync python scripts/governance/sync_agent_env.py --check
```

## Out of scope

`.cursor/hooks.json`, shared Git hooks, CI workflows, `IGX_HOOKS=off`, `IGX_CURSOR_PUSH_PRECHECK_SKIP`, candidate publisher, Claude Code surface, branch protection, force push policy, `igx_integrator.py` / `igx_publish_candidate.py` clean-tree checks.

## Lifecycle state

in_progress