# T-IGX-058-C — Cursor direct development push preflight correction

## Scope

Repair the existing Cursor-only direct development push path under WF-01. Resolve the actual shell checkout (including an explicit single Git -C), run existing governance gates through the checkout's uv project environment without syncing, verify the exact origin/development..HEAD range, retain clean-tree and fast-forward checks, and provide actionable diagnostics. Reject unsupported global Git overrides, ambiguous compound push commands, multiple refspecs and hook bypass flags. Standard Cursor sessions must finish authorized publication themselves, without routine Operator handoff. Keep the separate scoped candidate publisher unchanged.

## Plan digest

Plan digest: n/a (bounded scope and write set in this ticket)

## Approved

Approved: operator @ 2026-11-24T12:44:00+01:00

Approval source: Operator chat instruction to implement the proposed direct development push correction in the repository directly on development. Timestamp records the approval minute. No independent certification is claimed.

## Write set

```text
.cursor/rules/igx-cursor-direct-execution.mdc
agent-env/rules/igx-cursor-direct-execution.md
governance/tickets/T-IGX-058-C.md
scripts/hooks/cursor_development_push.py
scripts/hooks/igx_cursor_integration_guard.py
tests/unit/governance/test_cursor_development_push.py
tests/unit/governance/test_cursor_integration_guard.py
```

## Test contract

```text
uv run --no-sync pytest tests/unit/governance/test_cursor_development_push.py tests/unit/governance/test_cursor_integration_guard.py tests/unit/governance/test_cursor_candidate_publication.py tests/unit/governance/test_claude_surface_unchanged.py -q
uv run --no-sync python scripts/governance/sync_agent_env.py --check
uv run --no-sync ruff check scripts/hooks/cursor_development_push.py scripts/hooks/igx_cursor_integration_guard.py tests/unit/governance/test_cursor_development_push.py tests/unit/governance/test_cursor_integration_guard.py
```

## Out of scope

Claude Code, shared Git hooks, branch protections, shared role registry, IIAC/ACT-N11, product code, candidate publisher, Software Factory and CI workflow definitions. No gate bypass or history rewrite. Actual Windows IDE qualification remains distinct from isolated regression tests and GitHub CI.

## Lifecycle state

in_progress
