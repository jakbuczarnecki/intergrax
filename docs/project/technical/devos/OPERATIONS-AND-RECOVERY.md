# Operations and recovery (DCP)

## Stale advisory ticket lock

Locks live in `.tmp/session/ticket-locks/*.json` (advisory only).

```powershell
uv run python scripts/governance/ticket_lock.py status T-IGX-001-A
uv run python scripts/governance/ticket_lock.py release T-IGX-001-A
```

If holder session crashed, operator may `release` or delete lock file under `.tmp/`.

## Failed push gate

1. Read stderr from pre-push (ticket Ref vs regression).
2. Fix issues locally; rerun `uv run python -m scripts.governance.igx_gov ticket`.
3. Emergency only: set bypass env vars documented in DEVELOPER-QUALITY-PROCESS.md with operator approval.

## Doctor failures

```powershell
uv run python -m scripts.governance.igx_gov bootstrap
uv run python -m scripts.governance.igx_gov doctor
```

Fix missing skills/YAML reported by verify scripts before continuing delivery.

## Inventory

```powershell
uv run python -m scripts.governance.igx_gov inventory
```
