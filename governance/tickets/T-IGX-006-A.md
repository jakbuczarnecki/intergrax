# T-IGX-006-A — validate_policy_registry + governance-fast CI

## Scope

Wave W3 in [`DCP-CLOSURE-BACKLOG.md`](../DCP-CLOSURE-BACKLOG.md): `validate_policy_registry.py`, registry tests, governance-fast CI tier wiring.

## Lifecycle state

closed

## Test contract

```bash
uv run python scripts/governance/validate_policy_registry.py
uv run pytest tests/unit/governance/ -q
```

## Checkpoint

```
[INFO] validate_policy_registry: OK
75 passed tests/unit/governance/ (closure batch)
CI: governance-fast job on PR/push development|main
```

## Out of scope

FRZ checklist edits
