# T-IGX-006-A — validate_policy_registry + governance-fast CI

## Scope

Wave W3 in [`DCP-CLOSURE-BACKLOG.md`](../DCP-CLOSURE-BACKLOG.md): `validate_policy_registry.py`, registry tests, governance-fast CI tier wiring.

## Lifecycle state

draft

## Test contract

```bash
uv run python scripts/governance/validate_policy_registry.py
uv run pytest tests/unit/governance/ -q
```

## Out of scope

FRZ checklist edits
