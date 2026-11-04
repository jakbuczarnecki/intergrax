# T-IGX-007-A — Plan-verify hardening (OD-05)

## Scope

Wave W4 in [`DCP-CLOSURE-BACKLOG.md`](../DCP-CLOSURE-BACKLOG.md): enforce ticket FSM plan guards beyond procedural YAML; operator OD-05 attestation when binding.

## Lifecycle state

draft

## Test contract

```bash
uv run pytest tests/unit/governance/test_dcp_control_plane_registry.py -q
uv run python -m scripts.governance.igx_gov doctor
```

## Out of scope

PostgreSQL / external lease store
