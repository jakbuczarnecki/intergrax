# T-IGX-007-A — Plan-verify hardening (OD-05)

## Scope

Wave W4 in [`DCP-CLOSURE-BACKLOG.md`](../DCP-CLOSURE-BACKLOG.md): enforce ticket FSM plan guards beyond procedural YAML; operator OD-05 attestation when binding.

## Lifecycle state

closed

## Test contract

```bash
uv run pytest tests/unit/governance/test_dcp_control_plane_registry.py tests/unit/governance/test_agent_role_routing.py -q
uv run python -m scripts.governance.igx_gov doctor
uv run python scripts/governance/verify_ticket_lifecycle.py governance/tickets/T-IGX-007-A.md
```

## Checkpoint

```
[INFO] verify_ticket_lifecycle: OK (T-IGX-007-A)
task_lifecycle.yaml guards documented + verify_ticket_lifecycle.py
validate_policy_registry wired into igx_gov ticket
```

## Out of scope

PostgreSQL / external lease store
