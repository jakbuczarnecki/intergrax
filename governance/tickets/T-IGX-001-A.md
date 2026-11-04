# T-IGX-001-A — DCP Phase 0–1: tickets + igx_gov + commit Ref verifier

## Scope

Delivery DevOS foundation: governance manifest, policy registry, ticket SSOT, commit `Ref:` enforcement, `scripts/governance/igx_gov` quality CLI, hook integration with existing push gate.

## Plan digest

Plan digest: (record when plan file is pinned)

## Approved

Approved: (operator) @ (timestamp)

## Test contract

```bash
uv run pytest tests/unit/governance/test_dcp_commit_ticket_refs.py tests/unit/governance/test_dcp_control_plane_registry.py -q
uv run python -m scripts.governance.igx_gov doctor
```

## Out of scope

- PostgreSQL / gateway lease
- GitHub Issues integration
- FRZ-* status changes
- Docker / vendor migration

## Lifecycle state

closed

## Implementation

- Git: `2eddd42230da93f5f26dd648840c3c8af629f7ba` on `development`
- Phases 0–7 delivered in same program batch (control plane, skills, docs hub)

## Checkpoint

- pytest DCP unit tests: pass
- `igx_gov doctor`: OK
