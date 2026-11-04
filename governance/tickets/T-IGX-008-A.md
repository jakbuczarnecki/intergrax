# T-IGX-008-A — DCP closure remainder (W5–W8)

## Scope

Waves W5–W8 in [`DCP-CLOSURE-BACKLOG.md`](../DCP-CLOSURE-BACKLOG.md): optional sync-adapters (Claude/Codex), `test_agent_role_routing`, branch protection + hooks rollout checkpoints, G6 handoff prep (manifest flags stay false until audit).

## Lifecycle state

draft

## Test contract

```bash
uv run pytest tests/unit/governance/test_dcp_*.py -q
uv run python -m scripts.governance.igx_gov doctor
```

## Out of scope

Implementer-declared G6 PASS; `global_certification.*` true without external audit
