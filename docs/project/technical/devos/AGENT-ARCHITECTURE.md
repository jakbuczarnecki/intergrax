# Agent architecture (Delivery DevOS)

## Control plane SSOT

- [`governance/agent-control-plane/role_registry.yaml`](../../../../governance/agent-control-plane/role_registry.yaml) — seven roles, `default_role: orchestrator`, `subagent_role_map`.
- [`governance/agent-control-plane/skill_routing_registry.yaml`](../../../../governance/agent-control-plane/skill_routing_registry.yaml) — skill id → role + `.cursor/skills/igx-devos-*`.
- [`governance/agent-control-plane/task_lifecycle.yaml`](../../../../governance/agent-control-plane/task_lifecycle.yaml) — ticket FSM (procedural).

Generated matrix: [`MATRIX.md`](../../../../governance/agent-control-plane/MATRIX.md) via `generate_control_plane_matrix.py`.

## Role catalog (summary)

| Role | Commits? | Notes |
|------|----------|-------|
| orchestrator | No | Routes intent, subagents |
| architect-planner | No | Plan digest, scope |
| implementer | Yes | Ticket-bound code |
| tester | No | Verification checkpoint |
| reviewer | No | Not G6 |
| delivery-closure | No | `gh pr create` only |
| independent-quality-auditor | No | Separate session |

## Cursor integration

- Rules: `@intergrax-agent-roles`, `@intergrax-independent-audit`
- Skills: `.cursor/skills/igx-devos-*/SKILL.md`
- Token budget subagent exception when control plane active (see `intergrax-token-budget.mdc`)

## Verification

```powershell
uv run python scripts/governance/verify_agent_control_plane.py
uv run python scripts/governance/verify_control_plane_consistency.py
uv run python scripts/governance/verify_agent_skills.py
uv run python -m scripts.governance.igx_gov doctor
```
