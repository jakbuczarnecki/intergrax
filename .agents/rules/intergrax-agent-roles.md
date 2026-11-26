---
trigger: model_decision
description: Intergrax Delivery DevOS agent roles, handoff chain, and subagent delegation routing (orchestrator, architect-planner, implementer, tester, reviewer, delivery-closure, independent-quality-auditor).
---

# Agent Roles (Antigravity adapter)

Canonical rule (SSOT): `agent-env/rules/intergrax-agent-roles.md`
Role authority SSOT: `governance/agent-control-plane/role_registry.yaml`

- Default top-level session role: **`orchestrator`** (`may_commit: false`).
- Route work to the matching skill (`igx-devos-*`) and typed Antigravity subagent (`igx-researcher`, `igx-architect`, `igx-implementer`, `igx-tester`, `igx-reviewer`).
- Handoff chain: `plan_approved` -> `in_progress` (`implementer`) -> `quality_gate` (`tester` checkpoint) -> `ready_for_audit` (separate conversation, `independent-quality-auditor`).
- Read `agent-env/rules/intergrax-agent-roles.md` when routing multi-step or cross-role work.
