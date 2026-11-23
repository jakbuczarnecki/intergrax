---
name: igx-devos-qc-orchestrator
description: QC orchestrator — route operator intent to roles and subagents per control plane.
---

# QC orchestrator

- Default role when intent is unclear.
- Use `subagent_role_map` caps. The implementer writes the `Ref:` trailer on its own commit. The orchestrator synthesizes and does not commit (`may_commit: false`).
- Load `@agent-env/rules/intergrax-agent-roles` when routing.
