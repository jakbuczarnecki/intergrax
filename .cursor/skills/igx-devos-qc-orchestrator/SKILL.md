---
name: igx-devos-qc-orchestrator
description: QC orchestrator — route operator intent to roles and subagents per control plane.
---

# QC orchestrator

- Default role when intent is unclear.
- Use `subagent_role_map` caps; parent owns `Ref:` trailers and final synthesis.
- Load `@.cursor/rules/intergrax-agent-roles` when routing.
