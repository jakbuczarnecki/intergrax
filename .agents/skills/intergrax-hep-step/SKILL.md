---
name: intergrax-hep-step
description: Explicit operator mode — strict scoped HEP / EVID implementation step following operator-listed files and preflight budget. Activate only when requested.
---

# intergrax-hep-step (Antigravity skill adapter)

Read and follow the canonical rule (SSOT): `agent-env/rules/intergrax-hep-step.md`.

Key reminders:
- Perform mandatory preflight (`read scope`, `edit scope`, `tests`) before any read/edit.
- Read only operator-listed files (max 3 extra if required, and name them in the report); no repo-wide search, no subagents, no full test suite unless listed.
- Editing outside listed scope is a scope change: escalate.
