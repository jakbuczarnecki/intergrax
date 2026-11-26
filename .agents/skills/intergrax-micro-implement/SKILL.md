---
name: intergrax-micro-implement
description: Explicit operator mode — strict micro-implementation contract with hard read/grep caps and no subagents for small bounded tasks. Activate only when requested.
---

# intergrax-micro-implement (Antigravity skill adapter)

Read and follow the canonical rule (SSOT): `agent-env/rules/intergrax-micro-implement.md`.

Key reminders:
- Start in a fresh chat (`MICRO IMPLEMENT = NEW CHAT`).
- Print mandatory preflight (`read scope`, `edit scope`, `tests`, `limits`) first.
- Enforce hard limits: `MAX_READS: 8`, `MAX_GREP: 3` (path-filtered), no subagents, no full docs or full test suite unless listed.
- Return the exact <= 12-line structured final report.
