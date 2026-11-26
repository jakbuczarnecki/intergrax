---
name: intergrax-independent-audit
description: Independent G6 audit and IIAC certification protocol — strict organizational separation; implementer/tester/reviewer sessions never declare PASS.
---

# intergrax-independent-audit (Antigravity skill adapter)

Read and follow the canonical rule (SSOT): `agent-env/rules/intergrax-independent-audit.md`.

Key reminders:
- Implementer, tester, and reviewer sessions (and subagents spawned inside them) MUST NOT declare G6 PASS, IIAC PASS, certification, or FRZ closure.
- Organizational independence requires a **separate new Antigravity conversation** acting as `independent-quality-auditor` with operator authorization.
- Implementers stop at `quality_gate` with `igx_gov ticket` checkpoint; the operator transitions to `ready_for_audit`.
