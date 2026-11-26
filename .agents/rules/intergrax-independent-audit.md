---
trigger: model_decision
description: Independent G6 audit and IIAC certification boundary — implementer, tester, and reviewer sessions must never declare G6/IIAC PASS; separate conversation required.
---

# Independent Audit (G6) and IIAC (Antigravity adapter)

Canonical rule (SSOT): `agent-env/rules/intergrax-independent-audit.md`

- Implementer, tester, reviewer sessions and their subagents MUST NOT declare G6 PASS, IIAC PASS, certification, FRZ closure, or flip `IIAC_POLICY_ACTIVATION_RECORD.json` to `IN_FORCE`.
- Formal audit requires a **separate new Antigravity conversation** acting as `independent-quality-auditor` with operator authorization.
- Read `agent-env/rules/intergrax-independent-audit.md` whenever audit, G6, IIAC, or certification topics arise.
