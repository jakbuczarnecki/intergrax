---
description: Independent G6 audit — implementer must not PASS; separate session only
loading: auto
---

# Independent audit (G6) and IIAC

- **Implementer / tester / reviewer sessions MUST NOT** declare G6 PASS, IIAC PASS, certification, FRZ closure, or set [`IIAC_POLICY_ACTIVATION_RECORD.json`](../../docs/project/maintainers/qualification/IIAC_POLICY_ACTIVATION_RECORD.json) to `IN_FORCE`.
- Organizational independence requires a **new chat** with role `independent-quality-auditor` and operator authorization.
- **IIAC (INTEGRAX Internal Independent Agent Certification):** after Session A governance implementation, formal EVOL/ENT closure follows [`INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md`](../../docs/project/maintainers/qualification/INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md) once policy is activated (Session B governance audit first).
- Session prompts: [`IIAC_SESSION_B_INDEPENDENT_GOVERNANCE_AUDITOR_INSTRUCTIONS.md`](../../docs/project/maintainers/qualification/IIAC_SESSION_B_INDEPENDENT_GOVERNANCE_AUDITOR_INSTRUCTIONS.md), Session C/D EVOL/ENT instructions in same qualification folder.
- Follow [`docs/audit_results/AUDIT_PROTOCOL.md`](../../docs/audit_results/AUDIT_PROTOCOL.md) when operator requests legacy audit — do not auto-load audit dossiers.
- Cursor `security-review` or `bugbot` subagents are **tools**, not G6 or IIAC sign-off.

Ticket lifecycle: implementer stops at `quality_gate` with `igx_gov ticket` checkpoint; operator moves to `ready_for_audit` for external review.

See [`docs/project/technical/devos/INDEPENDENT-REVIEW-HANDOFF.md`](../../docs/project/technical/devos/INDEPENDENT-REVIEW-HANDOFF.md).
