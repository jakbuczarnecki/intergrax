# DELEGATION-T-IGX-071-A-INT — Operator commission for plan approval (T-IGX-071-A)

| Field | Value |
| --- | --- |
| **Delegation ID** | DELEGATION-T-IGX-071-A-INT |
| **Campaign** | IGX-CURSOR-INTEGRATOR-CORR-001 |
| **Ticket** | `T-IGX-071-A` |
| **Plan path** | `governance/tickets/plans/PLAN_T-IGX-071-A.md` |
| **Expected plan digest** | `09db804be84a502ea1cfb7ae2c973a86199fd824f92ca7a28612c95c0a5f7a2e` |
| **Basis** | IIAC Amendment 01, rules R-1..R-6 (`governance/tickets/README.md`) |
| **Date** | 2026-11-23 |
| **Source of commission** | Operator chat commission **IGX-CURSOR-INTEGRATOR-CORR-001** |
| **Assurance** | `process_enforced` — delegate may refuse if verification fails |

## Operator commission

The operator commissioned a **new, independent, dedicated operator-delegate session** to review and approve governance for `T-IGX-071-A` (Cursor integrator corrections INT-F01..INT-F03). This is **not** permission for the implementer to approve its own work.

The delegate independently verifies plan scope, digest, and exclusions (no Claude Code change; no SFX product implementation).

## Scope of the delegation

- Exactly one approval block in `governance/tickets/T-IGX-071-A.md`.
- Bounded to integrator correction in `scripts/cursor/igx_integrator.py` and listed tests.
- Delegate session must differ from `Implementer session:` on the ticket.

## Explicit prohibited operations

- Disabling CI or branch protection
- Using the integrator under repair to authorize its own publication
- SFX R2 canonical adoption or product implementation