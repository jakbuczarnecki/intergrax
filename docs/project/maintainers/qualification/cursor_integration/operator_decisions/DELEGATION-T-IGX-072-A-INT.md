# DELEGATION-T-IGX-072-A-INT — Operator commission for plan approval (T-IGX-072-A)

| Field | Value |
| --- | --- |
| **Delegation ID** | DELEGATION-T-IGX-072-A-INT |
| **Campaign** | IGX-DELIVERY-UNBLOCK-001 |
| **Ticket** | `T-IGX-072-A` |
| **Plan path** | `governance/tickets/plans/PLAN_T-IGX-072-A.md` |
| **Expected plan digest** | `00a49d58b53f26b2bc7d806125f311252d66af117102f1fa7628b17536d1b91f` |
| **Basis** | IIAC Amendment 01, rules R-1..R-6 (`governance/tickets/README.md`) |
| **Date** | 2026-11-23 |
| **Source of commission** | Operator chat commission **IGX-DELIVERY-UNBLOCK-001** |
| **Assurance** | `process_enforced` — delegate may refuse if verification fails |

## Operator commission

The operator commissioned a **new, independent, dedicated operator-delegate session** to review and approve governance for `T-IGX-072-A` (Cursor integrator corrections INT-F01..F03). This identifier supersedes the contested integrator use of `T-IGX-071-A`. EQO-WB02 `T-IGX-071-A` remains separate and unchanged.

This is **not** permission for the implementer to approve its own work.

## Scope of the delegation

- Exactly one approval block in `governance/tickets/T-IGX-072-A.md`.
- Bounded to integrator correction in `scripts/cursor/igx_integrator.py` and listed tests.
- Delegate session must differ from `Implementer session:` on the ticket.

## Explicit prohibited operations

- Disabling CI or branch protection
- Using the integrator under repair to authorize its own publication
- SFX R2 canonical adoption or product implementation