# OD-MCP-P6-TICKET-01 — T-IGX-035-G formal scoped closure

| Field | Value |
| --- | --- |
| **Decision ID** | OD-MCP-P6-TICKET-01 |
| **Status** | **REQUESTED** (unsigned; sign only after the conditions below hold) |
| **Ticket** | `governance/tickets/T-IGX-035-G.md` |
| **Audit verdicts** | G6 `ACCEPT_SCOPED`; IIIC `PASS_SCOPED` |

## Question

Authorize lifecycle transition **`T-IGX-035-G` → `closed` (scoped)** with AC-P6-01..08 satisfied @ the documented SHAs and F-MCP-P6-GOV-001 / F-MCP-P6-GOV-002 on record (not waived)?

## Conditions (all must be true at signing)

1. G6 `ACCEPT_SCOPED` and IIIC `PASS_SCOPED` evidence published on `origin/development`.
2. `IIAC_POLICY_ACTIVATION_RECORD.json` bound to the P6 audit pin with `verify_iiac_policy_activation --require-in-force` exit 0 on `origin/development`.
3. Required CI checks green on the publication merge and on `development`.
4. Activation OD (`IIAC_OD_ACTIVATION_MCP_ACT_N11_P6_2026-11-11.md`) and OD-MCP-P6-GOV-01 signed.

## Operator line (required; not written by an agent)

```text
Authorized: operator @ ________________
Ticket closed: T-IGX-035-G — scoped P6; F-MCP-P6-GOV-001/002 preserved; post-publication independent G6 and IIIC records retained
```

**Explicit non-actions:** no retroactive pre-merge approval for PR #459; no FRZ promotion; no MCP-SERVER-X stage CLOSED; no P7 start.
