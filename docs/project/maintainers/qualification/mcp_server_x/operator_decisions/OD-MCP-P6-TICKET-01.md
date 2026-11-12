# OD-MCP-P6-TICKET-01 — T-IGX-035-G formal scoped closure

| Field | Value |
| --- | --- |
| **Decision ID** | OD-MCP-P6-TICKET-01 |
| **Status** | **APPROVED** |
| **Ticket** | `governance/tickets/T-IGX-035-G.md` |
| **Audit verdicts** | G6 `ACCEPT_SCOPED`; IIIC `PASS_SCOPED` |

## Question

Authorize lifecycle transition **`T-IGX-035-G` → `closed` (scoped)** with AC-P6-01..08 satisfied @ the documented SHAs and F-MCP-P6-GOV-001, F-MCP-P6-GOV-002 and F-MCP-P6-GOV-003 on record (not waived)?

## Conditions (all must be true at signing)

1. G6 `ACCEPT_SCOPED` and IIIC `PASS_SCOPED` evidence published on `origin/development`.
2. `IIAC_POLICY_ACTIVATION_RECORD.json` bound to the P6 audit pin with `verify_iiac_policy_activation --require-in-force` exit 0 on `origin/development`.
3. Required CI checks green on the publication merge and on `development`.
4. Activation OD (`IIAC_OD_ACTIVATION_MCP_ACT_N11_P6_2026-11-11.md`), OD-MCP-P6-GOV-01 and OD-MCP-P6-GOV-02 signed (option A).
5. The activation PR #464 is merged as a merge commit (`66dcd1bfe260b33a41881c7b9911b2a2abb752f9`) and the `Regression gate` push run on `origin/development` at or after `dcdb35601305a2ad9a3ee22894b26c29800a9640` shows `governance-fast`, `ci-smoke` and `evol-x-b4-preflight` successful.

## Operator line (written by a designated signing session; see provenance note)

```text
Authorized: operator @ 2026-11-12T07:35:55+01:00
Ticket closed: T-IGX-035-G — scoped P6; F-MCP-P6-GOV-001/002/003 preserved; post-publication independent G6 and IIIC records retained
```

Provenance note: the line above was written by an operator-designated signing session (IIAC Amendment 01, R-1..R-6) under the delegation record `docs/project/maintainers/qualification/IIAC_OD_DELEGATION_MCP_ACT_N11_P6_2026-11-12.md` (addendum 2, commit b35f43e75), on the operator's chat commission of 2026-11-12 and the operator's direct confirmation in chat to the signing session. It is not claimed to be operator-written. The five conditions above were verified read-only by the signing session before signing (details in addendum 2). The timestamp is the designated session's system clock at the moment of signing. This signature authorizes the transition named in the second line; it does not itself edit the ticket file.

**Explicit non-actions:** no retroactive pre-merge approval for PR #459 or PR #460; no FRZ promotion; no MCP-SERVER-X stage CLOSED; no P7 start.
