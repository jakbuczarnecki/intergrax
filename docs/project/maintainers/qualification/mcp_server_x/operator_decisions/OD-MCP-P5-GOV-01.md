# OD-MCP-P5-GOV-01 — P5 merge sequencing and prospective G6 gate

| Field | Value |
| --- | --- |
| **Decision ID** | OD-MCP-P5-GOV-01 |
| **Status** | **APPROVED — Option A** |
| **Ticket** | `governance/tickets/T-IGX-035-F.md` |
| **Findings** | F-MCP-P5-GOV-001, F-MCP-P5-GOV-002 |
| **Audit execution SHA** | `c5989282937f5e592748cb606b2272d04837c601` |
| **G6 evidence publication SHA** | `7fd899aaea873aac6ec9ae3fe6ce5e3ef9ec9c8c` |
| **Closeout merge** | PR #457 @ `9ae3fdbe42831153c42846ca8231e4627973d3e3` |

## Question

Does the operator accept **post-publication** independent G6 scoped technical acceptance @ `c5989282937f5e592748cb606b2272d04837c601` while **preserving on record** that PR #456 merged before organizational G6 acceptance, and adopt the **prospective** merge gate in [`GOVERNANCE_RECONCILIATION.md`](../../iiac_evidence/MCP-SERVER-X/p5_g6_closeout/GOVERNANCE_RECONCILIATION.md)?

## Options

| Option | Meaning |
| --- | --- |
| **A (recommended)** | Accept `ACCEPT_SCOPED_WITH_GOVERNANCE_FINDING`; no retroactive pre-merge approval; adopt prospective corrective |
| **B** | Reject post-publication acceptance — require corrective replay-only path without ticket close |

## Operator record

```text
Authorized: operator @ 2026-11-11T16:24:00+01:00
Selected option: A — ACCEPT_SCOPED_WITH_GOVERNANCE_FINDING
```

**Explicit non-actions:** does not retroactively approve PR #456 before its merge; does not waive governance controls; does not promote FRZ; does not declare MCP-SERVER-X CLOSED.
