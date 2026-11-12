# OD-MCP-P6-GOV-01 — P6 premature merge governance acknowledgment

| Field | Value |
| --- | --- |
| **Decision ID** | OD-MCP-P6-GOV-01 |
| **Status** | **REQUESTED** (unsigned) |
| **Ticket** | `governance/tickets/T-IGX-035-G.md` |
| **Findings** | F-MCP-P6-GOV-001 (CRITICAL), F-MCP-P6-GOV-002 (HIGH) |
| **PR** | #459 @ `a695c9c835bb5b4b206e6e3e00bb2aee23ca7b6b` (merged 2026-11-11T17:08:34Z) |
| **Audit execution SHA (G6)** | `9b8c19d75d5df0f4b7c87d2707811f3715805102` |
| **G6 evidence publication SHA** | `2d3eceb6cf7e93dccd4d96919319a6c969b4444b` |

## Question

Does the operator accept **post-publication** independent scoped acceptance (G6 `ACCEPT_SCOPED`, IIIC `PASS_SCOPED`) while **preserving on record** that PR #459 merged before independent G6 publication, before the ACT-N11 P6 rebind and while `governance-fast` was red, and adopt the prospective discipline of [`GOVERNANCE_RECONCILIATION.md`](../../iiac_evidence/MCP-SERVER-X/p6_g6_closeout/GOVERNANCE_RECONCILIATION.md)?

## Options

| Option | Meaning |
| --- | --- |
| **A (recommended)** | Accept `ACCEPT_SCOPED_WITH_GOVERNANCE_FINDING`. No retroactive pre-merge approval. F-MCP-P6-GOV-001 is recorded as a **recurrence** of F-MCP-P5-GOV-001: the prospective gate of OD-MCP-P5-GOV-01 was in force and did not prevent it. The mechanical gap (the `development` ruleset requires no status checks) is cross-referenced to OPEN-BP-01. |
| **B** | Revert the PR #459 product merge. Not recommended: a product-path change creates a new ACT-N11 delta requiring new independent audits, and it does not remove the finding. |

## Operator line (required; not written by an agent)

```text
Approved: OD-MCP-P6-GOV-01 option __ @ ________________
```

**Explicit non-actions (both options):** does not retroactively approve PR #459 before its merge; does not waive any governance control; does not promote FRZ; does not declare MCP-SERVER-X CLOSED; does not authorize P7.
