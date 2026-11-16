# OD-MCP-P6-GOV-02 — recovery PR #460 merged with failing governance-fast (F-MCP-P6-GOV-003)

| Field | Value |
| --- | --- |
| **Decision ID** | OD-MCP-P6-GOV-02 |
| **Status** | **APPROVED — Option A** |
| **Ticket** | `governance/tickets/T-IGX-035-G.md` |
| **Finding** | F-MCP-P6-GOV-003 (CRITICAL) |
| **PR** | #460 merged as `8ed2bdcee6d244c128f3c5d95be3d7817418433b` on 2026-11-12T05:26:38Z (head `b0b415e92082f306d2ee7de698c5f2e25822f04f`) |
| **Related** | OD-MCP-P6-GOV-01 (signed, option A, covers F-MCP-P6-GOV-001/002) |

## Facts

PR #460 (evidence publication for the P6 recovery) was merged while `governance-fast` was failing (ACT-N11: three product paths changed after the audited SHA `8fff85e3bfc5190929ff464b00fbe759dadcb5f8`) and before `IIAC_POLICY_ACTIVATION_RECORD.json` was rebound. `origin/development` therefore stays red until the activation PR (branch `mcp-p6-gov-signing`) lands. The evidence and unsigned decision drafts are published; no product path, activation record or auditor artifact was changed by the merge.

## Question

Does the operator accept the continuation of the P6 recovery on branch `mcp-p6-gov-signing` (two-phase activation, green mandatory CI before merge) while **preserving on record** that PR #460 was merged with failing `governance-fast` and before activation (F-MCP-P6-GOV-003), as a recurrence of F-MCP-P5-GOV-001 and F-MCP-P6-GOV-001?

## Options

| Option | Meaning |
| --- | --- |
| **A (recommended)** | Accept with the governance finding. No retroactive approval of the merge of PR #460. Prospective discipline: the activation PR is merged (merge commit) only after every required check is green on its head. The mechanical gap (no required status checks on the `development` ruleset) stays cross-referenced to OPEN-BP-01. |
| **B** | Do not accept the continuation: halt P6 closure until the operator gives a further directive. |

## Operator line (written by a designated signing session; see provenance note)

```text
Approved: OD-MCP-P6-GOV-02 option A @ 2026-11-12T06:46:49+01:00
```

Provenance note: the line above was written by an operator-designated signing session (IIAC Amendment 01, R-1..R-6) under the delegation record `docs/project/maintainers/qualification/IIAC_OD_DELEGATION_MCP_ACT_N11_P6_2026-11-12.md` (addendum 1, commit 94a858308), on the operator's chat commission of 2026-11-12 and the operator's direct confirmation in chat to the signing session (option A as answered by the operator in chat). It is not claimed to be operator-written. The timestamp is the designated session's system clock at the moment of signing.

**Explicit non-actions (both options):** does not retroactively approve the merge of PR #459 or PR #460; does not waive any governance control; does not promote FRZ; does not declare MCP-SERVER-X CLOSED; does not authorize P7.
