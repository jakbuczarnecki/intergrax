# MCP-SERVER-X P6 — operator closure record

| Field | Value |
| --- | --- |
| **Ticket** | `T-IGX-035-G` — closure to **`closed` (scoped)** authorized by `OD-MCP-P6-TICKET-01`; the lifecycle flip is applied in the formal closure commit `b18a7afc41e25234ca7e2adc903cfea1e0bf63f0`, not in the commit that creates this record |
| **Authorized (ticket closure)** | operator @ **2026-11-12T07:35:55+01:00** (`OD-MCP-P6-TICKET-01`) |
| **G6 verdict** | `ACCEPT_SCOPED` (post-publication; dossier and `p6_g6_closeout/` evidence) |
| **G6 operator disposition** | `ACCEPT_SCOPED_WITH_GOVERNANCE_FINDING` (`OD-MCP-P6-GOV-01` option A) |
| **IIIC ACT-N11 P6 verdict** | `PASS_SCOPED` (`act_n11_mcp_p6/`) |
| **Certifier preflight** | `CERTIFIER_PREFLIGHT_PASS` |
| **G6 audit execution SHA** | `9b8c19d75d5df0f4b7c87d2707811f3715805102` |
| **G6 / IIIC evidence publication SHA** | `2d3eceb6cf7e93dccd4d96919319a6c969b4444b` |
| **IIIC certification publication SHA** | `276c64305799fa4d7f1a25018cc1cccf061fbafd` |
| **Product merge (PR #459)** | `a695c9c835bb5b4b206e6e3e00bb2aee23ca7b6b` |
| **Recovery evidence merge (PR #460)** | `8ed2bdcee6d244c128f3c5d95be3d7817418433b` |
| **Activation commits (two-phase)** | phase 1 `92ad3555775649742299826e3680ec8ce703cc8a`, phase 2 `19c617365cda85d88b474575e77b7d083739b529` |
| **Activation PR merge** | `66dcd1bfe260b33a41881c7b9911b2a2abb752f9` |
| **Formal closure commit** | `b18a7afc41e25234ca7e2adc903cfea1e0bf63f0` |
| **Closure PR merge** | `6e53fb36f39b4a57331e6c079f2fcbe6094e2786` (PR #468) |
| **Terminal verification** | `P6_TERMINAL_VERIFIED` @ `6e53fb36f39b4a57331e6c079f2fcbe6094e2786` - [`MCP_SERVER_X_P6_TERMINAL_VERIFICATION_REPORT.md`](./MCP_SERVER_X_P6_TERMINAL_VERIFICATION_REPORT.md) |

SHA roles are defined once in [`MCP_SERVER_X_SHA_LADDER.md`](./MCP_SERVER_X_SHA_LADDER.md); this table does not redefine them.

## Operator decisions

| Decision | Status | Operator line / timestamp |
| --- | --- | --- |
| **IIAC activation OD** `IIAC_OD_ACTIVATION_MCP_ACT_N11_P6_2026-11-11.md` | `AUTHORIZED` | `2026-11-12T06:35:45+01:00` (written by the operator-designated signing session) |
| **OD-MCP-P6-GOV-01** | APPROVED — Option A (`ACCEPT_SCOPED_WITH_GOVERNANCE_FINDING`) | `2026-11-12T06:35:45+01:00` (designated signing session) |
| **OD-MCP-P6-GOV-02** | APPROVED — Option A | `2026-11-12T06:46:49+01:00` (designated signing session) |
| **OD-MCP-P6-TICKET-01** | APPROVED | 2026-11-12T07:35:55+01:00 |
| **OD-MCP-P6-P7-01** | APPROVED — P7 entry effective only at the `origin/development` SHA where the P6 terminal verification PASS is published | 2026-11-12T07:35:55+01:00 |

Signing provenance: the activation OD, `OD-MCP-P6-GOV-01`, `OD-MCP-P6-GOV-02`, `OD-MCP-P6-TICKET-01` and `OD-MCP-P6-P7-01` lines were written by an operator-designated signing session under [`IIAC_OD_DELEGATION_MCP_ACT_N11_P6_2026-11-12.md`](../IIAC_OD_DELEGATION_MCP_ACT_N11_P6_2026-11-12.md) (IIAC Amendment 01, R-1..R-6) and are not claimed to be operator-written. Provenance of the `OD-MCP-P6-TICKET-01` and `OD-MCP-P6-P7-01` lines is whatever each file's own provenance note states.

## Findings preserved (not waived, not back-dated)

- **F-MCP-P6-GOV-001** (CRITICAL) — PR #459 merged before independent G6 publication, before the ACT-N11 rebind and while `governance-fast` was red; recorded as a recurrence of F-MCP-P5-GOV-001. No retroactive pre-merge approval ([`GOVERNANCE_RECONCILIATION.md`](../iiac_evidence/MCP-SERVER-X/p6_g6_closeout/GOVERNANCE_RECONCILIATION.md)). Covered by `OD-MCP-P6-GOV-01`.
- **F-MCP-P6-GOV-002** (HIGH) — ticket `T-IGX-035-G` was stale at merge of PR #459. Covered by `OD-MCP-P6-GOV-01`.
- **F-MCP-P6-GOV-003** (CRITICAL, governance) — recovery PR #460 merged as `8ed2bdcee6d244c128f3c5d95be3d7817418433b` (2026-11-12T05:26:38Z, head `b0b415e92082f306d2ee7de698c5f2e25822f04f`) with `governance-fast` failing and before `IIAC_POLICY_ACTIVATION_RECORD.json` was rebound. No retroactive approval of that merge. Covered by `OD-MCP-P6-GOV-02`.
- Structural gap: the `development` ruleset requires no status checks, so mandatory CI is policy-only (cross-reference OPEN-BP-01; not changed in P6).
- Cross-stage: `F-MCP-P5-MEM-001` / `LIM-RESEARCH-STRICT-FACTORY-MEMORY` (PROD-Q) remains OPEN, non-blocking.

## Explicit non-claims

MCP-SERVER-X stage **not CLOSED**; this record claims nothing beyond "P6 closed (scoped)"; no FRZ promotion; no IIIC P8 bypass; P6 closure is scoped to the audited P6 inventory only; P7 entry is effective only as stated in `OD-MCP-P6-P7-01`; none of the three governance findings is retroactively approved.
