# MCP-SERVER-X P6 — terminal dossier (scoped, recovery in progress)

**Ticket:** `T-IGX-035-G`  
**Product publication:** `a695c9c835bb5b4b206e6e3e00bb2aee23ca7b6b`  
**Disposition:** **NOT CLOSED** — governance recovery published; operator activation pending

| Gate | Status |
| --- | --- |
| G6 | **`ACCEPT_SCOPED`** @ `2d3eceb6c` evidence |
| IIIC ACT-N11 P6 | **`PASS_SCOPED`** @ `act_n11_mcp_p6/` |
| Certifier | **`CERTIFIER_PREFLIGHT_PASS`** |
| IIIC `IN_FORCE` rebind | **PENDING** operator |
| `governance-fast` | **Expected FAIL** until activation |

## Governance (immutable)

F-MCP-P6-GOV-001: merge before ACT-N11 activation — [`GOVERNANCE_RECONCILIATION.md`](../iiac_evidence/MCP-SERVER-X/p6_g6_closeout/GOVERNANCE_RECONCILIATION.md)

**MCP-SERVER-X stage:** not CLOSED. **FRZ:** not promoted.

## Recovery verification notes (W1)

- **Pin-only certifier edits:** the diff `2d3eceb6c..276c64305` on auditor-authored files (IIIC dossier, IIIC findings, G6 manifest, G6 dossier) changes only `PENDING` placeholders to the evidence publication SHA `2d3eceb6cf7e93dccd4d96919319a6c969b4444b`. Independence of G6 auditor, IIIC auditor and preflight certifier is a `process_enforced` attestation; the activation certifier is a separate new session.
- **Inventory erratum:** the cumulative ACT-N11 inventory is **17** paths (P4: 16, plus `intergrax/applications/_shared/mcp_developer_api.py`), recomputed from `11ac8ad85..a695c9c83` and equal to `audit_scope.json`. Earlier P6 drafts say "19" / "16 + 3". Auditor files are not edited.
- **Minor:** the G6 dossier states ~225 s replay; the stored log shows 97.57 s; auditor logs carry no command/SHA header.
- **Structural:** the `development` ruleset requires no status checks; mandatory CI is policy-only (cross-reference OPEN-BP-01).
- **Activation OD:** the draft `IIIC_OD_ACTIVATION_MCP_ACT_N11_P6_2026-11-11.md` is superseded by the gate-compliant, still unsigned `IIAC_OD_ACTIVATION_MCP_ACT_N11_P6_2026-11-11.md`. Operator decisions OD-MCP-P6-GOV-01, OD-MCP-P6-TICKET-01 and OD-MCP-P6-P7-01 are REQUESTED (unsigned).
