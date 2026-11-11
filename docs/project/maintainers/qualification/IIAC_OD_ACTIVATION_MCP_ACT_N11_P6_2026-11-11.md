# IIAC operator decision — MCP ACT-N11 product re-bind (P6 cumulative)

Decision ID: IIAC-OD-MCP-ACT-N11-P6-2026-11-11
Policy version: IIAC-2026-11-05.1
Decision scope: IIAC policy activation (MCP P6 product delta vs `8fff85e3bfc5190929ff464b00fbe759dadcb5f8`)
Decision status: PENDING
Audit report digest: c405c65faf53243a0511b0b9584a9f3abda4db08c09ee9b9266c86fe9f6b7a4e

Product independent audit dossier digest (MCP ACT-N11 P6 re-audit): f4b339ab2695a1f162cf4085a8123ffe6c5e6c568990212111b4647548f21a66

________________

Drafting note: the date in the Decision ID is the drafting date, not a signing timestamp. The operator line above is intentionally unsigned; no agent writes or composes it.

Scope: three-path P6 delta vs the P4 pin (`applications/research_application/mcp/server.py`, `intergrax/applications/_shared/mcp_application_server.py`, `intergrax/applications/_shared/mcp_developer_api.py`), of which one path (`mcp_developer_api.py`) is new to the ACT-N11 inventory. Cumulative inventory: 17 paths. Erratum: earlier P6 drafts and dossiers say "19 paths" / "16 + 3"; the recomputed inventory (`11ac8ad85..a695c9c83`) is 17 and equals `audit_scope.json`. Auditor-authored files are not edited.

Independent audit: `PASS_SCOPED` (auditor session `b8d2e7f4-1a3c-4f9e-8d62-5c1a0b9e7f3d`). Certifier preflight: `CERTIFIER_PREFLIGHT_PASS` (session `c4e8a1f2-6b3d-4e9a-8f71-2d5c6b0a9e3f`).
Evidence publication SHA: `2d3eceb6cf7e93dccd4d96919319a6c969b4444b`. Certifier publication SHA: `276c64305799fa4d7f1a25018cc1cccf061fbafd`.

Finding F-MCP-P6-GOV-001 (PR #459 merged before ACT-N11 activation and green `governance-fast`) is preserved. This decision is not a retroactive approval of PR #459.

Session provenance (operator confirms by signing): the P6 implementer (`T-IGX-035-G-MCP-P6-IMPLEMENTER`), the G6 auditor (`77e4a696-c2e5-4a47-94dc-58aff28ccb8d`), the IIIC auditor (`b8d2e7f4-1a3c-4f9e-8d62-5c1a0b9e7f3d`), the preflight certifier (`c4e8a1f2-6b3d-4e9a-8f71-2d5c6b0a9e3f`) and the activation certifier (new dedicated session, id recorded in the activation binding manifest) are separate sessions; none is the implementer.

Authorization applies only to this audited scope. It does not bypass governance checks, does not promote FRZ, does not close the MCP-SERVER-X stage and does not authorize P7.

## Provenance

| Item | Value |
| --- | --- |
| Ticket | `governance/tickets/T-IGX-035-G.md` |
| Prior `audit_execution_sha` | `8fff85e3bfc5190929ff464b00fbe759dadcb5f8` |
| P6 product merge | `a695c9c835bb5b4b206e6e3e00bb2aee23ca7b6b` |
| Product manifest | `iiac_evidence/MCP-SERVER-X/act_n11_mcp_p6/iiac_audit_session_manifest.MCP-SERVER-X.act_n11_mcp.reaudit_p6.json` |
| G6 dossier | `mcp_server_x/MCP_SERVER_X_P6_INDEPENDENT_AUDIT_DOSSIER.md` |
| Supersedes (draft) | `IIIC_OD_ACTIVATION_MCP_ACT_N11_P6_2026-11-11.md` (not gate-compliant) |
