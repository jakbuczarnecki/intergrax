# IIAC operator decision — MCP ACT-N11 product re-bind (P2–P4)

Decision ID: IIAC-OD-MCP-ACT-N11-2026-11-11  
Policy version: IIAC-2026-11-05.1  
Decision scope: IIAC policy activation (MCP product Δ vs `11ac8ad85f…`)  
Decision status: AUTHORIZED  
Audit report digest: c405c65faf53243a0511b0b9584a9f3abda4db08c09ee9b9266c86fe9f6b7a4e

Product independent audit dossier digest (MCP ACT-N11 re-audit): d2fea1ee3d65a7a7b96ba0cc59c3b1fb2387ef74a6d58256d41b86e514a33cb7

Approved IIAC policy activation IIAC-2026-11-05.1: operator @ 2026-11-11T13:43:00+01:00

Scope: MCP-SERVER-X P2–P4 cumulative 16-path product rebind under ACT-N11.  
Independent audit: PASS_SCOPED.  
Certifier preflight: PASS.  
Evidence publication SHA: 8fff85e3bfc5190929ff464b00fbe759dadcb5f8.

Authorization applies only to the audited MCP ACT-N11 scope. It does not authorize bypassing governance checks, promoting FRZ, or declaring MCP-SERVER-X CLOSED.

## Provenance

| Item | Value |
| --- | --- |
| Operator routing (required) | `Approved: D-MCP-IIIC-01=a` |
| Ticket | `governance/tickets/T-IGX-035-E.md` |
| Prior `audit_execution_sha` | `11ac8ad85f62a6925d25ea840c92d5c6a936cfe4` |
| New `audit_execution_sha` / evidence publication | `8fff85e3bfc5190929ff464b00fbe759dadcb5f8` |
| Activation binding manifest | `iiac_evidence/MCP-SERVER-X/act_n11_mcp/iiac_audit_session_manifest.IIAC-ACT-N11-MCP-ACTIVATION.json` |
| Product manifest | `iiac_audit_session_manifest.MCP-SERVER-X.act_n11_mcp.reaudit_p4.json` |

## Scope and limitations

- Re-bind `audit_execution_sha` after MCP P2–P4 product paths landed post ENT Session B.
- Does **not** close MCP-SERVER-X stage, P5 qualification, or FRZ.
- Does **not** bypass governance checks or authorize merge without green `verify_iiac_policy_activation --require-in-force`.
