# DELEGATION-MCP-RC-1 — Operator commission for plan-approval signing (MCP-RC tickets T-IGX-038-A..F)

| Field | Value |
| --- | --- |
| **Record ID** | DELEGATION-MCP-RC-1 |
| **Stage** | MCP-SERVER-X corrective campaign (MCP-RC) |
| **Basis** | IIAC Amendment 01, rules R-1..R-6 (as summarized in `governance/tickets/README.md` §Approved); `OD-MCP-RC-01` (D-3, D-8); precedents `DELEGATION-PDC-WAVE-*.md` |
| **Date** | 2026-11-12 |

## Commission (transcription)

1. In the plan decision pack the Operator answered D-3 in chat, English rendering: **"Yes; delegates as before"** (Polish option label: "Tak; delegaci jak dotychczas"): a dedicated non-implementing session may sign the lines that the Operator would otherwise sign, on the `DELEGATION-PDC-WAVE-*` pattern.
2. The Operator answered D-8 in chat, English rendering: **"I approve"** (Polish: "Zatwierdzam"): ticket series `T-IGX-038-A..F` and the registration of MCP-RC as an additional writer of the `intergrax/applications/_shared/` hotspot.
3. The Operator then instructed, in chat, English rendering: **"After the decisions are given you will carry out all the fixes from start to finish"** (Polish: "po ich udzieleniu wykonasz wszystkie poprawki od początku do końca").
4. The binding decisions of `OD-MCP-RC-01` apply: one PDC-P7-00 window, branch `mcp-rc`, merge to `development` only together with the rebind.

## Scope of the delegation

- A dedicated delegate session may write `Approved: operator-delegate @ <ISO8601>` followed by `Delegation: <this file>@<commit sha>` in **T-IGX-038-A, -B, -C, -D, -E and -F** only.
- The delegate writes these lines only after confirming all of the following:
  - each ticket's recorded plan digest matches `scripts/governance/plan_digest.py` on its plan file;
  - each plan stays inside the master plan's batch definition (`agent-env/plans/pasted-content-id-703a-integrax-prancy-fiddle.md`, sections D and E) and `OD-MCP-RC-01`;
  - no plan creates a second identity, execution, deadline, trace, policy or registry authority, and none weakens an ACT-N11, IIAC or governance invariant;
  - the plans state that nothing merges to `development` before independent audit, certification and the PDC-P7-00 rebind.
- **Not delegated:** IIAC activation lines (a separate commission after the independent audit and certification), merges, FRZ promotion, any ruleset change, any other ticket.

## Separation

- The implementer session and its subagents never write these lines.
- The delegate session does not implement any MCP-RC ticket.
