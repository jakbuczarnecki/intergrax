# OD-MCP-RC-03 — the MCP-RC IIAC rebind waits for the integrated baseline of the PDC-P7-00 window

| Field | Value |
| --- | --- |
| **Decision ID** | OD-MCP-RC-03 |
| **Status** | **RECORDED (operator chat answer; not a signed `Authorized:` line)** |
| **Related** | `OD-MCP-RC-01` (D-1: one PDC-P7-00 window, one delta audit, one two-phase rebind; D-1b: branch `mcp-rc`, merge only together with the rebind), `DELEGATION-MCP-RC-2`, `MCP_SERVER_X_RC_ACTIVATION_RUNBOOK.md` (question Q1, branch `mcp-rc-activation`) |

## Question

ACT-N11 compares every product path under `intergrax/`, `agents/` and `applications/` with the pin `audit_execution_sha`. At drafting, `origin/development` was 99 commits and 125 product paths ahead of the campaign base, none of them one of the 23 MCP-RC paths. A pin on the `mcp-rc` line would be green only on a tree without those 125 paths. Should the MCP-RC rebind (signing of the activation line, binding manifest, two phases) wait for the integrated baseline of the PDC-P7-00 window, so that it is done once for the MCP-RC and PDC-X product deltas together?

## Operator record

Operator chat, 2026-11-13, answer to that question (Polish option label: "Tak, czekaj na okno"). English rendering: "Yes, wait for the window."

## Disposition

1. The activation line of the MCP-RC decision is **not** signed now, and no rebind phase is run, before the PDC-P7-00 window publishes its integrated baseline.
2. `mcp-rc` (audit evidence, last product change `b0e19bf61`) and `mcp-rc-activation` (unsigned decision, `DELEGATION-MCP-RC-2`, runbook) stay as they are. Nothing merges to `development` before the rebind (OD-MCP-RC-01).
3. When the window baseline exists, the runbook steps (a) to (f) resume, and the pins in the unsigned decision and the runbook candidates are re-derived against the integrated baseline first (runbook questions Q1, Q2). A changed pin means a new unsigned digest and a re-checked `DELEGATION-MCP-RC-2`.
4. The audit of the non-MCP product paths of `development` belongs to PDC-X; MCP-RC does not take over that ownership.

## Explicit non-actions

No signature, no ticket approval, no rebind, no merge, no product change, no FRZ change (counter stays 225).
