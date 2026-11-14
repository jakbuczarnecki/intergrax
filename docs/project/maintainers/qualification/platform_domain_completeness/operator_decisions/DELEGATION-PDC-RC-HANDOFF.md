# DELEGATION-PDC-RC-HANDOFF — Operator commission for ticket T-IGX-040-A (MCP-RC handoff repairs F-49, F-50, F-41)

| Field | Value |
| --- | --- |
| **Record ID** | DELEGATION-PDC-RC-HANDOFF |
| **Stage** | PDC-X P6 remainder (batch B-P6-J) |
| **Basis** | IIAC Amendment 01, rules R-1..R-6 (as summarized in `governance/tickets/README.md` §Approved); `DELEGATION-PDC-STANDING.md`; `DELEGATION-PDC-WAVE2.md` |
| **Date** | 2026-11-14 |

## Commission (transcription)

1. The MCP-SERVER-X Release-Candidate session (MCP-RC) asked PDC-X, in the handoff `docs/project/maintainers/qualification/mcp_server_x/MCP_SERVER_X_RC_PDC_X_HANDOFF_F49_F41.md` (branch `mcp-rc`, commit `c2fa6ebb9`), to carry two product repairs: the single `openai` catalog row must dispatch on `integration_category` (F-49, with F-50 test pins), and the impact map needs a rule for the Tier-3 root `applications/` (F-41). The Operator pasted the handoff in chat and told the orchestrator to run it as a separate sub-agent task (original wording in Polish; English rendering: "run this task as a separate sub-agent").
2. The sub-agent produced ticket **T-IGX-040-A**, its plan and WBS batch **B-P6-J** (PR 544). The ticket id differs from the first proposal (`T-IGX-036-AA`) because the 036 series is exhausted (A..Z) and `governance/dcp-commit-ref-config.yaml` accepts only a single-letter scope.
3. The orchestrator asked the Operator who signs the plan. The Operator answered in English rendering: **"Delegate"** (original wording in Polish: "Delegat").

## Scope of the delegation

- A dedicated delegate session may write `Approved: operator-delegate @ <ISO8601>` and `Delegation: <this file>@<commit sha>` in **T-IGX-040-A** only, after confirming: the plan digest matches; the plan matches batch B-P6-J in `PDC_IMPLEMENTATION_WBS.md` (write set equal); no write path lies under `intergrax/contracts/`, `intergrax/applications/_shared/`, `intergrax/integrations/registry/`, `scripts/governance/`, `.github/`, and none of the 23 MCP-RC paths named in the handoff is in the write set; the existing `xd_12` and `p6_impact` tests stay byte-identical (no gate narrowed; a second typed table instead of editing the prefix test); the plan keeps one catalog row per slug and adds no new registry, scheduler, engine, trust, policy or audit authority; the plan states it is not a certification verdict.
- **Merge gate.** The signer states in the ticket which merge gate governs. The orchestrator's position, which the delegate confirms or rejects with evidence: product pull requests of PDC-X already merge under authorization 3 of `DELEGATION-PDC-STANDING.md` (for example PR 537), so the gate of OD-PDC-SEQ-01 is treated as satisfied; the MCP-RC request to merge before its rebind (`OD-MCP-RC-05`, recorded on `mcp-rc` only) is a scheduling request and not an additional gate of PDC-X. If the delegate finds that OD-PDC-SEQ-01 is **not** satisfied, it writes nothing and reports.
- The implementer merges the batch pull request under authorization 3 of `DELEGATION-PDC-STANDING.md` (green CI, only the known IIAC activation red class, head equals the verified SHA, normal merge for conflicts) and under decision 8 of `DELEGATION-PDC-WAVE2.md`.
- Write-set amendments of T-IGX-040-A are **not** delegated: any need to touch another product file, the registry, a hotspot or one of the 23 MCP-RC paths returns to the Operator.

## Not delegated (unchanged)

OD signatures; FRZ promotion and the PDC-X roadmap closure; IIAC activation lines other than as stated in `DELEGATION-PDC-STANDING.md`; lifting C10/C15; the MCP-RC rebind, addendum and activation record (owned by MCP-RC); any change under `scripts/governance/` or `.github/`.

## Separation

The implementer session and its implementer subagents never write `Approved:` lines. A delegate session does not implement the ticket it signs.
