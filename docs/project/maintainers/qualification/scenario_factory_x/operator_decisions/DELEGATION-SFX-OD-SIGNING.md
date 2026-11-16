# DELEGATION-SFX-OD-SIGNING — Operator commission for signing OD-SFX-01..04, 15 and 16

| Field | Value |
| --- | --- |
| **Record ID** | DELEGATION-SFX-OD-SIGNING |
| **Stage** | SCENARIO-FACTORY-X (SFX) |
| **Basis** | IIAC Amendment 01, rules R-1..R-6. Precedent: `IIAC_OD_DELEGATION_MCP_P7_P8_AUTH_2026-11-12.md` (OD-MCP-P7-P8-AUTH-01). |
| **Date** | 2026-11-12 |
| **Provenance** | Transcribed by the SFX orchestrator session. Not operator-signed. Assurance: `process_enforced`. |

## Commission (transcription)

The orchestrator asked the operator in chat: "Who signs the records OD-SFX-01..04, 15 and 16?" The two options offered were "Sesja podpisująca" (a signing session) and "Podpiszę sam" (I will sign myself). The operator answered: **"Sesja podpisująca"** (English rendering: "A signing session").

## Scope of the delegation

- A new, dedicated signing session may write the operator line in these six files under `docs/project/maintainers/qualification/scenario_factory_x/operator_decisions/`: **OD-SFX-01, OD-SFX-02, OD-SFX-03, OD-SFX-04, OD-SFX-15, OD-SFX-16**.
  - Use the plain form of the OD-MCP-P7-P8-AUTH-01 precedent: `Approved: <OD id> @ <ISO8601>`.
  - Precede the line with a provenance paragraph that cites this record as `<path>@<commit sha>`.
  - Set the record's status to **APPROVED**.
- Before signing a record, the session confirms all of the following:
  1. The decision text is a faithful English rendering of what the operator said in the SFX planning chat. Use the chat transcript where the session tools allow it. If it cannot be checked, the session says so in the provenance paragraph or refuses.
  2. The decision does not widen the scope beyond what the operator said.
  3. No record authorizes SFX-P5 or later, accepts an ADR, changes FRZ, or bypasses SCENARIO-GATE or PROD-Q.
- **Not delegated:**
  - any other OD;
  - ADR acceptance;
  - ticket approvals, which are covered by DELEGATION-SFX-PRE-FREEZE;
  - IIAC activation;
  - merges;
  - GATE-SFX-ENTRY.

## Separation

- The orchestrator, its implementer subagents, its reviewer sessions and the ticket delegate (`sfx-delegate-signer`) never write these lines.
- The signing session implements and reviews nothing.
