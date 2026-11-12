# OD-PDC-SEQ-01 — Sequencing PDC product merges against MCP-SERVER-X

| Field | Value |
| --- | --- |
| **Decision ID** | OD-PDC-SEQ-01 |
| **Stage** | PDC-X |
| **Status** | **APPROVED — Option A** |
| **Related** | ACT-N11 (`scripts/governance/iiac_activation_gate.py`): any change under `intergrax/`, `agents/` or `applications/` after `audit_execution_sha` invalidates the active IIAC activation |

## Question

When may PDC-P5 product changes merge to `development`, given that MCP-SERVER-X P6–P8 certification depends on the IIAC activation being current?

## Recommended option (A)

- PDC-P0..P4 are documentation and planning only, outside material governance paths. They proceed in parallel with MCP-SERVER-X.
- PDC-P5 implementation may start in isolated worktrees. **Product merges wait** until the MCP-SERVER-X P8 certification is published.

## Alternate option (B)

- PDC product merges proceed earlier. Each one obliges MCP-SERVER-X to re-run its ACT-N11 rebind on a new audit SHA, which costs time and can repeat the audit.

## Operator record

Approved: operator @ 2026-11-12T07:16:15+01:00

**Selected variant:** Option A (recommended).

Provenance: the Operator answered "Approve the three remaining ODs (FRZ-01, ARCHF-01, SEQ-01) as option A?" in chat with the words "Approve all A". Transcribed on the Operator's instruction under T-IGX-036-F.
