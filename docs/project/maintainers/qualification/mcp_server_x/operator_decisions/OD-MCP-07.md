# OD-MCP-07 — FRZ strategy for MCP-SERVER-X closure

| Field | Value |
| --- | --- |
| **Decision ID** | OD-MCP-07 |
| **Status** | **APPROVED — Option A** |
| **ADR** | [`ADR_MCP_07_QUALIFICATION_MCP_ACC_INCREMENTAL_FRZ.md`](../../../architecture/ADR_MCP_07_QUALIFICATION_MCP_ACC_INCREMENTAL_FRZ.md) |

## Question

How does MCP-SERVER-X relate to the FRZ checklist (225 criteria)?

## Recommended option (A) — default campaign

- Close stage via **MCP-ACC-01..20** on qualification SHA.
- Attach incremental evidence to **existing** FRZ families only.
- **Do not** add `FRZ-MCP-*` rows or change the 225 counter in this campaign.

## Alternate option (B) — not default

- New FRZ-MCP family + validator/ECAP migration — requires separate operator ticket.

## Operator record

Approved: operator @ 2026-11-11T11:24:00+01:00

**Selected variant:** Option A — MCP-ACC closure without new `FRZ-MCP-*` family; incremental evidence on existing FRZ rows; 225 counter unchanged. ADR-MCP-07 remains **Proposed**.
