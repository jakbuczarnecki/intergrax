# OD-MCP-07 — FRZ strategy for MCP-SERVER-X closure

| Field | Value |
| --- | --- |
| **Decision ID** | OD-MCP-07 |
| **Status** | **PENDING** |
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

_Decision status: **PENDING**. Agent MUST NOT write `Approved`._
