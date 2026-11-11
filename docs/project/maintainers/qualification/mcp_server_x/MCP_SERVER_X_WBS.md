# MCP-SERVER-X — work breakdown (summary)

Full task detail: campaign master plan (digest `5dd3c72e497c72764ccc0c08d3df2dddfdbe19b3f36550ba5d87c4091f9833e7`). This file is the repo-local WBS index.

| Phase | ID range | Gate | Status |
| --- | --- | --- | --- |
| P0 | MCP-P0-01..10 | Roadmap amended | **Complete** (`ROADMAP_AMENDMENT_SHA` `bd5d04d1…`) |
| P1 | MCP-P1-01..04 | ADR Proposed + OD templates | **Complete** (OD unsigned — P2 blocked) |
| P2–P4 | MCP-P2..P4 | OD-MCP-01,02,04,05 | **Blocked** |
| P5–P6 | MCP-P5..P6 | Engine frozen | **Blocked** |
| P7 | MCP-P7-01..09 | Qual package | **Blocked** |
| P8 | MCP-P8-01..10 | Independent audit + IIIC | **Blocked** — no self-cert |

## P1 tasks (T-IGX-035-B)

| Task | Outcome |
| --- | --- |
| MCP-P1-01 | ADR-MCP-01..07 **Proposed** in `docs/project/maintainers/architecture/` |
| MCP-P1-02 | Server definition field spec in ADR-MCP-01 (no product code) |
| MCP-P1-03 | OD-MCP-01..07 **PENDING** in `operator_decisions/` |
| MCP-P1-04 | [`MCP_SERVER_X_P1_INDEPENDENT_REVIEW_REPORT.md`](./MCP_SERVER_X_P1_INDEPENDENT_REVIEW_REPORT.md) |

## P2–P4 refinements (post-OD)

| Phase | Focus | Key tasks |
| --- | --- | --- |
| P2 | Definition & publish | P2-01 contract v1; P2-03 exposure; P2-04 engine composition; P2-06 shadow-registry gate |
| P3 | Transport & host | P3-01 adapter consolidation; P3-02 HTTP L3 pin; P3-04 CONFIG mount |
| P4 | Lifecycle & COMPAT | P4-03 fingerprint; P4-04 eight-app facades; P4-05 inventory |

## P5–P6 refinements

| Phase | Focus | Key tasks |
| --- | --- | --- |
| P5 | Security & execution | P5-02 cross-tenant; P5-04 ToolRuntime path; P5-06 cancel/disconnect; P5-07 adversarial pack |
| P6 | DX & migration | P6-01 public API; P6-02 reference qual server; P6-03 eight apps; P6-04 migration guide |

## P7–P8 refinements

| Phase | Focus | Key tasks |
| --- | --- | --- |
| P7 | Qualification | MCP-ACC 01–20; L2/L3 split; EVOL/ENT delta; exact-SHA package |
| P8 | Audit & close | Independent audit; IIIC; roadmap CLOSED row; ARCH-FREEZE handoff |

## Parallelism notes

- SSOT docs: **serial** (orchestrator).
- Engine P2+: single owner `_shared` + `intergrax/contracts` after OD gate.
- Audit / IIIC: **separate sessions** only.
