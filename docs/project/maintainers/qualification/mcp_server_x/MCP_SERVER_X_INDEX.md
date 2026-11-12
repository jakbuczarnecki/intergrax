# MCP-SERVER-X — campaign index (SSOT hub)

**Stage ID:** `MCP-SERVER-X`  
**Roadmap status:** `PLANNED / MANDATORY / CURRENT`  
**Baseline pin:** `c228799785db8f35826f47ba0790d6a21d97fbe3`  
**ROADMAP_AMENDMENT_SHA:** `bd5d04d1a3244ccb9c4464007b147dc54fc0cfd6`  
**Tickets:** P0 [`T-IGX-035-A.md`](../../../../governance/tickets/T-IGX-035-A.md) · P1 [`T-IGX-035-B.md`](../../../../governance/tickets/T-IGX-035-B.md)

## Authority map (one SSOT per concern)

| Concern | SSOT artifact |
| --- | --- |
| Program sequence & stage status | [`PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md`](../../plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md) §3 / §5 |
| Architecture & ownership (P1) | [`MCP_SERVER_X_ARCHITECTURE_AND_OWNERSHIP.md`](./MCP_SERVER_X_ARCHITECTURE_AND_OWNERSHIP.md) |
| Campaign WBS & tasks | [`MCP_SERVER_X_WBS.md`](./MCP_SERVER_X_WBS.md) |
| Existing code inventory (closed world @ pin) | [`MCP_SERVER_X_EXISTING_IMPLEMENTATION_INVENTORY.md`](./MCP_SERVER_X_EXISTING_IMPLEMENTATION_INVENTORY.md) |
| Requirements gaps REQ-MCP-* | [`MCP_SERVER_X_GAP_REGISTER.md`](./MCP_SERVER_X_GAP_REGISTER.md) |
| Acceptance matrix MCP-ACC-01..20 | [`MCP_SERVER_X_QUALIFICATION_MATRIX.md`](./MCP_SERVER_X_QUALIFICATION_MATRIX.md) |
| Contract & protocol | [`MCP_SERVER_X_CONTRACT_AND_PROTOCOL_MATRIX.md`](./MCP_SERVER_X_CONTRACT_AND_PROTOCOL_MATRIX.md) |
| SHA ladder | [`MCP_SERVER_X_SHA_LADDER.md`](./MCP_SERVER_X_SHA_LADDER.md) |
| ADR-MCP-01..07 | `docs/project/maintainers/architecture/ADR_MCP_0N_*.md` (**Proposed**) |
| Operator decisions OD-MCP-01..07 | [`operator_decisions/OD-MCP-0N.md`](./operator_decisions/) — **PENDING** |
| P1 handoff | [`MCP_SERVER_X_P1_CONSOLIDATED_DOSSIER.md`](./MCP_SERVER_X_P1_CONSOLIDATED_DOSSIER.md) |
| P1 → P2 gate | [`MCP_SERVER_X_P1_TO_P2_GATE_REPORT.md`](./MCP_SERVER_X_P1_TO_P2_GATE_REPORT.md) |
| Capability pair (hub pointers) | [`docs/project/capabilities/architecture/MCP_SERVER.md`](../../../capabilities/architecture/MCP_SERVER.md) · [`docs/project/capabilities/plan/MCP_SERVER.md`](../../../capabilities/plan/MCP_SERVER.md) |
| ARCH-FREEZE handoff (published @ MCP close) | `MCP_SERVER_X_ARCH_FREEZE_HANDOFF.md` (_P8_) |
| Final closure report | `MCP_SERVER_X_FINAL_CLOSURE_REPORT.md` (_P8_) |

## Terminal state (target — not claimed @ P1)

`ROADMAP AMENDED → ARCHITECTURE APPROVED → ENGINE IMPLEMENTED → L1/L2/L3 QUALIFIED → INDEPENDENT AUDIT PASS → IIAC CERTIFICATION ACCEPTED → MCP-SERVER-X CLOSED → ARCH-FREEZE HANDOFF PUBLISHED`

## Deliverable status

| Wave | Status |
| --- | --- |
| P0 | **Complete** |
| P1 docs / ADR / OD templates | **Complete** (OD unsigned) |
| P2–P4 engine | **Complete (scoped)** @ P4 terminal `7930c256f` |
| P5 security / execution qual | **Closed (scoped)** — `T-IGX-035-F` closed; [`P5 terminal dossier`](./MCP_SERVER_X_P5_TERMINAL_DOSSIER.md) · [`operator closure`](./MCP_SERVER_X_P5_OPERATOR_CLOSURE_RECORD.md) |
| P6 | **Closed (scoped)** — `T-IGX-035-G` closed; guide [`MCP_SERVER_X_P6_DEVELOPER_GUIDE.md`](./MCP_SERVER_X_P6_DEVELOPER_GUIDE.md) · [`terminal dossier`](./MCP_SERVER_X_P6_TERMINAL_DOSSIER.md) · [`operator closure`](./MCP_SERVER_X_P6_OPERATOR_CLOSURE_RECORD.md); F-MCP-P6-GOV-001/002/003 preserved; stage **not CLOSED** |
| P7 | **Ready after terminal verification** — entry `OD-MCP-P6-P7-01`; [`P7 handoff`](./MCP_SERVER_X_P7_HANDOFF.md); until the P6 terminal verification PASS is published: Blocked |
| P8 | **Blocked** (stage not CLOSED) |
