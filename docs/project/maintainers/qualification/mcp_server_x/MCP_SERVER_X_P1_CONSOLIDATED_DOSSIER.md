# MCP-SERVER-X — P1 consolidated dossier (operator / ChatGPT handoff)

**Purpose:** Single-page summaries of seven operator decisions + architecture pointers.  
**Status:** Handoff @ P1 PR — **not** closure.

## Campaign context

- Stage `MCP-SERVER-X` on roadmap: `PLANNED / MANDATORY / CURRENT`
- `ROADMAP_AMENDMENT_SHA`: `bd5d04d1a3244ccb9c4464007b147dc54fc0cfd6`
- Architecture SSOT: [`MCP_SERVER_X_ARCHITECTURE_AND_OWNERSHIP.md`](./MCP_SERVER_X_ARCHITECTURE_AND_OWNERSHIP.md)

---

### OD-MCP-01 — Public API placement

**Ask:** Approve contract in `intergrax/contracts/` and composition engine in `_shared` (no per-app engines, no `runtime/mcp` execution package).  
**If yes:** P2 can add `mcp_server_definition.py` + engine module.  
**Status:** PENDING → [`operator_decisions/OD-MCP-01.md`](./operator_decisions/OD-MCP-01.md)

---

### OD-MCP-02 — Capability scope

**Ask:** Approve **Tools only** v1; Resources and Prompts UNSUPPORTED_BY_DESIGN.  
**If yes:** Protocol matrix and L3 tests exclude resources/prompts.  
**Status:** PENDING → [`operator_decisions/OD-MCP-02.md`](./operator_decisions/OD-MCP-02.md)

---

### OD-MCP-03 — Transport

**Ask:** Approve **streamable HTTP only** (FastMCP 3.x pin); stdio/SSE out of scope.  
**If yes:** Qualification uses HTTP client only.  
**Status:** PENDING → [`operator_decisions/OD-MCP-03.md`](./operator_decisions/OD-MCP-03.md)

---

### OD-MCP-04 — Definition & exposure

**Ask:** Approve typed declarative definition + thin builder; exposure changes via CONFIG recompose, not registry mutation; no shadow registry.  
**If yes:** P2 builds allowlist exposure policy.  
**Status:** PENDING → [`operator_decisions/OD-MCP-04.md`](./operator_decisions/OD-MCP-04.md)

---

### OD-MCP-05 — Versioning & facades

**Ask:** Approve `mcp-server-definition.v1` + keep legacy `build_nexus_mcp_server` facades until migration guide retires them.  
**If yes:** P4 COMPAT fingerprint + eight-app migration.  
**Status:** PENDING → [`operator_decisions/OD-MCP-05.md`](./operator_decisions/OD-MCP-05.md)

---

### OD-MCP-06 — Identity

**Ask:** Approve harness principal SSOT; client tenant/user as assertions; fail-closed when auth required.  
**If yes:** P5 cross-tenant negatives gate release.  
**Status:** PENDING → [`operator_decisions/OD-MCP-06.md`](./operator_decisions/OD-MCP-06.md)

---

### OD-MCP-07 — FRZ strategy

**Ask:** Approve default path **A** — MCP-ACC matrix + incremental FRZ notes, **no** new `FRZ-MCP-*` family (225 unchanged).  
**Alternate B:** new FRZ family — separate ticket.  
**Status:** PENDING → [`operator_decisions/OD-MCP-07.md`](./operator_decisions/OD-MCP-07.md)

---

## ADR index

| ADR | File |
| --- | --- |
| 01 | [`ADR_MCP_01_PLATFORM_OWNERSHIP_AND_PUBLIC_API.md`](../../architecture/ADR_MCP_01_PLATFORM_OWNERSHIP_AND_PUBLIC_API.md) |
| 02 | [`ADR_MCP_02_CAPABILITY_SCOPE_TOOLS_ONLY.md`](../../architecture/ADR_MCP_02_CAPABILITY_SCOPE_TOOLS_ONLY.md) |
| 03 | [`ADR_MCP_03_TRANSPORT_STREAMABLE_HTTP.md`](../../architecture/ADR_MCP_03_TRANSPORT_STREAMABLE_HTTP.md) |
| 04 | [`ADR_MCP_04_DECLARATIVE_DEFINITION_AND_EXPOSURE.md`](../../architecture/ADR_MCP_04_DECLARATIVE_DEFINITION_AND_EXPOSURE.md) |
| 05 | [`ADR_MCP_05_COMPATIBILITY_FACADES_AND_V1.md`](../../architecture/ADR_MCP_05_COMPATIBILITY_FACADES_AND_V1.md) |
| 06 | [`ADR_MCP_06_TRUST_BOUNDARY_HARNESS_PRINCIPAL.md`](../../architecture/ADR_MCP_06_TRUST_BOUNDARY_HARNESS_PRINCIPAL.md) |
| 07 | [`ADR_MCP_07_QUALIFICATION_MCP_ACC_INCREMENTAL_FRZ.md`](../../architecture/ADR_MCP_07_QUALIFICATION_MCP_ACC_INCREMENTAL_FRZ.md) |

## Next gate line

See [`MCP_SERVER_X_P1_TO_P2_GATE_REPORT.md`](./MCP_SERVER_X_P1_TO_P2_GATE_REPORT.md) (terminal line).
