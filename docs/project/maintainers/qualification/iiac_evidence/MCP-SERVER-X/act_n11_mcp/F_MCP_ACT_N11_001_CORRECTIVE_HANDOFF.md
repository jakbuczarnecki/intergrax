# F-MCP-ACT-N11-001 — implementer corrective handoff (product)

| Field | Value |
|---|---|
| Finding | `F-MCP-ACT-N11-001` |
| Ticket | `governance/tickets/T-IGX-035-E.md` |
| Branch | `mcp-server-x-p4` |
| PR | **New corrective PR** vs `origin/development` @ `7562a069` (PR #454 **merged** — see [`CORRECTIVE_PUBLICATION_RECONCILIATION.md`](./CORRECTIVE_PUBLICATION_RECONCILIATION.md)) |
| Prior audit verdict | `PARTIAL_EXACT_BLOCKERS` @ `7c401fd98` |
| Corrective scope | `intergrax/applications/_shared/mcp_server_engine.py` — `host_custom` FastMCP registration only |

## Product fix (summary)

- Replaced `**kwargs` inner wrapper with `_bind_custom_host_tool_for_fastmcp`, mirroring each handler's explicit public signature for FastMCP 3.3.1 `ParsedFunction`.
- Registration uses `mcp.tool(name=…)(registered)` per binding so multiple custom tools retain distinct handlers.
- Concurrency admission preserved via mirrored wrapper when `max_concurrent_tool_calls` is set.
- Handlers using `*args` / `**kwargs` fail closed at composition with `McpServerDefinitionValidationError`.

## Regression evidence

| Command | Result |
|---------|--------|
| T-IGX-035-E ticket contract (5 modules) | **28 pass, 1 fail** — fail is pre-P4 residual `LIM-RESEARCH-MOUNT-MEMORY-ADMISSION` only |
| `test_research_mcp_pipeline_reaches_orchestration_strategy` | **PASS** |
| New `test_mcp_server_engine.py` host_custom tests | **PASS** |

Raw log: [`implementer_f_mcp_act_n11_001_corrective.log`](./implementer_f_mcp_act_n11_001_corrective.log)

**`corrective_candidate_sha`:** `c699889b0` (publication HEAD; implementer; not auditor PASS).

## Residual (classified separately — not PASS)

**`LIM-RESEARCH-MOUNT-MEMORY-ADMISSION`:** `test_research_application_exposes_mcp_mount` → `MemoryProviderAdmissionError` (`qualification_evidence_missing`, `sqlite.user_profile`). Observed on P4 head before and after this corrective; unchanged by `host_custom` registration fix. Baseline: `auditor_mcp_ticket_test_contract.log` @ `7c401fd98`.

## Finding disposition (implementer — not self-certification)

| Finding | Implementer disposition | Auditor re-verification |
|---------|----------------------|-------------------------|
| F-MCP-ACT-N11-001 | **CORRECTIVE_CANDIDATE_PUBLISHED** | **REQUIRED** — new independent IIIC session; do not reuse prior `PARTIAL_EXACT_BLOCKERS` verdict |

## Operator boundary

- No IIAC activation record edit.
- No operator signature.
- No FRZ promotion.
- No PR merge.
- **Request:** schedule **new** independent IIIC audit @ `corrective_candidate_sha`.
