# TLX-TL11-TRUST-A-CANCEL — MCP Principal Binding & Execution Lifecycle

| Field | Value |
| --- | --- |
| **Task** | `TLX-TL11-TRUST-A-CANCEL` |
| **Parent** | `TOOL-LIFE-X` §3.0.5 — **IN PROGRESS / BLOCKED** |
| **Operator authorization** | `OPERATOR APPROVED — MCP TRUST-A` |
| **START_HEAD** | `33345c383d36ece80b19305aa2c19efbed3baa65` |
| **Disposition** | **`TLX-TL11-TRUST-A-CANCEL — READY FOR INDEPENDENT AUDIT`** (scoped; CANCEL-B documented) |

**Non-closure:** No global `FRZ-*` PASS; no `TOOL-LIFE-X` parent CLOSED; no `DIST-SCALE-X`; parent cancel→Tool on MCP path remains **BLOCKED** (L3-R2 unchanged).

---

## TRUST-A implementation contract

| Element | Owner |
| --- | --- |
| HTTP ingress auth | `HarnessApiKeyMiddleware` / `resolve_harness_authenticated_principal_for_state` |
| MCP auth config wire | `couple_fastapi_with_mcp` → `register_nexus_mcp_harness_auth` |
| FastMCP request scope | `fastmcp.server.context.request_ctx` → Starlette `Request` |
| Identity resolution | `resolve_mcp_run_agent_identity` |
| Conflict rejection | `reject_identity_assertion_conflicts` |
| Execution admission | `HostTaskExecutionPort.execute` only |
| Strict caller principal | `bind_strict_serving_http_caller_principal` when verified |

**Lab:** in-process FastMCP without harness auth wiring retains client-supplied tenant/user (not production PASS).

---

## Cancellation gate

**CANCEL-B (scoped):** MCP client disconnect/timeout does not create a competing `CancellationCoordinator`. Host `ExecutionDeadlineAuthority` remains bounded on the canonical path; L3-R2 subprocess matrix **reused** @ L2. No claim of physical provider cancellation via MCP alone.

---

## TA-01–TA-20 (summary)

| ID | Result |
| --- | --- |
| TA-01–TA-09 | **PASS** @ this batch (TRUST-A wiring + negatives) |
| TA-10 | **ALREADY PROVEN** — `HostTaskExecutionPort` only |
| TA-11–TA-14 | **CANCEL-B** — host deadline owner unchanged; no new MCP cancel authority |
| TA-15–TA-17 | **ALREADY PROVEN** — no catalog invoke / shadow executor |
| TA-18 | **VALIDATION REQUIRED** — scoped FRZ; no global PASS |
| TA-19 | **PASS** — discovery evidence @ `adef74d2…` remains valid |
| TA-20 | **READY FOR INDEPENDENT AUDIT** |

---

## Tests

| Layer | Command | Result |
| --- | --- | --- |
| L1 | `uv run pytest tests/unit/applications/test_mcp_trust_a_principal_binding.py -q` | **7 passed** |
| L2 | consolidated MCP + harness + L3-R2 + strict principal (see session log) | **48 passed** |
| L3 | L3-R2 subprocess scenarios in L2 batch + HTTP ingress 401 | **reused / scoped** |

Session logs: `.tmp/session/TLX-TL11-TRUST-A-CANCEL/`

---

## Remaining blockers

- Parent **cancel → Tool** on MCP host path (TL11 parent matrix).
- Full streamable-HTTP ASGI qualification of `run_agent` identity on live uvicorn (optional hardening; resolver proven via `request_ctx` + ingress 401).
- Independent GitHub SHA audit (TA-20).
