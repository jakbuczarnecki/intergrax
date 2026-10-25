# TLX-TL11-GATE-B-P07-CANCEL-C — MCP Catalog Invoke (P07) & Parent Cancel Propagation

| Field | Value |
| --- | --- |
| **Task** | `TLX-TL11-GATE-B-P07-CANCEL-C` |
| **Parent** | `TOOL-LIFE-X` §3.0.5 — **not CLOSED** |
| **Operator authorization** | `OPERATOR APPROVED — TLX-TL11 GATE B — SCOPED CATALOG INVOCATION & CANCEL-C QUALIFICATION` |
| **START_HEAD** | `480d89cf7d644daf3718330e59167378466094ca` |
| **IMPLEMENTATION_HEAD** | *(see publication commit after push)* |
| **Disposition** | **`TLX-TL11-GATE-B-P07-CANCEL-C — READY FOR INDEPENDENT AUDIT`** (scoped) |

**Non-closure:** No global `FRZ-*` PASS; no `TOOL-LIFE-X` parent CLOSED; no Gate A; no TL8 positive recovery; no production deployment; no new cancellation/deadline authority; subprocess L3 for **catalog invoke + physical HTTP cancel** remains **explicitly unproven** (fixture unchanged — see §7).

---

## 1. Git identity gate

| Check | Result |
| --- | --- |
| Repository | `jakbuczarnecki/intergrax` @ `origin` |
| Branch | `development` |
| START_HEAD | `480d89cf7d644daf3718330e59167378466094ca` |
| Ancestry | START_HEAD ancestor of publication HEAD — **yes** (pre-commit) |

---

## 2. Call graph

### Before (discovery-only Gate B)

```text
MCP client → list_catalog_tools / describe_catalog_tool → ToolRegistry exporter (no RuntimeToolInvoker)
```

### After (authorized P07 + CANCEL-C transport coupling)

```text
Authenticated MCP client (harness principal when wired)
  → invoke_catalog_tool (FastMCP)
  → resolve_mcp_run_agent_identity (TRUST-A reuse)
  → resolve_mcp_catalog_execution_binding (ActiveTaskRegistry — authoritative run/tenant)
  → execution-bound step/agent from task.metadata / binding (fail-closed)
  → asyncio host work + optional run_host_execution_with_mcp_transport_cancel
       → governed_cancel_active_task on transport disconnect (CANCEL-A reuse)
  → ExecutionBoundCatalogToolInvoker.invoke (host-injected; read-only contract)
       → RuntimeToolInvoker (canonical — unchanged)
  → bounded MCP structured result
```

**Preserved:** `list_catalog_tools`, `describe_catalog_tool` unchanged when `catalog_tool_invoker` omitted.

---

## 3. Closed-world caller inventory

| Surface | When present |
| --- | --- |
| `list_agents` | always (`build_nexus_mcp_server`) |
| `run_agent` | always |
| `list_catalog_tools` | `tool_registry` set |
| `describe_catalog_tool` | `tool_registry` set |
| `invoke_catalog_tool` | `tool_registry` **and** `catalog_tool_invoker` set |

No per-tool-id MCP invoke aliases; no direct `ToolRegistry` handler execution from MCP.

---

## 4. Changed files and symbols

| File | Symbols |
| --- | --- |
| `intergrax/applications/_shared/mcp_catalog_invocation_bridge.py` | **new** — `McpCatalogInvocationComposition`, `execute_mcp_catalog_tool_invocation`, `resolve_mcp_catalog_execution_binding` |
| `intergrax/applications/_shared/mcp_catalog_tools.py` | `mount_catalog_tools_on_mcp(..., catalog_invocation=)`; `invoke_catalog_tool` MCP tool |
| `intergrax/applications/_shared/mcp_nexus_server.py` | `build_nexus_mcp_server(..., catalog_tool_invoker=, tool_enablement=)` |
| `tests/unit/applications/test_mcp_catalog_gate_b_cancel_c_qualification.py` | **new** — Gate B + CANCEL-C adversarial L2 |
| `tests/unit/applications/test_mcp_catalog_discovery_qualification.py` | invoke absent without invoker; rename discovery guard test |
| `docs/project/maintainers/qualification/TOOL_LIFE_X_TLX_TL11_GATE_B_P07_CANCEL_C_QUALIFICATION_EVIDENCE.md` | this record |

**Read-only (unchanged):** `ExecutionBoundCatalogToolInvoker`, `RuntimeToolInvoker`, `CancellationCoordinator`, `HostTaskExecutionPort`.

---

## 5. Ownership and contracts

| Concern | Semantic owner | Composition owner | Contract |
| --- | --- | --- | --- |
| Catalog invoke admission | `execute_mcp_catalog_tool_invocation` | Tier-3 `build_nexus_mcp_server` | `ExecutionBoundCatalogToolInvokeRequest` |
| Tool runtime | `RuntimeToolInvoker` | host-injected `NexusExecutionBoundCatalogToolInvoker` | `ToolExecutionResult` |
| Active execution identity | `ActiveTaskRegistry` | host execution register/unregister | `ActiveTaskBinding` |
| Caller identity | `resolve_mcp_run_agent_identity` | harness HTTP / lab defaults | `RequestIdentity` |
| Transport cancel | `run_host_execution_with_mcp_transport_cancel` | MCP HTTP | `McpHostExecutionTransportCancelContext` |
| Governed cancel | `governed_cancel_active_task` | task control boundary | `TaskControlResult` |

---

## 6. Invariant matrix (selected)

| Invariant | Owner | Test | Layer | FRZ (scoped) | Result |
| --- | --- | --- | --- | --- | --- |
| Authorized invoke succeeds | bridge + invoker | `test_gate_b_successful_authorized_invoke` | L2 | FRZ-BND-03 | PASS |
| Missing caller identity | identity resolver | `test_gate_b_missing_caller_identity` | L2 | FRZ-SEC | PASS |
| Wrong tenant | binding + principal | `test_gate_b_wrong_tenant` | L2 | FRZ-TEN | PASS |
| Unknown/disabled tool | registry + enablement | `test_gate_b_unknown_tool`, `test_gate_b_disabled_tool` | L2 | — | PASS |
| Missing/stale binding | ActiveTaskRegistry | `test_gate_b_missing_execution_binding`, `test_gate_b_stale_run_binding` | L2 | FRZ-BND-03 | PASS |
| Side-effect gate | task.metadata | `test_gate_b_unauthorized_side_effect_tool` | L2 | FRZ-GOV | PASS |
| Parent cancel before invoke | CancellationCoordinator | `test_gate_b_parent_cancel_before_invoke` | L2 | FRZ-REL-04 | PASS |
| In-flight invoke + parent cancel signal | coordinator + invoker | `test_cancel_c_parent_cancel_during_in_flight_invoke` | L2 | FRZ-REL-04 | PASS (cooperative; not L3 HTTP) |
| No shadow invoker | composition | `test_gate_b_no_second_runtime_tool_invoker_in_bridge` | L1 | FRZ-BND-03 | PASS |
| Discovery unchanged | mount | discovery qual batch | L2 | FRZ-BND-03 | PASS |
| Hung MCP discovery subprocess | TL11 fixture | e2e hung batch | L3 | FRZ-REL-06 | PASS (discovery only) |

---

## 7. Physical / subprocess proof gaps

| Scenario | Status | Reason |
| --- | --- | --- |
| Catalog **invoke** over streamable HTTP subprocess | **NOT PROVEN (L3)** | `tool_life_x_tl11_nexus_catalog_qualification_server.py` unchanged (operator fixture lock); no `catalog_tool_invoker` in subprocess |
| Parent cancel → in-flight **invoke** on physical HTTP | **NOT PROVEN (L3)** | Same fixture gap; CANCEL-C proven at **L2** in-process with transport cancel module reuse |
| Root deadline exceed during catalog invoke (full host stack) | **L2 reuse** | Deadline authority remains on host Execution (MCP16 MC-04/05); no new MCP deadline parameter |

---

## 8. Validation

```text
uv run pytest -p no:xdist -q \
  tests/unit/applications/test_mcp_catalog_gate_b_cancel_c_qualification.py \
  tests/unit/applications/test_mcp_catalog_discovery_qualification.py \
  tests/unit/applications/test_mcp_catalog_e2e_hung_server_qualification.py
```

| Result @ implementation | **45 passed**, 0 failed (~134s) |
| Log | `.tmp/session/tlx-tl11-gate-b-p07-cancel-c/consolidated-pytest.log` (operator may capture on audit) |

---

## 9. ECAP scoped check

| ECAP slice | Outcome |
| --- | --- |
| P1 Ownership | Single ToolRuntime path via injected invoker — **preserved** |
| P2 Composition | Tier-3 wires invoker; no Tier-3 `RuntimeToolInvoker` construction — **preserved** |
| P3 Typing | `ExecutionBoundCatalogToolInvokeRequest` + Pydantic tool input — **preserved** |
| P4 Authority | No synthetic tenant/run without `ActiveTaskRegistry` — **preserved** |
| P5 Cancel | Transport → governed cancel reuse; no MCP-only business cancel — **preserved** |
| P6 Evidence | This record + GitHub SHA — **VALIDATION REQUIRED** (independent) |

---

## 10. FRZ (no global PASS)

| ID | Applicability | Status |
| --- | --- | --- |
| FRZ-BND-03 | No shadow MCP dispatcher | Scoped **preserved** — VALIDATION REQUIRED |
| FRZ-SEC-* | MCP principal on invoke | Scoped — VALIDATION REQUIRED |
| FRZ-TEN-* | Tenant binding | Scoped — VALIDATION REQUIRED |
| FRZ-REL-04/05 | Cancel / external op | L2 invoke path — **partial**; L3 invoke subprocess **OPEN** |
| FRZ-GOV-* | Governed cancel | Reuse CANCEL-A — VALIDATION REQUIRED |
| FRZ-REG-* | Global freeze | **OPEN** |

---

## 11. Remaining TL11 closure conditions

1. Independent exact-SHA audit of this publication.
2. L3 subprocess extension or new qual package for **catalog invoke** + HTTP cancel (fixture change requires operator authorization).
3. Gate A / TL8 — **explicitly out of scope**.
4. `TOOL-LIFE-X` parent — **not CLOSED**.

---

## 12. Gate A / TL8 exclusions

No Gate A activation; no TL8 positive recovery; no production credentials; no new cloud resources.
