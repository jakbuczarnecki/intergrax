# TLX-TL11-CANCEL-A — MCP → Canonical Execution Cancellation

| Field | Value |
| --- | --- |
| **Task** | `TLX-TL11-CANCEL-A` |
| **Operator authorization** | `OPERATOR APPROVED — MCP CANCEL-A` |
| **START_HEAD** | `bb43430ec6bb28c8ee67d3488508e5aba9b1f753` |
| **Gate** | **A1** (governed propagation where boundary + verified principal) |
| **Disposition** | **`TLX-TL11-CANCEL-A — READY FOR INDEPENDENT AUDIT`** (scoped) |

**Non-closure:** No global `FRZ-*` PASS; no `TOOL-LIFE-X` parent CLOSED; no physical Tool provider kill claim; TL8/TL7 blockers preserved.

---

## Canonical cancellation owner graph

```text
MCP transport (disconnect / asyncio cancel)
        │
        ▼
execute_mcp_agent_task + mcp_host_execution_transport_cancel
        │  (no CancellationCoordinator.request from MCP directly)
        ▼
governed_cancel_active_task  [task_control.py]
        │  ControlPlaneMutationAuthorizationBoundary
        ▼
CancellationCoordinator.request(binding.task)  [runtime/cancellation/coordinator.py]
        ▼
Execution / ToolRuntime cooperative cancel + deadline authority (unchanged)
```

**Admission path:** Same as HTTP `POST /v1/tasks/{task_id}/cancel` — `governed_cancel_active_task` with verified `RequestIdentity`, exact active `task_id`/`run_id` from `ActiveTaskRegistry`, and harness control-plane boundary.

**Correlation:** `task_id` from MCP intake `Task`; `run_id` from active binding only (never client-supplied).

---

## CA-01–CA-24 closure (summary)

| ID | Classification | Notes |
| --- | --- | --- |
| CA-01 | ALREADY PROVEN | TRUST-A principal @ `bb43430…` |
| CA-02 | ALREADY PROVEN | Single owner: `governed_cancel` → `CancellationCoordinator` |
| CA-03 | **PASS (this batch)** | MCP transport has no independent cancel authority |
| CA-04 | **PASS** | Binding lookup by platform `task_id` |
| CA-05 | **PASS** | Governed mutation + tenant authority |
| CA-06 | **PASS** | Cross-tenant negative `test_cancel_a_cross_tenant_*` |
| CA-07 | VALIDATION REQUIRED | Pre-admission: no binding → no cooperative signal (bounded) |
| CA-08 | **PASS** | Active execution cancel via governed path |
| CA-09–CA-11 | ALREADY PROVEN | Execution terminal CAS (reuse Batch C / host) |
| CA-12–CA-13 | ALREADY PROVEN | UNKNOWN / no-blind-replay unchanged |
| CA-14–CA-15 | ALREADY PROVEN | `ExecutionDeadlineAuthority`; MCP ≠ client timeout |
| CA-16 | N/A — WITH EVIDENCE | Physical provider cancel not claimed |
| CA-17–CA-18 | VALIDATION REQUIRED | L3-R2 subprocess reuse + host shield drain |
| CA-19 | **PASS** | Tenant authority on governed cancel |
| CA-20 | **PASS** | No new coordinator/registry |
| CA-21 | **PASS** | TRUST-A preserved |
| CA-22 | **PASS** | L1/L2 batch below |
| CA-23 | **PASS** | Scoped FRZ/ECAP |
| CA-24 | VALIDATION REQUIRED | Independent SHA audit |

---

## Validation

| Layer | Command | Result |
| --- | --- | --- |
| L1 | `uv run pytest tests/unit/applications/test_mcp_cancel_a_transport_governed_cancel.py -q` | **3 passed** |
| L2 | consolidated: cancel-a + trust-a + task_control_governed_cancel + L3-R2 MCP host | **43 passed** |
| L3 | streamable HTTP TRUST-A in L2; L3-R2 subprocess matrix reused | **included** |

Logs: `.tmp/session/TLX-TL11-CANCEL-A/` (`l1-cancel-a.log`, `l2-consolidated.log`).

---

## FRZ (scoped)

| Tag | Status |
| --- | --- |
| FRZ-BND-03 | Preserved — `HostTaskExecutionPort` only |
| FRZ-TEN | Scoped PASS — cross-tenant cancel denied |
| FRZ-GOV | Scoped PASS — control-plane mutation on cancel |
| FRZ-REL | Scoped PASS — transport ≠ physical Tool kill |
| FRZ-SEC | TRUST-A reuse |

---

## TL11 / TL1–TL15 rollup

| Item | Status |
| --- | --- |
| TL11 CANCEL-A | **READY FOR INDEPENDENT AUDIT** |
| TL11 TRUST-A | Unchanged READY |
| TL11 MCP→Tool physical cancel | **Not claimed** |
| TOOL-LIFE-X parent | **IN PROGRESS / BLOCKED** (TL8/TL7) |

**Next executable stage:** Independent SHA audit of CANCEL-A batch; then TL11 catalog E2E / remaining L3 physical host scenarios per roadmap §3.0.5.
