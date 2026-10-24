# TLX-TL11-OWN-A2 — Canonical Supervised Execution Handoff

| Field | Value |
| --- | --- |
| **Task** | `TLX-TL11-OWN-A2` |
| **Operator authorization** | `OPERATOR APPROVED — ADR OWN-A2` |
| **START_HEAD** | `9580615d4785c64a04bf404b63bc61d76ae2c858` |
| **IMPLEMENTATION_HEAD** | *(set at publication commit)* |
| **PUBLICATION_HEAD** | *(set at publication commit)* |
| **Disposition** | **`TLX-TL11-OWN-A2 — READY FOR INDEPENDENT AUDIT`** (scoped) |

**Non-closure:** No global `FRZ-*` PASS; no `TOOL-LIFE-X` parent CLOSED; no TL8 positive recovery; no DIST-SCALE-X; process crash does not invent persistent recovery.

---

## Architecture (ratified)

| Element | Location |
| --- | --- |
| Typed contract | `intergrax/contracts/root_execution_transport_detachment_supervision.py` — `RootExecutionTransportDetachmentSupervisionPort` |
| Canonical owner / implementation | `HostRootExecutionTransportDetachmentSupervisor` in `intergrax/runtime/execution/root_execution_transport_detachment_supervision.py` — private supervised-work map on host composition |
| Host composition | `HarnessHostRuntime.transport_detachment_supervision` — one supervisor per harness build |
| MCP coupling | `mcp_transport_detachment_supervision_coupling.py` + `fastapi_mcp.couple_*` app state + shutdown lifespan drain |
| Handoff linearization | `request_transport_detachment_handoff` returns `ACCEPTED` only after strong reference + observer task registered under supervisor lock (before MCP `_await_host_task_bounded_after_transport_waiter_lost` returns) |
| MCP transport integration | `mcp_host_execution_transport_cancel._await_host_task_bounded_after_transport_waiter_lost` — handoff when `ActiveTaskRegistry` binding present; `REFUSED` retains MCP waiter via `shield(host_coro)` until completion |

**Predecessor:** CANCEL-A OWN-A implicit detach superseded by explicit ACCEPTED handoff when supervision port is wired (harness MCP path).

---

## SA-01–SA-28 closure matrix (summary)

| ID | Classification | Layer | Notes |
| --- | --- | --- | --- |
| SA-01 | PASS (scoped) | L1 | Single owner: `HostTaskExecutionPort` + supervisor strong ref after ACCEPT |
| SA-02 | PASS | L1 | Typed contract in `intergrax/contracts/` |
| SA-03 | PASS | L2 | `HarnessHostRuntime` + MCP app state |
| SA-04 | PASS | L1 | MCP monitor only; no independent supervision authority |
| SA-05–SA-06 | PASS | L1 | ACCEPTED/REFUSED + lock linearization |
| SA-07–SA-08 | PASS | L1 | Handoff serialized; MCP returns only after ACCEPT or legacy pre-binding path |
| SA-09 | PASS | L1 | REFUSED → shield await (no orphan) |
| SA-10–SA-12 | PASS | L1 | Observer retains/ref releases; `exception()` retrieved |
| SA-13–SA-15 | ALREADY PROVEN | L2 | Deadline/terminal unchanged |
| SA-16–SA-18 | PASS | L1/L2 | Governed cancel preserved; duplicate idempotent ACCEPT |
| SA-19–SA-20 | PASS | L1 | Shutdown refuses handoff; lifespan drain |
| SA-21 | EXPLICIT N/A | — | No crash recovery invented |
| SA-22–SA-23 | ALREADY PROVEN | L2 | UNKNOWN/retry owners unchanged; no shadow registry |
| SA-24 | PASS | L3 | `test_cancel_a_l3_physical_http_client_abandon_admits_governed_cancel` + supervision wired |
| SA-25–SA-26 | PASS | L2 | Supervisor cleanup; TRUST-A suite rerun |
| SA-27 | PASS | L1/L2/L3 | This record + pytest logs under `.tmp/session/tlx-tl11-own-a2/` |
| SA-28 | VALIDATION REQUIRED | — | Publication SHA + independent audit |

---

## L1 / L2 / L3 commands

```text
uv run pytest tests/unit/runtime/execution/test_root_execution_transport_detachment_supervision.py tests/unit/applications/test_mcp_cancel_a_transport_governed_cancel.py -q
uv run pytest tests/unit/runtime/execution/test_root_execution_transport_detachment_supervision.py tests/unit/applications/test_mcp_cancel_a_transport_governed_cancel.py tests/unit/applications/test_mcp_trust_a_principal_binding.py -q
```

**Results:** 14/14 then 25/25 passed (logs: `.tmp/session/tlx-tl11-own-a2/l1-l2-batch.log`, `l2-consolidated.log`).

**L3 physical:** included in CANCEL-A suite — streamable HTTP abandon + governed cancel with `transport_detachment_supervision` on MCP app state.

---

## FRZ / ECAP (scoped)

| Ref | Scoped disposition |
| --- | --- |
| FRZ-BND-03 | No second coordinator/registry; supervisor is private host state |
| FRZ-TEN | Cross-tenant handoff refused (`PRINCIPAL_TENANT_MISMATCH`) |
| FRZ-GOV | Governed cancel path unchanged |
| FRZ-REL | Shutdown bounded drain via lifespan |
| ECAP | Host MCP harness composition only |

---

## TL11 / TL1–TL15 rollup

| Item | Status |
| --- | --- |
| TL11 OWN-A2 | **READY FOR INDEPENDENT AUDIT** (this batch) |
| TL11 CANCEL-A | Prior evidence `88c4ec30…` — extended by OWN-A2 handoff |
| TL1–TL15 parent | **IN PROGRESS** — OWN-A2 blocker cleared; TL8/TL7 unchanged |

---

## Remaining blockers

- Independent GitHub audit of publication SHA
- Global FRZ qualification unchanged
- `TOOL-LIFE-X` parent not CLOSED

---

## Next mandatory TOOL-LIFE-X stage

Independent audit sign-off on OWN-A2 publication SHA; then resume TL11 downstream items per roadmap §3.0.5 without TL8 positive recovery.
