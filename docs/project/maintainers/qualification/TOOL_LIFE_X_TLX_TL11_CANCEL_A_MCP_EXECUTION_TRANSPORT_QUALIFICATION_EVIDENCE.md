# TLX-TL11-CANCEL-A — MCP → Canonical Execution Cancellation

| Field | Value |
| --- | --- |
| **Task** | `TLX-TL11-CANCEL-A-TERMINATION-SAFETY` |
| **Operator authorization** | `OPERATOR APPROVED — MCP CANCEL-A` |
| **START_HEAD** | `275fd3a5cc8e098d8d3a9bedaea47df5eb4992c5` |
| **IMPLEMENTATION_HEAD** | `88c4ec3008d3e770c1d3e0acce1488f4c37f3215` |
| **PUBLICATION_HEAD** | `88c4ec3008d3e770c1d3e0acce1488f4c37f3215` |
| **Gate** | **A1** (governed propagation where boundary + verified principal) |
| **Disposition** | **`TLX-TL11-CANCEL-A-TERMINATION-SAFETY — READY FOR INDEPENDENT AUDIT`** (scoped) |

**Non-closure:** No global `FRZ-*` PASS; no `TOOL-LIFE-X` parent CLOSED; no physical Tool provider kill claim; TL8/TL7 blockers preserved.

---

## OWN-A / OWN-B / OWN-C

| Decision | **OWN-A** — canonical execution ownership transfer |
| --- | --- |
| **Source contract** | `HostTaskExecutionPort.execute` / root `Execution` + `ActiveTaskRegistry` binding; MCP transport waiter is not lifecycle owner |
| **Mechanism** | After transport waiter loss, bounded drain; if governed cancel admitted, **do not** `cancel()` host asyncio task for HTTP waiter teardown; host work remains under canonical port until cooperative cancel / deadline |
| **OWN-B** | N/A — not used (no separate bounded terminal kill of authoritative Execution) |
| **OWN-C** | Not reached |

---

## CANCEL-F01–F06 R2 (termination-safety batch)

| ID | Disposition | Action |
| --- | --- | --- |
| **F01-R2** | **IMPLEMENTATION REQUIRED → mitigated** | Removed unbounded `await host_coro` after `cancel()` when host suppresses `CancelledError`; bounded `wait_for` for non-admitted path; OWN-A detach when governed cancel admitted |
| **F02** | **PASS (scoped)** | `cancel_admission_accepted` only on `governed_cancel_active_task.accepted=True` |
| **F03** | **PASS (scoped)** | Terminal result linearization during drain; `CancelledError` to MCP caller when no terminal |
| **F04-R2** | **VALIDATION REQUIRED → PASS (scoped)** | `test_cancel_a_l3_physical_http_client_abandon_admits_governed_cancel` — real streamable HTTP client cancel + context teardown (no `_client_disconnected` patch) |
| **F05-R2** | **IMPLEMENTATION REQUIRED → mitigated** | MCP transport observation uses module-local bounds; **not** `EventDeliveryPolicy`; root deadline remains `ExecutionDeadlineAuthority` only |
| **F06** | **ALREADY PROVEN** | Eight factory roots unchanged |

---

## Closure matrix (15 criteria)

| Criterion | Semantic owner | Contract | Gap | Classification | Correction | Negative test | Layer |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1. Transport cancel observation | MCP transport monitor | `mcp_host_execution_transport_cancel` | — | PASS (scoped) | `is_disconnected` + asyncio cancel | patched disconnect unit + L3 abandon | L1/L3 |
| 2. Governed cancel admission | `governed_cancel_active_task` | `task_control.py` | — | PASS | mutation boundary | rejection / no boundary | L1 |
| 3. Active Execution correlation | `ActiveTaskRegistry` | task_id/run_id binding | — | PASS | lookup before cancel | pre-admission | L1 |
| 4. Root deadline | `ExecutionDeadlineAuthority` | HARNESS-02 / host | — | ALREADY PROVEN | no MCP extension | deadline hierarchy tests (L2 reuse) | L2 |
| 5. Cleanup ownership | `HostTaskExecutionPort` | UE-11G host task | F01-R2 closed | PASS (scoped) | OWN-A detach | suppress-cancel adversarial | L1 |
| 6. Completion/cancel linearization | host terminal + waiter | transport cancel module | — | PASS | drain + result capture | complete-during-drain | L1 |
| 7. Idempotent cancel | governed cancel | mutation_id prefix | — | PASS | lock + admission flag | repeated transport signals | L1 |
| 8. Refused cancel | control plane | `TaskControlResult.accepted` | — | PASS | no coordinator signal | governance rejection | L1 |
| 9. Late results / terminal CAS | Execution terminal | Batch C host | — | ALREADY PROVEN | unchanged | host qualification | L2 |
| 10. UNKNOWN side effects | provider cancel | W4-D | — | N/A — WITH EVIDENCE | not claimed | — | — |
| 11. Physical disconnect | ASGI / streamable HTTP | Starlette `is_disconnected` | F04-R2 closed | PASS (scoped) | L3 abandon | client timeout ≠ disconnect (not conflated) | L3 |
| 12. Host shutdown | app lifespan | harness MCP | — | ALREADY PROVEN | uvicorn test fixture | — | L2 |
| 13. Multi-tenant authority | TRUST-A principal | verified identity | — | PASS | tenant on principal | cross-tenant | L1 |
| 14. No shadow lifecycle owner | FRZ-BND-03 | no new coordinator | — | PASS | single governed path | arch tests (L2 reuse) | L2 |
| 15. Cancellation time budgets | MCP observation constants | local module bounds | F05-R2 closed | PASS (scoped) | decoupled from EventDeliveryPolicy | bounded cleanup timing | L1 |

---

## Canonical cancellation owner graph

```text
MCP transport (disconnect / asyncio cancel)
        │
        ▼
execute_mcp_agent_task + mcp_host_execution_transport_cancel  [transport waiter only]
        │
        ▼
governed_cancel_active_task  [task_control.py]
        ▼
CancellationCoordinator.request(binding.task)
        ▼
HostTaskExecutionPort.execute / Execution + ExecutionDeadlineAuthority
```

**After HTTP transport loss (OWN-A):** MCP waiter ends within `_HOST_DRAIN_AFTER_TRANSPORT_CANCEL_S`; canonical Execution continues under `HostTaskExecutionPort` until cooperative cancel or root deadline — not orphaned, not unbounded transport `await`.

---

## Race / negative matrix (summary)

| Scenario | Expected | Test / evidence |
| --- | --- | --- |
| Task finishes before cancel | terminal returned | L1 asyncio cancel + complete-during-drain |
| Cancel wins | `CancelledError` to caller | L1 transport cancel |
| Completion during governed admission | single linearization | governed rejection test |
| Completion during cleanup | terminal returned | `test_cancel_a_host_completes_during_transport_cancel_drain` |
| Governance refusal | no cooperative signal | rejection test |
| No active binding | no false cancel | pre-admission test |
| Host suppresses `CancelledError` | bounded waiter; host owned | `test_cancel_a_transport_cleanup_bounded_when_host_suppresses_cancellation` |
| Cross-tenant cancel | denied | cross-tenant test |
| Physical client abandon | governed cancel | L3 physical abandon |
| Provider UNKNOWN | unchanged | not claimed |

---

## Validation

| Layer | Command | Result |
| --- | --- | --- |
| L1 | `uv run pytest tests/unit/applications/test_mcp_cancel_a_transport_governed_cancel.py -q` | **10 passed** |
| L2 | cancel-a + task_control + trust-a + active_task_registry + L3-R2 MCP host (paths in batch) | **68 passed** |
| L3 | `test_cancel_a_l3_physical_http_client_abandon_admits_governed_cancel` + streamable HTTP boundary smoke | **included in L1** |

Logs: `.tmp/session/TLX-TL11-CANCEL-A/` (`l1-termination-safety.log`, `l2-termination-safety.log`).

---

## FRZ (scoped ECAP)

| Tag | Status |
| --- | --- |
| FRZ-BND-03 | Preserved — `HostTaskExecutionPort` only; no reverse edge |
| FRZ-TEN | Scoped PASS |
| FRZ-GOV | Scoped PASS |
| FRZ-REL | Scoped PASS — transport ≠ Tool kill |
| FRZ-SEC | TRUST-A reuse |

---

## TL11 / TL1–TL15 rollup

| Item | Status |
| --- | --- |
| TL11 CANCEL-A termination safety | **READY FOR INDEPENDENT AUDIT** |
| TL11 TRUST-A | Unchanged READY |
| TL11 MCP→Tool physical cancel | **Not claimed** |
| TOOL-LIFE-X parent | **IN PROGRESS / BLOCKED** (TL8/TL7) |

**Remaining blockers:** Independent SHA audit; TL8/TL7 parent gates; optional expanded physical disconnect matrix.

**Next executable batch:** Independent audit of publication SHA; TL11 catalog E2E per roadmap §3.0.5.
