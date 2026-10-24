# TLX-TL11-OWN-A2-FINAL-SAFETY — Canonical Supervised Execution Handoff

| Field | Value |
| --- | --- |
| **Task** | `TLX-TL11-OWN-A2-FINAL-SAFETY` |
| **Operator authorization** | `OPERATOR APPROVED — ADR OWN-A2` |
| **START_HEAD** | `8d3c6a21d6a57841f41f8e631988183c83a20a02` |
| **IMPLEMENTATION_HEAD** | `eb4c1d8536a853ce2dc6942396cec12d05bf801d` |
| **PUBLICATION_HEAD** | `eb4c1d8536a853ce2dc6942396cec12d05bf801d` |
| **Disposition** | **`TLX-TL11-OWN-A2-FINAL-SAFETY — READY FOR INDEPENDENT AUDIT`** (scoped) |

**Non-closure:** No global `FRZ-*` PASS; no `TOOL-LIFE-X` parent CLOSED; no TL8 positive recovery; no DIST-SCALE-X; process crash does not invent persistent recovery.

---

## Independent audit remediation (SA-F01–SA-F06)

| ID | Severity | Finding | Remediation | Evidence |
| --- | --- | --- | --- | --- |
| SA-F01 | HIGH | `REFUSED` handoff → unbounded `shield(host_coro)` | Bounded observe/retry loop; no indefinite waiter; no unsanctioned cancel on refusal | `mcp_host_execution_transport_cancel._await_host_task_bounded_after_transport_waiter_lost` |
| SA-F02 | CRITICAL | `wait_for(gather(*pending))` could cancel Execution on shutdown timeout | `asyncio.wait(..., ALL_COMPLETED)` observation only; explicit log when pending remain | `HostRootExecutionTransportDetachmentSupervisor.shutdown_supervised_work` |
| SA-F03 | HIGH | `governed_cancel_admitted` early return without supervision | Removed silent detach; bounded wait; lab path without supervision does not `cancel()` when govern cancel admitted | `test_cancel_a_governed_admitted_without_supervision_waits_bounded_not_detach` |
| SA-F04 | HIGH | Duplicate `(task_id, run_id)` auto-ACCEPT without task identity check | `DUPLICATE_HANDOFF_CONFLICT` + idempotent ACCEPT only for same `host_asyncio_task` | `test_duplicate_handoff_conflicting_host_task_refused` |
| SA-F05 | HIGH | Observer/terminal races | Re-check host done before register; observer cleanup keyed to task identity; terminal exception logged | supervisor `_observe_supervised_work` + tests |
| SA-F06 | VALIDATION | Real MCP HTTP + shutdown | L3 streamable HTTP + abandon tests rerun in consolidated batch | `test_cancel_a_l3_*` |

---

## Architecture (ratified)

| Element | Location |
| --- | --- |
| Typed contract | `intergrax/contracts/root_execution_transport_detachment_supervision.py` — `RootExecutionTransportDetachmentSupervisionPort`, `DUPLICATE_HANDOFF_CONFLICT` |
| Canonical owner | `HostRootExecutionTransportDetachmentSupervisor` — private supervised map + non-cancelling shutdown observation |
| Host composition | `HarnessHostRuntime.transport_detachment_supervision` |
| MCP coupling | `mcp_transport_detachment_supervision_coupling.py` + `fastapi_mcp.couple_*` lifespan drain |
| Linearization | Handoff `ACCEPTED` only after lock + observer registered; duplicate key validated against live asyncio task identity |
| MCP waiter | Bounded handoff/retry/observe; `ACCEPTED` releases waiter; supervised completion may still be observed for terminal result |

---

## SA-01–SA-28 closure matrix

| ID | Owner | Canonical contract | Proof | Gap (pre-batch) | Correction | Negative test | Layer | FRZ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SA-01 | Host execution | Single owner after ACCEPT | L1 handoff test | — | — | cross-tenant | L1 | FRZ-BND-03 |
| SA-02 | Contract | Typed port | contract module | — | `DUPLICATE_HANDOFF_CONFLICT` | invalid REFUSED without reason | L1 | FRZ-BND-03 |
| SA-03 | Harness | One supervisor per build | harness wiring | — | — | — | L2 | FRZ-BND-03 |
| SA-04 | MCP monitor | No shadow authority | CANCEL-A | — | — | — | L1 | FRZ-GOV |
| SA-05 | Supervisor | ACCEPT/REFUSED | handoff tests | — | — | shutdown refuse | L1 | FRZ-REL |
| SA-06 | Supervisor | Lock linearization | unit tests | — | done-before-accept check | concurrent duplicate | L1 | FRZ-REL |
| SA-07 | MCP waiter | No detach without ACCEPT/safe terminal | drain tests | F03 | bounded loop | governed no detach | L1 | FRZ-SEC |
| SA-08 | MCP waiter | Handoff before release (continuing work) | suppress-cancel test | F01 | bounded not infinite shield | REFUSED budget | L1 | FRZ-REL |
| SA-09 | MCP waiter | REFUSED ≠ orphan | final-safety | F01 | observe/retry | — | L1 | FRZ-REL |
| SA-10 | Observer | Strong ref until terminal | retention test | F05 | identity-keyed cleanup | — | L1 | FRZ-REL |
| SA-11 | Observer | Exception visibility | observer log | F05 | `exception()` logged | — | L1 | FRZ-REL |
| SA-12 | Observer | No independent terminal writer | architecture | — | — | — | L1 | FRZ-BND-03 |
| SA-13 | Execution deadline | Unchanged owner | prior evidence | — | — | — | L2 | FRZ-REL |
| SA-14 | Terminal service | Unchanged | prior evidence | — | — | — | L2 | FRZ-BND-03 |
| SA-15 | Governed cancel | CANCEL-A path | cancel suite | — | — | cross-tenant | L2 | FRZ-GOV |
| SA-16 | Duplicate | Idempotent same task | duplicate test | F04 | conflict refuse | conflicting coroutine | L1 | FRZ-SEC |
| SA-17 | Tenant | Verified binding | cross-tenant | — | — | principal mismatch | L1 | FRZ-TEN |
| SA-18 | Shutdown | Refuse new handoff | shutdown test | — | — | — | L1 | FRZ-REL |
| SA-19 | Shutdown | No unauthorized cancel | shutdown observe test | F02 | `asyncio.wait` | suppress-cancel host | L1 | FRZ-REL |
| SA-20 | Lifespan | Drain on exit | L3 HTTP | — | — | — | L2/L3 | FRZ-REL |
| SA-21 | Crash recovery | N/A | — | — | no invention | — | — | — |
| SA-22 | UNKNOWN | Prior owners | prior | — | — | — | L2 | FRZ-REL |
| SA-23 | Registry | ActiveTaskRegistry only | prior | — | — | — | L2 | FRZ-BND-03 |
| SA-24 | Physical MCP | Streamable HTTP | L3 tests | F06 | rerun | abandon client | L3 | FRZ-SEC |
| SA-25 | Cleanup | Supervisor map | observer finally | F05 | — | — | L1 | FRZ-REL |
| SA-26 | TRUST-A | Principal binding | trust suite | — | — | forged tenant | L2 | FRZ-TEN |
| SA-27 | Evidence | This record | publication | — | — | — | L1/L2/L3 | — |
| SA-28 | Audit | Independent | — | open | publication SHA | — | — | — |

---

## REFUSED disposition matrix

| Reason | Transport waiter | Execution ownership | Cancel authority |
| --- | --- | --- | --- |
| `shutdown_in_progress` | Bounded observe; no new ACCEPT | Existing supervised refs retained | Not bypassed |
| `no_active_binding` | Bounded retry/observe | Host coroutine until binding/terminal | Governed path only |
| `run_id_mismatch` | REFUSED; bounded observe | Unchanged | Not bypassed |
| `tenant_mismatch` / `principal_tenant_mismatch` | REFUSED | Unchanged | Not bypassed |
| `host_task_already_done` | Safe terminal retrieve | Terminal | N/A |
| `duplicate_handoff_conflict` | REFUSED; existing supervision unchanged | Prior supervised task | Not bypassed |

---

## Ownership / linearization graph

```text
MCP transport waiter
  → (optional) governed_cancel_active_task
  → bounded observe / retry handoff
  → RootExecutionTransportDetachmentSupervisionPort.request_transport_detachment_handoff
       [lock: refuse if shutdown | duplicate validate | register observer]
  → ACCEPTED: waiter may release; supervisor holds strong ref
  → REFUSED: waiter stays within observation budget only (no shield forever, no unsanctioned cancel)
Host shutdown lifespan
  → shutdown_supervised_work: stop accepting; asyncio.wait(pending) without cancelling Execution
```

**Linearization point:** supervisor `_lock` at register/remove of `(task_id, run_id)` → observer task creation.

---

## L1 / L2 / L3 validation

| Tier | Command | Result |
| --- | --- | --- |
| L1/L2 consolidated | `uv run pytest tests/unit/runtime/execution/test_root_execution_transport_detachment_supervision.py tests/unit/applications/test_mcp_cancel_a_transport_governed_cancel.py tests/unit/applications/test_mcp_trust_a_principal_binding.py -q` | **30 passed** |
| Log | `.tmp/session/tlx-tl11-own-a2-final-safety/l2-consolidated.log` | exit 0 |

L3 physical cases included: `test_cancel_a_l3_streamable_http_run_agent_with_task_control_boundary`, `test_cancel_a_l3_physical_http_client_abandon_admits_governed_cancel`.

---

## FRZ / ECAP (scoped)

| Ref | Disposition |
| --- | --- |
| FRZ-BND-03 | No second coordinator; supervisor remains private host state |
| FRZ-TEN | Cross-tenant / principal mismatch refused |
| FRZ-SEC | No unsupervised detach on governed cancel admission |
| FRZ-GOV | Governed cancel path unchanged |
| FRZ-REL | Shutdown observation does not cancel Execution tasks |
| ECAP | Harness MCP composition + streamable HTTP qualification |

---

## TL11 / parent rollup

| Item | Status |
| --- | --- |
| TLX-TL11-OWN-A2-FINAL-SAFETY | **READY FOR INDEPENDENT AUDIT** |
| TL11 CANCEL-A / TRUST-A | Regressed in consolidated batch (pass) |
| TOOL-LIFE-X parent | **IN PROGRESS** — not CLOSED |

---

## Remaining blockers

- Independent audit of publication SHA
- Global `FRZ-*` qualification unchanged
- Process-level shutdown with still-pending supervised Execution after drain timeout (explicit warning only; no false graceful claim)

---

## Next mandatory TOOL-LIFE-X stage

Independent audit sign-off on OWN-A2-FINAL-SAFETY publication SHA; continue TL11 downstream per roadmap §3.0.5.
