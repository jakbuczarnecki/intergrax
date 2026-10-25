# TLX-TL11-OWN-A2-FINAL-SAFETY — Canonical Supervised Execution Handoff

| Field | Value |
| --- | --- |
| **Task** | `TLX-TL11-OWNERSHIP-ADMISSION` / `TLX-TL11-OWN-A2-OWNERSHIP-INVARIANT` |
| **Operator authorization** | `OPERATOR APPROVED — ADR OWN-A2` |
| **START_HEAD** | `346d3ec5334fe7a11fbacdb80aeb7507b39b2eb1` |
| **IMPLEMENTATION_HEAD** | `75643a97633bc27c9cb8333daaea0f21e14e40eb` |
| **PUBLICATION_HEAD** | `75643a97633bc27c9cb8333daaea0f21e14e40eb` |
| **Disposition** | **`TLX-TL11-OWNERSHIP-ADMISSION — READY FOR INDEPENDENT AUDIT`** (scoped) |

**Non-closure:** No global `FRZ-*` PASS; no `TOOL-LIFE-X` parent CLOSED; no TL8 positive recovery; no DIST-SCALE-X; process crash does not invent persistent recovery.

---

## Independent audit remediation (SA-F01–SA-F06)

| ID | Severity | Finding | Remediation | Evidence |
| --- | --- | --- | --- | --- |
| SA-F01 / OWN-FINAL-01 | CRITICAL | Handoff-only ownership → live orphan after REFUSED / invariant | `admit_root_execution_supervision` at MCP launch; `holds_root_execution_supervision` releases waiter only when supervisor retains strong ref; shutdown refuses admission before Execution | `test_admission_supervision_without_active_binding_retains_host`, `test_cancel_a_pre_admission_*`, `test_own_a2_refused_handoff_*` |
| SA-F02 | CRITICAL | `wait_for(gather(*pending))` could cancel Execution on shutdown timeout | `asyncio.wait(..., ALL_COMPLETED)` observation only; explicit log when pending remain | `HostRootExecutionTransportDetachmentSupervisor.shutdown_supervised_work` |
| SA-F03 / OWN-FINAL-02 | HIGH | Protected path without supervision / governed-only release | `McpProtectedHostExecutionAdmissionError` before `create_task`; unsupervised coupling raises ownership invariant | `test_own_a2_protected_http_missing_supervision_fail_closed_before_execution`, `test_cancel_a_governed_admitted_without_supervision_raises_ownership_invariant` |
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
| Linearization | **Admission:** `admit_root_execution_supervision` at `execute_mcp_agent_task` / protected transport entry before waiter detach; **handoff:** binding validation + idempotent ACCEPT when same `host_asyncio_task` already supervised |
| MCP waiter | TERMINAL, admission-held supervision, or handoff `ACCEPTED` releases waiter; live+unowned → `McpHostExecutionTransportOwnershipInvariantError` |
| Protected admission | Requires `transport_detachment_supervision` + verified principal; admission `REFUSED` → bounded cancel observe + `McpProtectedHostExecutionAdmissionError` (no Execution start) |

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
| `shutdown_in_progress` | Retry/observe budget; invariant if still live+unsupervised | Existing supervised refs retained | Not bypassed |
| `no_active_binding` | Transient retry; waiter may release if admission supervision holds | Supervisor strong ref from admission | Governed path only |
| `run_id_mismatch` | Permanent REFUSED; invariant if live+unsupervised | Unchanged | Not bypassed |
| `tenant_mismatch` / `principal_tenant_mismatch` | Permanent REFUSED; invariant if live+unsupervised | Unchanged | Not bypassed |
| `host_task_already_done` | Safe terminal retrieve | Terminal | N/A |
| `duplicate_handoff_conflict` | Permanent REFUSED; invariant if conflicting host live+unsupervised | Prior supervised task | Not bypassed |

---

## Ownership / linearization graph

```text
execute_mcp_agent_task (protected)
  → create_task(host execute)
  → admit_root_execution_supervision (verified principal tenant)
MCP transport waiter
  → (optional) governed_cancel_active_task
  → bounded observe / retry handoff
  → request_transport_detachment_handoff (binding validation; idempotent if admitted)
  → waiter release: TERMINAL | holds_root_execution_supervision | handoff ACCEPTED
  → live+unowned → McpHostExecutionTransportOwnershipInvariantError
Host shutdown lifespan
  → shutdown_supervised_work: stop accepting; asyncio.wait(pending) without cancelling Execution
```

**Linearization point:** supervisor `_lock` at admission/handoff register of `task_id` → observer task creation (before MCP transport waiter may detach).

---

## L1 / L2 / L3 validation

| Tier | Command | Result |
| --- | --- | --- |
| L1/L2/L3 consolidated | `uv run pytest tests/unit/runtime/execution/test_root_execution_transport_detachment_supervision.py tests/unit/applications/test_mcp_cancel_a_transport_governed_cancel.py tests/unit/applications/test_mcp_trust_a_principal_binding.py -q` | **33 passed** (2026-10-24 admission batch) |
| Log | `.tmp/session/tlx-tl11-own-admission/l2-consolidated.log`, `l3.log` | exit 0 |

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
