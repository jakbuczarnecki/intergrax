# TLX-TL11-MCP16-CANCEL-C — `run_agent` Execution Deadline & Governed Cancellation

| Field | Value |
| --- | --- |
| **Task** | `TLX-TL11-MCP16-CANCEL-C` |
| **Parent** | `TOOL-LIFE-X` §3.0.5 — **not CLOSED** |
| **Predecessor** | `TLX-TL11-PRODUCTION-CATALOG-PHYSICAL-QUALIFICATION` — independently accepted (scoped) |
| **Operator authorization** | Roadmap §3.0.5 + prior `OPERATOR APPROVED — MCP CANCEL-A` / `ADR OWN-A2` (no new deadline or cancel authority) |
| **START_HEAD** | `20e2225b1bae367ee8c3a365fd167c629ba924c6` |
| **IMPLEMENTATION_HEAD** | `20e2225b1bae367ee8c3a365fd167c629ba924c6` (**0** production delta) |
| **PUBLICATION_HEAD** | `d8a5eb739b2a737a251e8d4ed33e358a2a180e4a` |
| **Disposition** | **`TLX-TL11-MCP16-CANCEL-C — READY FOR INDEPENDENT AUDIT`** (scoped) |

**Non-closure:** No global `FRZ-*` PASS; no `TOOL-LIFE-X` parent CLOSED; no TL8 positive recovery; no DIST-SCALE-X; **parent Execution cancel → catalog Tool on MCP** remains **BLOCKED** (out of scope — not a new coordinator).

**Stale evidence invalidated (substantive):** [`TOOL_LIFE_X_TLX_TL11_MCP_FINAL_DISCOVERY_TRUST_CANCEL_QUALIFICATION_EVIDENCE.md`](TOOL_LIFE_X_TLX_TL11_MCP_FINAL_DISCOVERY_TRUST_CANCEL_QUALIFICATION_EVIDENCE.md) **MCP-16** row (*client disconnect not proven on host path*) — superseded for **run_agent governed cancel + host deadline reuse** by CANCEL-A publication and this reconciliation @ START_HEAD.

---

## 1. Git identity gate

| Check | Result |
| --- | --- |
| Repository | `jakbuczarnecki/intergrax` @ `origin` |
| Branch | `development` |
| START_HEAD | `20e2225b1bae367ee8c3a365fd167c629ba924c6` (= operator EXPECTED_START_HEAD) |
| Ancestry | START_HEAD ancestor of publication HEAD — **yes** |
| Worktree @ qualification | clean before publication commit |

---

## 2. Canonical call graph (protected HTTP `run_agent`)

```text
MCP streamable HTTP (couple_fastapi_with_mcp)
  → HarnessApiKeyMiddleware (when configured)
  → FastMCP run_agent
  → resolve_mcp_run_agent_identity (verified principal + conflict reject when wired)
  → execute_mcp_agent_task
       → admit_protected_mcp_root_execution_supervision (OWN-A2; gate before business work)
       → run_host_execution_with_mcp_transport_cancel [transport waiter only]
            → is_disconnected / asyncio waiter cancel
            → governed_cancel_active_task → CancellationCoordinator.request(binding.task)
       → HostTaskExecutionPort.execute(Task)
            → ExecutionRuntime / ExecutionDeadlineAuthorityResolver (root deadline)
            → terminal CAS / idempotency (Batch C — reused)
```

**Client disconnect path:** `governed_cancel_active_task` — not direct `Task.cancel()` on business Execution.

**Deadline authority:** singular — `ExecutionDeadlineAuthority` on host Execution; MCP module constants (`_ACTIVE_BINDING_WAIT_S`, `_HOST_DRAIN_AFTER_TRANSPORT_CANCEL_S`) are **transport observation only** (see `mcp_host_execution_transport_cancel.py` comment L198).

---

## 3. MC-01–MC-24 closure matrix

| ID | Stage | Owner | Contract | Evidence SHA | Classification | Gap | Required action | Negative test | L1/L2/L3 | FRZ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MC-01 | Admission | `execute_mcp_agent_task` | `HostTaskExecutionPort.execute` | `20e2225…` + L3-R2 | **ALREADY PROVEN** | — | — | bypass Nexus | L2/L3 | FRZ-BND-03 scoped |
| MC-02 | Identity | `resolve_mcp_run_agent_identity` | harness principal binding | TRUST-A @ prior SHA | **ALREADY PROVEN** | lab default without auth | — | forged tenant w/ auth | L1/L3 | FRZ-SEC scoped |
| MC-03 | Supervision | OWN-A2 supervisor | `admit_root_execution_supervision` | OWN-A2 evidence | **ALREADY PROVEN** | — | — | missing supervision | L1/L3 | FRZ-BND-03 |
| MC-04 | Root deadline | `ExecutionDeadlineAuthority` | host bind at execute | H02 + W1 @ `20e2225…` | **ALREADY PROVEN** | no MCP deadline param | reuse only | MCP param extend | L2 | FRZ-REL scoped |
| MC-05 | Propagation | deadline projection | `remaining_budget` | Batch A / H02 | **ALREADY PROVEN** | — | — | child > parent | L2 | FRZ-REL |
| MC-06 | Transport vs root | MCP transport cancel | no client timeout reset | CANCEL-A F05 | **ALREADY PROVEN** | — | — | client `wait_for` ≠ terminal | L1/L3 | FRZ-REL |
| MC-07 | Disconnect | `mcp_host_execution_transport_cancel` | `is_disconnected` | CANCEL-A + L3 abandon | **VALIDATION REQUIRED→PASS** | — | — | abandon client | L3 | FRZ-REL |
| MC-08 | Gov denial | `governed_cancel_active_task` | `accepted=False` | `test_cancel_a_governed_rejection_*` | **ALREADY PROVEN** | — | — | no coordinator signal | L1 | FRZ-GOV scoped |
| MC-09 | Accepted cancel | `CancellationCoordinator` | cooperative metadata | L3 physical abandon | **VALIDATION REQUIRED→PASS** | — | — | hang until flag | L3 | FRZ-REL |
| MC-10 | Non-cooperative host | OWN-A detach | bounded drain | `test_cancel_a_transport_cleanup_bounded_*` | **ALREADY PROVEN** | — | — | suppress CancelledError | L1 | FRZ-REL |
| MC-11 | No resurrection | terminal CAS | Execution terminal | Batch C reuse | **ALREADY PROVEN** | not L3 MCP full stack | — | late CAS | L2 | FRZ-REL |
| MC-12 | Late result | terminal authority | reject/reconcile | Batch C / host | **ALREADY PROVEN** | full MCP+real host late path | — | post-timeout result | L2 | FRZ-REL |
| MC-13 | UNKNOWN retry | retry eligibility | no blind retry | Batch A/E reuse | **ALREADY PROVEN** | — | — | retry after UNKNOWN | L2 | FRZ-REL |
| MC-14 | Retry vs deadline | retry budget | within `remaining_budget` | Batch A reuse | **ALREADY PROVEN** | — | — | extend D | L2 | FRZ-REL |
| MC-15 | Duplicate cancel | governed cancel | mutation_id prefix | CANCEL-A idempotent | **ALREADY PROVEN** | — | — | repeat disconnect | L1 | — |
| MC-16 | Shutdown race | OWN-A2 supervisor | shutdown + handoff | `test_own_a2_refused_handoff_*` | **ALREADY PROVEN** | — | — | shutdown during run | L1 | FRZ-REL |
| MC-17 | Pre-binding cancel | active registry wait | `_wait_for_active_binding` | `test_cancel_a_pre_admission_*` | **ALREADY PROVEN** | — | — | false cooperative | L1 | — |
| MC-18 | Cross-tenant | task control governance | tenant on principal | `test_cancel_a_cross_tenant_*` | **ALREADY PROVEN** | — | — | other tenant | L1 | FRZ-TEN scoped |
| MC-19 | No shadow authority | FRZ-BND-03 | single governed path | arch + this graph | **ALREADY PROVEN** | — | — | second coordinator | L2 | FRZ-BND-03 |
| MC-20 | Physical HTTP | streamable HTTP uvicorn | `couple_fastapi_with_mcp` | `test_cancel_a_l3_*` @ `20e2225…` | **VALIDATION REQUIRED→PASS** | — | — | missing API key | L3 | FRZ-REL-06 scoped |
| MC-21 | Diagnostics | logs / TaskControlResult | actionable detail | governed_cancel tests | **ALREADY PROVEN** | — | — | — | L1 | — |
| MC-22 | ECAP / FRZ | scoped ECAP | checklist rows | freeze checklist read | **VALIDATION REQUIRED** | global FRZ OPEN | independent audit | — | — | FRZ-* OPEN globally |
| MC-23 | Prior evidence | CANCEL-A / OWN-A2 / TRUST-A | SHA lineage | see §8 | **ALREADY PROVEN** | MCP-FINAL MCP-16 row | this record | — | — | — |
| MC-24 | Independent audit | GitHub SHA | publication HEAD | **pending** | **VALIDATION REQUIRED** | — | external auditor | — | — | — |

**CANCEL-C (parent cancel → Tool on MCP transport):** **BLOCKED** — unchanged; not authorized; no synthetic provider proof (roadmap propagation slice for catalog Tool invoke).

---

## 4. Adversarial state / race matrix (summary)

| Initial state | Competing ops | Owner | Expected | Forbidden | Test | FRZ |
| --- | --- | --- | --- | --- | --- | --- |
| RUNNING before deadline | normal complete | Host port | terminal success | transport owns terminal | L3 streamable mock host | REL |
| RUNNING, provider wait | root deadline fires | ExecutionDeadlineAuthority | TIMED_OUT terminal | retry extends D | H02 / W1 (L2) | REL |
| ADMIT pending | client disconnect | supervisor + gate | no business start | orphan host | `test_af06_*` | BND-03 |
| RUNNING bound | disconnect | governed cancel | coordinator set | raw cancel() | L3 abandon | REL |
| Terminal completing | disconnect | terminal CAS | single terminal | double success | L1 drain tests | REL |
| Supervisor shutting down | new run_agent | admission | REFUSED / fail-closed | silent execute | `test_own_a2_refused_*` | REL |
| Governance accepts | cancel | governed path | cooperative flag | bypass gov | L3 abandon | GOV |
| Governance refuses | disconnect | transport only | no coordinator | unauthorized cancel | rejection test | GOV |
| Duplicate cancel | second disconnect | lock + mutation_id | idempotent safe | double side effect | L1 | — |
| Client timeout only | no disconnect | transport | no false cancel | ≡ terminal fail | documented invariant | REL |
| Host suppresses cancel | transport lost | OWN-A drain | bounded waiter | infinite await | suppress test | REL |
| Admission rejected | — | supervisor | no execute | shadow run | ADM-02 | BND-03 |
| UNKNOWN outcome | retry request | retry policy | no blind retry | auto retry | Batch A/E | REL |
| Late completion | after timeout | terminal | reject/reconcile | resurrect | Batch C L2 | REL |
| Concurrent tenants | cancel other | governance | denied | cross cancel | cross-tenant L1 | TEN |

---

## 5. Deadline / cancellation source matrix

| Source | Authority | Affects root `deadline_at`? | Terminal owner |
| --- | --- | --- | --- |
| Execution admission / bind | `ExecutionDeadlineAuthority` | **yes** (authoritative) | Execution terminal |
| MCP client `wait_for` / read timeout | MCP client only | **no** | — |
| MCP transport disconnect | `governed_cancel_active_task` | **no** | Execution cancel path |
| Governance deny cancel | control plane | **no** | Execution unchanged |
| MCP `_HOST_DRAIN_AFTER_TRANSPORT_CANCEL_S` | transport waiter | **no** | waiter release only |
| Platform shutdown | OWN-A2 supervisor | **no** (refuse/handoff) | admission fail-closed |

---

## 6. Validation (consolidated @ `20e2225b1bae367ee8c3a365fd167c629ba924c6`)

**L1:** not required (no production semantic delta).

**L2/L3 (single batch):**

```text
uv run pytest \
  tests/unit/applications/test_mcp_cancel_a_transport_governed_cancel.py \
  tests/unit/applications/test_mcp_canonical_execution.py \
  tests/unit/applications/test_mcp_trust_a_principal_binding.py \
  tests/unit/tools/providers/test_tool_life_x_l3_r2_mcp_host_lifecycle_qualification.py \
  tests/unit/runtime/execution/test_enterprise_scale_resilience_w1.py::test_root_execution_global_deadline_fixed_at_bind \
  tests/qualification/harness_02/test_harness_02_gates.py::test_harness_02_grandchild_preserves_root_global_deadline \
  tests/unit/applications/test_task_control_governed_cancel.py -q
```

| Result | **66 passed**, 0 failed |
| Log | `.tmp/session/tlx-tl11-mcp16-cancel-c/l2-l3-consolidated-pytest.log` |
| Duration | ~70s |

**L3 physical scenarios covered in batch:** authenticated streamable `run_agent`; client abandon → governed cancel; governance rejection; cross-tenant deny; supervision admission/refusal; non-cooperative host drain.

**L3 not claimed:** full-stack MCP `run_agent` with **real** `HostTaskExecutionPort` implementation expiring root deadline on the wire (deadline proven on same canonical port via L2 H02/W1 — **reuse policy** per qualification-first).

---

## 7. FRZ / ECAP (scoped)

| Area | Disposition |
| --- | --- |
| FRZ-BND-03 | scoped — no shadow MCP lifecycle owner |
| FRZ-REL / REL-04/05 | scoped PASS for run_agent cancel/deadline **reuse**; global checklist **OPEN** |
| FRZ-TEN | scoped — binding when harness auth wired; **TRUST-C** lab-without-auth remains separate |
| FRZ-GOV | scoped — governed cancel admission |
| ECAP | P1/P2/P4/P6 — evidence record only; no production delta |

---

## 8. Evidence carry-forward

| Artifact | Status @ START_HEAD |
| --- | --- |
| CANCEL-A transport qual | **valid** — [`TOOL_LIFE_X_TLX_TL11_CANCEL_A_MCP_EXECUTION_TRANSPORT_QUALIFICATION_EVIDENCE.md`](TOOL_LIFE_X_TLX_TL11_CANCEL_A_MCP_EXECUTION_TRANSPORT_QUALIFICATION_EVIDENCE.md) |
| OWN-A2 supervision | **valid** — [`TOOL_LIFE_X_TLX_TL11_OWN_A2_SUPERVISED_EXECUTION_HANDOFF_QUALIFICATION_EVIDENCE.md`](TOOL_LIFE_X_TLX_TL11_OWN_A2_SUPERVISED_EXECUTION_HANDOFF_QUALIFICATION_EVIDENCE.md) |
| TRUST-A principal | **valid** — [`TOOL_LIFE_X_TLX_TL11_TRUST_A_CANCEL_MCP_PRINCIPAL_BINDING_EVIDENCE.md`](TOOL_LIFE_X_TLX_TL11_TRUST_A_CANCEL_MCP_PRINCIPAL_BINDING_EVIDENCE.md) |
| TL11 production catalog physical | **valid** — independent acceptance carried forward |
| MCP-FINAL discovery MCP-16 | **partially invalidated** — see header |

---

## 9. TL11 / TL1–TL15 rollup

| Slice | Status |
| --- | --- |
| TLX-TL11-MCP16-CANCEL-C (`run_agent` deadline + governed cancel) | **READY FOR INDEPENDENT AUDIT** (scoped) |
| TL11 parent cancel → Tool (CANCEL-C propagation) | **BLOCKED** |
| TL7 heartbeat / TL8 recovery | **OPEN** |
| TL13 Docker PG | **OPEN** |
| L3-R5-B | **OPEN** |
| TRUST-C (unauthenticated MCP lab) | **OPERATOR DECISION** (unchanged) |
| TOOL-LIFE-X parent | **IN PROGRESS / BLOCKED** |

---

## 10. Remaining blockers & next stage

| Blocker | Notes |
| --- | --- |
| Parent cancel → catalog Tool on MCP | Architecture / operator — not run_agent transport |
| L3 full host deadline on MCP wire | Optional strengthen — not required for 0-delta qual |
| TL8, TL7, TL13, L3-R5-B, TRUST-C | Roadmap §3.0.5 / §5 rollup |

**Next mandatory executable (roadmap):** independent SHA audit of this publication; then continue TL11 / TOOL-LIFE-X slices per §3.0.5 (TL8, TL7, TRUST-C decision, L3-R5-B) — **does not** authorize DIST-SCALE-X or ARCH-FREEZE.
