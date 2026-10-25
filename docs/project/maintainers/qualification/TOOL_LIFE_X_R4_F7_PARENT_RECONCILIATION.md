# TOOL-LIFE-X R4 — F7 parent reconciliation (remediation wave)

| Field | Value |
| --- | --- |
| **EXPECTED_START_HEAD** | `a63646bb8f0dc2fd6849547d33820bd58dc394ab` |
| **IMPLEMENTATION_SHA** | _(set at commit — `git rev-parse HEAD`)_ |
| **Disposition** | `TOOL-LIFE-X R4 — PARTIAL / EXACT BLOCKERS` |

## Consolidated verification

| Command | Result | Log |
| --- | --- | --- |
| R4 remediation selection (20 modules) | **213 passed**, 0 failed | `.tmp/session/TOOL-LIFE-X-R4/pytest-r4-remediation-consolidated.log` |
| R6 pollution repro (gate_a → r6 negatives) | **5 passed** | session |
| F3 streamable HTTP physical L3 | **2 passed** | same remediation log |
| F6 scenario 24 qual | **3 passed** | `test_tool_life_x_r4_f6_scenario_24_recovery.py` |

## Test contamination fix (205/208 → green)

| Root cause | Fix |
| --- | --- |
| `compose_tool_life_durable_single_host_certified_owner_loss` mutates module-global `_COMPOSITION` issuer binding | `tests/unit/runtime/external_operations/conftest.py` autouse `reset_certified_owner_loss_composition_for_tests()` (+ revocation trust reset) |

## F3 qualification layers

| Layer | Module | Contract |
| --- | --- | --- |
| L2 / integration | `test_mcp_catalog_f3_cross_loop_cancel_c_physical_l3.py` | In-process FastMCP `Client`, simulated host drain |
| **L3 physical HTTP** | `test_mcp_catalog_f3_streamable_http_physical_l3.py` | uvicorn loopback + `streamablehttp_client`; `build_harness_host_task_execution` + `ExecutionBoundary` + OWN-A2 (`admit_protected_mcp_root_execution_supervision` via coupled boundary); cross-loop ingress on canonical `HostTaskExecution` |

## Gate A per sink (R4 evidence map)

| Sink | R4 rows | Evidence |
| --- | --- | --- |
| DB / durable | Revoke vs CAS, stale epoch, UNKNOWN terminal | R8 suite + `test_gate_a_*` + in-memory CAS under lock |
| DB / PostgreSQL | Physical transactions | `test_tool_life_x_l3_physical_postgresql_qualification.py` (Docker-gated; not in remediation batch) |
| HTTP | Auth fence, lifecycle, UNKNOWN | Batch D `test_tool_life_x_batch_d_provider_resource_lifecycle.py` |
| MCP | Subprocess host lifecycle | R2 `test_tool_life_x_l3_r2_mcp_host_lifecycle_qualification.py` |

Authorization fencing ≠ remote atomic commit fencing (operator honesty retained).

## G-A07

| Check | Status |
| --- | --- |
| Default-off env | `test_wiring_disabled_by_default` |
| Topology fail-closed | `test_wiring_fail_closed_wrong_topology` |
| Enabled `DURABLE_SINGLE_HOST` materializes composition | `test_wiring_materializes_when_enabled` (+ bundle/store/revocation fields) |
| No prod recovery activation | Wiring only; no issuer auto-activate |

## F6 scenario 24

| Test | Status |
| --- | --- |
| Positive UNKNOWN recovery after revoke + certified loss | `test_scenario_24_positive_recovery_after_revocation_and_certified_loss` |
| Stale epoch physical deny | `test_scenario_24_stale_epoch_cannot_resume_physical_effect` |
| Forged recovery rejected | `test_scenario_24_forged_recovery_without_staged_evidence_rejected` |

Qual-only; no production recovery default-on.

## G-A01–G-A09 (snapshot)

| ID | Status |
| --- | --- |
| G-A01 | PASS scoped — R6–R10 + durable composition |
| G-A02 | PARTIAL — per-sink qual mapped; HTTP/MCP B-class auth-only |
| G-A03 | PASS scoped — UNKNOWN / reconcile paths in Batch B + R8 |
| G-A04–G-A06 | PASS scoped — Batch A/B/D + R8 |
| G-A07 | WIRED default-off; **OD-R4-G-A07-PROD-ACTIVATE** for live activation |
| G-A08 | PASS scoped — remediation consolidated 213/213 |
| G-A09 | PASS scoped — scenario 24 qual module @ remediation SHA |

## Scenarios 1–24

| Scenario | Status |
| --- | --- |
| 1–23 | PASS scoped — historical L3 reuse in remediation batch |
| 24 | **PASS qual** — `test_tool_life_x_r4_f6_scenario_24_recovery.py` |

## EC1–EC15 (summary)

| EC | Status |
| --- | --- |
| EC1–EC7 | PASS scoped (EC7 record unchanged) |
| EC8–EC12 | PASS scoped — batch modules in remediation log |
| EC13–EC15 | PARTIAL — prod activation / independent audit gates |

## Remaining mandatory blockers

1. **OD-R4-G-A07-PROD-ACTIVATE** — operator authorization for effective production wiring/recovery (distinct from qual proof).
2. **Production MCP HTTP catalog** — remains Option C (disabled).
3. **PostgreSQL physical L3** — optional Docker qual not rerun in remediation batch.
4. **Formal TOOL-LIFE-X CLOSED / EXT-L4** — independent parent audit only; Cursor must not self-certify.
