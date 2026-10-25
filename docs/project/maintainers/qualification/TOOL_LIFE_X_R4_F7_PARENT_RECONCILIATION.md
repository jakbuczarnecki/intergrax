# TOOL-LIFE-X R4 — F7 parent reconciliation (remediation wave)

| Field | Value |
| --- | --- |
| **EXPECTED_START_HEAD** | `c9fad321a7dc26195237da05a9e6a5c09b159f2c` |
| **IMPLEMENTATION_SHA** | `50f0b667906045691d2e064306937ea748458605` — [`TOOL_LIFE_X_R4_EVIDENCE_GAP_CLOSURE.md`](TOOL_LIFE_X_R4_EVIDENCE_GAP_CLOSURE.md) |
| **Disposition** | `TOOL-LIFE-X R4 — PARTIAL / EXACT ARCHITECTURE BLOCKERS` |

## Consolidated verification

| Command | Result | Log |
| --- | --- | --- |
| R4 remediation selection (20 modules) | **213 passed**, 0 failed | `.tmp/session/TOOL-LIFE-X-R4/pytest-r4-remediation-consolidated.log` |
| R6 pollution repro (gate_a → r6 negatives) | **5 passed** | session |
| F3 streamable HTTP physical L3 | **3 passed** (incl. transport disconnect) | evidence-gap log |
| F6 scenario 24 qual | **4 passed** (positive re-drive **BLOCKED** proof) | `test_tool_life_x_r4_f6_scenario_24_recovery.py` |

## Test contamination fix (205/208 → green)

| Root cause | Fix |
| --- | --- |
| `compose_tool_life_durable_single_host_certified_owner_loss` mutates module-global `_COMPOSITION` issuer binding | `tests/unit/runtime/external_operations/conftest.py` autouse `reset_certified_owner_loss_composition_for_tests()` (+ revocation trust reset) |

## F3 qualification layers

| Layer | Module | Contract |
| --- | --- | --- |
| L2 / integration | `test_mcp_catalog_f3_cross_loop_cancel_c_physical_l3.py` | In-process FastMCP `Client`, simulated host drain |
| **L3 physical HTTP** | `test_mcp_catalog_f3_streamable_http_physical_l3.py` | uvicorn loopback + `streamablehttp_client`; sanctioned `root_child_work_ingress=` on `build_host_task_execution` (no `__setattr__`); OWN-A2 `task_id`; disconnect L3 |

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
| RUNNING → UNKNOWN after certified loss | `test_scenario_24_certified_loss_transitions_running_to_unknown` |
| Positive re-drive after reconciliation | **BLOCKED** — `test_scenario_24_positive_redrive_blocked_until_redrive_authority_contract` |
| Stale epoch physical deny | `test_scenario_24_stale_epoch_cannot_resume_physical_effect` |
| Forged recovery rejected | `test_scenario_24_forged_recovery_without_staged_evidence_rejected` |

Qual-only; no production recovery default-on. UNKNOWN transition ≠ positive recovery.

## G-A01–G-A09 (snapshot)

| ID | Status |
| --- | --- |
| G-A01 | PASS scoped — R6–R10 + durable composition |
| G-A02 | PARTIAL — per-sink qual mapped; HTTP/MCP B-class auth-only |
| G-A03 | PASS scoped — UNKNOWN / reconcile paths in Batch B + R8 |
| G-A04–G-A06 | PASS scoped — Batch A/B/D + R8 |
| G-A07 | WIRED default-off; **OD-R4-G-A07-PROD-ACTIVATE** for live activation |
| G-A08 | PASS scoped — remediation consolidated 213/213 |
| G-A09 | PARTIAL — scenario 24 negatives + UNKNOWN; positive re-drive BLOCKED |

## Scenarios 1–24

| Scenario | Status |
| --- | --- |
| 1–23 | PASS scoped — historical L3 reuse in remediation batch |
| 24 | **PARTIAL** — UNKNOWN qual; positive re-drive **ARCHITECTURE DECISION REQUIRED** |

## EC1–EC15 (summary)

| EC | Status |
| --- | --- |
| EC1–EC7 | PASS scoped (EC7 record unchanged) |
| EC8–EC12 | PASS scoped — batch modules in remediation log |
| EC13–EC15 | PARTIAL — prod activation / independent audit gates |

## Remaining mandatory blockers

1. **TL8 scenario 24 positive re-drive** — sanctioned redrive authority after reconciliation (worker epoch still bound + revoked blocks `mark_physical_running`).
2. **G-A02 HTTP** — atomic provider-side fence for non-idempotent remote effects (**ARCHITECTURE DECISION REQUIRED**).
3. **OD-R4-G-A07-PROD-ACTIVATE** — operator authorization for effective production wiring/recovery (distinct from qual proof).
4. **Production MCP HTTP catalog** — remains Option C (disabled).
5. **PostgreSQL physical L3** — optional Docker qual not rerun in evidence-gap batch.
6. **Formal TOOL-LIFE-X CLOSED / EXT-L4** — independent parent audit only; Cursor must not self-certify.
