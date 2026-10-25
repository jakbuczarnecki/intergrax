# TOOL-LIFE-X R4 — Gate A closure wave evidence (partial)

| Field | Value |
| --- | --- |
| **START_HEAD** | `34777f9c7e1a66c672d3322b83ff2bbf65c4df81` |
| **Disposition** | `TOOL-LIFE-X R4 — PARTIAL / EXACT BLOCKERS` |

## G-A07 (OD-R4-G-A07-WIRE)

| Layer | Status |
| --- | --- |
| Module / factory | **AVAILABLE** — `wire_tool_life_durable_single_host_execution_recovery` |
| Host bootstrap | **WIRED** — `resolve_tool_life_host_bootstrap_wiring` + `HarnessHostRuntime.tool_life_execution_recovery` |
| Configuration | **Default-off** — `INTERGRAX_TOOL_LIFE_DURABLE_WIRING_ENABLED` |
| Topology gate | **DURABLE_SINGLE_HOST** only |
| Recovery activation | **NOT AUTHORIZED** — composition only |
| L2 | `test_tool_life_host_bootstrap_wiring.py` |

## G-A02 / G-A03 (wave delta)

| Sink | Delta | L3 |
| --- | --- | --- |
| DB / durable ext-op | `mark_physical_running` refuses revoked epoch; in-memory CAS revoke under lock | R8 + `test_gate_a_inmemory_cas_revoke_race.py` |
| HTTP | Authorization fence (existing Batch D) | **75+** batch D L3 |
| MCP | Existing R2 subprocess qual | Reused |

New L2: `test_gate_a_mark_physical_running_revocation.py`

## F3 CANCEL-C physical

| Item | Status |
| --- | --- |
| Ingress shutdown/cancel race L2 | **PASS** — `test_cross_loop_cancel_before_drain_returns_shutdown_or_denied` |
| Full MCP HTTP + supervised root + parent cancel L3 | **PASS qual** — `test_mcp_catalog_f3_cross_loop_cancel_c_physical_l3.py`; production catalog HTTP remains Option C |

## F6 scenario 24

**NOT RUN** — positive recovery requires full Gate A matrix @ dedicated R4 L3 rows; G-A07 wiring qualified but recovery not activated.

## Consolidated L2/L3 log

`.tmp/session/TOOL-LIFE-X-R4/pytest-r4-final-consolidated.log` — **205 passed** (208 collected; 3 R6 order-pollution failures — R6 file alone 28/28). F7: [`TOOL_LIFE_X_R4_F7_PARENT_RECONCILIATION.md`](TOOL_LIFE_X_R4_F7_PARENT_RECONCILIATION.md).

## G-A01–G-A09 snapshot

| ID | Status |
| --- | --- |
| G-A01 | PARTIAL — epoch/revoke via store (R6–R8) |
| G-A02 | PARTIAL — per-sink; HTTP/MCP physical B = auth-only |
| G-A03 | PARTIAL — UNKNOWN paths in Batch B/R8 |
| G-A04–G-A06 | PROVEN scoped reuse |
| G-A07 | **WIRED default-off**; effective prod activation **BLOCKED** (OD-R4-G-A07-PROD-ACTIVATE) |
| G-A08 | PARTIAL — L3 subset green |
| G-A09 | BLOCKED positive until F6 |

## OD-EC7

[`TOOL_LIFE_X_R4_OD_EC7_RECORD.md`](TOOL_LIFE_X_R4_OD_EC7_RECORD.md)
