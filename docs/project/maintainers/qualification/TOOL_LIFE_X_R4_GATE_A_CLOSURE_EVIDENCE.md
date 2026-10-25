# TOOL-LIFE-X R4 — Gate A closure wave evidence (partial)

| Field | Value |
| --- | --- |
| **START_HEAD** | `a9b81f6d0c7a82cf02806729323fcf2de37ba23b` |
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
| DB / durable ext-op | `mark_physical_running` refuses revoked epoch | R8 + PG qual reused |
| HTTP | Authorization fence (existing Batch D) | **75+** batch D L3 |
| MCP | Existing R2 subprocess qual | Reused |

New L2: `test_gate_a_mark_physical_running_revocation.py`

## F3 CANCEL-C physical

| Item | Status |
| --- | --- |
| Ingress shutdown/cancel race L2 | **PASS** — `test_cross_loop_cancel_before_drain_returns_shutdown_or_denied` |
| Full MCP HTTP + supervised root + transport cancel L3 | **OPEN** — production catalog HTTP remains Option C |

## F6 scenario 24

**NOT RUN** — positive recovery requires full Gate A matrix @ dedicated R4 L3 rows; G-A07 wiring qualified but recovery not activated.

## Consolidated L2/L3 log

`.tmp/session/TOOL-LIFE-X-R4/pytest-gate-a-wave.log` — **128 passed** (last full wave run).

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
