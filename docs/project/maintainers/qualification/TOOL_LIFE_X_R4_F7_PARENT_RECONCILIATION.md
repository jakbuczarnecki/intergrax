# TOOL-LIFE-X R4 — F7 parent reconciliation (partial)

| Field | Value |
| --- | --- |
| **EXPECTED_START_HEAD** | `34777f9c7e1a66c672d3322b83ff2bbf65c4df81` |
| **IMPLEMENTATION_SHA** | `128ecdc8e62d6c0caaa3a3ef8a965b77af9bdc47` |
| **Disposition** | `TOOL-LIFE-X R4 — PARTIAL / EXACT BLOCKERS` |

## Consolidated verification

| Command | Result | Log |
| --- | --- | --- |
| R4 final wave (18 modules, 208 tests) | **205 passed**, 3 failed (R6 issuer-binding pollution when run after R7+; R6 file alone **28 passed**) | `.tmp/session/TOOL-LIFE-X-R4/pytest-r4-final-consolidated.log` |
| F3 + Gate A delta | **3 passed** (CAS in-memory, F3 HTTP L3 ×2) | `.tmp/session/TOOL-LIFE-X-R4/pytest-r4-final-wave2.log` |
| Cross-loop + Gate B cancel | **32 passed** (incl. bridge cancel parity) | same wave2 log |
| Ruff (changed sources) | **PASS** | session |

## Gate A atomicity audit (`mark_physical_running` → CAS)

| Finding | Remediation |
| --- | --- |
| SQLite durable path already serializes revoke vs CAS under `BEGIN IMMEDIATE` | Reused R8/PG qual |
| In-memory `InMemoryExternalOperationStateStore._compare_and_set_locked` lacked revoke guard under store lock | **Fixed** — `cas_update_blocked_by_bound_worker_revocation` under lock |
| Fast-path `mark_physical_running` pre-CAS revoke read | Retained; complements locked CAS |
| No new fencing manager | **Compliant** |

## F3 MCP HTTP CANCEL-C L3 (qual ingress)

| Criterion | Evidence |
| --- | --- |
| Real MCP HTTP client (`fastmcp.Client`) | `test_mcp_catalog_f3_cross_loop_cancel_c_physical_l3.py` |
| Supervised root + cross-loop ingress | `_IngressQualHostExecution` + `enable_cross_loop_catalog_child=True` |
| Child under active root | `test_f3_http_cross_loop_child_admitted_under_active_root` |
| Parent cancel during in-flight child | `test_f3_http_cross_loop_parent_cancel_during_child` |
| Cooperative cancel after cross-loop invoke | Bridge + `asyncio.to_thread` on sync catalog invoker |
| Production Option C | **Unchanged** — default off |

## G-A01–G-A09

| ID | Status | Evidence / blocker |
| --- | --- | --- |
| G-A01 | PARTIAL | R6–R10 + durable composition; dedicated R4 race matrix not exhaustive |
| G-A02 | PARTIAL | DB revoke deny + in-memory CAS fix; HTTP/MCP B-class = **authorization fence only** |
| G-A03 | PARTIAL | UNKNOWN paths Batch B/R8; no false SUCCESS after revoke (scoped) |
| G-A04–G-A06 | PASS scoped | Reuse Batch A/B/D + R8 |
| G-A07 | WIRED default-off | `test_tool_life_host_bootstrap_wiring.py`; **OD-R4-G-A07-PROD-ACTIVATE** for effective prod |
| G-A08 | PARTIAL | 205/208 consolidated; PG qual not in this batch |
| G-A09 | BLOCKED | F6 scenario 24 positive not re-run @ this SHA |

## F6 scenario 24

**NOT RUN** @ this batch — recovery default-off; positive resume requires operator acceptance of Gate A class + explicit qual run. Reuse candidates: `test_tool_life_x_batch_c_stalled_abandoned_recovery.py`, R8 `recover_running_operation_after_certified_owner_loss` (historical green).

## EC1–EC15 (summary)

| EC | Status |
| --- | --- |
| EC1–EC6 | PASS scoped — reuse R1–R6 L3 |
| EC7 | PASS scoped — [`TOOL_LIFE_X_R4_OD_EC7_RECORD.md`](TOOL_LIFE_X_R4_OD_EC7_RECORD.md) |
| EC8–EC12 | PARTIAL — Batch B/D subset in consolidated log |
| EC13–EC15 | PARTIAL — F3/F5 gaps per blockers below |

## Scenarios 1–24 (summary)

| Range | Status |
| --- | --- |
| 1–23 | PASS scoped — historical L3 reuse (Batch A–E, R1–R10) unless superseded by delta |
| 24 | **BLOCKED** — not executed this batch |

## Exact blockers (mandatory)

1. **OD-R4-G-A07-PROD-ACTIVATE** — effective production wiring / recovery activation.
2. **G-A02** — dedicated per-sink R4 L3 rows (HTTP atomic remote fence not claimed).
3. **F6 scenario 24** — positive qual-only recovery @ frozen SHA.
4. **Formal TOOL-LIFE-X CLOSED** / **EXT-L4** — independent parent audit only.

## Operator decisions

| ID | Decision |
| --- | --- |
| OD-R4-G-A07-PROD-ACTIVATE | Pending |
| OD-R4-CONSOLIDATED | Partial closure accepted; not READY audit |
| Production MCP HTTP catalog | Remains Option C |
