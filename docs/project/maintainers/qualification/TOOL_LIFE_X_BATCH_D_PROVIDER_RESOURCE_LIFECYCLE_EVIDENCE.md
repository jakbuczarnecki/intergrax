# TOOL-LIFE-X Batch D — Provider Resource, Bounds & Cancellation Qualification Evidence

| Field | Value |
| --- | --- |
| **Task** | `TOOL-LIFE-X-BATCH-D` |
| **Parent** | `TOOL-LIFE-X` (§3.0.5) — **not CLOSED** |
| **START_HEAD** | `792064ce3127fda505e51fdf987c64ef495d451c` |
| **FINAL_COMMIT** | `54bb2ba88` |
| **Test execution SHA** | `54bb2ba88` |
| **Branch** | `development` |
| **Origin** | `https://github.com/jakbuczarnecki/intergrax.git` |
| **Disposition** | **`TOOL-LIFE-X BATCH D — READY FOR INDEPENDENT AUDIT`** |

**Non-closure:** Does not close `TOOL-LIFE-X`, does not promote global `FRZ-*` PASS, does not start Batch E. Parent L3 real-infrastructure certification (Docker DB cancel, live MCP process, HTTP fault injection) remains deferred.

## Prerequisites (acceptance evidence)

| Batch | Disposition | Evidence |
| --- | --- | --- |
| **A** | READY FOR INDEPENDENT AUDIT | [`TOOL_LIFE_X_BATCH_A_DEADLINE_HIERARCHY_EVIDENCE.md`](TOOL_LIFE_X_BATCH_A_DEADLINE_HIERARCHY_EVIDENCE.md) — L2 **65/65** |
| **B** | READY FOR INDEPENDENT AUDIT | [`TOOL_LIFE_X_BATCH_B_RETRY_IDEMPOTENCY_UNKNOWN_OUTCOME_EVIDENCE.md`](TOOL_LIFE_X_BATCH_B_RETRY_IDEMPOTENCY_UNKNOWN_OUTCOME_EVIDENCE.md) |
| **C** | READY FOR INDEPENDENT AUDIT | [`TOOL_LIFE_X_BATCH_C_STALLED_ABANDONED_RECOVERY_EVIDENCE.md`](TOOL_LIFE_X_BATCH_C_STALLED_ABANDONED_RECOVERY_EVIDENCE.md) @ `792064ce…` ancestry |

## MASTER INSTRUCTION v2 — pre-test justification (L1/L2)

1. **Why these tests now:** Batch D implementation delta touches SQL/HTTP/browser/MCP adapter ceilings; L1 proves new bounds; L2 regresses A/B/C invoker semantics unchanged.
2. **Why not full platform pytest:** Token/process budget; scoped provider + TOOL-LIFE-X nexus inventory only.
3. **Why mocks suffice here:** Architecture-scoped Path B — physical cancel/lock proof remains L3; unit bounds ≠ production vendor PASS.
4. **What would be false confidence:** Claiming MCP hung-server or PostgreSQL `pg_cancel_backend` qualification from in-memory doubles.
5. **What unchanged suites were skipped:** Full Batch A **65** standalone re-run (deadline tests included in L2 **50** subset); `test_t14_t17_builder_tools` (sandbox env authority); U5 gate (unrelated env flake @ HEAD).
6. **What blocks parent close:** TL7 heartbeat/STALLED, TL13 live-infra matrix, TL15 rollup, distributed owner-loss L3.

## Closed-world provider inventory (production entry)

| Family | Registration | Handler | Transport | Cancel / bound owner |
| --- | --- | --- | --- | --- |
| **SQL** | `database/bundle.py` | `DatabaseQueryHandler` → `database_query` | `ConfiguredRelationalStoreExecutionPort` / adapter | Row+byte ceilings @ tool service; cancel = driver/port (declared in `provider_transport_capabilities`) |
| **HTTP** | `http/bundle.py` | `HttpRequestHandler` → `http_request` | `HttpClientBackend` (`AllowlistHttpClient`) | Invoker timeout + `timeout_s`; response bytes @ tool service |
| **MCP** | `mcp_catalog_tools.py` + `mcp_nexus_server.py` | Host task execution → ToolRuntime (no shadow invoker) | FastMCP transport | `serialize_bounded_mcp_json` on describe payload; invoke via canonical host |
| **Browser** | `browser/bundle.py` + `sandbox` `browser.run` | `BrowserFetchPageHandler` / `browser_run` | `BrowserAutomation` / sandbox session | Text char ceiling @ tool service; sandbox already has `timeout_s`/`max_chars` |

**Canonical path preserved:** `ExecutionRuntime` → `ToolRuntime` / gateway → `catalog_dispatch` → `RuntimeToolInvoker` → `ToolExecutor` → provider adapter.

## Provider qualification matrix (summary)

| Family | Owner | Op class | Deadline | Admission | Resource bounds | Cancel capability | Physical certainty | FRZ (scoped) | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SQL query | Tool service + invoker | READ_ONLY | Invoker `timeout_ms` | Integration pin | `max_rows`, `max_result_bytes` | QUERY_CANCEL (port-dependent) | COMPLETED / fail closed byte | FRZ-REL-10 | **VALIDATION REQUIRED** (L3 DB) |
| SQL execute | Tool service + invoker | NON_IDEMPOTENT_WRITE | Invoker | Pin | N/A rows | QUERY_CANCEL | UNCERTAIN on timeout (Batch B) | FRZ-REL-03 | Reuse Batch B |
| HTTP | Tool service + invoker | NON_IDEMPOTENT_WRITE default | Invoker + `timeout_s` | Allowlist host | `max_response_bytes` | CONNECTION_DROP | UNCERTAIN on lost ack (L3) | FRZ-REL-06 | **VALIDATION REQUIRED** (L3) |
| MCP catalog | Host execution | READ_ONLY describe | Host task deadline | Tenant on task | MCP JSON byte cap | COOPERATIVE_ONLY | N/A describe | FRZ-REL-06 | **VALIDATION REQUIRED** (live MCP) |
| Browser fetch | Tool service | READ_ONLY | Invoker + declared nav timeout | Integration | `max_text_chars` | CONNECTION_DROP | COMPLETED/truncated | FRZ-REL-06 | **VALIDATION REQUIRED** (L3) |

## TL1–TL15 (Batch D reconciliation)

| ID | Status | Notes |
| --- | --- | --- |
| TL1–TL2 | VALIDATION REQUIRED | Batch A evidence; unchanged by D delta |
| TL3 | VALIDATION REQUIRED | Physical DB/HTTP/MCP cancel — L3 |
| TL4–TL9 | Reuse A/B/C | No invoker ownership change |
| **TL10** | **VALIDATION REQUIRED** | SQL row/byte contract + tests @ this batch |
| **TL11** | **VALIDATION REQUIRED** | MCP canonical path + payload bound; live process qual L3 |
| **TL12** | **VALIDATION REQUIRED** | HTTP response byte ceiling + timeout param |
| TL13 | BLOCKED | Parent L3 real infrastructure |
| TL14 | ALREADY PROVEN | No second invoker / shadow owner |
| TL15 | BLOCKED | Parent rollup |

## Validation

### L1 (critical — provider bounds)

```text
uv run pytest -p no:xdist \
  tests/unit/tools/providers/test_tool_life_x_batch_d_provider_resource_lifecycle.py \
  tests/unit/tools/providers/database/test_database_tools.py -q
```

| Metric | Value |
| --- | --- |
| **Passed** | **12** |
| **Failed** | **0** |
| **Log** | `.tmp/session/TOOL-LIFE-X-BATCH-D/l1-batch-d.log` |

### L2 (consolidated Batch D + A/B/C regression subset)

```text
uv run pytest -p no:xdist \
  tests/unit/tools/providers/test_tool_life_x_batch_d_provider_resource_lifecycle.py \
  tests/unit/tools/providers/database/test_database_tools.py \
  tests/unit/runtime/nexus/tools/test_tool_life_x_batch_a_deadline_hierarchy.py \
  tests/unit/runtime/nexus/tools/test_tool_life_x_batch_b_retry_idempotency_unknown_outcome.py \
  tests/unit/runtime/nexus/tools/test_tool_life_x_batch_b_real_governance_authorization.py \
  tests/unit/runtime/nexus/tools/test_tool_life_x_batch_c_stalled_abandoned_recovery.py -q
```

| Metric | Value |
| --- | --- |
| **Passed** | **50** |
| **Failed** | **0** |
| **Log** | `.tmp/session/TOOL-LIFE-X-BATCH-D/l2-batch-d.log` |

**Excluded (reason):** `test_t14_t17_builder_tools` sandbox authority env; U5 zero-bypass — not in Batch D delta (FRZ-BND-03 unchanged).

## ECAP / FRZ (scoped)

| ID | Finding |
| --- | --- |
| FRZ-BND-03 | No new invoker, registry, or MCP lifecycle manager |
| FRZ-REL-10 | SQL result bounds enforced at tool adapter |
| FRZ-REL-06 | HTTP/browser/MCP bounds — architecture scoped; live PASS **0** |
| ECAP-P4 | Global FRZ PASS **0** |

## Changed files (implementation)

- `intergrax/tools/providers/provider_resource_enforcement.py`
- `intergrax/tools/providers/provider_transport_capabilities.py`
- `intergrax/tools/mcp_result_bounds.py`
- `intergrax/tools/providers/database/contracts.py`, `service.py`
- `intergrax/tools/providers/http/contracts.py`, `service.py`
- `intergrax/tools/providers/browser/contracts.py`, `service.py`
- `intergrax/applications/_shared/mcp_catalog_tools.py`
- `tests/unit/tools/providers/test_tool_life_x_batch_d_provider_resource_lifecycle.py`
- `tests/unit/tools/providers/database/test_database_tools.py`

## Remaining before Batch E / parent L3

- Live Docker DB: query cancel, lock timeout, connection teardown evidence.
- HTTP fault service: lost ack, 429 under tool retry budget.
- MCP hung-server process qualification.
- Browser remote termination vs local close semantics.
- TL7 STALLED/heartbeat operator certification.

---

**TOOL-LIFE-X BATCH D — READY FOR INDEPENDENT AUDIT**

Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.
