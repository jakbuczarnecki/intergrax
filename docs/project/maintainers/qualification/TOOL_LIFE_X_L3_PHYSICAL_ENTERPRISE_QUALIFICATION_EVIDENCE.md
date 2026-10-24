# TOOL-LIFE-X L3-PHYSICAL — Enterprise Physical Lifecycle Qualification Evidence

| Field | Value |
| --- | --- |
| **Task** | `TOOL-LIFE-X-L3-PHYSICAL` |
| **Parent** | `TOOL-LIFE-X` (§3.0.5) — **not CLOSED** |
| **START_HEAD** | `b82e352336ccaa223060c5780b807dd1c6e6e1e1` |
| **IMPLEMENTATION BASELINE** | `712665150530286aeb4b3b6f43ad10957af8f040` |
| **Branch** | `development` |
| **Origin** | `https://github.com/jakbuczarnecki/intergrax.git` |
| **Operator authorization** | `OPERATOR APPROVED — TLX-L3-PHYSICAL` |
| **Production delta** | Server-side PostgreSQL bounded fetch; qualification fixture probe alignment; coexistent tenant isolation tests; adversarial bounded-fetch matrix |
| **Disposition** | **`TLX-L3-PHYSICAL — READY FOR INDEPENDENT AUDIT (CORRECTED PHYSICAL BOUNDS)`** |

**Non-closure:** Does **not** close `TOOL-LIFE-X` parent, **not** global `FRZ-*` PASS, **not** positive TL8 recovery, **not** `DIST-SCALE-X`, **not** TL15 closure.

---

## 1. Git identity gate

| Check | Result |
| --- | --- |
| Repository | `jakbuczarnecki/intergrax` |
| Branch | `development` |
| START_HEAD | `b82e352336ccaa223060c5780b807dd1c6e6e1e1` |
| Baseline ancestry | `712665150530286aeb4b3b6f43ad10957af8f040` ancestor of publication HEAD — **yes** |
| Worktree @ publication | clean after commit |

| SHA role | Value |
| --- | --- |
| **START_HEAD** | `b82e352336ccaa223060c5780b807dd1c6e6e1e1` |
| **Publication SHA** | Git `development` tip commit containing this evidence revision (independent audit SHA) |

---

## 2. PHYS-F01–F05 remediation closure matrix

| Finding | Canonical TL | Semantic owner | Contract | Prior behavior | Valid evidence SHA | Classification | Correction | Negative tests | L1/L2/L3 | FRZ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **PHYS-F01** | TL10 | `_PostgreSQLRelationalStore` | `RelationalStoreBoundedQueryPort` | Client `execute` + row iteration (full result risk) | `71266515…` (insufficient) | **IMPLEMENTATION REQUIRED → corrected** | `BEGIN READ ONLY` + named server-side cursor, `itersize=1` | row/byte/exact/empty/cancel | L1+L3 | FRZ-REL-03 scoped |
| **PHYS-F02** | TL10 | same | same | No transport-level proof | — | **VALIDATION REQUIRED → corrected** | Early-stop timing + named cursor guard test | slow `generate_series`+`pg_sleep`, byte before row | L3 | FRZ-REL-05 scoped |
| **PHYS-F03** | TL13 | qual fixture | env `INTERGRAX_POSTGRESQL_*` | Hardcoded DSN probe vs execution config | — | **IMPLEMENTATION REQUIRED → corrected** | Probe uses `resolve_tool_life_postgresql_config()`; `pytest.fail` not `skip` | unreachable host timeout | L3 | FRZ-BND-03 |
| **PHYS-F04** | TL12/P12 | `PostgreSQLConnectionProvider` | `search_path` tenant schema | Sequential tenants; weak isolation proof | — | **VALIDATION REQUIRED → scoped PASS + gap** | Dual-tenant coexistence; unqualified cross-tenant empty; **qualified cross-schema succeeds with shared DB role** | dual fixture | L3 | FRZ-TEN scoped gap |
| **PHYS-F05** | TL3 | `_PostgreSQLRelationalStore.cancel_backend` | adapter only | Adapter cancel proven; no ToolRuntime wire | `71266515…` partial | **VALIDATION REQUIRED (unchanged)** | Preserve `conn.cancel()`; **no** new cancellation authority | sleep + cancel; bounded-stream cancel | L3 | FRZ-REL-06 partial |

---

## 3. P01–P14 physical exit matrix (reconciled)

| ID | Criterion | Canonical owner | Production path | Evidence | Classification | Missing / notes |
| --- | --- | --- | --- | --- | --- | --- |
| **P01** | PG statement timeout, cancel, recovery | `PostgreSQLSession` / adapter | `database_query` → bounded port | @ publication | **scoped PASS** | `cancel_backend` on connection; **no** ToolRuntime→PG E2E |
| **P02** | Lock wait, rollback | `PostgreSQLConnectionProvider` | session | @ publication | **scoped PASS** | — |
| **P03** | Bounded rows/bytes | `RelationalStoreBoundedQueryPort` | server-side cursor fetch | @ publication | **scoped PASS** | `physical_bounds_enforced=True`; adversarial L3 |
| **P04** | Connect acquisition timeout | psycopg | `connect_timeout` | @ publication | **VALIDATION REQUIRED (partial)** | Bad host bound; no pool saturation |
| **P05** | MCP subprocess lifecycle | L3-R1/R2 | reused | historical | **ALREADY PROVEN (scoped)** | — |
| **P06** | MCP hung / cancel | MCP client | reused | historical | **VALIDATION REQUIRED (partial)** | — |
| **P07** | ToolRuntime catalog + deadline | `RuntimeToolInvoker` | MCP catalog | L3-R1 HTTP | **VALIDATION REQUIRED** | MCP catalog path not E2E |
| **P08–P10** | HTTP lifecycle | Batch E | reused | historical | **ALREADY PROVEN (scoped)** | — |
| **P11** | Cross-adapter cancel | `CancellationCoordinator` | W4-C | historical | **ALREADY PROVEN (scoped)** | Not live PG via invoker |
| **P12** | Tenant / Governance | `search_path` | PG qual | @ publication | **scoped PASS + explicit gap** | Unqualified tenant scoping; **not** DB ACL for qualified SQL |
| **P13** | Evidence @ SHA | this doc | — | @ publication | **PASS** | Independent GitHub audit mandatory |
| **P14** | Parent TL rollup | §5 | — | — | **PASS** | No TL8/TL15 closure |

---

## 4. TL3 / TL10 / TL11 / TL12 / TL13

| TL | Requirement | Owner | Classification | L3 evidence |
| --- | --- | --- | --- | --- |
| **TL3** | Cancellation propagation | Coordinator + adapters | **VALIDATION REQUIRED** | PG `cancel_backend` + cancel during bounded stream; **no** ToolRuntime SQL cancel E2E |
| **TL10** | DB limits / lifecycle | Bounded query port | **scoped PASS (PG)** | Server-side cursor + cleanup |
| **TL11** | MCP lifecycle | MCP host | **VALIDATION REQUIRED** | Reused L3-R1/R2 |
| **TL12** | HTTP/provider | AllowlistHttpClient | **ALREADY PROVEN (scoped)** | Batch E |
| **TL13** | Real-infra failures | Cross-provider | **VALIDATION REQUIRED (improved)** | PG qual fail-closed; Docker @ 5434 |

---

## 5. TL1–TL15 parent rollup

| TL | Classification |
| --- | --- |
| TL1–TL2 | **ALREADY PROVEN** (Batch A) |
| TL3 | **VALIDATION REQUIRED** (adapter cancel only) |
| TL4–TL6 | **ALREADY PROVEN** (Batch B + L3-R1 HTTP) |
| TL7 | **OPERATOR DECISION REQUIRED** |
| TL8 | **BLOCKED** |
| TL9 | **ALREADY PROVEN** (Batch C + W4-C) |
| TL10 | **scoped PASS (PG physical bounds corrected)** |
| TL11 | **VALIDATION REQUIRED** |
| TL12 | **ALREADY PROVEN (scoped)** + tenant **gap** for qualified SQL |
| TL13 | **VALIDATION REQUIRED (improved)** |
| TL14 | **ALREADY PROVEN** |
| TL15 | **BLOCKED** |

---

## 6. PostgreSQL streaming / fetch proof

| Mechanism | Detail |
| --- | --- |
| Transaction | `BEGIN READ ONLY` per bounded fetch |
| Cursor | Named server-side (`igx_bq_*`), `itersize=1` |
| Truncation | `accumulate_bounded_sql_rows` row/byte ceilings |
| Cleanup | `cursor.close()` in `finally`; `commit` on success / `rollback` on error |
| Proof tests | `test_pg_bounded_query_stops_remote_work_early`; `test_pg_bounded_fetch_uses_named_server_side_cursor`; L1 `test_postgresql_bounded_fetch.py` |

**Not claimed:** arbitrary SQL rewritten with `LIMIT`; full client buffer before loop.

---

## 7. Tenant isolation proof

| Check | Result |
| --- | --- |
| Coexistent schemas A + B | **PASS** (`tool_life_postgresql_dual_tenant_stores`) |
| A reads own rows | **PASS** |
| B reads own rows | **PASS** |
| Unqualified cross-tenant row (id swap) | **PASS** (empty) |
| Qualified `schema_b.table` from A session | **SUCCEEDS** — shared qualification role |
| Security model | **`search_path` session convention**; **not** privilege-enforced multi-tenant ACL for arbitrary qualified SQL |

---

## 8. Cancellation reachability

| Path | Status |
| --- | --- |
| `store.cancel_backend()` → `conn.cancel()` | **PASS** (L3) |
| Cancel during bounded streamed fetch | **PASS** |
| `CancellationCoordinator` → `database_query` → PG cancel | **NOT WIRED** — no production E2E claim |
| Logical timeout vs backend still running | **VALIDATION REQUIRED** (statement_timeout separate) |

---

## 9. Adversarial matrix (L3-PHYSICAL)

| Case | Status |
| --- | --- |
| Large result >> max_rows | **PASS** |
| Single row > max_result_bytes | **PASS** (raises) |
| Byte limit before row limit | **PASS** |
| Exact row boundary | **PASS** |
| Empty result | **PASS** |
| Unicode/multibyte (`ż`) | **PASS** |
| Cancel during streamed fetch | **PASS** |
| Lock timeout / rollback | **PASS** |
| SQL error rollback | **PASS** |
| Dual-tenant coexistence | **PASS** |
| Named server-side cursor guard | **PASS** |

---

## 10. Validation results

| Layer | Command | Result | Log |
| --- | --- | --- | --- |
| **L1** | `uv run pytest tests/unit/integrations/providers/relational_store/test_postgresql_bounded_fetch.py tests/unit/integrations/providers/relational_store/test_postgresql_session.py -q` | **13 passed** | `.tmp/session/TOOL-LIFE-X-L3-PHYSICAL/l1-pytest.log` |
| **L2** | `uv run pytest` L3-PHYSICAL + `test_postgresql*.py` + `test_database_tools.py` + `test_tool_life_x_batch_d_provider_resource_lifecycle.py` | **69 passed** | `.tmp/session/TOOL-LIFE-X-L3-PHYSICAL/l2-consolidated-pytest.log` |
| **L3** | `uv run pytest tests/unit/tools/providers/test_tool_life_x_l3_physical_postgresql_qualification.py -q` | **included in L2 (15 tests)** | `.tmp/session/TOOL-LIFE-X-L3-PHYSICAL/l3-pg-pytest.log` |

Docker: `intergrax-postgresql` healthy on host port **5434**.

---

## 11. FRZ / ECAP (scoped)

| ID | Disposition |
| --- | --- |
| FRZ-BND-03 | **scoped PASS** — sanctioned adapter boundary; no shadow cancel manager |
| FRZ-REL-03/05 | **improved** — physical PG bounds |
| FRZ-REL-06 | **partial** — adapter cancel only |
| FRZ-TEN | **gap documented** — qualified SQL + shared role |
| FRZ-SEC / FRZ-GOV | **unchanged** — governance HTTP reused from L3-R1 |
| Global FRZ | **OPEN** |

---

## 12. Remaining blockers

1. **TL11 / P07** — ToolRuntime MCP catalog E2E with deadline.
2. **TL3 / PHYS-F05** — wire `CancellationCoordinator` to SQL `cancel_backend` if product requires full propagation.
3. **P12 / tenant** — DB privilege or SQL admission for qualified cross-schema reads (architecture decision).
4. **TL8** — positive owner-loss recovery **BLOCKED**.
5. **P04** — pool saturation under load.
6. **P01** — TCP mid-query drop.

---

## 13. Resource cleanup

| Action | Result |
| --- | --- |
| Container | `intergrax-postgresql` **left running** |
| Ephemeral schemas | `DROP SCHEMA` per fixture |
| Volumes | **Preserved** |
| Logs | `.tmp/session/TOOL-LIFE-X-L3-PHYSICAL/` |

---

## 14. Parent readiness

**`TLX-L3-PHYSICAL — READY FOR INDEPENDENT AUDIT (CORRECTED PHYSICAL BOUNDS)`**

**`TOOL-LIFE-X` — IN PROGRESS / BLOCKED** — mandatory independent audit at exact GitHub SHA.
