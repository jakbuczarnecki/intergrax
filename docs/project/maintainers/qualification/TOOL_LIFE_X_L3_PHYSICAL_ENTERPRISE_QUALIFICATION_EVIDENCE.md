# TOOL-LIFE-X L3-PHYSICAL — Enterprise Physical Lifecycle Qualification Evidence

| Field | Value |
| --- | --- |
| **Task** | `TOOL-LIFE-X-L3-PHYSICAL` |
| **Parent** | `TOOL-LIFE-X` (§3.0.5) — **not CLOSED** |
| **START_HEAD** | `7018f019732a9d44d14b0129ad95e9eccf9d0948` |
| **IMPLEMENTATION BASELINE** | `7018f019732a9d44d14b0129ad95e9eccf9d0948` |
| **Branch** | `development` |
| **Origin** | `https://github.com/jakbuczarnecki/intergrax.git` |
| **Operator authorization** | `OPERATOR APPROVED — TLX-L3-PHYSICAL` |
| **Production delta** | PHYS-F07 connection-state gate (fail-closed UNKNOWN/INERROR/ACTIVE/closed); PHYS-F08 cursor-close-before-commit ordering; adversarial L1/L3 matrix; prior F06 INTRANS scope preserved |
| **Disposition** | **`TLX-L3-PHYSICAL — READY FOR INDEPENDENT AUDIT (PG TRANSACTION/CURSOR SAFETY)`** |

**Non-closure:** Does **not** close `TOOL-LIFE-X` parent, **not** global `FRZ-*` PASS, **not** positive TL8 recovery, **not** `DIST-SCALE-X`, **not** TL15 closure.

---

## 1. Git identity gate

| Check | Result |
| --- | --- |
| Repository | `jakbuczarnecki/intergrax` |
| Branch | `development` |
| START_HEAD | `7018f019732a9d44d14b0129ad95e9eccf9d0948` |
| Baseline ancestry | `7018f019732a9d44d14b0129ad95e9eccf9d0948` ancestor of publication HEAD — **yes** |
| Worktree @ publication | clean after commit |

| SHA role | Value |
| --- | --- |
| **START_HEAD** | `7018f019732a9d44d14b0129ad95e9eccf9d0948` |
| **Tested SHA** | `d9a5d282c8f45b4bdcca2cff0bb145f6440ca50f` (L1/L2/L3 on implementation tree) |
| **Implementation SHA** | `d9a5d282c8f45b4bdcca2cff0bb145f6440ca50f` (PHYS-F07/F08 batch) |
| **Publication SHA** | `d4147579da0f9fb7554f13657cf72fba7a08b494` (`development` tip — independent audit SHA) |

---

## 2. PHYS-F01–F08 remediation closure matrix

| Finding | Canonical TL | Semantic owner | Contract | Prior behavior | Valid evidence SHA | Classification | Correction | Negative tests | L1/L2/L3 | FRZ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **PHYS-F01** | TL10 | `_PostgreSQLRelationalStore` | `RelationalStoreBoundedQueryPort` | Client `execute` + row iteration (full result risk) | `71266515…` (insufficient) | **IMPLEMENTATION REQUIRED → corrected** | `BEGIN READ ONLY` + named server-side cursor, `itersize=1` | row/byte/exact/empty/cancel | L1+L3 | FRZ-REL-03 scoped |
| **PHYS-F02** | TL10 | same | same | No transport-level proof | — | **VALIDATION REQUIRED → corrected** | Early-stop timing + named cursor guard test | slow `generate_series`+`pg_sleep`, byte before row | L3 | FRZ-REL-05 scoped |
| **PHYS-F03** | TL13 | qual fixture | env `INTERGRAX_POSTGRESQL_*` | Hardcoded DSN probe vs execution config | — | **IMPLEMENTATION REQUIRED → corrected** | Probe uses `resolve_tool_life_postgresql_config()`; `pytest.fail` not `skip` | unreachable host timeout | L3 | FRZ-BND-03 |
| **PHYS-F04** | TL12/P12 | `PostgreSQLConnectionProvider` | `search_path` tenant schema | Sequential tenants; weak isolation proof | — | **VALIDATION REQUIRED → scoped PASS + gap** | Dual-tenant coexistence; unqualified cross-tenant empty; **qualified cross-schema succeeds with shared DB role** | dual fixture | L3 | FRZ-TEN scoped gap |
| **PHYS-F05** | TL3 | `_PostgreSQLRelationalStore.cancel_backend` | adapter only | Adapter cancel proven; no ToolRuntime wire | `71266515…` partial | **VALIDATION REQUIRED (unchanged)** | Preserve `conn.cancel()`; **no** new cancellation authority | sleep + cancel; bounded-stream cancel | L3 | FRZ-REL-06 partial |
| **PHYS-F06** | TL10 | `_PostgreSQLRelationalStore` + `PostgreSQLConnectionProvider` | bounded port + session ownership | Unconditional `BEGIN READ ONLY` / `commit` / `rollback` on shared connection | `dd23e5751` (INTRANS scope) | **corrected (scoped)** | Caller INTRANS: named cursor only; adapter commit/rollback only on owned RO | L1+L3 caller write/RO | L1+L3 | FRZ-BND-03, FRZ-REL-03 |
| **PHYS-F07** | TL10 | `classify_postgresql_bounded_fetch_transaction_scope` | connection state gate | Boolean gate treated UNKNOWN as IDLE | audit disposition | **IMPLEMENTATION REQUIRED → corrected** | Typed scope + `PostgreSQLBoundedFetchSafetyError`; IDLE/INTRANS only; UNKNOWN fail-closed | INERROR/ACTIVE/UNKNOWN/closed | L1+L3 | FRZ-BND-03 |
| **PHYS-F08** | TL10 | `fetch_query_bounded` | cursor vs transaction lifetime | `commit` before `finally` cursor close | audit disposition | **IMPLEMENTATION REQUIRED → corrected** | Close server cursor before adapter `commit`; primary fetch error over close error; owned rollback on failure | close/commit/rollback error paths | L1+L3 | FRZ-REL-03 |

### Transaction status truth table (psycopg `pq.TransactionStatus`)

| Status | Bounded fetch | Adapter BEGIN RO | Adapter commit/rollback |
| --- | --- | --- | --- |
| **IDLE** | Allowed (adapter-owned) | Yes | On success/failure of owned path only |
| **INTRANS** | Allowed (caller-owned) | No | **Never** |
| **INERROR** | **Rejected** (`PostgreSQLBoundedFetchSafetyError`) | No | No |
| **ACTIVE** | **Rejected** | No | No |
| **UNKNOWN** | **Rejected** (fail-closed) | No | No |
| **closed** | **Rejected** | No | No |

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
| **TL10** | DB limits / lifecycle | Bounded query port | **scoped PASS (PG)** | Server-side cursor + tx ownership |
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
| TL10 | **scoped PASS (PG physical bounds + transaction ownership corrected)** |
| TL11 | **VALIDATION REQUIRED** |
| TL12 | **ALREADY PROVEN (scoped)** + tenant **gap** for qualified SQL |
| TL13 | **VALIDATION REQUIRED (improved)** |
| TL14 | **ALREADY PROVEN** |
| TL15 | **BLOCKED** |

---

## 6. PostgreSQL streaming / fetch + transaction ownership

| Mechanism | Detail |
| --- | --- |
| Ownership gate | `classify_postgresql_bounded_fetch_transaction_scope` → `ADAPTER_OWNED_READONLY` or `CALLER_OWNED` |
| Idle connection (A) | `BEGIN READ ONLY` → named cursor → fetch → **`cursor.close()`** → `commit`; owned failure → `rollback` |
| Caller transaction (B–D) | Named server-side cursor only — **no** `BEGIN`, **no** `commit`, **no** `rollback` |
| Cursor | `igx_bq_*`, `itersize=1` |
| Truncation | `accumulate_bounded_sql_rows` row/byte ceilings |
| Cleanup | Close cursor before adapter-owned `commit`; fetch error preserved over secondary close error |
| Proof tests | L1 `test_postgresql_bounded_fetch.py` (idle + INTRANS + rollback ownership); L3 `test_pg_bounded_fetch_does_not_commit_caller_owned_write_transaction`; `test_pg_bounded_fetch_inside_caller_read_only_transaction` |

**Invariant:** bounded read never commits or rolls back another component's transaction without adapter-owned READ ONLY scope.

**Not claimed:** arbitrary SQL rewritten with `LIMIT`; full client buffer before loop; nested `BEGIN READ ONLY` inside caller transactions.

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
| FRZ-TEN full PASS | **`OPERATOR DECISION REQUIRED — POSTGRESQL QUALIFIED-SCHEMA TENANT ISOLATION`** |

---

## 8. Cancellation reachability

| Path | Status |
| --- | --- |
| `store.cancel_backend()` → `conn.cancel()` | **PASS** (L3) |
| Cancel during bounded streamed fetch | **PASS** |
| `CancellationCoordinator` → `database_query` → PG cancel | **NOT WIRED** — no production E2E claim |
| Logical timeout vs backend still running | **VALIDATION REQUIRED** (statement_timeout separate) |

---

## 9. Adversarial matrix (L1 + L3-PHYSICAL)

| # | Invariant | Owner | Expected | Test layer | Status |
| --- | --- | --- | --- | --- | --- |
| 1 | Idle → bounded read | adapter | owned RO tx + commit | L1+L3 | **PASS** |
| 2 | Caller uncommitted write → bounded read | adapter | no commit; rollback still clears write | L3 | **PASS** |
| 3 | Caller READ ONLY tx → bounded read | adapter | no nested BEGIN | L3 | **PASS** |
| 4–5 | Bounded error/success must not roll back/commit unrelated writes | adapter | INTRANS: no adapter rollback/commit | L1 | **PASS** |
| 6–7 | Cursor closed **before** commit on owned path | adapter | L1 order + close/commit error paths | L1 | **PASS** |
| 7b | INERROR / ACTIVE rejected | session gate | typed safety error | L1+L3 | **PASS** |
| 7c | UNKNOWN fail-closed | session gate | no `BEGIN READ ONLY` | L1 | **PASS** |
| 8–9 | Cancel during / before close | adapter | `QueryCanceled` | L3 | **PASS** |
| 10 | Timeout during fetch | session | `statement_timeout` | L3 | **PASS** |
| 11–16 | Row/byte/empty/unicode/exact/large | bounded port | truncation semantics | L3 | **PASS** |
| 17–19 | Error / recovery / next request | session | manual rollback patterns | L3 | **PASS** |
| 20 | Connection ownership transfer | adapter | `release_connection_*` unchanged | historical | **PASS** |
| 21 | Concurrent same connection | — | not supported (single conn store) | — | **explicit** |
| 22–23 | Tenant `search_path` / SQL governance | provider | unchanged context | L3 | **PASS** |
| 24–27 | UNKNOWN retry / open tx / cursor leak / schema | platform | no regression | L2 | **PASS** |
| 28 | Docker PG unavailable | fixture | `pytest.fail` not skip | fixture | **PASS** |

---

## 10. Validation results

| Layer | Command | Result | Log |
| --- | --- | --- | --- |
| **L1** | `uv run pytest tests/unit/integrations/providers/relational_store/test_postgresql_bounded_fetch.py -q` | **18 passed**, exit 0 | `.tmp/session/TOOL-LIFE-X-L3-PHYSICAL/l1-pytest.log` |
| **L2** | bounded fetch + `test_postgresql_session.py` + `test_postgresql.py` + database tools + Batch D lifecycle | **71 passed**, exit 0 | `.tmp/session/TOOL-LIFE-X-L3-PHYSICAL/l2-consolidated-pytest.log` |
| **L3** | `uv run pytest tests/unit/tools/providers/test_tool_life_x_l3_physical_postgresql_qualification.py -q` @ 5434 | **19 passed**, exit 0 | `.tmp/session/TOOL-LIFE-X-L3-PHYSICAL/l3-pytest.log` |

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

**`TLX-L3-PHYSICAL — READY FOR INDEPENDENT AUDIT (PG TRANSACTION/CURSOR SAFETY)`**

**`TOOL-LIFE-X` — IN PROGRESS / BLOCKED** — mandatory independent audit at exact GitHub SHA.
