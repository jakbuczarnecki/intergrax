# TLX-TL11-PRODUCTION-CATALOG-PHYSICAL-QUALIFICATION — Evidence

| Field | Value |
| --- | --- |
| **Task** | `TLX-TL11-PRODUCTION-CATALOG-PHYSICAL-QUALIFICATION` |
| **Parent** | `TOOL-LIFE-X` §3.0.5 — **IN PROGRESS / BLOCKED** |
| **Predecessor** | `TLX-TL11-CATALOG-E2E-HUNG-SERVER` — physical wiring gap (duplicate qual mount) |
| **Repository** | `jakbuczarnecki/intergrax` |
| **Branch** | `development` |
| **EXPECTED_START_HEAD** | `f73c6e4660c9b4ff948ffd578c832bdba54197c2` |
| **Disposition** | **`TLX-TL11-PRODUCTION-CATALOG-PHYSICAL-QUALIFICATION — READY FOR INDEPENDENT AUDIT`** (scoped) |

**Non-closure:** No `TOOL-LIFE-X` parent CLOSED; no global `FRZ-*` PASS; TL8/TL7/CANCEL-C/TRUST-C blockers unchanged.

**Ledger:** Prior hung-server batch listed **Publication SHA** `bdcdf05fc8332ab5a1bd8388d8feaa9a054bbeea` on an earlier implementation line — **superseded for PC-01/PC-02** by this batch; canonical **GitHub publication** for START is `f73c6e4660c9b4ff948ffd578c832bdba54197c2` (implementation/test SHA updated in §1 after commit).

---

## 1. Git identity

| Check | Result |
| --- | --- |
| Remote | `origin` → `https://github.com/jakbuczarnecki/intergrax.git` |
| Branch | `development` |
| START_HEAD | `f73c6e4660c9b4ff948ffd578c832bdba54197c2` |
| Clean worktree @ START | **yes** |
| Ancestry | `f73c6e46…` ancestor of batch commit |

| SHA role | Value |
| --- | --- |
| **START_HEAD** | `f73c6e4660c9b4ff948ffd578c832bdba54197c2` |
| **Implementation / test SHA** | `c6bda9b2e4f0b86efb42ecf63cf1a59832247f9f` |
| **Publication SHA** | `fdb8c21b3f4bf5bfbbfe17d46c578a0fdd5599a8` (docs SHA pin only; no code delta) |

**Reused evidence (unchanged SHA, still valid):** MCP-FINAL discovery @ `adef74d2…`; L3-R2 MCP host lifecycle @ `8b33d9a…`; TRUST-A / CANCEL-A / OWN-A2 qualification records on `development` line; `describe_catalog_tool` `registry.has` fix in MCP-FINAL batch.

---

## 2. Canonical production catalog mount graph

```text
build_nexus_mcp_server(..., tool_registry=ToolRegistry)
  └─ mount_catalog_tools_on_mcp(mcp, tool_registry)   # intergrax/applications/_shared/mcp_catalog_tools.py
       ├─ list_catalog_tools → to_mcp_tools(registry) → emit_bounded_mcp_list
       └─ describe_catalog_tool → registry.has/get → to_mcp_tools(contract) → bound_mcp_structured_result
```

| Edge | Owner | Contract | Composition root | Tenant/principal |
| --- | --- | --- | --- | --- |
| Registry → exporter | Tier-0 `ToolRegistry` / `to_mcp_tools` | `RegisteredTool` / `ToolContract` | Host `tool_wiring.registry` | N/A on discovery |
| Exporter → MCP tools | `mcp_catalog_tools` | bounded list/dict | `build_nexus_mcp_server` | N/A |
| HTTP ingress | `couple_fastapi_with_mcp` + `HarnessAuthState` | API key middleware | Tier-3 FastAPI app | harness principal tenant on auth path |
| Client | MCP streamable HTTP | `ClientSession.call_tool` | test subprocess | optional `X-Api-Key` |

**Production vs prior fixture:** Removed `_mount_qual_catalog_tools` duplicate handlers. Physical server now passes `tool_registry` into `build_nexus_mcp_server`, exercising **`mount_catalog_tools_on_mcp`** unchanged. Fault injection: `_Tl11FaultInjectingToolRegistry.list()` / `.get()` delays or `os._exit` **before** delegating to `ToolRegistry` (canonical read boundary for `to_mcp_tools` / describe path).

---

## 3. PC-01–PC-22 closure matrix

| ID | Canonical owner | Contract | Evidence SHA | Classification | Missing proof | Negative test | L | FRZ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PC-01 | `mcp_nexus_server` | `mount_catalog_tools_on_mcp` | **This batch** L3 | **VALIDATION REQUIRED→PASS** | — | — | L3 | FRZ-BND-03 |
| PC-02 | same | no duplicate mount | **This batch** | **PASS** | — | — | L3 | FRZ-BND-03 |
| PC-03 | `ToolRegistry` + exporter | `to_mcp_tools` / `has`/`get` | **This batch** + MCP-FINAL | **PASS** | — | — | L2/L3 | — |
| PC-04 | operator Gate B | P07 N/A discovery | MCP-FINAL | **ALREADY PROVEN** | — | handler `AssertionError` | L2 | FRZ-BND-03 |
| PC-05 | `mcp_catalog_tools` | list + describe | **This batch** L3 | **PASS** | — | — | L3 | FRZ-REL-06 |
| PC-06 | `describe_catalog_tool` | `unknown_tool:*` | MCP-FINAL L2 + **L3** `test_tl11_e2e_unknown_tool_id_bounded_error` | **PASS** | — | missing id | L2/L3 | — |
| PC-07 | `mcp_result_bounds` | byte ceilings | MCP-FINAL L2 | **ALREADY PROVEN** | — | huge payload | L2 | FRZ-REL-06 |
| PC-08 | harness auth | `X-Api-Key` | **This batch** L3 | **PASS** | — | — | L3 | FRZ-SEC |
| PC-09 | harness auth | fail-closed | **This batch** L3 | **PASS** | missing/invalid key | 401 class | L3 | FRZ-SEC |
| PC-10 | TRUST-A / MCP intake | tenant unchanged | prior TRUST-A | **ALREADY PROVEN** | — | — | L2 | FRZ-TEN scoped |
| PC-11 | client `asyncio.wait_for` | hung registry read | **This batch** (fault at `.list()`/`.get()`) | **PASS** | — | hang_list/describe | L3 | FRZ-REL |
| PC-12 | client semantics | timeout ≠ kill | **This batch** | **PASS** | — | server alive until terminate | L3 | FRZ-GOV |
| PC-13 | subprocess | crash on read | **This batch** | **PASS** | — | crash_list/describe | L3 | FRZ-REL |
| PC-14 | asyncio cancel | not Execution cancel | **This batch** | **PASS** | — | `CancelledError` | L3 | FRZ-REL scoped |
| PC-15 | fresh subprocess | new session | **This batch** | **PASS** | — | no replay | L3 | FRZ-REL |
| PC-16 | fixture | terminate/kill | **This batch** | **PASS** | — | — | L3 | FRZ-REL |
| PC-17 | catalog read-only | parallel list/describe | **This batch** | **PASS** | — | — | L3 | — |
| PC-18 | `mcp_result_bounds` | limits | MCP-FINAL L2 | **ALREADY PROVEN** | — | — | L2 | FRZ-REL-06 |
| PC-19 | platform UNKNOWN | no blind retry | roadmap + Batch A | **ALREADY PROVEN** | — | — | L2 | FRZ-GOV |
| PC-20 | freeze checklist | scoped ECAP | this record | **VALIDATION REQUIRED→PASS** | independent audit | — | — | FRZ-* scoped |
| PC-21 | predecessors | reuse | §1 | **ALREADY PROVEN** | — | — | — | — |
| PC-22 | operator | exact-SHA audit | — | **OPERATOR DECISION REQUIRED** | pending | — | — | — |

---

## 4. CE-01–CE-28 reconciliation (no auto-promotion)

| ID | Prior class | This batch |
| --- | --- | --- |
| CE-01–CE-04 | ALREADY PROVEN / N/A | **Unchanged** — L2 discovery |
| CE-05–CE-06 | PASS (qual duplicate mount) | **Re-qualified** — production mount L3 |
| CE-07–CE-14 | PASS | **Re-qualified** — fault at registry boundary, production handlers |
| CE-15 | PASS | **Extended** — invalid API key L3 |
| CE-16–CE-23 | ALREADY PROVEN / PASS | **Unchanged** (L2 bounds/auth semantics) |
| CE-24–CE-26 | PASS | **Unchanged** |
| CE-27 | N/A — WITH EVIDENCE | **Unchanged** |
| CE-28 | OPERATOR DECISION | **Unchanged** — mandatory independent audit |

---

## 5. Physical E2E matrix (mandatory scenarios)

| # | Scenario | Physical mechanism | Production code exercised | Owner | Expected | Forbidden | Test | FRZ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Normal list | subprocess ok | `mount_catalog_tools_on_mcp` → list | catalog mount | success | handler invoke | `test_tl11_e2e_catalog_normal_list_and_describe_subprocess` | FRZ-REL-06 |
| 2 | Normal describe | subprocess ok | describe path | catalog mount | schema | handler invoke | same | — |
| 3 | Unknown tool | subprocess ok | `registry.has` | `mcp_catalog_tools` | `unknown_tool:*` | KeyError | `test_tl11_e2e_unknown_tool_id_bounded_error` | — |
| 4 | Large / bounds | L2 patch tight limit | `emit_bounded_mcp_list` | bounds | limit error | unbounded | `test_discovery_oversized_*` (L2) | FRZ-REL-06 |
| 5 | No business handler | qual handler raises | discovery only | registry | success | `AssertionError` | L2 `test_discovery_no_direct_*` + qual handler | FRZ-BND-03 |
| 6 | Auth success | harness + key | HTTP ingress | harness | success | anonymous | `test_tl11_e2e_http_ingress_accepts_authenticated_*` | FRZ-SEC |
| 7 | Missing key | no header | middleware | harness | error | 200 | `test_tl11_e2e_http_ingress_rejects_missing_api_key` | FRZ-SEC |
| 8 | Invalid key | wrong key | middleware | harness | error | 200 | `test_tl11_e2e_http_ingress_rejects_invalid_api_key` | FRZ-SEC |
| 9 | Concurrent | asyncio.gather | parallel MCP calls | transport | both OK | cross-talk | `test_tl11_e2e_concurrent_catalog_discovery_isolated` | — |
| 10 | Hung list | registry `.list()` sleep | production list handler | client | `TimeoutError` | provider killed claim | `test_tl11_e2e_hung_list_catalog_client_timeout_not_definite_failure` | FRZ-REL |
| 11 | Slow list | registry `.list()` 2.5s | production mount | client | success | — | `test_tl11_e2e_slow_list_within_deadline` | — |
| 12 | Client timeout | short `wait_for` | production mount | client | `TimeoutError` | — | hung/slow tests | FRZ-GOV |
| 13 | Client cancel | task.cancel | production mount | client | `CancelledError` | Execution cancel | `test_tl11_e2e_client_cancel_during_hung_catalog_list` | FRZ-REL scoped |
| 14 | Server crash list | `os._exit` in `.list()` | production mount | transport | error | — | `test_tl11_e2e_server_crash_mid_catalog_list` | FRZ-REL |
| 15 | Reconnect | new subprocess | fresh session | client | success | replay | `test_tl11_e2e_reconnect_after_catalog_timeout_fresh_session` | FRZ-REL |
| 16 | No implicit replay | timeout then new server | — | client | new success only | auto-retry | reconnect test | FRZ-GOV |
| 17 | Cleanup | fixture finally | subprocess | fixture | proc ended | zombie | `test_tl11_e2e_subprocess_cleanup_after_hung_catalog` | FRZ-REL |
| 18 | Unavailable | exit 2 | early main | fixture | code 2 | hang | `test_tl11_e2e_mcp_server_unavailable_exit_code` | — |
| 19 | Never ready | sleep in main | startup waiter | fixture | startup timeout | indefinite | `test_tl11_e2e_no_indefinite_wait_on_never_ready_startup` | FRZ-REL |
| 20 | Crash describe | `os._exit` in `.get()` | describe path | transport | error | — | `test_tl11_e2e_server_crash_mid_catalog_describe` | FRZ-REL |
| 21 | Hung describe | `.get()` sleep | describe path | client | `TimeoutError` | — | `test_tl11_e2e_hung_describe_catalog_client_timeout` | FRZ-REL |

---

## 6. L1 / L2 / L3 validation

| Tier | Trigger | Justification |
| --- | --- | --- |
| L1 | **Skipped** | No production authority/tenant/lifecycle code change — fixtures + tests + evidence only |

| Tier | Command | Result | Log |
| --- | --- | --- | --- |
| L2/L3 consolidated | `uv run pytest tests/unit/applications/test_mcp_catalog_e2e_hung_server_qualification.py tests/unit/applications/test_mcp_catalog_discovery_qualification.py -q` | **31 passed** | `.tmp/session/tlx-tl11-production-catalog-physical/l2-l3-pytest.log` |

---

## 7. FRZ / ECAP (scoped, Path B)

| Ref | Disposition |
| --- | --- |
| FRZ-BND-03 | Single host graph; discovery-only catalog; production `mount_catalog_tools_on_mcp` on L3 |
| FRZ-REL-06 | Bounded payloads (L2) + transport timeouts (L3) |
| FRZ-SEC | Harness ingress fail-closed missing/invalid API key (L3) |
| FRZ-TEN | Unchanged; MCP principal bridge gaps remain from TRUST-A |
| FRZ-GOV | Client timeout/cancel does not assert Tool execution outcome |
| ECAP | Hung catalog client boundary documented; no Execution cancellation implied |

**Not claimed:** global `FRZ-*` PASS; live-scoped `FRZ-PRD-*`.

---

## 8. Changed artifacts

| Path | Change |
| --- | --- |
| `tests/fixtures/tool_life_x_tl11_nexus_catalog_qualification_server.py` | Production mount via `tool_registry=`; fault injection on registry reads |
| `tests/unit/applications/test_mcp_catalog_e2e_hung_server_qualification.py` | L3 unknown tool + invalid API key |
| `docs/project/maintainers/qualification/TOOL_LIFE_X_TLX_TL11_PRODUCTION_CATALOG_PHYSICAL_QUALIFICATION_EVIDENCE.md` | This record |
| `docs/project/maintainers/qualification/TOOL_LIFE_X_TLX_TL11_CATALOG_E2E_HUNG_SERVER_QUALIFICATION_EVIDENCE.md` | Cross-reference / gap note |

---

## 9. Remaining blockers

- PC-22 / CE-28 independent exact-SHA audit (mandatory).
- TL8 positive recovery; TL7 heartbeat (`TOOL-LIFE-X` parent).
- CANCEL-C / MCP execution deadline on `run_agent` — out of scope.
- TRUST-C MCP principal forwarding — operator decision.
- L3-R5-B ratification; parent L3 formal closure.

---

## 10. Next mandatory stage (roadmap §3.0.5 authority)

Independent audit of this publication SHA; then **TL8** / **TL7** blockers and **L3-R5-B** ratification before `DIST-SCALE-X` entry.
