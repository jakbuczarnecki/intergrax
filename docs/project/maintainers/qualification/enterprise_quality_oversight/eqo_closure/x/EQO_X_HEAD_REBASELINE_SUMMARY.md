# EQO-X Cross-Cutting Governance & Assurance HEAD Re-baseline Summary (`646e63953`)

- **Published under:** `governance/tickets/T-IGX-090-B.md` (`OP-EC-01`, `2026-11-26T17:26:01+01:00`, `OD-W2-DOCS=A`, `OD-EQO-09=A`)
- **Evaluated Analysis Baseline:** `development` @ `646e63953ae1a7eb28a229322590c9cdcbb96625` (ancestor of branch base `32005c212ed4debca91f607ee30b3abfab79ce6c`)
- **Source Analysis:** Wave 1 read-only re-evaluation (`G1-E` `EQO_X01_TO_X08_HEAD_REPORT.md` & `KR28_HEAD_SETTLEMENT_MAP.json`; zero pytest runs)
- **Strict Non-Claim Banner:** Re-baseline governance summary only — **NOT** `G2`, **NOT** `R`/`RB-2`, **NOT** `FCS`, **NOT** `M3`, **NO** FRZ promotion, **NO** `C01-C25 FULLY_MET`, **NO** `G6 PASS` / `IIAC PASS`, and **NO** `EQO_CLOSED` (`EQO_NOT_CLOSED_EXACT_BLOCKERS` preserved).

---

## 1. Executive Summary (`EQO-X-01..X-08`)

| Task | Scope / Anchor Findings | Stale EQO State (`2026-11-21`) | Verified HEAD State (`646e63953`) | Closure Owner (`AF-CLOSE` / `EQO`) | Pre-`G2` Action Required? |
| --- | --- | --- | --- | --- | --- |
| **`EQO-X-01`** | `EQOA-A4-003` (**BLOCKER**), `EQOA-A4-004` (`HIGH`) | `MERGED_TO_TRAIN` / `EXACT_BLOCKER` | **`FIXED_IN_CODE_PENDING_CI_AND_AUDIT`** (`check_intergrax_no_applications_imports.py` `rc=0`; `check_agent_registry_bypass.py` `rc=0` & in CI; `check_intergrax_no_applications_imports.py` **NOT** wired in `.github/workflows/unit-tests.yml` and `test_faudit_remediation.py` is `unit` only) | Pre-`G2`: `WL-K1-B-04`, `WL-K1-E-07`, `WL-K3-05`/`10` (`PT-S06`, `OD-A4-003-CI=A`), `WL-K3-06` (`PT-G01`) · Post-`G2`: `PT-Q13` / `T-B7-ECAP-2C` | **YES** (`.github/` & `scripts/` freeze at `G2` per `AD-AF-DEVTRAIN` `D-1`) |
| **`EQO-X-02`** | `EQOA-A2-002` (**BLOCKER**), `A2-001` (`HIGH`), `A1-017` (`MEDIUM`) | `BLOCKED` / `OWNER_DECISION` | **`STILL_OPEN_ON_HEAD`** (Roadmap L499 independence unmet for `SKILL-X` conditional/O-5, `TOOL-LIFE-X` `AUD-TOOLLIFE` STOP, `DIST-SCALE-X` operator-scoped, `RRX`, `EBH-7`; `FIND-AF2-03` OPEN; `RM499_ENTRY_MAP.md` 41/46 OPEN) | Post-`G2`: `PT-D03`, `PT-D09` (`DP-42`), `PT-AUD-DOC`, `PT-Q05`, `PT-Q08`, `PT-Q09`, `PT-Q11`, `PT-Q13`, `T-AF2-REQUAL-HIST-PASS` | No product edit; confirm `RM499` Observation O-5 (`SKILL-X`) before W6 |
| **`EQO-X-03`** | `EQOA-A0-007` (`HIGH`), `ACT-N11`, `RB-2` | `BLOCKED until RB-2` | **`ACT_N11_DRIFT_88_FILES_NO_PREMATURE_FLIP`** (`verify_iiac_policy_activation.py` `rc=1`, exactly **88** `ACT-N11` product paths since `82bbaacc8`, **0** non-ACT-N11 errors, **0** premature edits to record or material governance paths) | Pre-`G2`: `WL-K3-04` (tooling readiness only) · Strictly post-`G2`: `PT-Q02` -> **`PT-Q03` (`RB-2`)** -> `FCS` | **NO rebind before `G2`** (tooling readiness check only in `WL-K3-04`) |
| **`EQO-X-04`** | `EQOA-A5-007` (`HIGH`), `ROADMAP-REPLAY-X` counters | `2`/`3` & `1` non-zero @ old pins | **`DRIFTED_POST_T_IGX_066_A`** (live `all_counters()`: 4 counters = `0`, `historical_evidence_invalidated_by_later_changes = 2` [`intergrax/runtime/__init__.py`, `milvus/rag_store.py`], `vendor_neutrality_replay_failures = 1` [`reliability_wiring.py`], `prohibited_unresolved = 0`; +112 commits since `S07D` `35137143e`) | Post-`G2`: `PT-Q04` + **`PT-Q05`** (`KR-10` excluded from pre-`G2` `WORKLIST.md §4`) | No (zeroing pre-`G2` would drift again during `K1` merges) |
| **`EQO-X-05`** | `EQOA-A3-012` (`HIGH`), `TENANT-X` drift, `EQOA-A4-027` | `NO_PRODUCT_DELTA` (stale vs `258bbf6`) | **`HIGH_DRIFT_SINCE_5CCB72FC_AND_A4_027_RESIDUAL`** (**4,111** commits / **789** product files / **160** tenant-relevant files since `5ccb72fc`; **7** product files still hold `tenant_id = "default"` incl. LKW, dispute_sim, GCA, attestation_demo, checkpoint routes, scaffold) | Pre-`G2`: **`AF-CLOSE K1-C` (`WL-K1-C-05` / `K1-C B5`, runs last)** + `WL-K1-C-08` · Post-`G2`: `PT-Q07` + `T-B7-TEN-REVAL` | **YES** (owned in `WL-K1-C-05`; `GAP-10`/`GAP-11` already landed in code) |
| **`EQO-X-06`** | `KR-01..KR-28` (`20 FIX` / `8 ENV`) | `0/28 SETTLED` (`25 BLOCKED`, `3 WAIT_MEP_NODE`) | **`2_FIXED_IN_CODE_26_OPEN_OR_VERIFY_0_UNOWNED_IN_AF_CLOSE`** (`KR-03`, `KR-18` fixed in code; **14** FIX in `K1-A..F`; **4** FIX post-`G2` `PT-Q03/Q04/Q06/Q11`; **8** ENV in `WL-K3-12..13`, **0/8 `ENV_PROVEN`**; **5** ENV still `UNASSIGNED` in stale `known_red_register.v1.json`) | `K1-A` (1), `K1-B` (2), `K1-D` (4), `K1-F` (9), `K3` (8 ENV), post-`G2` `PT-Q*` (4) | **YES** (16 `K1` FIX + `WL-K3-12` CI secret + `WL-K3-13` ENV disposition + `WL-K1-F-10` register repair) |
| **`EQO-X-07`** | `EDER-N-0001..N-0004` | All 4 `INGESTED` / `UNKNOWN` | **`N0001_PATH_ON_HEAD` · `N0002_OPEN` · `N0003_CORE_LANDED_2_RESIDUALS` · `N0004_MANDATED_NO_EXEC`** (owners confirmed under `T-IGX-090-B` `E-03`; impact remains `UNKNOWN` until triage/closure) | `N-0001`/`N-0002`: `ARCH-FREEZE`/`DUP-X` decision · `N-0003`: `WL-K1-C-09` (`Q-A7C-08` deferral retained per `OD-TKT-META=A`) · `N-0004`: `WL-K3-02` (`T-IGX-052-E`, mandated by `OD-DP54=A`) | **YES** for `N-0004` (`WL-K3-02` before `G2` `scripts/` freeze) |
| **`EQO-X-08`** | 8 open `BLOCKS` findings in `FINDINGS.jsonl` (+1 `MERGED`) | 8 `BLOCKED` / `BLOCKS_ARCH_FREEZE` | **`2_OPEN_SEVERITY_BLOCKERS · 7_ACTIVE_BLOCKS + 1_SUPERSEDED_CLOSED`** (`A2-017` PDC-X superseded-closed on HEAD; `A4-003` fixed in code pending CI/audit; `A7-029` 6/7 decisions resolved; 5 still open) | `A2-017`: EQO bookkeeping · `A4-003`: `K1-B/K3` + `PT-Q13` · `A2-002/A2-001/A1-017`: `PT-Q05..Q13` · `A3-021/A7-005/A7-029`: `AF-CLOSE` W3–W8 + Operator | **YES** (`A4-003` CI wiring, `A7-029` `PT-S01..S06`, `A3-021`/`A7-005` `K1` prerequisites) |

**Headline Counters on `646e63953`:**
- **EQO-X tasks re-evaluated:** `8 / 8`
- **Open `severity=BLOCKER` findings:** **`2`** (`EQOA-A2-002`, `EQOA-A4-003`; `EQOA-A2-025` is `MERGED` into `EQOA-A2-002`)
- **Open `arch_freeze_impact=BLOCKS` findings (`FINDINGS.jsonl`):** **`8`** recorded (`7` active/partial + `1` `EQOA-A2-017` `SUPERSEDED_CLOSED_ON_HEAD`)
- **Unowned KR rows (`KR-01..KR-28`) in `AF-CLOSE`:** **`0`** (5 ENV rows remain `UNASSIGNED` in stale `known_red_register.v1.json` pending `WL-K1-F-10` / `WL-K3-12..13`)

---

## 2. Complete `KR-01..KR-28` Settlement Matrix on `HEAD` (`646e63953`)

| KR ID | Disp. | HEAD Static Status | `AF-CLOSE` Owner | Register Owner Field | `env_proof` | Verified Fact on `HEAD` (`646e63953`) |
| --- | --- | --- | --- | --- | --- | --- |
| `KR-01` | `ENV` | `ENV_UNPROVEN` | `WL-K3-13` (`K3`) | `UNASSIGNED` | `PENDING` | `unit-tests.yml:286,658` omits `--group platform-proofs-vpi-dataset` (`pyarrow`). |
| `KR-02` | `ENV` | `ENV_UNPROVEN` | `WL-K3-13` (`K3`) | `UNASSIGNED` | `PENDING` | `unit-tests.yml:286,658` omits `--extra integrations-postgresql` (`psycopg`). |
| `KR-03` | `FIX` | **`ALREADY_FIXED_IN_CODE`** | `WL-K1-F-09` (`K1-F`) | `T-AF2-RW-GATES` | `n/a` | `reference_harness_runtime.py` exports `lab_harness_context_from_modality_tooling` (`0202d3125` ancestor); record proof. |
| `KR-04` | `FIX` | `STILL_DEFECTIVE_ON_HEAD` | `WL-K1-A-04` (`K1-A`) + `WL-K1-F-10` | `T-AF2-RW-GATES` | `n/a` | All 3 `test_acp_token_*.py` files lack governance identity fixture. |
| `KR-05` | `ENV` | `ENV_CI_FIX_NEEDED_PRE_G2` | `WL-K3-12` (`K3`) | `UNASSIGNED` | `PENDING` | `INTERGRAX_DIAGNOSTIC_PROBLEM_LIST_CURSOR_SECRET` absent from `unit-tests.yml`. |
| `KR-06` | `ENV` | `ENV_UNPROVEN` | `WL-K3-13` (`K3`) | `UNASSIGNED` | `n/a` | OTel collector container reachability; Docker restricted under C10/C15. |
| `KR-07` | `FIX` | `STILL_DEFECTIVE_ON_HEAD` | `WL-K1-D-01` (`K1-D`) + `WL-K1-F-10` | `T-AF2-RW-SUPPLY` | `n/a` | `mongodb/integration.py` lacks CAS methods (`put_if_absent`/`replace_if_match`/`delete_if_match`); stale register path. |
| `KR-08` | `FIX` | `POST_G2_ACT_N11_DRIFT` | **`PT-Q03`** (excl. pre-`G2`; `WL-K3-04` prep) | `PT-Q03` | `n/a` | 88 `ACT-N11` product paths drifted since `9d3a042`; closes strictly via post-`G2` `RB-2`. |
| `KR-09` | `FIX` | `POST_G2_PIN_STALE` | **`PT-Q11`** (excl. pre-`G2`) | `PT-Q11` | `n/a` | EBH-7 F10 matrix pin stale; closes at `M3` (`PT-Q11`). |
| `KR-10` | `FIX` | `POST_G2_RRX_DRIFT` | **`PT-Q04` / `PT-Q05`** (excl. pre-`G2`) | `PT-Q04` | `n/a` | Live RRX counters on HEAD: invalidation=`2`, VBN replay=`1` (`prohibited_unresolved=0`). |
| `KR-11` | `FIX` | `POST_G2_VERIFY_PRE_G2_PATH` | **`PT-Q06`** (excl. pre-`G2`) | `PT-Q06` | `n/a` | `impact_map.py` unmentioned in `tests/qualification/trace_x/`; verify pre-`G2` whether product edit needed. |
| `KR-12` | `ENV` | `ENV_C10_C15_SCOPED` | `WL-K3-13` (`K3`) + `PT-Q09` | `PT-Q09` | `n/a` | 3 `dist_scale_x` PG/Docker tests under C10/C15 freeze; envelope proof stays with `PT-Q09`. |
| `KR-13` | `FIX` | `VERIFY_ON_HEAD` | `WL-K1-F-01` (`K1-F`) | `T-AF2-RW-GATES` | `n/a` | 77 incident scenario reds (`EX-L-01`); re-run after `K0.1`. |
| `KR-14` | `FIX` | `VERIFY_ON_HEAD` | `WL-K1-F-02` (`K1-F`) | `T-AF2-RW-GATES` | `n/a` | `test_execute_scenario_task_persists_finalized_trace_for_orchestration` (`EX-L-13`); re-run after `K0.1`. |
| `KR-15` | `FIX` | `VERIFY_ON_HEAD` | `WL-K1-F-03` (`K1-F`) | `T-B7-CMP-REVAL` | `n/a` | 14 `COMPAT-X` order-dependent reds (global registry polluter); sequenced with `WL-K1-F-12`. |
| `KR-16` | `FIX` | `STILL_DEFECTIVE_ON_HEAD` | `WL-K1-F-08` (`K1-F`) | `PT-WB` | `n/a` | `test_fixture_dataset_ingestion.py` still passes `tenant_id=0` across 4 `load_documents()` calls. |
| `KR-17` | `FIX` | `STILL_DEFECTIVE_ON_HEAD` | `WL-K1-B-03` (`K1-B`) + `WL-K1-F-10` | `T-AF2-RW-RUNTIME` | `n/a` | `mongodb/client.py:323` imports `pymongo.errors` outside declared opens; stale register path. |
| `KR-18` | `FIX` | **`ALREADY_FIXED_IN_CODE`** | `WL-K1-B-01` (`K1-B`) + `WL-K1-F-10` | `PT-WA` | `n/a` | Fixed by `e79f17f8a` & `5a13d33cc` (both ancestors of `HEAD`); `root.py:82` is docstring only. |
| `KR-19` | `FIX` | `VERIFY_ON_HEAD` | `WL-K1-F-04` (`K1-F`) | `PT-WF` | `n/a` | Register path `observability_otlp/` stale; actual file is `intergrax/runtime/observability/export_boundary.py`. |
| `KR-20` | `FIX` | `VERIFY_ON_HEAD` | `WL-K1-F-05` (`K1-F`) | `PT-WF` | `n/a` | `trace_x` model-call surface gates (4+) + LKW hybrid daemon; needs HEAD re-run. |
| `KR-21` | `ENV` | `ENV_QUOTED_UNPROVEN` | `WL-K3-13` (`K3`) | `none needed` | `QUOTED` | Optional extras (`duckdb`, `anthropic`, `langchain_core`, `fastmcp`, `celery`); proving run at `M3`. |
| `KR-22` | `ENV` | `ENV_UNPROVEN` | `WL-K3-13` (`K3`) | `UNASSIGNED` | `n/a` | `ent_audit_x` `launcher_05` FREEZE timeout; needs timed run to separate timeout vs defect. |
| `KR-23` | `FIX` | `VERIFY_ON_HEAD` | `WL-K1-D-02` (`K1-D`) | `PT-Q07` | `n/a` | MCP `R-P7-02` (11 tests across 3 files); MCP P8 window. |
| `KR-24` | `FIX` | `STILL_DEFECTIVE_ON_HEAD` | `WL-K1-D-03` (`K1-D`) | `T-IGX-039-A` | `n/a` | `tests/unit/applications/test_mcp_surface_opt_in.py:162` carries `pytest.mark.xfail` on `[product]`. |
| `KR-25` | `ENV` | `ENV_QUOTED_UNPROVEN` | `WL-K3-13` (`K3`) | `none needed` | `QUOTED` | 27 materialization `MAX_PATH` reds; short `--basetemp` proving run at `M3`. |
| `KR-26` | `FIX` | `STILL_DEFECTIVE_ON_HEAD` | `WL-K1-F-06` (`K1-F`, touches `K3`) | `PT-WD` | `n/a` | `intergrax/platform/enterprise_audit/validators.py:236-238` requires `qualification_at_<TES>` prefix (`ACT-N11` sensitive). |
| `KR-27` | `FIX` | `VERIFY_LIKELY_FIXED_PRE_BASE` | `WL-K1-D-04` (`K1-D`) | `PT-G01` | `n/a` | Ollama embedding catalog fixes `c3d5b7005`/`f209febbe`/`df436b7d1` landed pre-baseline; confirm green on HEAD. |
| `KR-28` | `FIX` | `STILL_DEFECTIVE_ON_HEAD` | `WL-K1-F-07` (`K1-F`) | `PT-G01` | `n/a` | `_create_base_llm_adapter` removed from `llm_resolver.py`; 3 E2E routing test files still patch it. |

---

## 3. All 8 Open `BLOCKS` Findings Verification (`EQO-X-08`)

| # | Finding ID | Severity | Verified HEAD Verdict (`646e63953`) | Closure Owner (`AF-CLOSE` / `EQO`) |
| ---: | --- | --- | --- | --- |
| 1 | **`EQOA-A4-003`** | **BLOCKER** | `FIXED_IN_CODE_PENDING_CI_AND_AUDIT` | Pre-`G2`: `WL-K1-B-04`, `WL-K1-E-07`, `WL-K3-05`/`10` (`OD-A4-003-CI=A`) · Post-`G2`: `PT-Q13` / `T-B7-ECAP-2C` |
| 2 | **`EQOA-A2-002`** *(incl. merged `A2-025`)* | **BLOCKER** | `STILL_OPEN_ON_HEAD` | Post-`G2`: `PT-D03`, `PT-D09`, `PT-AUD-DOC`, `PT-Q05/08/09/11/13`, `T-AF2-REQUAL-HIST-PASS` |
| 3 | **`EQOA-A2-001`** | `HIGH` | `STILL_OPEN_ON_HEAD` | Post-`G2`: `PT-Q08` (`TOOL-LIFE-X` RA-1) + `PT-D09` + `PT-AUD-DOC` |
| 4 | **`EQOA-A1-017`** | `MEDIUM` | `STILL_OPEN_ON_HEAD` | Post-`G2`: `PT-Q08` + `PT-D09` + `PT-AUD-DOC` (joint with `A2-001`/`A2-002`) |
| 5 | **`EQOA-A2-017`** | `INFO` | `SUPERSEDED_CLOSED_ON_HEAD` | EQO bookkeeping reconciliation (`PDC-X` P8 `969cbd964` / P9 `feaa2a87d`; residuals in `FIND-AF2-09..12`) |
| 6 | **`EQOA-A3-021`** | `MEDIUM` | `STILL_OPEN_ON_HEAD` | Pre-`G2` `K1`/`K3` + Post-`G2` `PT-D10` + `T-AF2-REQUAL-HIST-PASS` + `T-B7-PROMOTE-{OBS,RPL,SEC,REL,REG}` |
| 7 | **`EQOA-A7-005`** | `INFO` | `STILL_OPEN_ON_HEAD` | Pre-`G2` `K1`/`K3` + Post-`G2` `PT-VBN` + `T-AF2-MANIFESTS` + `T-B7-PROMOTE-*` + `T-B7-COORD-ARCH-FRZ` |
| 8 | **`EQOA-A7-029`** | `INFO` | `PARTIALLY_RESOLVED_ON_HEAD` | Operator + `AF-CLOSE`: `PT-S01..S06` + W8 `DS-ADM` (`OD-ARCH-FREEZE-01`) |

**Recorded:** 2026-11-26 · `Ref: governance/tickets/T-IGX-090-B.md`
