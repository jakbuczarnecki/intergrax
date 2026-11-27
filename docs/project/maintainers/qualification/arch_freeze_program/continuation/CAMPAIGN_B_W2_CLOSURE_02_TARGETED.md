# Campaign B V5 — AF-W2-CLOSURE-02 targeted historical diagnostics

**Context:** `HISTORICAL_ALPHA` @ baseline **B** `7960ae7f6c82da60ca9acdc0ceebf63810002a42`.  
**Worktree (pytest cwd):** `D:/Projekty/_wt/intergrax/af-v5-p2-b` (no baseline commits).  
**Isolated venv:** `D:/Projekty/_wt/intergrax/_scratch/af-v5-b06/venv` (`UV_LINK_MODE=copy`).  
**Overlay:** vendored probes copied to `.tmp/campaign-b-overlay/` (gitignored under B).  
**Does not** close PT-B01, PT-B02, or SCOPE.

**AF-W2-CLOSURE-03 note:** Measurements below are **pinned historical diagnostics** @ B. They are not re-run in closure-03 and do not certify a green baseline.

## Environment (declared extras only)

`uv sync --frozen` from baseline worktree with extras: **`mcp`**, **`dev-ci-rag`** (`langchain-core`), **`llm-langchain-ollama`**, **`llm-gemini`** (`google-genai` for R24 skips). No product tree edits.

## R04 — tenant probe (`pp_ten.py`)

| Field | Value |
| --- | --- |
| Execution SHA | `7960ae7f6c82da60ca9acdc0ceebf63810002a42` |
| Command | `python -m pytest .tmp/campaign-b-overlay/probes/pp_ten.py -q --tb=line` |
| Collected | **13** |
| Passed / failed / skipped / errors | **5 / 8 / 0 / 0** |
| Pytest exit code | **1** |
| Wall time | ~23s (with langchain-core) |
| Environment extras | `dev-ci-rag` (`langchain-core`) |
| Log | `.tmp/session/af-w2-closure-02/HISTORICAL_ALPHA/R04/combined_with_langchain.log` |
| Log SHA256 | `f07db0aed7830b29edb47c58a470c26d20de43c7b0519075c8cf8117acf160b4` |

Failed nodes (8) and assertion summaries:

| Node | Assertion failure (summary) |
| --- | --- |
| `test_nt11a_from_langchain_document_foreign_tenant` | foreign metadata minted `scope.tenant_id` without caller-bound tenant |
| `test_nt11b_from_legacy_rag_hit_foreign_tenant` | legacy RAG hit minted foreign `scope.tenant_id` |
| `test_nt12a_tier1_routing_record_not_shared` | cross-tenant routing record consumption (`model-b`) |
| `test_nt12b_tier3_routing_record_not_shared` | cross-tenant routing record consumption (`model-b`) |
| `test_nt13a_profile_projection_no_silent_drop` | silent drop of 7 configured profile values |
| `test_nt15b_debug_app_experiment_registry_tenant` | tenant B saw tenant A experiment |
| `test_nt15d_debug_router_guard_honours_identity_idp` | anonymous caller served with IdP configured |
| `test_nt15e_erl_ahi_tenant_reference_scan` | ERL/AHI store lacks tenant identity |

Langchain skip resolved via declared **`dev-ci-rag`** extra; remaining failures are historical tenant-isolation assertions (diagnostic FAIL, not baseline repair).

## R24 — NT-19 probe (`pp_nt19.py`)

| Field | Value |
| --- | --- |
| Execution SHA | `7960ae7f6c82da60ca9acdc0ceebf63810002a42` |
| Command | `python -m pytest .tmp/campaign-b-overlay/probes/pp_nt19.py -q --tb=line` |
| Collected | **10** |
| Passed / failed / skipped / errors | **9 / 1 / 0 / 0** |
| Pytest exit code | **1** |
| Wall time | ~13s |
| Environment extras | `llm-langchain-ollama`, `llm-gemini` |
| Log | `.tmp/session/af-w2-closure-02/HISTORICAL_ALPHA/R24/combined_final.log` |
| Log SHA256 | `0f965124866640d62dc36fe43fbeda7b400274df623a82dd87833c11f0e93052` |

Remaining **FAIL:** `test_nt19_a_persistent_sdk_failure_is_bounded_by_max_retries[langchain-ollama]` — `Failed: DID NOT RAISE <class 'Exception'>` (historical NT-19 diagnostic; not missing-dependency).  
**google.genai** skips cleared via declared **`llm-gemini`** extra.

## S03 — harness_01 full node run

| Field | Value |
| --- | --- |
| Execution SHA | `7960ae7f6c82da60ca9acdc0ceebf63810002a42` |
| Command | `python -m pytest tests/qualification/harness_01 -q --maxfail=3 --tb=line` |
| Pytest exit code | **0** |
| Collected / passed / failed / skipped / errors | **127 / 127 / 0 / 0 / 0** |
| Wall time | **80.35s** |
| Environment extras | baseline B venv (closure-02 session) |
| Log | `.tmp/session/af-w2-closure-02/HISTORICAL_ALPHA/S03/combined.log` |
| Log SHA256 | `a2d2a8bc99799c224a6f1c4c742e8af5e48e9056f85b657b42b53abbdd91973f` |

## HEAD_DIAGNOSTIC (supplementary @ development HEAD)

Does not back-fill baseline **B**. Historical @ B: R01a/R02/R05/R20 = **`NOT_EXECUTABLE_AT_BASELINE`**.

| Case | Command | Outcome | Log |
| --- | --- | --- | --- |
| R01a / R20 | `uv run python scripts/governance/verify_rw_plan_phase_check3.py` | exit **0**, `PLAN_STRUCTURE=PASS` | `.tmp/session/af-w2-closure-02/HEAD_DIAGNOSTIC/R01a_R20_stdout.txt` |
| R02 | `uv run pytest tests/unit/governance/test_arch_freeze_plan_traceability.py -q` | PASS @ closure-02 HEAD (per [`CAMPAIGN_B_MISSING_CASE_MAP.md`](CAMPAIGN_B_MISSING_CASE_MAP.md)) | closure-02 HEAD session |
| R05 | `uv run python scripts/governance/lint_iiac_closure_regex.py` | PASS @ closure-02 HEAD (per missing-case map) | closure-02 HEAD session |

Supersedes runner-only kill evidence (`4294967295`) for **node completeness**; outer campaign runner kill remains historical context in [`CAMPAIGN_B_V5_ALPHA.md`](CAMPAIGN_B_V5_ALPHA.md). Does not certify product.