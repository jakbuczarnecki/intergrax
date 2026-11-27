# Campaign B V5 — AF-W2-CLOSURE-02 targeted historical diagnostics

**Context:** `HISTORICAL_ALPHA` @ baseline **B** `7960ae7f6c82da60ca9acdc0ceebf63810002a42`.  
**Worktree (pytest cwd):** `D:/Projekty/_wt/intergrax/af-v5-p2-b` (no baseline commits).  
**Isolated venv:** `D:/Projekty/_wt/intergrax/_scratch/af-v5-b06/venv` (`UV_LINK_MODE=copy`).  
**Overlay:** vendored probes copied to `.tmp/campaign-b-overlay/` (gitignored under B).  
**Does not** close PT-B01, PT-B02, or SCOPE.

## Environment (declared extras only)

`uv sync --frozen` from baseline worktree with extras: **`mcp`**, **`dev-ci-rag`** (`langchain-core`), **`llm-langchain-ollama`**, **`llm-gemini`** (`google-genai` for R24 skips). No product tree edits.

## R04 — tenant probe (`pp_ten.py`)

| Field | Value |
| --- | --- |
| Command | `python -m pytest .tmp/campaign-b-overlay/probes/pp_ten.py -q --tb=line` |
| Pytest exit code | **1** |
| Passed / failed / skipped | **5 / 8 / 0** (after `langchain-core`; prior alpha: 7 failed + 1 skip) |
| Wall time | ~23s (with langchain-core) |
| Log | `.tmp/session/af-w2-closure-02/HISTORICAL_ALPHA/R04/combined_with_langchain.log` |

Langchain skip resolved via declared **`dev-ci-rag`** extra; remaining failures are historical tenant-isolation assertions (diagnostic FAIL, not baseline repair).

## R24 — NT-19 probe (`pp_nt19.py`)

| Field | Value |
| --- | --- |
| Command | `python -m pytest .tmp/campaign-b-overlay/probes/pp_nt19.py -q --tb=line` |
| Pytest exit code | **1** |
| Passed / failed / skipped | **9 / 1 / 0** (after extras; prior alpha: 6 passed, 4 skipped) |
| Wall time | ~13s |
| Log | `.tmp/session/af-w2-closure-02/HISTORICAL_ALPHA/R24/combined_final.log` |

Remaining **FAIL:** `test_nt19_a_persistent_sdk_failure_is_bounded_by_max_retries[langchain-ollama]` — `DID NOT RAISE` (historical diagnostic).  
**google.genai** skips cleared via declared **`llm-gemini`** extra.

## S03 — harness_01 full node run

| Field | Value |
| --- | --- |
| Execution SHA | `7960ae7f6c82da60ca9acdc0ceebf63810002a42` |
| Command | `python -m pytest tests/qualification/harness_01 -q --maxfail=3 --tb=line` |
| Pytest exit code | **0** |
| Collected / passed | **127 passed** (summary line) |
| Failed / skipped / errors | **0** |
| Wall time | **80.35s** |
| Log | `.tmp/session/af-w2-closure-02/HISTORICAL_ALPHA/S03/combined.log` |

Supersedes runner-only kill evidence (`4294967295`) for **node completeness**; outer campaign runner kill remains historical context in [`CAMPAIGN_B_V5_ALPHA.md`](CAMPAIGN_B_V5_ALPHA.md). Does not certify product.