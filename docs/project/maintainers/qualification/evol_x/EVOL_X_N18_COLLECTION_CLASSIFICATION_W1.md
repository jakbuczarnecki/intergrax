# EVOL-X W1 — N-18 collection error classification

**Command (W1 session):** `uv run pytest tests/unit -m "gate and not no_ci" --collect-only -q`  
**Log:** `.tmp/session/evol-x-w1/pytest-collect-gate-unit.log` (gitignored; reproduce locally)  
**Outcome:** 13398 collected, **32 errors**, 27522 deselected  
**W1 code delta:** `git diff 5a4e922..HEAD --name-only` → **docs only** → errors are **not** New W1 regression.

## START_HEAD reproduction

| Evidence | Conclusion |
|----------|------------|
| No test/source files changed between `5a4e922` and `cac5dfd91` | Same tree for all 32 modules @ START_HEAD |
| Same default `uv` env (without `--extra dev-ci`) | Errors **reproduce** at START_HEAD under identical environment |
| Formal re-run @ START_HEAD checkout | Not executed (no detached checkout); inference **VERIFIED** via diff |

## Unique root causes (7)

| RC-ID | Root cause class | Missing / broken dependency | Module count | Affected modules |
|-------|------------------|----------------------------|--------------|------------------|
| **RC-1** | Environment / optional extra | `fastmcp` (in `dev-ci` extra, not default sync) | 11 | `test_fastapi_mcp.py`, `test_mcp_*` (9), `test_tool_life_x_l3_r2_mcp_host_lifecycle_qualification.py` |
| **RC-2** | Environment / optional extra | `langchain_core` | 9 | `test_documents.py`, `test_*ollama*`, `test_structured_output_flags.py`, `test_adapter_conformance.py`, RAG/nexus/tools `test_*` (4) |
| **RC-3** | Environment / optional extra | `psycopg` (+ `IntegrationConfigurationError` message) | 6 | `test_postgresql_bounded_fetch.py`, VPI `test_postgresql_*` (4), related |
| **RC-4** | Environment / optional extra | `pgvector` | 2 | `test_pgvector_vector_storage_adapter.py`, `test_storage_bootstrap_operator.py` |
| **RC-5** | Environment / optional extra | `qdrant_client` | 2 | `test_source_ownership_contract.py`, `test_vectorstore_cross_tenant_isolation.py` |
| **RC-6** | Environment / optional extra | `anthropic`, `sentence_transformers` | 3 | `test_adapter_conformance.py`, `test_structured_output_flags.py`, `test_vpi_data_pack_proof_runner.py` |
| **RC-7** | Import / architecture drift (baseline defect) | `lab_harness_context_from_modality_tooling` removed from `reference_harness_runtime.py` | 1 | `tests/unit/agents/test_reference_harness_strict.py` |

**Unique root cause count:** **7** (32 ERROR lines = 32 modules, not 32 independent platform defects).

## W2 qualification entry impact

| Question | Answer |
|----------|--------|
| Does N-18 block W2 **code** start? | **No** — EVOL W2 does not require full gate-unit collect |
| Does N-18 block **CONSERVATIVE_FULL** definition as `-m "gate and not no_ci"` on default local sync? | **Yes** — until **OD-EVOL-10** pins manifest to **CI-equivalent** extras (`uv sync --extra dev-ci --frozen`) **or** explicit node list (QUAL-X P5 pattern) |
| Hide failures via skip/xfail/marker shrink? | **Forbidden** |

## Minimal correction strategy (no W1 execution)

1. **OD-EVOL-10:** Define EVOL `CONSERVATIVE_FULL` = enumerated pytest nodes + CI `dev-ci` sync profile — **not** raw local marker collect without extras.  
2. **RC-7:** Track as **existing baseline defect** — fix in separate authorized change (restore export or update test); **not** EVOL-X W1 scope.  
3. **RC-1..6:** Document as **environment parity** — not EVOL inventory omissions; do not exclude from platform, exclude from **local default collect oracle**.

## Decision required

**OD-EVOL-10** (operator): approve CI parity collect command and authorize W2 `evol_x_p3_l3` workflow edit + eventual `main` sync.
