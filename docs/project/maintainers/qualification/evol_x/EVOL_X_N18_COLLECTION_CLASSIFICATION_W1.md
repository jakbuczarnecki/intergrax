# EVOL-X W1 — N-18 collection error classification (F-01 exact)

**Command:** `uv run pytest tests/unit -m "gate and not no_ci" --collect-only -q`  
**Session log:** `.tmp/session/evol-x-w1/pytest-collect-gate-unit.log` (gitignored; not part of publication SHA)  
**Session HEAD when log captured:** `5a4e922ffacabef0ee24eb676e5c8705c20ef4a1` (inferred same as `@ 54e0de377` for listed modules — see baseline column)  
**ERROR module count:** **32** (one ERROR line = one module)

**Prior arithmetic error (corrected):** Summing RC “module counts” as 11+9+6+… yielded **34** because **langchain** and **anthropic** modules were merged into one bucket and **fastmcp** was under-counted. This table assigns **exactly one primary cause per module**; primary causes sum to **32**.

## Baseline reproducibility legend

| Code | Meaning |
|------|---------|
| **EXEC@log** | Observed in W1 session log at capture time |
| **INF-TREE** | Not re-run @ START_HEAD checkout; `git diff 5a4e922..54e0de377` shows **no** changes to `tests/unit/**` → same import graph inferred |
| **INF-ENV** | Environment unchanged assumption: default `uv` sync without `--extra dev-ci` (same as log capture) |

**No START_HEAD checkout replay was executed** for N-18 (git safety / scope).

## CONSERVATIVE_FULL impact (per module)

These modules are **not** EVOL-X qualification evidence. They affect only the **invalid** oracle definition “local `-m gate and not no_ci` collect without CI extras”. **OD-EVOL-10:** CONSERVATIVE_FULL must use CI `dev-ci` manifest / explicit node list — not this collect.

| # | Module ID | Primary RC | Secondary | Error evidence (log) | Baseline | CONSERVATIVE_FULL |
|---|-----------|------------|-----------|----------------------|----------|-------------------|
| 1 | `tests/unit/agents/test_reference_harness_strict.py` | **RC-7** Import/architecture drift | — | `ImportError: cannot import name 'lab_harness_context_from_modality_tooling'` from `reference_harness_runtime.py` | EXEC@log; INF-TREE @ pub SHA | Excluded from EVOL manifest; baseline defect tracked separately |
| 2 | `tests/unit/applications/test_fastapi_mcp.py` | **RC-1** `fastmcp` missing | — | `ModuleNotFoundError: No module named 'fastmcp'` | EXEC@log; INF-TREE; INF-ENV | Resolved by `uv sync --extra dev-ci` for CI parity oracle |
| 3 | `tests/unit/applications/test_mcp_cancel_a_transport_governed_cancel.py` | RC-1 | — | `No module named 'fastmcp'` | j.w. | j.w. |
| 4 | `tests/unit/applications/test_mcp_canonical_execution.py` | RC-1 | — | j.w. | j.w. | j.w. |
| 5 | `tests/unit/applications/test_mcp_catalog_child_execution_qualification.py` | RC-1 | — | j.w. | j.w. | j.w. |
| 6 | `tests/unit/applications/test_mcp_catalog_cross_loop_ingress_qualification.py` | RC-1 | — | j.w. | j.w. | j.w. |
| 7 | `tests/unit/applications/test_mcp_catalog_discovery_qualification.py` | RC-1 | — | j.w. | j.w. | j.w. |
| 8 | `tests/unit/applications/test_mcp_catalog_f3_cross_loop_cancel_c_physical_l3.py` | RC-1 | — | j.w. | j.w. | j.w. |
| 9 | `tests/unit/applications/test_mcp_catalog_f3_streamable_http_physical_l3.py` | RC-1 | — | j.w. | j.w. | j.w. |
| 10 | `tests/unit/applications/test_mcp_catalog_gate_b_cancel_c_qualification.py` | RC-1 | — | j.w. | j.w. | j.w. |
| 11 | `tests/unit/applications/test_mcp_trust_a_principal_binding.py` | RC-1 | — | j.w. | j.w. | j.w. |
| 12 | `tests/unit/tools/providers/test_tool_life_x_l3_r2_mcp_host_lifecycle_qualification.py` | RC-1 | — | j.w. | j.w. | j.w. |
| 13 | `tests/unit/compat/langchain/test_documents.py` | **RC-2** `langchain_core` | — | `No module named 'langchain_core'` | EXEC@log; INF-TREE; INF-ENV | CI extra / manifest — not omission in EVOL graph |
| 14 | `tests/unit/llm_adapters/test_native_ollama_adapter.py` | RC-2 | — | j.w. | j.w. | j.w. |
| 15 | `tests/unit/llm_adapters/test_ollama_structured_output.py` | RC-2 | — | j.w. | j.w. | j.w. |
| 16 | `tests/unit/llm_adapters/test_ollama_tool_calling.py` | RC-2 | — | j.w. | j.w. | j.w. |
| 17 | `tests/unit/rag/indexing/test_indexing_manager_indexes_documents.py` | RC-2 | — | j.w. | j.w. | j.w. |
| 18 | `tests/unit/rag/vectorstore/test_source_ownership_contract.py` | **RC-5** `qdrant_client` | — | `No module named 'qdrant_client'` | EXEC@log; INF-TREE | j.w. |
| 19 | `tests/unit/rag/vectorstore/test_vectorstore_cross_tenant_isolation.py` | RC-5 | — | j.w. | j.w. | j.w. |
| 20 | `tests/unit/runtime/nexus/tools/test_parallel_semantic_batch_pattern.py` | RC-2 | — | `langchain_core` | j.w. | j.w. |
| 21 | `tests/unit/runtime/nexus/tools/test_tool_catalog_embedder.py` | RC-2 | — | j.w. | j.w. | j.w. |
| 22 | `tests/unit/tools/providers/rag/test_rag_scope.py` | RC-2 | — | j.w. | j.w. | j.w. |
| 23 | `tests/unit/llm_adapters/test_adapter_conformance.py` | **RC-3** `anthropic` | import via `claude_adapter` | `No module named 'anthropic'` | EXEC@log; INF-TREE | j.w. |
| 24 | `tests/unit/llm_adapters/test_structured_output_flags.py` | RC-3 | — | j.w. | j.w. | j.w. |
| 25 | `tests/unit/integrations/providers/relational_store/test_postgresql_bounded_fetch.py` | **RC-4** `psycopg` | — | `No module named 'psycopg'` | EXEC@log; INF-TREE | j.w. |
| 26 | `tests/unit/platform_proofs/.../test_postgresql_exact_identifier_lookup.py` | RC-4 | `IntegrationConfigurationError` message | psycopg / integration extra | j.w. | j.w. |
| 27 | `.../test_postgresql_lexical_search.py` | RC-4 | — | j.w. | j.w. | j.w. |
| 28 | `.../test_postgresql_structured_contains_capability.py` | RC-4 | — | j.w. | j.w. | j.w. |
| 29 | `.../test_postgresql_structured_search.py` | RC-4 | — | j.w. | j.w. | j.w. |
| 30 | `.../test_pgvector_vector_storage_adapter.py` | **RC-6** `pgvector` | — | `No module named 'pgvector'` | EXEC@log; INF-TREE | j.w. |
| 31 | `.../test_storage_bootstrap_operator.py` | RC-6 | — | j.w. | j.w. | j.w. |
| 32 | `.../test_vpi_data_pack_proof_runner.py` | **RC-8** `sentence_transformers` | — | `No module named 'sentence_transformers'` | EXEC@log; INF-TREE | j.w. |

**Primary RC totals (modules):** RC-1 **11** · RC-2 **8** · RC-3 **2** · RC-4 **5** · RC-5 **2** · RC-6 **2** · RC-7 **1** · RC-8 **1** → **32**.

**Unique primary root-cause kinds:** **8** (not 32 independent platform defects).

## W2 impact

Does **not** block W2 code. Blocks defining CONSERVATIVE_FULL as unqualified local gate collect until **OD-EVOL-10** approves CI-parity manifest.

## Forbidden mitigations

No xfail/skip/marker shrink/exclusion to hide RC-7 or env gaps in EVOL qualification artifacts.
