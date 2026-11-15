# RAG / context red classification (T-IGX-041-B, F-D2-08, F-D2-09, F-D2-10)

Command (one run, default environment, no optional extras installed, Windows):

```text
uv run pytest tests/unit/rag tests/integration/rag tests/integration/context tests/unit/runtime/architecture/test_ideal_harness_l3_w2_depth_gate.py -q --continue-on-collection-errors
```

| Run | Result |
| --- | --- |
| Audit run 5 (c08efe46a) | 48 failed, 746 passed, 25 skipped, 13 collection errors |
| Before the repairs of this ticket (current base) | 47 failed, 747 passed, 26 skipped, 12 collection errors |
| After | 8 failed, 771 passed, 50 skipped, 0 collection errors (log `.tmp/proof/pdc-x-041b/rag_after_run2.log`, not committed) |

Operator decision 1 (installing the CI RAG extras) was not answered, so no extra was installed. Modules that
need an extra now carry an explicit `pytest.importorskip(<module>, reason=...)` naming the extra. They are
reported as skipped (not green) here and are executed by the CI RAG profile
(`uv sync --extra dev-ci --extra dev-ci-rag`, `rag-guard.yml`).

## Classes

| Class | Action | Files |
| --- | --- | --- |
| Stale double: `list_source_record_ids(..., root_document_id=)` (contract `ae14377c9`) | double updated | `tests/unit/rag/indexing/test_single_index_strategy.py`, `tests/unit/rag/ingest/test_ingest_lci2b_loader_boundary.py`, `tests/unit/rag/ingest/test_ingest_pipeline_graph_wiring.py` |
| Stale double: abstract `load()` / moved `PipelineDocumentHandler` | double updated | `tests/unit/rag/document_loaders/parsers/test_parser_pipeline.py`, `tests/unit/rag/document_loaders/registry/test_document_handler_registry.py` |
| Stale double: `supports_native_hybrid_search` | double updated | `tests/unit/rag/retrievers/test_hybrid_retriever.py` |
| Stale expectation: the loader metadata now carries the freshness stamp `source_modified_at` (PDC-D2-R005) | expectation updated | `test_ingest_lci2b_loader_boundary.py` |
| Removed API: `RagProfile(graph_store_backend=...)` (the graph store is injected in these tests) | argument removed | `tests/integration/rag/test_graph_reingest_qualification.py`, `tests/integration/rag/test_same_source_reingest_serialization.py` |
| Docstring-coupled test | replaced by a typed check | `tests/integration/context/test_mem_xint6_cross_layer_e2e_certification.py` |
| Missing extra (langchain-core, qdrant-client, python-docx, pandas, PyMuPDF, docling-core, unstructured, sentence-transformers) | explicit `importorskip` with the extra named | the `tests/unit/rag/indexing`, `vectorstore`, `document_loaders`, `document_splitters` files and the `tests/integration/rag` parsers, embedding, retrievers and rerankers files of the write set |
| Syntax error | fixed, plus the syntax gate | `tests/integration/rag/test_neo4j_live_qualification.py`, `tests/qualification/pdc_x/d2_repair/test_d2r_test_tree_syntax_gate.py` |
| Stale gate (relocated UAEP) | audit roots repaired, loop reports every failing script | `intergrax/runtime/policy/pre_context_policy_audit.py`, `tests/unit/runtime/architecture/test_ideal_harness_l3_w2_depth_gate.py` |

## Still red after this ticket (8 ids): product defects or open items outside the write set

| Id | Cause | Why not repaired here |
| --- | --- | --- |
| `test_parser_pipeline.py::test_handler_attaches_native_handle_as_private_runtime_state` | Product: two parallel `_KnowledgeDocumentWithParserRuntime` classes (`contracts/parser_native_handle.py` attaches, `compat/legacy_runtime_document.py::get_parser_native_handle` reads another class) | `intergrax/rag/document_loaders/` is not in the write set |
| `test_fixture_dataset_ingestion.py::test_docx_ingestion`, `::test_txt_ingestion` | Product: `handlers/doc_smart_document_handler.py:11` imports `PipelineDocumentHandler` from `contracts/base_document_handler.py`, where it no longer exists (it lives in `pipeline/document_handler_loading.py`); the handler registry import fails | not in the write set |
| `test_hierarchical_dual_index_wiring.py::test_rag_stack_wires_toc_store_when_hierarchical_enabled` | Product: `InmemoryVectorStoreIntegration` has no `list_collections`, which `VectorstoreManager.list_collections` and `hierarchical_bootstrap.py:38` call (`@pytest.mark.no_ci`) | `intergrax/integrations/` is not in the write set |
| `test_ebh_2g_r1_retrieval_contract_typing_gate.py::test_rag_guard_gate_tests_collect_without_rag_local_embeddings` | `--collect-only` over `tests/unit/tools/providers/rag/` still reports a collection error (`test_rag_scope.py` imports a missing extra) | file not in the write set |
| `test_graph_reingest_qualification.py::test_graphrag_source_reingest_replaces_vectors_and_graph_safely`, `test_same_source_reingest_serialization.py::test_stale_inflight_publication_cannot_become_active_after_takeover`, `::test_stale_graph_topology_is_inactive_after_generation_takeover` | Behavioural assertions fail after the removed `graph_store_backend` argument is gone (active publication generation `None`, graph topology not inactive); root cause not yet established (stale test or reingest defect) | open; needs a follow-up investigation |
