<!--
© Artur Czarnecki. All rights reserved.
Intergrax framework – proprietary and confidential.
Use, modification, or distribution without written permission is prohibited.
-->

# LANGCHAIN_INDEPENDENCE — dependency inventory

**Parent hub:** [`../LANGCHAIN_INDEPENDENCE.md`](../LANGCHAIN_INDEPENDENCE.md)
**Feature plan:** [`../../plan/LANGCHAIN_INDEPENDENCE.md`](../../plan/LANGCHAIN_INDEPENDENCE.md)

---

## A. Inventory methodology

| Field | Value |
|-------|-------|
| **Audit date** | 2026-08-02 |
| **Branch** | `development` |
| **Commit** | `0d01a6eb96ff29d1e6479fdd2d968d7505f82f94` |
| **Scope** | `intergrax`, `agents`, `applications`, `tests`, `scripts`, `pyproject.toml`, `uv.lock` (excludes `docker/runtime-context` copies) |
| **Patterns** | Top-level `from` / `import` of `langchain*`, `langgraph` in Python sources; `pyproject.toml` declarations |
| **Classifications** | `CORE_CONTRACT_LEAK`, `CORE_IMPLEMENTATION_DEPENDENCY`, `PROVIDER_BOUND_DEPENDENCY`, `OPTIONAL_COMPATIBILITY`, `LEGACY_OPTIONAL`, `TOOLING_DEPENDENCY`, `TEST_ONLY`, `PACKAGING_DEPENDENCY`, `GENERATED_LOCK_ENTRY` |
| **Tooling definition** | Executable repository tooling or generators import LangChain, but the dependency is not part of production runtime or a documentation-only textual mention. |
| **Generated files** | `uv.lock` summarized as one `GENERATED_LOCK_ENTRY` row (resolver output; individual transitive packages are not separate migration decisions). |

## LCI-4A closeout

Active retrievers and the retrieval tool now use the native immutable `RetrievalHit` contract. Graph, ingest and auxiliary legacy rows remain assigned to LCI-4C and LCI-4D as applicable.

## LCI-4B closeout

## LCI-4D closeout

Memory indexing, multimedia document loaders, legacy RAG answer contracts, evaluation harnesses and soak tooling use the canonical KnowledgeDocument boundary. Auxiliary runtime paths preserve canonical identity, tenant, namespace, workspace and provenance without using user metadata as system transport. No active LCI-4D production module imports or exposes LangChain Document. Provider-local loaders and embedding dependencies remain assigned to LCI-5 and LCI-6. LCI-4D is APPROVED; LCI-5A is APPROVED; LCI-5B is READY_FOR_REVIEW; LCI-5C is PLANNED / NEXT AFTER ACCEPTANCE.


Reranker candidate/result contracts and reranker provider adapters now use native Intergrax knowledge documents; the seven eliminated LangChain document import rows were removed from the detailed inventory.


## B. Summary

| Metric | Count |
|--------|------:|
| direct production/runtime imports | 11 |
| direct test imports | 46 |
| direct tooling imports | 1 |
| direct LangGraph imports | 2 |
| packaging declaration rows | 10 |
| generated lock rows | 1 |
| core contract leaks | 0 |
| core implementation dependencies | 0 |
| provider-bound dependencies | 9 |
| optional compatibility paths | 0 |
| legacy optional paths | 2 |
| tooling dependencies | 1 |
| test-only | 46 |
| documentation-only | 0 |
| unclassified occurrences | 0 |
| total detailed inventory rows | 69 |

## C. Detailed inventory table

| Inventory ID | Package/module | Path | Line | Symbol or usage | Layer/domain | Dependency exposure | Classification | Current requirement status | Target state | Migration task | Evidence/notes |
|--------------|----------------|------|-----:|-----------------|--------------|---------------------|----------------|----------------------------|--------------|----------------|----------------|
| LCI-INV-0001 | `langchain_core.documents` | `applications/local_workspace_application/tests/workspaces/test_workspace_lifecycle.py` | 3 | `Document` | APPLICATION / test | test-only | TEST_ONLY | test only | Tests use native fixtures; compatibility tests under LCI-7C | LCI-3C | verified import |
| LCI-INV-0007 | `langchain_core.documents` | `intergrax/integrations/providers/document_parser/openpyxl/opens.py` | 127 | `Document` | INTEGRATIONS / production | runtime | PROVIDER_BOUND_DEPENDENCY | optional (`rag-langchain-loaders`) | Provider-local LangChain use; lazy import; optional extra | LCI-5C | verified lazy import |
| LCI-INV-0008 | `langchain_community.document_loaders` | `intergrax/integrations/providers/document_parser/pymupdf/opens.py` | 14 | `PyMuPDFLoader` | INTEGRATIONS / production | runtime | PROVIDER_BOUND_DEPENDENCY | optional (`rag-langchain-loaders`) | Provider-local LangChain use; lazy import; optional extra | LCI-5C | verified lazy import |
| LCI-INV-0009 | `langchain_community.document_loaders` | `intergrax/integrations/providers/document_parser/python_docx/opens.py` | 27 | `Docx2txtLoader` | INTEGRATIONS / production | runtime | PROVIDER_BOUND_DEPENDENCY | optional (`rag-langchain-loaders`) | Provider-local LangChain use; lazy import; optional extra | LCI-5C | verified lazy import |
| LCI-INV-0010 | `langchain_core.documents` | `intergrax/integrations/providers/document_parser/python_docx/opens.py` | 69 | `Document` | INTEGRATIONS / production | runtime | PROVIDER_BOUND_DEPENDENCY | optional (`rag-langchain-loaders`) | Provider-local LangChain use; lazy import; optional extra | LCI-5C | verified lazy import |
| LCI-INV-0011 | `langchain_community.document_loaders` | `intergrax/integrations/providers/document_parser/unstructured/opens.py` | 11 | `UnstructuredHTMLLoader` | INTEGRATIONS / production | runtime | PROVIDER_BOUND_DEPENDENCY | optional (`rag-langchain-loaders`) | Provider-local LangChain use; lazy import; optional extra | LCI-5C | verified lazy import |
| LCI-INV-0029 | `langchain_ollama` | `intergrax/llm_adapters/providers/ollama_adapter.py` | 39 | `ChatOllama` | LLM_ADAPTERS / production | runtime | PROVIDER_BOUND_DEPENDENCY | optional (`llm-langchain-ollama`) | Compatibility-only provider; lazy import; explicit construction; not default | LCI-6E | verified lazy import and missing-extra error |
| LCI-INV-0030 | `langchain_core.messages` | `intergrax/llm_adapters/providers/ollama_adapter.py` | 264 | `AIMessage, HumanMessage, SystemMessage, ToolMessage` | LLM_ADAPTERS / production | runtime | PROVIDER_BOUND_DEPENDENCY | optional (`llm-langchain-ollama`; core retained elsewhere) | Compatibility-only provider; lazy import; explicit construction; not default | LCI-6E | verified lazy import |
| LCI-INV-0066 | `langchain_text_splitters` | `intergrax/rag/document_splitters/strategies/langchain_recursive_chunking_strategy.py` | 34 | `RecursiveCharacterTextSplitter` | RAG / production | runtime | PROVIDER_BOUND_DEPENDENCY | optional | Optional provider loaded lazily; explicit registry registration | LCI-2E | verified lazy import |
| LCI-INV-0075 | `langchain_ollama` | `intergrax/rag/embedding/providers/ollama_embedding_provider.py` | 48 | `OllamaEmbeddings` | RAG / production | runtime | PROVIDER_BOUND_DEPENDENCY | optional (`rag-langchain-embeddings`; package retained for LCI-6) | Provider-local LangChain use; lazy import; optional extra | LCI-5C | verified lazy import |
| LCI-INV-0104 | `langgraph.graph` | `intergrax/supervisor/supervisor_to_state_graph.py` | 198 | `END, StateGraph` | ORCHESTRATION / production | runtime | LEGACY_OPTIONAL | required (default install) | Legacy path retired or isolated under optional extra | LCI-8A | verified import |
| LCI-INV-0105 | `langgraph.graph.message` | `intergrax/websearch/integration/langgraph_nodes.py` | 11 | `add_messages` | ORCHESTRATION / production | runtime | LEGACY_OPTIONAL | required (default install) | Legacy path retired or isolated under optional extra | LCI-8A | verified import |
| LCI-INV-0106 | `langchain_core.documents` | `scripts/docs/generate_integration_usage_docs.py` | 422 | `Document` | PLATFORM_FOUNDATION / tooling | tooling | TOOLING_DEPENDENCY | tooling only | Generator uses native types or optional LangChain extra | LCI-7D | verified import |
| LCI-INV-0107 | `langchain_core.documents` | `tests/e2e/llama_cpp/test_llama_cpp_stack_e2e.py` | 19 | `Document` | TEST / test | test-only | TEST_ONLY | test only | Tests use native fixtures; compatibility tests under LCI-7C | LCI-3A | verified import |
| LCI-INV-0108 | `langchain_core.documents` | `tests/e2e/rag/test_rag_full_runtime_e2e.py` | 8 | `Document` | TEST / test | test-only | TEST_ONLY | test only | Tests use native fixtures; compatibility tests under LCI-7C | LCI-2F | verified import |
| LCI-INV-0109 | `langchain_core.documents` | `tests/integration/applications/test_memory_vector_ltm_wiring.py` | 19 | `Document` | TEST / test | test-only | TEST_ONLY | test only | Tests use native fixtures; compatibility tests under LCI-7C | LCI-4D | verified import |
| LCI-INV-0110 | `langchain_core.documents` | `tests/integration/rag/answers/test_rag_answer_pipeline.py` | 8 | `Document` | TEST / test | test-only | TEST_ONLY | test only | Tests use native fixtures; compatibility tests under LCI-7C | LCI-4D | verified import |
| LCI-INV-0116 | langchain_core.documents | 	ests/integration/rag/document_loaders/parsers/test_image_smart_document_handler.py | 10 | Document | TEST / test | test-only | TEST_ONLY | test only | Legacy modality path retains LangChain Document fixtures until LCI-4D migration. | LCI-4D | verified import |
| LCI-INV-0118 | langchain_core.documents | 	ests/integration/rag/document_loaders/parsers/test_video_smart_document_handler.py | 10 | Document | TEST / test | test-only | TEST_ONLY | test only | Legacy modality path retains LangChain Document fixtures until LCI-4D migration. | LCI-4D | verified import |
| LCI-INV-0119 | `langchain_core.documents` | `tests/integration/rag/document_splitters/test_chunking_integration.py` | 10 | `Document` | TEST / test | test-only | TEST_ONLY | test only | Tests use native fixtures; compatibility tests under LCI-7C | LCI-2D | verified import |
| LCI-INV-0120 | `langchain_core.documents` | `tests/integration/rag/embedding/test_hf_embedding_pipeline_integration.py` | 8 | `Document` | TEST / test | test-only | TEST_ONLY | test only | Tests use native fixtures; compatibility tests under LCI-7C | LCI-3A | verified import |
| LCI-INV-0121 | `langchain_core.documents` | `tests/integration/rag/embedding/test_ollama_embedding_pipeline_integration.py` | 9 | `Document` | TEST / test | test-only | TEST_ONLY | test only | Tests use native fixtures; compatibility tests under LCI-7C | LCI-3A | verified import |
| LCI-INV-0122 | `langchain_core.documents` | `tests/integration/rag/embedding/test_vllm_embedding_pipeline_integration.py` | 7 | `Document` | TEST / test | test-only | TEST_ONLY | test only | Tests use native fixtures; compatibility tests under LCI-7C | LCI-3A | verified import |
| LCI-INV-0123 | `langchain_core.documents` | `tests/integration/rag/retrievers/test_retrieval_integration.py` | 9 | `Document` | TEST / test | test-only | TEST_ONLY | test only | Tests use native fixtures; compatibility tests under LCI-7C | LCI-4A | verified import |
| LCI-INV-0124 | `langchain_core.documents` | `tests/integration/rag/retrievers/test_retriever_pipeline_basic.py` | 6 | `Document` | TEST / test | test-only | TEST_ONLY | test only | Tests use native fixtures; compatibility tests under LCI-7C | LCI-4A | verified import |
| LCI-INV-0125 | `langchain_core.documents` | `tests/integration/rag/vectorstore/test_vectorstore_real_backends.py` | 9 | `Document` | TEST / test | test-only | TEST_ONLY | test only | Tests use native fixtures; compatibility tests under LCI-7C | LCI-3C | verified import |
| LCI-INV-0126 | `langchain_core.documents` | `tests/unit/integrations/providers/test_provider_runtime_cutover.py` | 17 | `Document` | TEST / test | test-only | TEST_ONLY | test only | Tests use native fixtures; compatibility tests under LCI-7C | LCI-7C | verified import |
| LCI-INV-0127 | `langchain_core.documents` | `tests/unit/integrations/providers/vector_store/test_pinecone.py` | 13 | `Document` | TEST / test | test-only | TEST_ONLY | test only | Tests use native fixtures; compatibility tests under LCI-7C | LCI-3D | verified import |
| LCI-INV-0128 | `langchain_core.documents` | `tests/unit/integrations/providers/vector_store/test_qdrant_chroma.py` | 13 | `Document` | TEST / test | test-only | TEST_ONLY | test only | Tests use native fixtures; compatibility tests under LCI-7C | LCI-3D | verified import |
| LCI-INV-0129 | `langchain_core.documents` | `tests/unit/integrations/providers/vector_store/test_qdrant_point_id_normalization.py` | 13 | `Document` | TEST / test | test-only | TEST_ONLY | test only | Tests use native fixtures; compatibility tests under LCI-7C | LCI-3D | verified import |
| LCI-INV-0130 | `langchain_core.messages` | `tests/unit/llm_adapters/test_ollama_structured_output.py` | 11 | `AIMessage, HumanMessage, SystemMessage` | TEST / test | test-only | TEST_ONLY | test only | Tests use native fixtures; compatibility tests under LCI-7C | LCI-7C | verified import |
| LCI-INV-0131 | `langchain_core.messages` | `tests/unit/llm_adapters/test_ollama_tool_calling.py` | 12 | `AIMessage, HumanMessage, SystemMessage, ToolMessage` | TEST / test | test-only | TEST_ONLY | test only | Tests use native fixtures; compatibility tests under LCI-7C | LCI-7C | verified import |
| LCI-INV-0140 | `langchain_core.documents` | `tests/unit/rag/embedding/test_embedding_manager.py` | 8 | `Document` | TEST / test | test-only | TEST_ONLY | test only | Tests use native fixtures; compatibility tests under LCI-7C | LCI-3A | verified import |
| LCI-INV-0141 | `langchain_core.documents` | `tests/unit/rag/embedding/test_embedding_pipeline.py` | 8 | `Document` | TEST / test | test-only | TEST_ONLY | test only | Tests use native fixtures; compatibility tests under LCI-7C | LCI-3A | verified import |
| LCI-INV-0142 | `langchain_core.documents` | `tests/unit/rag/graph/test_community_report_graph_indexer.py` | 4 | `Document` | TEST / test | test-only | TEST_ONLY | test only | Tests use native fixtures; compatibility tests under LCI-7C | LCI-4C | verified import |
| LCI-INV-0143 | `langchain_core.documents` | `tests/unit/rag/graph/test_graph_indexer_plugin_registry.py` | 4 | `Document` | TEST / test | test-only | TEST_ONLY | test only | Tests use native fixtures; compatibility tests under LCI-7C | LCI-4C | verified import |
| LCI-INV-0144 | `langchain_core.documents` | `tests/unit/rag/graph/test_graph_lifecycle_delete_sync.py` | 4 | `Document` | TEST / test | test-only | TEST_ONLY | test only | Tests use native fixtures; compatibility tests under LCI-7C | LCI-4C | verified import |
| LCI-INV-0145 | `langchain_core.documents` | `tests/unit/rag/graph/test_graph_provenance_retrieval_trace.py` | 4 | `Document` | TEST / test | test-only | TEST_ONLY | test only | Tests use native fixtures; compatibility tests under LCI-7C | LCI-4C | verified import |
| LCI-INV-0146 | `langchain_core.documents` | `tests/unit/rag/graph/test_graph_rag_neo4j_prod_contract.py` | 8 | `Document` | TEST / test | test-only | TEST_ONLY | test only | Tests use native fixtures; compatibility tests under LCI-7C | LCI-4C | verified import |
| LCI-INV-0147 | `langchain_core.documents` | `tests/unit/rag/graph/test_graph_rag_retriever.py` | 6 | `Document` | TEST / test | test-only | TEST_ONLY | test only | Tests use native fixtures; compatibility tests under LCI-7C | LCI-4C | verified import |
| LCI-INV-0148 | `langchain_core.documents` | `tests/unit/rag/graph/test_graph_rag_retriever_hardening.py` | 4 | `Document` | TEST / test | test-only | TEST_ONLY | test only | Tests use native fixtures; compatibility tests under LCI-7C | LCI-4C | verified import |
| LCI-INV-0149 | `langchain_core.documents` | `tests/unit/rag/graph/test_hybrid_retrieval_graph_channel.py` | 4 | `Document` | TEST / test | test-only | TEST_ONLY | test only | Tests use native fixtures; compatibility tests under LCI-7C | LCI-4C | verified import |
| LCI-INV-0150 | `langchain_core.documents` | `tests/unit/rag/graph/test_llm_graph_indexer.py` | 6 | `Document` | TEST / test | test-only | TEST_ONLY | test only | Tests use native fixtures; compatibility tests under LCI-7C | LCI-4C | verified import |
| LCI-INV-0151 | `langchain_core.documents` | `tests/unit/rag/indexing/test_indexing_manager_indexes_documents.py` | 9 | `Document` | TEST / test | test-only | TEST_ONLY | test only | Tests use native fixtures; compatibility tests under LCI-7C | LCI-3B | verified import |
| LCI-INV-0153 | `langchain_core.documents` | `tests/unit/rag/ingest/test_hierarchical_dual_index_wiring.py` | 8 | `Document` | TEST / test | test-only | TEST_ONLY | test only | Tests use native fixtures; compatibility tests under LCI-7C | LCI-2F | verified import |
| LCI-INV-0156 | `langchain_core.documents` | `tests/unit/rag/profiles/test_rag_profile_query_expansion_wiring.py` | 8 | `Document` | TEST / test | test-only | TEST_ONLY | test only | Tests use native fixtures; compatibility tests under LCI-7C | LCI-4A | verified import |
| LCI-INV-0157 | `langchain_core.documents` | `tests/unit/rag/profiles/test_rag_profile_validator.py` | 6 | `Document` | TEST / test | test-only | TEST_ONLY | test only | Tests use native fixtures; compatibility tests under LCI-7C | LCI-4A | verified import |
| LCI-INV-0163 | `langchain_core.documents` | `tests/unit/rag/tracking/test_rag_otel_spans.py` | 9 | `Document` | TEST / test | test-only | TEST_ONLY | test only | Tests use native fixtures; compatibility tests under LCI-7C | LCI-4A | verified import |
| LCI-INV-0165 | `langchain_core.documents` | `tests/unit/rag/vectorstore/test_lexical_hybrid.py` | 6 | `Document` | TEST / test | test-only | TEST_ONLY | test only | Tests use native fixtures; compatibility tests under LCI-7C | LCI-3C | verified import |
| LCI-INV-0166 | `langchain_core.documents` | `tests/unit/rag/vectorstore/test_vectorstore_contract.py` | 3 | `Document` | TEST / test | test-only | TEST_ONLY | test only | Tests use native fixtures; compatibility tests under LCI-7C | LCI-3C | verified import |
| LCI-INV-0167 | `langchain_core.documents` | `tests/unit/rag/vectorstore/test_vectorstore_cross_tenant_isolation.py` | 11 | `Document` | TEST / test | test-only | TEST_ONLY | test only | Tests use native fixtures; compatibility tests under LCI-7C | LCI-3C | verified import |
| LCI-INV-0168 | `langchain_core.documents` | `tests/unit/runtime/nexus/tools/test_parallel_semantic_batch_pattern.py` | 12 | `Document` | TEST / test | test-only | TEST_ONLY | test only | Tests use native fixtures; compatibility tests under LCI-7C | LCI-2F | verified import |
| LCI-INV-0169 | `langchain_core.documents` | `tests/unit/runtime/nexus/tools/test_tool_catalog_embedder.py` | 11 | `Document` | TEST / test | test-only | TEST_ONLY | test only | Tests use native fixtures; compatibility tests under LCI-7C | LCI-2F | verified import |
| LCI-INV-0170 | `langchain_core.documents` | `tests/unit/tools/providers/rag/test_rag_index_lifecycle_tools.py` | 6 | `Document` | TEST / test | test-only | TEST_ONLY | test only | Tests use native fixtures; compatibility tests under LCI-7C | LCI-4A | verified import |
| LCI-INV-0171 | `langchain_core.documents` | `tests/unit/tools/providers/rag/test_rag_ingest.py` | 40 | `Document` | TEST / test | test-only | TEST_ONLY | test only | Tests use native fixtures; compatibility tests under LCI-7C | LCI-4A | verified import |
| LCI-INV-0172 | `langchain_core.documents` | `tests/unit/tools/providers/rag/test_rag_retrieve.py` | 247 | `Document` | TEST / test | test-only | TEST_ONLY | test only | Tests use native fixtures; compatibility tests under LCI-7C | LCI-4A | verified import |
| LCI-INV-0173 | `langchain_core.documents` | `tests/unit/tools/providers/rag/test_rag_scope.py` | 11 | `Document` | TEST / test | test-only | TEST_ONLY | test only | Tests use native fixtures; compatibility tests under LCI-7C | LCI-4A | verified import |
| LCI-INV-0174 | `langchain_core.documents` | `tests/unit/tools/providers/vector_store/test_vector_store_tools.py` | 6 | `Document` | TEST / test | test-only | TEST_ONLY | test only | Tests use native fixtures; compatibility tests under LCI-7C | LCI-3C | verified import |
| LCI-INV-0175 | `langchain` | `pyproject.toml` | ? | (removed) | PLATFORM_FOUNDATION / packaging | [project].dependencies | PACKAGING_DEPENDENCY | removed in LCI-0C | optional extra or removed from core | LCI-0C | direct meta-package removed in LCI-0C; zero exact root langchain imports |
| LCI-INV-0176 | `langchain-core` | `pyproject.toml` | 79 | `langchain-core>=0.3,<2.0` | PLATFORM_FOUNDATION / packaging | [project].dependencies | PACKAGING_DEPENDENCY | core | optional extra or removed from core | LCI-7A | declaration in [project].dependencies |
| LCI-INV-0177 | `langchain-community` | `pyproject.toml` | 80 | `langchain-community>=0.3,<0.5` | PLATFORM_FOUNDATION / packaging | [project.optional-dependencies].rag-langchain-loaders | PACKAGING_DEPENDENCY | optional extra | `rag-langchain-loaders` | LCI-5C | declaration in [project.optional-dependencies].rag-langchain-loaders |
| LCI-INV-0178 | `langchain-openai` | `pyproject.toml` | 81 | `langchain-openai>=0.3,<2.0` | PLATFORM_FOUNDATION / packaging | [project].dependencies | PACKAGING_DEPENDENCY | core | optional extra or removed from core | LCI-7A | declaration in [project].dependencies |
| LCI-INV-0179 | `langchain-ollama` | `pyproject.toml` | 82 | `langchain-ollama>=0.2,<2.0` | PLATFORM_FOUNDATION / packaging | [project].dependencies | PACKAGING_DEPENDENCY | core | optional extra or removed from core | LCI-7A | declaration in [project].dependencies |
| LCI-INV-0180 | `langchain-text-splitters` | `pyproject.toml` | 180 | `langchain-text-splitters>=0.3,<2.0` | PLATFORM_FOUNDATION / packaging | [project.optional-dependencies].rag-langchain-splitters | PACKAGING_DEPENDENCY | optional | optional extra: rag-langchain-splitters | LCI-2E | declaration in [project.optional-dependencies].rag-langchain-splitters |
| LCI-INV-0181 | `langgraph` | `pyproject.toml` | 187 | `langgraph>=0.0.40` | PLATFORM_FOUNDATION / packaging | [project.optional-dependencies].langgraph-legacy | PACKAGING_DEPENDENCY | optional extra | optional extra or removed from core | LCI-8A | declaration in [project.optional-dependencies].langgraph-legacy |
| LCI-INV-0182 | `langchain-ollama` | `pyproject.toml` | 191 | `langchain-ollama>=0.2,<2.0` | PLATFORM_FOUNDATION / packaging | [project.optional-dependencies].llm-ollama | PACKAGING_DEPENDENCY | llm-ollama extra | optional extra or removed from core | LCI-6E | declaration in [project.optional-dependencies].llm-ollama |
| LCI-INV-0183 | `langchain-core` | `pyproject.toml` | 191 | `langchain-core>=0.3,<2.0` | PLATFORM_FOUNDATION / packaging | [project.optional-dependencies].llm-ollama | PACKAGING_DEPENDENCY | llm-ollama extra | optional extra or removed from core | LCI-6E | declaration in [project.optional-dependencies].llm-ollama |
| LCI-INV-0184 | `langchain-ollama` | `pyproject.toml` | 208 | `langchain-ollama>=0.2,<2.0` | PLATFORM_FOUNDATION / packaging | [project.optional-dependencies].llm-all | PACKAGING_DEPENDENCY | llm-all extra | optional extra or removed from core | LCI-6E | declaration in [project.optional-dependencies].llm-all |
| LCI-INV-0185 | `uv.lock` | `uv.lock` | — | `langchain-core`, `langchain-community`, `langchain-openai`, `langchain-ollama`, `langchain-text-splitters`, `langgraph` | PLATFORM_FOUNDATION / lockfile | generated resolver output | GENERATED_LOCK_ENTRY | installed when core/extras resolve | lock regenerated on packaging change | LCI-7A | aggregate row; transitive entries not inventoried individually |

## D. Public contract leak register

Only real leaks through public or shared core contracts (LangChain types in Intergrax ABI). Provider-local LangChain messages inside `LangChainOllamaAdapter` are **not** public contract leaks — see §E.

| Leaked type | Contract signature / location | Producers | Consumers | Future native contract | Architecture prerequisite | Implementation migration | Migration risk |
|-------------|------------------------------|-----------|-----------|------------------------|----------------|----------------|
| `langchain_core.documents.Document` | `BaseDocumentParser.parse` | RAG parsers | Ingest/chunk/embed/index | Native knowledge document | LCI-1A | LCI-2A | High |
| `langchain_core.documents.Document` | `BaseDocumentLoader` / handler contracts | Loaders/handlers | Parser pipelines | Native loader contract | LCI-1A | LCI-2B | High |
| `langchain_core.documents.Document` | Normalizer/metadata contracts | Normalizers | Parser/metadata pipelines | Native normalization contract | LCI-1A | LCI-2C | High |
| `langchain_core.documents.Document` | `BaseChunkingStrategy` / splitter contracts | Chunking strategies | Indexing | Native chunking contract | LCI-1A | LCI-2D | High |
| `langchain_core.documents.Document` | `BaseEmbeddingManager.embed_documents` | Embedding layer | Indexing | Native embedding contract | LCI-1A | LCI-3A | High |
| `langchain_core.documents.Document` | `IndexStrategy` | Indexing strategies | Ingest | Native indexing contract | LCI-1A | LCI-3B | High |
| `langchain_core.documents.Document` | `VectorStore` CRUD/search | Vector providers | Retrieval/tools | Native vector contract | LCI-1A | LCI-3C | High |
| `langchain_core.documents.Document` | Vector tenant isolation | Vector layers | Retrieval | Native tenant-safe records | LCI-1A | LCI-3C | High |
| `langchain_core.documents.Document` | Graph isolation | Graph layers | Retrieval | Native graph document | LCI-1A | LCI-4C | High |
| `langchain_core.documents.Document` | `RerankerInput` / rerank contracts | Rerankers | Hybrid retrieval | Native rerank candidate | LCI-1A | LCI-4B | Medium |
| `langchain_core.documents.Document` | `RerankProviderContract` | Integration rerank | RAG rerank | Native integration boundary | LCI-1A | LCI-4B | Medium |
| `langchain_core.documents.Document` | Graph indexer contracts | Graph indexers | Graph retrieval | Native graph document | LCI-1A | LCI-4C | High |

## E. Provider and compatibility register

| Item | Location | Classification | Migration task | Notes |
|------|----------|----------------|----------------|-------|
| LangChain messages (`AIMessage`, etc.) | `LangChainOllamaAdapter` (provider-local) | PROVIDER_BOUND_DEPENDENCY | LCI-6B / LCI-6E | Not a public contract leak; stays inside provider until optionalized |
| `tool_calls_from_langchain_message` | `intergrax/llm_adapters/contracts/tool_call.py` | Migration candidate (no direct LangChain import) | LCI-6E | `Any`-typed helper; does not import `langchain*` |
| Integration shared bridges | `intergrax/integrations/_shared/*` | OPTIONAL_COMPATIBILITY | LCI-3D | Map native records at boundary |
| LangChain community loaders | document parser providers | PROVIDER_BOUND_DEPENDENCY | LCI-5C | Optional extras with lazy import |
| LangChain recursive splitter | `langchain_recursive_chunking_strategy.py` | PROVIDER_BOUND_DEPENDENCY | LCI-2E | Optional provider behind `rag-langchain-splitters`; native recursive strategy is baseline; explicit registry registration |

## F. Dependency package register

Direct import counts are from §C import rows only (not packaging rows).

| Package | Why installed | production/runtime | tests | tooling | total imports | Core today | Target | Task |
|---------|---------------|-------------------:|------:|--------:|--------------:|------------|--------|------|
| langchain | Meta alignment (no direct imports) | 0 | 0 | 0 | 0 | yes | remove from core / optional extra | LCI-7A |
| langchain-core | Document/messages ABI leak | 47 | 51 | 1 | 99 | yes | compat extra only | LCI-7A |
| langchain-community | Community loader bridges | 3 | 0 | 0 | 3 | no | rag-langchain-loaders extra | LCI-5C |
| langchain-openai | Embedding wrappers | 0 | 0 | 0 | 0 | yes | native/SDK path | LCI-5B |
| langchain-ollama | Chat/embeddings shim | 2 | 0 | 0 | 2 | yes | native Ollama + optional compat | LCI-6E |
| langchain-text-splitters | Recursive splitter optional provider | 1 | 0 | 0 | 1 | optional | rag-langchain-splitters extra | LCI-2E |
| langgraph | Legacy orchestration adapters | 2 | 0 | 0 | 2 | optional extra only | retirement review | LCI-8A |

## G. LangGraph register

Guard: `scripts/maintenance/check_langgraph_not_required.py`. Each lazy import has a dedicated §C row (`LCI-8A`). Optional extra: `langgraph-legacy`. Docker `runtime-context` copies mirror the same two modules but are excluded from §C to avoid double-counting.

| Path | Line | Symbol | Classification | Task |
|------|-----:|--------|----------------|------|
| `intergrax/supervisor/supervisor_to_state_graph.py` | 198 | `END, StateGraph` | LEGACY_OPTIONAL | LCI-8A |
| `intergrax/websearch/integration/langgraph_nodes.py` | 11 | `add_messages` | LEGACY_OPTIONAL | LCI-8A |

## H. LCI-0B boundary enforcement baseline

| Metric | Value |
|--------|------:|
| production files scanned | 4393 |
| allowed-zone imports | 15 |
| grandfathered guarded imports | 21 |
| grandfather entries | 21 |
| new forbidden imports | 0 |
| stale grandfather entries | 0 |
| register path | `scripts/maintenance/langchain_boundary_grandfather.json` |
| checker path | `scripts/maintenance/check_langchain_boundary.py` |

Detailed inventory remains the evidence register.
The JSON file is the executable grandfather subset for protected production zones.

## I. Unverified items

Ollama native parity (`LCI-6C`), embedding numeric parity, vector store live round-trip: **UNVERIFIED** until respective proof tasks execute.

## LCI-3C boundary note

The vector-store provider port is native under LCI-3D; providers map `VectorStoreRecord` directly to SDK payloads and return `VectorStoreHit`.
