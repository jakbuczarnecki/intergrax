<!--
© Artur Czarnecki. All rights reserved.
Intergrax framework – proprietary and confidential.
Use, modification, or distribution without written permission is prohibited.
-->

# LANGCHAIN_INDEPENDENCE — domain plan cross-references

**Parent plan:** [../LANGCHAIN_INDEPENDENCE.md](../LANGCHAIN_INDEPENDENCE.md)
**Feature architecture:** [../../architecture/LANGCHAIN_INDEPENDENCE.md](../../architecture/LANGCHAIN_INDEPENDENCE.md)

When an LCI-* task becomes active, add a concrete implementation row to the owning domain plan file listed below. This satellite does **not** add those rows in LCI-0A.

| LCI task | Owning domain plan file | Row topic (future) |
|----------|-------------------------|-------------------|
| LCI-0A | Feature plan only | Inventory and roadmap |
| LCI-0B | docs/plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md | LangChain boundary guard CI |
| LCI-0C | docs/plan/PLATFORM_FOUNDATION.md | Dependency range hardening and clean-install smoke |
| LCI-1A | docs/plan/RAG.md | Native knowledge document architecture |
| LCI-1B | docs/plan/RAG.md | Native knowledge document implementation |
| LCI-1C | docs/plan/RAG.md | LangChain document compatibility bridge |
| LCI-1D | docs/plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md | Knowledge document conformance gate |
| LCI-2A | docs/plan/RAG.md, docs/plan/INTEGRATIONS.md | Document parser contract migration |
| LCI-2B | docs/plan/RAG.md | Document loader and handler migration |
| LCI-2C | docs/plan/RAG.md | Normalization and metadata pipeline migration |
| LCI-2D | docs/plan/RAG.md | Chunking contract migration |
| LCI-2E | docs/plan/RAG.md | LangChain splitter optionalization |
| LCI-2F | docs/plan/RAG.md, docs/plan/ORCHESTRATION.md | Ingest pipeline and Nexus ingestion migration |
| LCI-3A | docs/plan/RAG.md, docs/plan/LLM_ADAPTERS.md | Embedding contract migration |
| LCI-3B | docs/plan/RAG.md | Indexing contract and strategy migration |
| LCI-3C | docs/plan/RAG.md | Vector store contract and tenant isolation |
| LCI-3D | docs/plan/RAG.md, docs/plan/INTEGRATIONS.md | Vector store provider adapter migration |
| LCI-4A | docs/plan/RAG.md | Retrieval result contract migration |
| LCI-4B | docs/plan/RAG.md, docs/plan/INTEGRATIONS.md | Reranking contract migration |
| LCI-4C | docs/plan/RAG.md | Graph RAG document contract migration |
| LCI-4D | docs/plan/MEMORY.md, docs/plan/MODALITY.md, docs/plan/RAG.md | Memory/multimedia/legacy/evaluation leak cleanup |
| LCI-5A | docs/plan/RAG.md | Native text document loader |
| LCI-5B | docs/plan/RAG.md, docs/plan/LLM_ADAPTERS.md | Native OpenAI embedding provider |
| LCI-5C | docs/plan/INTEGRATIONS.md | LangChain loaders and embeddings optionalization |
| LCI-6A | docs/plan/LLM_ADAPTERS.md | Native Ollama adapter architecture and parity matrix |
| LCI-6B | docs/plan/LLM_ADAPTERS.md | Native Ollama adapter implementation |
| LCI-6C | docs/plan/LLM_ADAPTERS.md | Native Ollama live parity proof |
| LCI-6D | applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md, docs/plan/LLM_ADAPTERS.md | LKW and Token Optimization native Ollama cutover |
| LCI-6E | docs/plan/LLM_ADAPTERS.md | LangChain Ollama compatibility optionalization |
| LCI-7A | docs/plan/PLATFORM_FOUNDATION.md | LangChain optional extras packaging |
| LCI-7B | docs/plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md | LangChain-free core installation gate |
| LCI-7C | docs/plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md | LangChain compatibility installation gate |
| LCI-7D | docs/plan/PLATFORM_FOUNDATION.md | Documentation and generator closeout |
| LCI-8A | docs/plan/ORCHESTRATION.md | LangGraph legacy retirement review |