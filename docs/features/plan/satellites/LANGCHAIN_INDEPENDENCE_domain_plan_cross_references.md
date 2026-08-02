<!--
© Artur Czarnecki. All rights reserved.
Intergrax framework – proprietary and confidential.
Use, modification, or distribution without written permission is prohibited.
-->

# LANGCHAIN_INDEPENDENCE — domain plan cross-references

**Parent plan:** [`../LANGCHAIN_INDEPENDENCE.md`](../LANGCHAIN_INDEPENDENCE.md)
**Feature architecture:** [`../../architecture/LANGCHAIN_INDEPENDENCE.md`](../../architecture/LANGCHAIN_INDEPENDENCE.md)

When an `LCI-*` task becomes active, add a concrete implementation row to the owning domain plan file listed below. This satellite does **not** add those rows in `LCI-0A`.

| LCI task | Owning domain plan file | Row topic (future) |
|----------|-------------------------|-------------------|
| LCI-0A | Feature plan only | Inventory and roadmap |
| LCI-0B | `docs/plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md` | LangChain boundary guard CI |
| LCI-0C | `docs/plan/PLATFORM_FOUNDATION.md` | Optional extras / minimal install design |
| LCI-1A | `docs/plan/RAG.md` | Native knowledge document architecture |
| LCI-1B | `docs/plan/RAG.md` | Native knowledge document implementation |
| LCI-1C | `docs/plan/RAG.md` | LangChain compat bridge module |
| LCI-1D | `docs/plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md` | Contract conformance tests |
| LCI-2A | `docs/plan/RAG.md`, `docs/plan/INTEGRATIONS.md` | Document loaders and parser bridges |
| LCI-2B | `docs/plan/RAG.md` | Chunking pipeline |
| LCI-2C | `docs/plan/RAG.md` | Recursive chunking native replacement |
| LCI-2D | `docs/plan/RAG.md` | Contextual enrichment |
| LCI-2E | `docs/plan/RAG.md` | Ingest pipeline |
| LCI-2F | `docs/plan/ORCHESTRATION.md` | Nexus ingestion service |
| LCI-3A | `docs/plan/RAG.md`, `docs/plan/LLM_ADAPTERS.md` | Embedding providers |
| LCI-3B | `docs/plan/RAG.md` | Indexing pipeline |
| LCI-3C | `docs/plan/RAG.md`, `docs/plan/INTEGRATIONS.md` | Vector store providers |
| LCI-3D | `docs/plan/RAG.md` | Tenant isolation contracts |
| LCI-4A | `docs/plan/RAG.md` | Retrieval pipeline |
| LCI-4B | `docs/plan/RAG.md`, `docs/plan/INTEGRATIONS.md` | Reranking |
| LCI-4C | `docs/plan/MEMORY.md` | Memory indexing |
| LCI-4D | `docs/plan/RAG.md` | Graph RAG indexers |
| LCI-5A | `docs/plan/MODALITY.md` | Multimedia loaders |
| LCI-5B | `docs/plan/INTEGRATIONS.md` | Document parser optional bridges |
| LCI-5C | `docs/plan/INTEGRATIONS.md` | Vector store shared bridges |
| LCI-6A | `docs/plan/RAG.md` | Legacy rag_answers isolation |
| LCI-6B | `docs/plan/RAG.md` | Evaluation harness |
| LCI-6C | `docs/plan/LLM_ADAPTERS.md` | Native Ollama adapter |
| LCI-6D | `docs/plan/LLM_ADAPTERS.md` | Tool-call helper boundary |
| LCI-6E | `applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md` | LKW proof scheduling (client) |
| LCI-7A | `docs/plan/PLATFORM_FOUNDATION.md` | Packaging closeout |
| LCI-7B | `docs/plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md` | CI conformance |
| LCI-7C | `docs/plan/PLATFORM_FOUNDATION.md` | Doc generator migration |
| LCI-7D | `docs/plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md` | Test fixture migration |
| LCI-8A | `docs/plan/ORCHESTRATION.md` | LangGraph retirement review |
