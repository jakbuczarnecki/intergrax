# RAG and Retrieval Architecture

**Status:** Canonical architecture  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Catalog:** Integration category `rag` in [`INTEGRATIONS.md`](../INTEGRATIONS.md)  
**Implementation plan:** [`plan/phases/rag-context-memory.md`](../plan/phases/rag-context-memory.md)

---

RAG and retrieval are Tier-0 platform capabilities consumed through Nexus policy and `ToolRuntime`. Agents MUST NOT embed vendor-specific vector stores directly.

## Design principles

- Retrieval is a **tool-backed** capability (`rag.*` tools), not ad-hoc agent HTTP calls.
- Indexing, chunking, and embedding profiles are integration contracts.
- Context assembly consumes retrieval results through the context engineering layer ([`CONTEXT_ENGINEERING.md`](CONTEXT_ENGINEERING.md)).
- Memory scopes ([`MEMORY_ARCHITECTURE.md`](../MEMORY_ARCHITECTURE.md)) and RAG indices are distinct namespaces.

## Related canon

- Platform foundation §7.1.2 (integration categories) — [`PLATFORM_FOUNDATION.md`](PLATFORM_FOUNDATION.md)
- Tool runtime — [`TOOLS_RUNTIME.md`](TOOLS_RUNTIME.md)
- Unified execution §42.12 ToolRuntime — [`UNIFIED_EXECUTION_RUNTIME.md`](UNIFIED_EXECUTION_RUNTIME.md)

See [`INTEGRAX_HARNESS_AUDIT_MAP.md`](../INTEGRAX_HARNESS_AUDIT_MAP.md) layer 14 (RAG and Retrieval).
