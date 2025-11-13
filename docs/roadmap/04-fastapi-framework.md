# FastAPI-Based Framework for Thematic AI Applications

**Status:** Planned  
**Owner:** Artur Czarnecki  
**Start Date:** _(add date)_  
**Target Version:** v1.x  
**Last Updated:** _(add date)_

---

## Goal
Provide a production-grade **FastAPI framework** that serves as the runtime backbone for intergrax applications (document chat, business agents, company intelligence, scoring).  
The framework must expose consistent HTTP and WebSocket interfaces, background task orchestration, dependency injection for core intergrax components, and secure multi-tenant session handling.

---

## Description
This milestone delivers `intergrax.api` — a modular FastAPI layer that standardizes how intergrax apps are built and deployed.  
It should offer reusable routers, middleware, and DI containers that wire together RAG, Agents, Supervisor, Tools, and Vectorstores, while remaining lightweight and testable.

Primary capabilities:
- **Unified API surface** for chat, file ingestion, retrieval, tool-calls, and agent workflows.
- **Streaming** via Server-Sent Events or WebSockets for token-by-token responses.
- **Background jobs** for ingestion, embedding, summarization, and long-running tasks.
- **Session & identity** to bind uploads, indices, and memory to users/tenants.
- **Config-driven bootstrapping** for environments (dev/stage/prod) and providers.

---

## Key Components

| Component | Role |
|-----------|------|
| **`ApiAppFactory`** | Builds the FastAPI app with configured routers, middleware, and DI. |
| **`DependencyContainer`** | Provides instances of Vectorstore, RAG, Supervisor, Tools, Memory. |
| **`ChatRouter`** | Endpoints for conversational flows, streaming, and tool-execution. |
| **`FilesRouter`** | Upload, ingestion, extraction, and indexing of documents. |
| **`RagRouter`** | Retrieval, re-ranking, and citation endpoints for testing and ops. |
| **`AgentsRouter`** | Plan/run agent workflows, inspect traces, and get artifacts. |
| **`AuthMiddleware`** | Token- or session-based auth; request-scoped tenant resolution. |
| **`TasksBackend`** | Background task runner (FastAPI built-ins or Celery/RQ/Arq adapter). |
| **`Observability`** | Structured logging, request IDs, metrics, tracing. |

---

## Implementation Plan

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Define API surface and DI contracts (interfaces for RAG, Tools, Supervisor). | ☐ |
| 2 | Implement `ApiAppFactory` and base middlewares (CORS, auth, logging). | ☐ |
| 3 | Add `ChatRouter` with streaming (SSE/WebSocket) and tool-call plumbing. | ☐ |
| 4 | Add `FilesRouter` for uploads, ingestion pipeline, and progress tracking. | ☐ |
| 5 | Add `RagRouter` and `AgentsRouter` for retrieval tests and workflow runs. | ☐ |
| 6 | Integrate `TasksBackend` for background embedding/summarization jobs. | ☐ |
| 7 | Hardening: rate limits, pagination, error model, OpenAPI docs, tests. | ☐ |

---

## API Endpoints (initial sketch)

- `POST /v1/chat` — start/continue a conversation; optional tool-calls.  
- `GET /v1/chat/stream` — SSE or WebSocket for streaming tokens and tool events.  
- `POST /v1/files/upload` — upload files; returns file IDs and queued task IDs.  
- `POST /v1/files/ingest` — trigger ingestion/indexing for uploaded files.  
- `GET /v1/rag/query` — retrieve passages with metadata and citations.  
- `POST /v1/agents/run` — execute a named agent workflow with inputs.  
- `GET /v1/health` — liveness/readiness; `GET /v1/version` — build info.

_All endpoints must be tenant-aware and auditable._

---

## Progress Journal

| Date | Commit / Ref | Summary |
|------|---------------|---------|
| YYYY-MM-DD |  | Baseline app factory and dependency container. |
| YYYY-MM-DD |  | Streaming chat route and unified response envelope. |
| YYYY-MM-DD |  | File ingestion pipeline wired to background tasks. |
| YYYY-MM-DD |  | RAG and Agents routers with integration tests. |
| YYYY-MM-DD |  |  |

---

## Notes & Dependencies

- **Reusability:** The API layer must only **compose** existing intergrax components (`intergrax-rag`, `intergrax-llm`, `intergrax-supervisor`, `intergrax-tools`, `intergrax-memory`) — no duplicated business logic.
- **Streaming:** Prefer WebSocket; provide SSE fallback. Ensure backpressure handling.
- **Background tasks:** Start with FastAPI `BackgroundTasks`; abstract to support Celery/RQ/Arq when needed.
- **Security:** JWT or session tokens; per-tenant isolation for files, indices, and memory; signed upload URLs if using object storage.
- **Observability:** Request/trace IDs, structured logs, latency and queue metrics, error taxonomy.
- **Docs & clients:** Generate OpenAPI; provide minimal Python/JS client stubs.
- **Compatibility:** Expose bridges for MCP tools (`intergraxMcpBridge`) so the same capabilities are available to internal/external agents.

---

## Related Documents

---

**Maintainer:** Artur Czarnecki  
**Repository:** [intergrax](https://github.com/jakbuczarnecki/intergrax-ai)
