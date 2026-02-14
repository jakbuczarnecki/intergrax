# Intergrax Roadmap

Last updated: 2026-02-14

This is a living engineering roadmap / TODO list.
It reflects current development priorities and may change frequently.

Legend:
[P0] — Production blockers (must-have before first E2E agent)
[P1] — Required for stable productization
[P2] — Important but not blocking first real use
[P3] — Nice-to-have / future

---

## CORE AGENT — MUST-HAVE BLOCKERS (subset of P0)

After completing this subset, we can start building the first E2E agent.

* [P0] Tooling — implement a formal tool/skill contract (input/output schema, error taxonomy, permissions)
* [P0] Tooling — implement permission scopes and auditing for tool usage
* [P0] Guardrails — implement minimal hard gates (output validation, tool gating, pii-safe logging)
* [P0] Security — implement PII redaction and multi-tenant isolation for logs, memory, artifacts and vector stores
* [P0] Sessions — implement production storage adapters for sessions and user profiles (DB-backed)
* [P0] Organization profiles — implement production storage and isolation per organization
* [P0] Runtime — implement run replay and inspection (ability to reconstruct a run from trace + artifacts)
* [P0] Runtime — implement idempotency and safe retry for tool calls with side effects
* [P0] Eval — implement an evaluation harness for agent quality, regressions and cost tracking

---

## DONE

[DONE] Logging — global settings and global contracts — simplify logging
[DONE] Diagnostics and tracing — in runtime and RuntimeState — replace dictionaries with typed structures

[DONE] Run record — persistent run record (trace + metadata + LLM cost) — in-memory MVP
[DONE] Trace events — persistent trace event storage — in-memory backend
[DONE] Runtime loop — retry and escalation at run level
[DONE] Budget control — LLM cost tracking per run (usage + stats)
[DONE] Error handling — typed error taxonomy and mapping to retry policies
[DONE] Runtime loop — implement timeout enforcement and fallback strategies
[DONE] Human-in-the-loop — implement base HITL escalation and clarification mechanism
[DONE] Budget control — create architecture for defining and enforcing budget policies (tokens, time, tool calls, replans)
[DONE] Prompting — all LLM instructions migrated to YAML prompt registry (no hard-coded prompts)
[DONE] Prompting — versioning, pinning and metadata for prompts implemented
[DONE] Prompting — prompts moved outside source code with nested folder support and backward-compatible lookup
[DONE] Artifacts — implement persistent artifact store and reference linking from trace

[PARTIAL] Tests — unit and integration coverage for runtime, trace, retry and cost
[PARTIAL] Tests — minimal unit and integration test coverage for all P0 foundations

---

## Production foundations (P0 — must be done before first E2E agent)

[P0] Runtime — implement run replay and inspection (ability to reconstruct a run from trace + artifacts)
[P0] Runtime — implement idempotency and safe retry for tool calls with side effects
[P0] Sessions — implement production storage adapters for sessions and user profiles (DB-backed)
[P0] Organization profiles — implement production storage and isolation per organization
[P0] Security — implement PII redaction and multi-tenant isolation for logs, memory, artifacts and vector stores
[P0] Tooling — implement a formal tool/skill contract (input/output schema, error taxonomy, permissions)
[P0] Tooling — implement permission scopes and auditing for tool usage
[P0] Guardrails — implement minimal hard gates (output validation, tool gating, pii-safe logging)
[P0] Eval — implement an evaluation harness for agent quality, regressions and cost tracking

---

## P0 additions: production-grade platform primitives (enterprise expectations)

[P0] Observability — define a first-class telemetry contract (traces, metrics, logs) and correlation IDs across all layers, compatible with OpenTelemetry
[P0] Observability — implement exporters/adapters: metrics to Prometheus, dashboards readiness for Grafana, error reporting adapter for Sentry
[P0] Runtime — implement “async run execution” contract (enqueue run, worker consumes, run status updates) with a strict run lifecycle model (PENDING/RUNNING/SUCCEEDED/FAILED/CANCELLED)
[P0] Queueing — implement a message broker port (publish/consume/ack/nack/delay/dead-letter) + in-memory backend for tests + at least one production backend (start with Redis Streams or RabbitMQ)
[P0] Reliability — implement outbox/inbox patterns for tool side effects + deduplication store (idempotency keys) + poison message handling (DLQ)
[P0] Runtime — implement cancellation/interrupt propagation (API -> queue -> worker -> running steps), including safe cleanup and final state persistence
[P0] Secrets — implement a SecretStore port + production adapter for HashiCorp Vault (fallback env adapter for dev)
[P0] Config — implement config layering and environment profiles (dev/stage/prod) with strict schema validation (no implicit defaults in prod)
[P0] Deployment readiness — health/readiness endpoints; structured logs with sampling; safe PII redaction; request/run IDs everywhere
[P0] Rate limiting — production-grade rate limiter port + adapter (Redis-based), supporting per-tenant/per-route/per-tool quotas
[P0] Data retention — retention policies for traces, artifacts, sessions, and evaluation datasets (TTL, archival)

---

## Productization & stability (P1)

[P1] Tests — convert notebooks into production-grade unit and integration tests
[P1] Runtime — create lifecycle events to notify users about reasoning and pipelines, and allow interruption when needed
[P1] Memory improvement — implement mechanisms for improving reasoning while history profiles grow (summaries, compression)
[P1] LLM Adapters — change generate_messages to return a custom object instead of a raw string
[P1] LLM Adapters — implement full-usage stream_messages
[P1] Runtime loop — handle long user questions by splitting them into manageable parts
[P1] Runtime loop — replace strategy flags with configuration-based pipeline selection
[P1] Skills — implement a skill mechanism similar to Claude
[P1] Pipelines — refactor pipeline architecture for customization (e.g. LangGraph-style), allow custom reasoning blocks
[P1] MCP — create foundations for MCP configurations for backend services
[P1] API / FastAPI — create foundations for API / FastAPI configurations
[P1] Logging — attach logger to other system components
[P1] Guardrails — extended policy and safety layer (advanced validators, classifiers, moderation)

[P1] RAG — advanced document parsing pipeline with streaming (ports/adapters for LlamaParse, Docling, Unstructured)
[P1] RAG — chunking strategy framework (semantic chunking + configurable policies + evaluation hooks)
[P1] RAG — metadata enrichment framework during ingestion (source typing, section/page anchors, timestamps, tags, entity hints)
[P1] RAG — vector store metadata management contract (typed metadata schema + normalization + validation)
[P1] RAG — metadata pre-filtering support end-to-end (query-time filters + index-time constraints + tenant isolation hooks)
[P1] RAG — hybrid retrieval contract (BM25/keyword + dense) with Reciprocal Rank Fusion (RRF) and deterministic scoring traces
[P1] RAG — reranker contract (typed interface + features + telemetry), pluggable backends (cross-encoder / LLM-based)
[P1] RAG — large context strategy layer (budgeted context assembly, dedupe, diversity, citations/anchors, “context packing” policies)

---

## P1 additions: production-scale execution, observability, scaling

[P1] Broker backends — add production adapters for Apache Kafka (event streaming) and RabbitMQ (task queue); document semantics and recommended use-cases
[P1] Worker framework — implement a worker runner abstraction; provide adapters for Celery (Python) and a thin “native worker” mode (no external deps)
[P1] Workflows — add an optional “durable workflow” port and adapter for Temporal for long-running, resumable agent processes
[P1] Observability — integrate LLM-level traces and prompt/version visibility via adapters (e.g., Langfuse / LangSmith), fed from existing Intergrax trace/run record (export pipeline, not replacement)
[P1] RAG observability — add RAG metrics (retrieval latency, hit rate, context length, rerank stats, groundedness signals) and connect to eval datasets; optional adapter to Arize Phoenix for investigations
[P1] Caching — implement semantic cache port (prompt+tools context hash) + Redis adapter + cache policy controls (TTL, tenant scope, invalidation)
[P1] Circuit breakers — implement per-provider and per-tool circuit breakers (open/half-open/closed) based on error rates and latency
[P1] Provider routing — implement model routing policy (cost/latency/quality tiers), fallback chains, and “safe degrade” modes
[P1] Security — integrate policy engine hooks (tool gating, data source gating) to support external policy engines later (e.g., OPA-style), while keeping core contracts typed
[P1] Infra integration — add Kubernetes-ready deployment patterns: worker scaling and queue depth metrics; optional autoscaling hook for Kubernetes environments
[P1] CI quality gates — implement CI pipelines for eval regressions (quality + cost) and load tests for critical endpoints and worker throughput

[P1] RAG scaling — define distributed ingestion/extraction execution model (queue-backed jobs, progress, retries, idempotency, per-tenant isolation)
[P1] RAG scaling — implement distributed indexing pipeline primitives (sharded ingestion, backpressure, incremental updates, reindex strategies)
[P1] RAG scaling — implement large-context handling under load (context assembly latency budgets, caching of retrieval/rerank, partial results streaming)

---

## Agent Factory (P1 — foundations for building specialized, product-grade agents)

[P1] Agent Factory — define a first-class “Agent Pipeline” concept (specialized pipeline = product agent behavior), with strict typed contracts and invariants (run -> RuntimeAnswer)
[P1] Agent Factory — introduce a typed AgentSpec/AgentProfile (agent identity, purpose, pipeline selection, tool scopes, budget policy, memory/RAG sources, output contract)
[P1] Agent Factory — implement prompt packs per pipeline phase (router/planner/step_decision/tool_use/critic/finalizer) using YAML prompt registry (versioned + pinned), with per-agent overrides
[P1] Agent Factory — create a pipeline base toolkit (reusable building blocks) for specialized pipelines (phase runner, stop conditions, clarification gates, progress checks), without restricting custom implementations
[P1] Agent Factory — implement per-agent tool permission scopes and auditing integration (agent -> allowed tools + allowed data sources), enforced by runtime policies
[P1] Agent Factory — implement per-agent memory/RAG/web configuration (which sources are available, retrieval limits, redaction rules), provided via typed context/facades (no raw dicts)
[P1] Agent Factory — add contract tests for pipelines/agents (behavioral invariants: tool gating, clarification behavior, output schema, stop reasons, budget limits)
[P1] Agent Factory — create an agent catalog/registry (discoverability, versioning, rollout strategy), enabling selection by config and safe upgrades/rollbacks

[P1] Agent Factory (RAG) — per-agent RAG profiles (ingestion pipeline selection, chunking policy, retrieval policy, rerank policy, context budget policy)
[P1] Agent Factory (RAG) — per-agent data source authorization (allowed corpora, allowed namespaces, metadata filter constraints, tenant boundaries)

---

## P1 additions: agent ops readiness

[P1] Agent Factory — add “deployment descriptors” for agents (resource needs, queue/worker class, concurrency limits, model routing policy, cache policy)
[P1] Agent Factory — add “SLO profile” per agent (latency target, max cost per run, acceptable tool failure budget)
[P1] Agent Factory — add offline replay suites (golden runs) + production “shadow mode” execution for safe rollouts

[P1] Agent Ops (RAG) — define RAG SLOs per agent (retrieval latency, context size, rerank time, index freshness) and enforce budgets via policies

---

## Agents & product demos (P1–P2)

[P1] Agent — design and implement a company profile agent (first E2E product)
[P1] Agent — design and implement an IT headhunter agent
[P2] Agent — create a virtual company team with a supervisor and inter-agent communication
[P2] Agent — implement an agent similar to Google NotebookLM as a demonstration product
[P2] Agent — implement an agent that searches project directories and creates summaries and comments
[P2] Agent / Tool — Text-to-SQL

---

## P2 additions: production-grade demos

[P2] Agent — “Ops Copilot” agent that inspects traces/metrics/logs and proposes mitigations (run replay + diagnostics-first demo)
[P2] Agent — “RAG Quality Analyst” agent that runs eval suites, finds regressions, and suggests dataset fixes (query rewrite, chunking, rerank)

[P2] Demo (RAG) — NotebookLM-class pipeline demo: multi-doc ingestion, citations/anchors, hybrid + RRF, rerank, and large-context assembly under strict budgets
[P2] Demo (RAG) — distributed ingestion/indexing demo: queue-backed parsing and indexing, incremental updates, and tenant-isolated namespaces

---

## Integrations & external systems (P2)

[P2] Integrations — Google Docs
[P2] Integrations — Google Drive
[P2] Integrations — Google Sheets
[P2] Integrations — Pinecone
[P2] Integrations — Firebase
[P2] Integrations — SerpAPI
[P2] Integrations — DuckDuckGo
[P2] Integrations — other useful and well-known APIs

[P2] Integrations (RAG parsing) — LlamaParse, Docling, Unstructured (document parsing/extraction) with streaming support
[P2] Integrations (Rerankers) — integrate SOTA rerank backends (cross-encoders and/or provider rerank APIs), exposed via the typed reranker port

---

## P2 additions: production toolchain integrations

[P2] Integrations (LLM providers) — OpenAI, Azure OpenAI (Microsoft Azure), Anthropic, Google (Gemini), Amazon Web Services (Bedrock), plus “local/self-hosted” adapter family (vLLM/llama.cpp/Ollama style)
[P2] Integrations (vector stores) — Qdrant, Milvus, Weaviate, Pinecone, PostgreSQL + pgvector
[P2] Integrations (observability) — Prometheus/Grafana, OpenTelemetry collectors, Sentry, Langfuse/LangSmith, Arize Phoenix
[P2] Integrations (queue/brokers) — Kafka, RabbitMQ, Redis Streams; optional NATS later
[P2] Integrations (storage) — S3-compatible object store (MinIO/S3) for artifacts and Claim-Check payloads
[P2] Integrations (CI/CD) — GitHub Actions/GitLab CI templates for eval gating and load tests; publish artifacts (traces/eval datasets)
[P2] Integrations (infra-as-code) — Terraform templates for minimal production stack (DB + Redis + broker + workers + OTel collector)

---

## Advanced capabilities (P3)

[P3] Cloud — create mechanisms for cloud computing integrations (Azure, AWS, etc.)
[P3] Voice agent — create an example voice chatbot
[P3] Large data handling — scalable reasoning over large datasets (source code, corpora)
[P3] Critics in CoT — implement self-awareness and auto-correction modules

[P3] RAG — agentic chunking (LLM/agent-guided segmentation, structure discovery, hierarchical chunk trees) with safe deterministic fallbacks
[P3] RAG — advanced “large context” solutions (hierarchical retrieval, multi-stage condensation, map-reduce retrieval plans, query decomposition)
[P3] RAG — adaptive retrieval strategies (intent classification -> retrieval plan selection; multi-hop retrieval; iterative retrieval with stop conditions)

---

## P3 additions: big-scale platform features

[P3] Multi-region — active/active patterns for runs, queues, and vector stores; latency-aware routing
[P3] Data governance — lineage for artifacts/context, audit trails, compliance exports
[P3] Advanced scaling — workload-aware autoscaling (queue depth, token usage, model latency) and cost-aware scheduling
[P3] Advanced safety — classifier ensembles, advanced policy engine integrations, content provenance/grounding guarantees

---

## Notes: platform stance (implicit requirements now captured by roadmap)

Intergrax core remains dependency-light; production tools integrate via typed ports/adapters.

Every “production primitive” must have: (1) typed contract, (2) in-memory test adapter, (3) at least one production adapter, (4) deterministic test coverage, (5) telemetry hooks.

Queueing/workers/workflows are treated as first-class runtime backends (not “nice to have”), because at scale you cannot run everything in-process.
