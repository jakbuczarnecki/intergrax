Intergrax Roadmap

Last updated: 2026-01-20

This is a living engineering roadmap / TODO list.
It reflects current development priorities and may change frequently.

Legend:
[P0] — Production blockers (must-have before first E2E agent)
[P1] — Required for stable productization
[P2] — Important but not blocking first real use
[P3] — Nice-to-have / future

DONE

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


[PARTIAL] Tests — unit and integration coverage for runtime, trace, retry and cost
[PARTIAL] Tests — minimal unit and integration test coverage for all P0 foundations

🟥 Production foundations (P0 — must be done before first E2E agent)

[P0] Artifacts — implement persistent artifact store and reference linking from trace
[P0] Runtime — implement run replay and inspection (ability to reconstruct a run from trace + artifacts)
[P0] Runtime — implement idempotency and safe retry for tool calls with side effects
[P0] Sessions — implement production storage adapters for sessions and user profiles (DB-backed)
[P0] Organization profiles — implement production storage and isolation per organization
[P0] Security — implement PII redaction and multi-tenant isolation for logs, memory, artifacts and vector stores
[P0] Tooling — implement a formal tool/skill contract (input/output schema, error taxonomy, permissions)
[P0] Tooling — implement permission scopes and auditing for tool usage
[P0] Guardrails — implement minimal hard gates (output validation, tool gating, pii-safe logging)
[P0] Eval — implement an evaluation harness for agent quality, regressions and cost tracking

🟧 Productization & stability (P1)

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

🟨 Agents & product demos (P1–P2)

[P1] Agent — design and implement a company profile agent (first E2E product)
[P1] Agent — design and implement an IT headhunter agent
[P2] Agent — create a virtual company team with a supervisor and inter-agent communication
[P2] Agent — implement an agent similar to Google NotebookLM as a demonstration product
[P2] Agent — implement an agent that searches project directories and creates summaries and comments
[P2] Agent / Tool — Text-to-SQL

🟦 Integrations & external systems (P2)

[P2] Integrations — Google Docs
[P2] Integrations — Google Drive
[P2] Integrations — Google Sheets
[P2] Integrations — Pinecone
[P2] Integrations — Firebase
[P2] Integrations — SerpAPI
[P2] Integrations — DuckDuckGo
[P2] Integrations — other useful and well-known APIs

🟩 Advanced capabilities (P3)

[P3] Cloud — create mechanisms for cloud computing integrations (Azure, AWS, etc.)
[P3] Voice agent — create an example voice chatbot
[P3] Large data handling — scalable reasoning over large datasets (source code, corpora)
[P3] Critics in CoT — implement self-awareness and auto-correction modules