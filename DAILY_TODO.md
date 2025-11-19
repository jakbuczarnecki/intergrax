# Intergrax — Daily Task Log

This file tracks daily micro-tasks related to the development of the Intergrax framework.
Tasks here are intentionally small, incremental, and actionable (e.g., “create script”, “add class”, “write example”, “test adapter”).

Status legend:
- [ ]  – Not started
- [*]  – In progress (actively working on it today)
- [-]  – Started but paused / postponed
- [+]  – Completed

---

## 2025-11-18
[+] Added notebook `notebooks/langgraph/hybrid_multi_source_rag_langgraph.ipynb` — a complete end-to-end demonstration of hybrid RAG combined with web search and LangGraph orchestration.


## 2025-11-19
[+] Implement `WebPageScraper` for structured HTML extraction in `websearch.fetcher`:
    - fetch raw HTML
    - extract readable content (readability-lxml style)
    - collect metadata (title, description, og:*, canonical, lang)
    - return unified `WebDocument` object for downstream RAG and LangGraph pipelines


## 2025-11-20
[ ] Implement the `UnifiedConversationEngine` — a single entrypoint class responsible for routing user messages through memory, attachments, RAG, web search, and tool execution.

Responsibilities:
    - ingest message (text or attachment)
    - embed + index files and conversational context
    - retrieve context (RAG + memory + web search if enabled)
    - select and call tools when relevant (OpenAI tools-compatible)
    - handle streaming responses
    - persist messages and memory to storage layer (SQLite for now)

Target location:
    intergrax/chat/unified_engine.py


---

## Backlog / Ideas
(Tasks that are not scheduled yet but should remain visible)

Core Framework Enhancements

[ ] Add unified storage layer abstraction (storages/) with initial adapters: SQLite, MySQL, PostgreSQL
[ ] Add VectorStore auto-migration and schema management
[ ] Add Redis-based ephemeral memory cache for fast session recall
[ ] Add configurable file-based persistence (.intergrax/session_id/*.json)

Advanced Conversational Memory

[ ] Add Long-term memory module with scoring: relevance, recency, importance
[ ] Add retrieval-based persona reconstruction (like GPT-4 conversational style persistence)
[ ] Add memory trimming policy engine (entropy-based summarization + embeddings)
[ ] Add plugin system for memory observers (track entities, TODOs, goals, documents)

Multimodal Context Handling (ChatGPT-Style Attachments)

[ ] Build AttachmentRouter for messages → handling text/pdf/docx/csv/xlsx/audio/video/images
[ ] Add automatic embedding + metadata injection on message send
[ ] Add delayed ingestion queue (processing happens async, not blocking the chat)
[ ] Add first-class support for multimodal RAG: images, tables, video transcript anchors
[ ] Add conversation-level citation tracking for multi-source RAG

Agents System

[ ] Build AgentProtocol and AgentRegistry for reusable agent behaviors
[ ] Add built-in Web Research Agent (LangGraph + Intergrax WebSearch + RAG)
[ ] Add Multimodal Analysis Agent (video+image+audio → structured summary)
[ ] Add API Tooling Agent that writes documentation or README based on repo context
[ ] Add task-level retry strategy, uncertainty scoring and confidence-based fallback responses

Chat + Tools Architecture

[ ] Add tool metadata introspection: “What tools can you run?” style self-awareness
[ ] Add automatic tool selection ranking (embedding similarity + schema match)
[ ] Add structured reasoning trace streaming (OpenAI-style logprobs/reasoning)
[ ] Build full OpenAI-compatible /v1/responses HTTP endpoint
[ ] Add event bus for tool execution hooks: before, after, failure, summary

Web Search & Knowledge Fusion

[ ] Add DuckDuckGo, Brave Search, You.com providers
[ ] Add semantic dedupe ranker using embedding similarity + document quality scoring
[ ] Add structured extraction layer (LLM-powered: facts, timelines, entities, relations)
[ ] Add Knowledge Graph builder from recurring searches

Notebook, Examples, Developer Experience

[ ] Add examples/api_fastapi_demo.py — production-ready served API
[ ] Add examples/cli_chat.py — terminal chat client with streaming
[ ] Add examples/langgraph_integration.ipynb — Intergrax as drop-in tool layer
[ ] Add template for building a custom agent: agent_template.py

Commercial-Level Features (Investor-Facing)

[ ] Add role and permissions layer (who can upload docs, run tools, delete memory)
[ ] Add audit trail and deterministic logs (compliance-ready mode)
[ ] Add encrypted doc store (AES-256 symmetric encryption per workspace)
[ ] Add SSO (OAuth2 + Keycloak + Supabase auth)
[ ] Add multi-tenant workspace isolation (for Mooff and SaaS launch)

Future / Optional Big Ticket Features

[ ] Build Knowledge Hub:
    Automatic ingestion + categorization + RAG indexing of user workspace files

[ ] Build Workflow Builder UI (agent orchestration visual editor)

[ ] Build Intergrax Cloud Hosting Layer (deploy → instant AI workspace + RAG + agents)