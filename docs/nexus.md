# nexus Mode – Runtime Design

## 1. Purpose

nexus Mode is a **high-level runtime layer** for Intergrax.

The goal is to provide **ChatGPT/Claude-style conversational capabilities** with:
- session awareness,
- file-aware conversations,
- dynamic RAG integration,
- tool/MCP/API orchestration,
- multi-source memory (session, user, organization),
- and configurable behavior for different apps (FastAPI, Streamlit, MCP, etc.).

Developers should prefer using this runtime over low-level components in most cases.

---

## 2. Core Principles

1. **High-level, low-friction API**  
   - One main entrypoint (e.g. `RuntimeEngine.run()`).
   - Hide complexity of RAG, tools, web search, memory, and attachments.
   - Easy to plug into FastAPI, Streamlit, FastMCP, CLI, etc.

2. **Configurable and modular**  
   - All subsystems are pluggable via adapters/config:
     - LLM adapters
     - Embedding models
     - Vector stores
     - Session stores
     - User profile stores
     - Tools / data sources / web search providers
   - Reasonable defaults, but everything can be overridden.

3. **Session-first design**  
   - Every interaction belongs to a `user_id` + `session_id`.
   - Sessions can be stored in different backends (SQLite, Postgres/MySQL, Supabase, files, etc.).
   - Sessions guarantee the ability to **resume conversations** without losing context.

4. **File-aware conversations**  
   - Users can attach files to any message (PDF, DOCX, XLSX, images, videos, code, etc.).
   - Files are:
     - stored,
     - processed via appropriate loaders,
     - embedded and indexed in vector stores,
     - linked to the session and user.
   - The runtime automatically retrieves relevant chunks from these files during later turns.

5. **Model-agnostic memory (beyond context window limits)**  
   - Intergrax works with multiple LLMs (different context sizes, providers).
   - The runtime cannot rely only on a single model’s context window.
   - Memory must combine:
     - short-term message history,
     - summarized history segments,
     - long-term facts about the user/workspace,
     - RAG over documents and databases.
   - Context selection must be dynamic and token-budget-aware.

---

## 3. Memory & Knowledge Layers

### 3.1. Session memory

- Stores the full conversation history for a given `user_id` + `session_id`.
- Not all messages are sent to the LLM on every call.
- Runtime uses strategies:
  - last N messages as short-term context,
  - plus summaries of older segments,
  - plus relevant RAG results.

### 3.2. Cross-session user profile

- Separate **User Profile Store**:
  - user preferences (language, tone, goals),
  - stable facts (role, company, tech stack, key projects).
- New sessions are **not anonymous**:
  - runtime loads the user profile and exposes it as part of system context.
- Profiles can be updated incrementally as conversations progress.

### 3.3. Organization / workspace knowledge

- Long-term, shared knowledge for a workspace or tenant:
  - company policies,
  - regulations and contracts,
  - product docs,
  - CRM/ERP data,
  - marketing materials.
- Retrieved via:
  - vector stores (RAG),
  - databases and APIs (tools),
  - web search (for external context).

### 3.4. Context selection

Runtime must **select** what goes into the LLM input:

- relevant parts of session history,
- relevant summaries,
- relevant documents (RAG),
- explicit user profile facts,
- optional web search results,
- tool outputs.

The behaviour should mimic the “infinite memory” feeling of ChatGPT, but implemented explicitly in Intergrax.

---

## 4. Supervisor & Agentic Behaviour

- nexus Mode can operate in two modes:

1. **Simple Mode**
   - Heuristic routing: memory + RAG + tools + web search.
   - Single LLM call per user request, no explicit multi-step planning.

2. **Agentic Mode (Supervisor-driven)**
   - Uses the existing Intergrax `supervisor`:
     - receives the user task,
     - decomposes it into steps,
     - calls specialized components (RAG, tools, web search, APIs),
     - synthesizes final answer.
   - Optional **human-in-the-loop**:
     - show the plan,
     - allow user to approve/modify steps,
     - execute with user oversight.

This design mirrors how ChatGPT-like systems internally chain multiple tool calls and reasoning steps.

---

## 5. Tools, Data Sources and Web Search

- Runtime should support **registration of tools and data sources** at:
  - framework level,
  - workspace/tenant level,
  - session level.
- Tools can expose:
  - internal APIs (e.g. CRM, ERP, invoices, KSeF),
  - external APIs (OpenAI, 3rd-party services),
  - database queries,
  - web search capabilities.

Key assumptions:

- Web search is treated as just another tool / provider (we already have `websearch` module).
- Tools and data sources must be discoverable and selectable by the runtime and/or supervisor.
- Configuration should define which tools are allowed in:
  - a given app (e.g. Mooff),
  - a given workspace,
  - a given session.

---

## 6. Reasoning, deduction and self-reflection

- The runtime must not behave like a naive “prompt in → answer out” wrapper.
- Instead, for complex tasks it should:
  - clarify the user’s goal,
  - identify required information (files, policies, APIs),
  - plan necessary steps (analyse code, read docs, call APIs, etc.),
  - only then generate an answer or code change proposal.

Example:
> “Modify my invoice issuing method so that every invoice is immediately sent to KSeF.”

Runtime behaviour:

1. Identify which method the user is talking about (attached file `invoices.py`).
2. Analyse current implementation of the method.
3. Retrieve or generate information on KSeF integration (docs, web search, internal tools).
4. Plan changes:
   - what needs to be added,
   - how to handle errors/responses,
   - how to keep the code maintainable.
5. Propose a concrete diff / code block with explanation.

Internally this can be implemented via:
- enriched prompts with “think step by step”,
- or explicit supervisor/agent flows (Agentic Mode).

---

## 7. Runtime Modularity & Configuration

The runtime must be **modular and configurable**:

- `RuntimeConfig` should allow:
  - selecting the LLM adapter and model,
  - choosing embedding model and vector store,
  - enabling/disabling RAG, web search, tools, long-term memory,
  - configuring token budgets and retrieval parameters,
  - selecting which tools and data sources are active for a given app/workspace.

Developers should be able to:

```python
runtime = RuntimeEngine(
    config=RuntimeConfig(
        enable_rag=True,
        enable_websearch=False,
        llm_adapter_name="openai_responses",
        embedding_model="gte-qwen2-1.5b",
        vectorstore_name="chroma",
        max_history_messages=20,
    )
)
