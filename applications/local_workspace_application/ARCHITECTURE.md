# Local Knowledge Workspace (LKW) — architecture

**Status:** Scaffold + architecture baseline (2026-06-07) — Wave 0 product reprioritization  
**Tier:** Tier-3 application (`local_workspace_application`)  
**Agents:** Tier-2 `local_indexer`, `local_search`, `local_synthesizer`  
**Canonical plan row:** [`docs/INTERGRAX_IMPLEMENTATION_PLAN.md` §6.3a LKW.*](../../docs/INTERGRAX_IMPLEMENTATION_PLAN.md#63a-business-backlog-register-consolidated)

---

## 1. Strategic purpose

**Local Knowledge Workspace (LKW)** is the first **business product environment** on Intergrax after harness platform maturity. Its role is dual:

1. **Product:** Give a user a local, safe assistant over their own files — search, gather context, produce structured outputs (reports, emails, estimates).
2. **Harness validation:** Exercise the Agent OS on a real, observable workload without external market APIs (unlike deferred K.1 Problem Radar / K.2 Vendor Discovery).

LKW validates: RAG ingest/retrieve, document parsing, shadow workspace, multi-agent orchestration, memory, policy, trace, MCP/HTTP serving, and Tier-3 composition — while surfacing platform gaps (e.g. read-only filesystem browse tools) early.

**Strategic frame:** [`docs/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../../docs/INTERGRAX_DEVELOPMENT_STRATEGY.md) — explicit product reprioritization after Appendix A sign-off.

---

## 2. Problem statement

Users store project knowledge across folders (PDF, DOCX, XLSX, TXT, email exports). They need to:

| Need | Example |
|------|---------|
| **Find** | „Znajdź dokumenty o projekcie X / rozliczeniu Y” |
| **Gather** | „Zbierz dane z folderów A i B dotyczące kosztorysu” |
| **Synthesize** | „Przygotuj mail / sprawozdanie / kosztorys wg szablonu” |
| **Safety** | Nic nie usuwać ani nie nadpisywać w oryginalnych plikach użytkownika |

LKW solves this with **read-heavy indexing + semantic retrieval + isolated write artifacts**, orchestrated by Nexus.

---

## 3. Solution overview

```text
┌─────────────────────────────────────────────────────────────────────────┐
│  User client (HTTP / MCP / future desktop shell)                        │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ POST /v1/local_workspace/run
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Tier-3  local_workspace_application                                    │
│  manifest · environment_profile · tool_wiring · factory · MCP           │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Tier-1  Nexus Agent OS                                                 │
│  Intake → Plan → Graph (index → search → synthesize) → Trace → Result   │
└───────┬─────────────────┬─────────────────┬─────────────────────────────┘
        │                 │                 │
        ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
│ local_indexer│  │ local_search │  │ local_synthesizer│
│ Tier-2       │  │ Tier-2       │  │ Tier-2           │
└──────┬───────┘  └──────┬───────┘  └────────┬─────────┘
       │                 │                    │
       └─────────────────┴────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Tier-0  Platform                                                       │
│  rag.* · document.parse · workspace.* · memory.* · cache.* · parsers    │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.1 Trust zones (filesystem safety)

| Zone | Purpose | Mechanism | Mutations |
|------|---------|-----------|-----------|
| **Read zone** | User documents (allowlisted paths) | `rag.ingest_document`, `document.parse`; future `filesystem.*` read-only | **None** on user FS |
| **Artifact zone** | Reports, drafts, exports | `workspace.*` on **shadow workspace** | Only under `INTERGRAX_SHADOW_ROOT` |
| **Sandbox zone** | Risky experiments | `sandbox.exec` (opt-in per task) | Isolated under `INTERGRAX_SANDBOX_ROOT` |

**Rule:** LKW agents MUST NOT write to user home directories. All deliverables go to shadow workspace unless the user explicitly promotes an export path in a future Wave.

---

## 4. Agent roster and capabilities

| Agent | Module | Capability | Responsibility |
|-------|--------|------------|----------------|
| **LocalIndexerAgent** | `agents/local_indexer/` | `local.workspace.index` | Discover paths (Wave 1: explicit), parse, chunk, embed, index via `rag.ingest_document` |
| **LocalSearchAgent** | `agents/local_search/` | `local.workspace.search` | Semantic + metadata-filtered retrieval via `rag.retrieve`; rank and package evidence |
| **LocalSynthesizerAgent** | `agents/local_synthesizer/` | `local.workspace.synthesize` | LLM synthesis from retrieved context; write artifacts to shadow workspace |

**Pipeline capability (graph-level):** `local.workspace.pipeline` — multi-step intent routing index → search → synthesize (Wave 2). Documented here; wired via Nexus `AgentGraph` / delegation like `research.pipeline`.

Agent architecture docs:

- [`agents/local_indexer/ARCHITECTURE.md`](../../agents/local_indexer/ARCHITECTURE.md)
- [`agents/local_search/ARCHITECTURE.md`](../../agents/local_search/ARCHITECTURE.md)
- [`agents/local_synthesizer/ARCHITECTURE.md`](../../agents/local_synthesizer/ARCHITECTURE.md)

---

## 5. Enabled platform capabilities

### 5.1 Tools (`host/tool_wiring.py`)

| tool_id | Role in LKW |
|---------|-------------|
| `rag.ingest_document` | Index local file into vector store |
| `rag.retrieve` | Semantic search over indexed chunks |
| `rag.list_collections` | Inspect index partitions |
| `document.parse` | Parse single file to text fragments (pre-ingest or ad-hoc) |
| `workspace.read_file` | Read artifact from shadow workspace |
| `workspace.write_file` | Write report/draft to shadow workspace |
| `workspace.list_files` | List shadow artifacts |
| `workspace.snapshot` | Point-in-time artifact snapshot |
| `memory.read` / `memory.write` / `memory.list_keys` | Task-scoped working memory |
| `cache.get` / `cache.set` | Dedup parse/embedding work |

**Explicitly disabled in baseline:** `websearch.*` — LKW is local-first.

**Planned Tier-0 (Wave 3):** `filesystem.list`, `filesystem.read_text`, `filesystem.glob` with `INTERGRAX_ALLOWED_READ_ROOTS`.

### 5.2 Integrations (`IntegrationProfile.legal_product()`)

| Slot | Provider | Notes |
|------|----------|-------|
| `relational_store` | SQLite | Trace, session, task memory |
| `vector_store` | In-memory (dev); Chroma/Qdrant (Wave 2+) | Per-user collection naming TBD |
| `document_parser` | Docling | PDF/DOCX/XLSX via `ParserPipeline` fallback |

Override via `INTERGRAX_INTEGRATION_PROFILE_JSON` for Chroma local persistence.

### 5.3 Environment profile

- `ApplicationEnvironmentProfile.product_defaults(profile_id="local_workspace.product")`
- `ContextProfile(enable_rag=True, enable_websearch=False)`
- `with_harness_memory()` — STM/LTM hooks for long sessions
- OTLP observability optional (`LOCAL_WORKSPACE_*` + `IntegrationProfile` OTEL slot)

See [`host/environment_profile.py`](host/environment_profile.py).

---

## 6. User-visible capabilities (product)

### 6.1 Wave 1 (current scaffold target)

- Accept **explicit file paths** or folder path list in task message/metadata
- Ingest → retrieve → answer or short summary
- Output written to **shadow workspace**; `shadow_workspace_id` in result metadata
- HTTP: `POST /v1/local_workspace/run`
- MCP: `list_agents`, `run_agent`, catalog tool describe

### 6.2 Wave 2 — multi-agent pipeline

- Single request triggers graph: indexer (if stale) → search → synthesizer
- `AgentGraph` on `ApplicationEnvironmentProfile.graph_spec`
- Delegation per [`docs/NEXUS_EXECUTION_FLOW_REFERENCE.md`](../../docs/NEXUS_EXECUTION_FLOW_REFERENCE.md)

### 6.3 Wave 3 — filesystem browse

- Allowlisted roots; safe list/glob/read tools
- Policy: read-only on user FS

### 6.4 Wave 4 — background indexing

- Queue-driven ingest (`message_bus.*` + worker)
- Incremental re-index on file change (watcher — new Tier-0 or Tier-3 daemon)

### 6.5 Wave 5 — desktop shell (out of harness scope)

- Tray app / file picker calling HTTP or MCP — separate client repo or Tier-3 extension

---

## 7. Request and data flows

### 7.1 Index flow

```text
Task(capability=local.workspace.index, metadata={source_paths: [...]})
  → LocalIndexerAgent UAEP steps
  → invoke rag.ingest_document per path
  → ParserPipeline + chunk + embed + vector store
  → StepOutput(metadata: {num_chunks, collection_id, parser_trace})
```

### 7.2 Search flow

```text
Task(capability=local.workspace.search, message="znajdź dokumenty o projekcie X")
  → LocalSearchAgent
  → rag.retrieve(query, metadata filters)
  → Package evidence chunks + citations (path, page, chunk_id)
```

### 7.3 Synthesize flow

```text
Task(capability=local.workspace.synthesize, metadata={template: "email"|"report"|...})
  → LocalSynthesizerAgent
  → LLM with retrieved context (from graph handoff or prior step)
  → workspace.write_file("draft.md", content)
  → metadata: {shadow_workspace_id, artifact_paths}
```

### 7.4 Pipeline flow (Wave 2)

```text
Task(capability=local.workspace.pipeline, intent=local_workspace_full)
  → Nexus graph: DELEGATES_TO indexer? → search → synthesizer
  → SharedTaskContext carries evidence + artifact refs
```

---

## 8. Tier-3 composition map

| File | Role |
|------|------|
| [`manifest.py`](manifest.py) | Roster, capabilities, `LOCAL_WORKSPACE_APPLICATION_MANIFEST` |
| [`host/environment_profile.py`](host/environment_profile.py) | RAG-on, websearch-off product profile |
| [`host/tool_wiring.py`](host/tool_wiring.py) | LKW tool allowlist |
| [`host/settings.py`](host/settings.py) | `LOCAL_WORKSPACE_*` env, RAG flags |
| [`host/wiring.py`](host/wiring.py) | Registry + `wire_application_environment` |
| [`host/factory.py`](host/factory.py) | FastAPI Core + MCP |
| [`serving/fastapi_router.py`](serving/fastapi_router.py) | `/run`, `/agents` |
| [`mcp/server.py`](mcp/server.py) | FastMCP mount |

**No agent logic in Tier-3** — only wiring. Domain steps live in `agents/*/steps/`.

Authoring guide: [`docs/AGENT_CREATION_GUIDE.md`](../../docs/AGENT_CREATION_GUIDE.md) · Tier-3: [`applications/USAGE.md`](../USAGE.md).

---

## 9. Configuration

### 9.1 Environment variables (`.env.example`)

| Variable | Default | Purpose |
|----------|---------|---------|
| `LOCAL_WORKSPACE_BACKEND_PORT` | `8020` | HTTP port |
| `LOCAL_WORKSPACE_DEFAULT_AGENT_ID` | `local_search` | Default roster agent |
| `LOCAL_WORKSPACE_ENABLE_RAG` | `true` | Enable `rag.retrieve` |
| `LOCAL_WORKSPACE_ENABLE_RAG_INGEST` | `true` | Enable `rag.ingest_document` |
| `INTERGRAX_SHADOW_ROOT` | `build/shadow_workspaces` | Artifact isolation root |
| `INTERGRAX_ALLOWED_READ_ROOTS` | *(planned Wave 3)* | Comma-separated read allowlist |

### 9.2 Task metadata conventions (Wave 1+)

```python
Task(
    metadata={
        "shadow_workspace": True,
        "source_paths": ["D:/Docs/project_a/report.pdf"],
        "collection_id": "user_u1_workspace",
        "synthesis_template": "email",
    }
)
```

---

## 10. Security and governance

- **Read-only user FS** in Waves 1–2 (ingest reads; no writes)
- **Shadow workspace** mandatory for synthesizer outputs
- **HITL** optional for sensitive exports (`REQUEST_HUMAN`) — [`docs/AGENT_CREATION_GUIDE.md` Appendix A](../../docs/AGENT_CREATION_GUIDE.md#appendix-a--human-in-the-loop)
- **Cost governance:** `CostProfile` on environment; embedding batch limits per ingest job
- **Trace:** all tool calls via Nexus trace DB — debug with `intergrax.debug` CLI

---

## 11. Observability and verification

```bash
# Host smoke
uv run pytest applications/local_workspace_application/local_workspace_application_tests -q

# Agent smoke
uv run pytest agents/local_indexer/tests agents/local_search/tests agents/local_synthesizer/tests -q

# Run host
uv run uvicorn local_workspace_application.host.main:app --host 127.0.0.1 --port 8020
```

Deploy triad: `docker/`, `BUILD_AND_DEPLOY.md` — gate `test_application_deploy_triad.py`.

---

## 12. Implementation waves (plan traceability)

| ID | Deliverable | Status |
|----|-------------|--------|
| **LKW.0** | Scaffold agents + application + architecture doc | **Done** (this document) |
| **LKW.1** | Wave 1 — single-path ingest + search smoke | Planned |
| **LKW.2** | Multi-agent graph + pipeline capability | Planned |
| **LKW.3** | Tier-0 `filesystem.*` read tools + allowlist policy | Planned |
| **LKW.4** | Background ingest queue + incremental index | Planned |
| **LKW.5** | Desktop client / file picker | Deferred product |

Registered in [`docs/INTERGRAX_IMPLEMENTATION_PLAN.md` §6.3a](../../docs/INTERGRAX_IMPLEMENTATION_PLAN.md#63a-business-backlog-register-consolidated).

---

## 13. Known platform gaps (honest audit)

| Gap | Impact | Mitigation (Wave) |
|-----|--------|-------------------|
| No `filesystem.list/glob` tools | User must pass explicit paths | LKW.1 manual paths; LKW.3 tools |
| No file watcher | No auto re-index | LKW.4 worker |
| In-memory vector store default | Index lost on restart | Chroma + `INTEGRATION_PROFILE_JSON` |
| Windows path / OneDrive edge cases | Parser failures | Test matrix in LKW.1 acceptance |

These gaps are **expected** — LKW exists to discover and close them without Nexus forks.

---

## 14. References

| Topic | Document |
|-------|----------|
| Agent workflow | [`docs/AGENT_CREATION_GUIDE.md`](../../docs/AGENT_CREATION_GUIDE.md) |
| Tools catalog | [`docs/TOOLS.md`](../../docs/TOOLS.md) |
| RAG control plane | [`docs/AGENT_CREATION_GUIDE.md` Appendix K](../../docs/AGENT_CREATION_GUIDE.md#appendix-k--integration--rag-control-plane) |
| Shadow workspace | [`docs/AGENT_CREATION_GUIDE.md` Appendix B](../../docs/AGENT_CREATION_GUIDE.md#appendix-b--shadow-workspace-and-sandbox) |
| Multi-agent graphs | [`docs/AGENT_CREATION_GUIDE.md` Appendix C](../../docs/AGENT_CREATION_GUIDE.md#appendix-c--multi-agent-graphs) |
| Nexus execution flow | [`docs/NEXUS_EXECUTION_FLOW_REFERENCE.md`](../../docs/NEXUS_EXECUTION_FLOW_REFERENCE.md) |
| Implementation plan | [`docs/INTERGRAX_IMPLEMENTATION_PLAN.md`](../../docs/INTERGRAX_IMPLEMENTATION_PLAN.md) |
| Quickstart | [`README.md`](README.md) · [`BUILD_AND_DEPLOY.md`](BUILD_AND_DEPLOY.md) |
